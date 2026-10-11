"""Exact physical surfaces, faithful change definitions and offline packaging."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import fakect_two_stage_3d as viewer


class TwoStage3DTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.before = np.zeros((5, 7, 9), dtype=bool)
        self.before[1:4, 2:5, 3:6] = True
        self.after = self.before.copy()
        self.after[2, 2, 3] = False
        self.after[2, 4, 6] = True
        self.path = self.root/'prediction.npz'

    def prediction(self, chained=True, threshold=.7):
        # Probability .6 is intentionally below a nondefault reference threshold.
        reference = self.before.astype(np.float32)*.9
        reference[2, 4, 6] = .6
        masks = dict(before=self.before, after=self.after,
                     added=self.after & ~self.before, removed=self.before & ~self.after,
                     oracle_reference_probability=reference,
                     oracle_added_mask=self.after & ~(reference >= threshold),
                     oracle_removed_mask=(reference >= threshold) & ~self.after,
                     sample_id=np.asarray('case:example'), spacing_ijk_mm=np.array([.5, .75, 1.]))
        if chained:
            segmentation = self.after.copy(); segmentation[0, 0, 0] = True; segmentation[2, 3, 3] = False
            masks.update(stage1_aorta_probability=segmentation.astype(np.float32),
                         chained_reference_probability=reference,
                         chained_added_mask=segmentation & ~(reference >= threshold),
                         chained_removed_mask=(reference >= threshold) & ~segmentation)
        np.savez_compressed(self.path, **masks)
        return {'sample_id': 'case:example', 'is_control': False,
                'prediction_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()}

    def test_exact_surface_coordinates_and_input_unchanged(self):
        original = self.before.copy()
        scene = viewer.build_scene({'before': self.before}, [.5, .75, 1.], sample_id='test')
        self.assertTrue(np.array_equal(original, self.before))
        surface = scene['surfaces']['before']
        xyz = np.frombuffer(base64.b64decode(surface['xyz']), '<f4').reshape(-1, 3)
        np.testing.assert_allclose(xyz.min(axis=0), [1.5, 1.5, 1.])
        np.testing.assert_allclose(xyz.max(axis=0), [3., 3.75, 4.])
        self.assertEqual(surface['voxels'], 27)
        self.assertEqual(scene['bounds_ijk_mm'], [[0, 4.5], [0, 5.25], [0, 5.]])
        self.assertEqual(scene['triangles'], surface['triangles'])

    def test_derived_binary_changes_and_segmentation_errors(self):
        row = self.prediction()
        scene = viewer._prediction_scene(self.path, row, .7, .5)
        fields = scene['surfaces']
        self.assertEqual(fields['oracle_added']['voxels'], 1)
        self.assertEqual(fields['oracle_removed']['voxels'], 1)
        self.assertEqual(fields['segmentation_fp']['voxels'], 1)
        self.assertEqual(fields['segmentation_fn']['voxels'], 1)
        self.assertEqual([mode['key'] for mode in scene['modes']], ['oracle', 'chained', 'segmentation'])
        self.assertIn('include segmentation errors', scene['modes'][1]['explanation'])

    def test_oracle_only_prediction_needs_no_model1(self):
        row = self.prediction(chained=False)
        scene = viewer._prediction_scene(self.path, row, .7, .5)
        self.assertEqual([mode['key'] for mode in scene['modes']], ['oracle'])
        self.assertNotIn('segmentation', scene['surfaces'])

    def test_hash_and_identity_mismatch_rejected(self):
        row = self.prediction()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            viewer._prediction_scene(self.path, dict(row, prediction_sha256='0'*64), .7, .5)
        with self.assertRaisesRegex(ValueError, 'another sample'):
            viewer._prediction_scene(self.path, dict(row, sample_id='other'), .7, .5)

    def test_binary_and_probability_validation(self):
        with self.assertRaisesRegex(ValueError, 'binary'):
            viewer.build_scene({'bad': self.before.astype(float)*.5}, [1, 1, 1], sample_id='bad')
        row = self.prediction()
        # Different threshold conflicts with the stored binary labels.
        with self.assertRaisesRegex(ValueError, 'binary changes disagree'):
            viewer._prediction_scene(self.path, row, .5, .5)

    def test_oversized_surface_is_explicitly_omitted_without_pooling(self):
        with patch.object(viewer, 'MAX_SURFACE_TRIANGLES', 10):
            scene = viewer.build_scene({'before': self.before}, [1, 1, 1], sample_id='big')
        self.assertTrue(scene['surfaces']['before']['omitted'])
        self.assertEqual(scene['surfaces']['before']['voxels'], 27)
        self.assertEqual(scene['triangles'], 0)
        self.assertIn('display budget', scene['warnings'][0])

    def test_empty_mask_retained_with_zero_counts(self):
        scene = viewer.build_scene({'empty': np.zeros_like(self.before)}, [1, 1, 1], sample_id='empty')
        entry = scene['surfaces']['empty']
        self.assertEqual(entry['voxels'], 0)
        self.assertEqual(entry['triangles'], 0)
        self.assertFalse(entry['omitted'])

    def test_file_protocol_packaging_and_safe_sample_text(self):
        scene = viewer.build_scene({'before': self.before}, [1, 1, 1], sample_id='</script><script>bad()</script>')
        result = viewer.render_viewer(self.root/'view', [scene])
        output = Path(result['viewer']).parent
        self.assertEqual(result['scenarios'], 1)
        self.assertTrue((output/'plotly.min.js').is_file())
        index = (output/'index.html').read_text()
        self.assertIn('Reset full-crop view', index)
        self.assertIn('copy this entire viewer directory and open index.html', index)
        self.assertIn('also copy its parent report', index)
        self.assertNotIn('report-3d', index)
        script = (output/'viewer.js').read_text()
        self.assertNotIn('fetch(', script)
        self.assertNotIn('report-3d', script)
        self.assertIn('Copy this entire viewer directory', script)
        self.assertIn("document.createElement('script')", script)
        self.assertIn("Plotly.purge(plot)", script)
        self.assertIn('scene.bounds_ijk_mm', script)
        self.assertIn('15%, 45%, or 80%', index)
        payload = (output/'scene-0000.js').read_text()
        self.assertNotIn('</script>', payload)
        decoded = json.loads(payload.removeprefix('window.fakeCTReceiveScene(').removesuffix(');\n'))
        self.assertEqual(decoded['sample_id'], scene['sample_id'])
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['scenes'][0]['sha256'], hashlib.sha256(payload.encode()).hexdigest())
        self.assertIn('no pooling', manifest['sampling'])

    def test_native_atlas_viewer_overrides_scope_and_hides_model_metrics(self):
        scene = viewer.build_scene({'aorta': self.before}, [.5, .75, 1.], sample_id='atlas:260602')
        coordinates = 'Native voxel geometry, millimeters from the lower crop face; orientation unverified.'
        note = 'Provisional T12-midpoint/T1-upper thoracic selection. No model predictions. <review needed>'
        result = viewer.render_viewer(self.root/'atlas', [scene], title='Atlas anatomy review',
                                      coordinate_description=coordinates, footer_note=note, show_metrics=False)
        output = Path(result['viewer']).parent
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['display'], dict(coordinate_description=coordinates, footer_note=note, show_metrics=False))
        self.assertEqual(manifest['coordinates'], coordinates)
        self.assertIsNone(manifest['scenes'][0]['is_control'])
        document = (output/'index.html').read_text()
        self.assertNotIn('ROI-selected', document)
        self.assertNotIn('saved model grid', document)
        self.assertIn('Provisional T12-midpoint/T1-upper', document)
        self.assertIn('&lt;review needed&gt;', document)
        self.assertIn('Native voxel geometry', document)
        script = (output/'viewer.js').read_text()
        self.assertIn("document.getElementById('metrics').hidden=!shown", script)
        self.assertIn("text('metrics','');return;", script)

    def test_scene_display_overrides_and_default_model_behavior(self):
        default = viewer.build_scene({'aorta': self.before}, [1, 1, 1], sample_id='default')
        custom = viewer.build_scene({'aorta': self.before}, [1, 1, 1], sample_id='native',
                                   coordinate_description='Native crop millimeters',
                                   footer_note='Native atlas labels; no predictions.', show_metrics=False)
        self.assertEqual(default['display'], {})
        self.assertEqual(custom['display']['show_metrics'], False)
        self.assertEqual(custom['display']['coordinate_description'], 'Native crop millimeters')
        result = viewer.render_viewer(self.root/'mixed', [default, custom])
        output = Path(result['viewer']).parent
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertTrue(manifest['display']['show_metrics'])
        self.assertEqual(manifest['display']['footer_note'], viewer.DEFAULT_FOOTER)
        payload = (output/'scene-0001.js').read_text()
        decoded = json.loads(payload.removeprefix('window.fakeCTReceiveScene(').removesuffix(');\n'))
        self.assertEqual(decoded['display'], custom['display'])
        script = (output/'viewer.js').read_text()
        self.assertIn('{...manifest.display,...(scene.display||{})}', script)

    def test_display_options_reject_ambiguous_types(self):
        for kwargs in (dict(show_metrics='false'), dict(coordinate_description=1), dict(footer_note=['text'])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                viewer.build_scene({'aorta': self.before}, [1, 1, 1], sample_id='bad', **kwargs)


if __name__ == '__main__':
    unittest.main()
