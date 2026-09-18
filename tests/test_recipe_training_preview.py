"""Recipe training previews distinguish selected source lineage from full ID masks."""
import base64
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_study_preview import render_training_target
from fakect_preview_report import write_preview_report

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jH9sAAAAASUVORK5CYII=')


class RecipeTrainingTargetTests(unittest.TestCase):
    def test_recipe_uses_original_roi_selection_while_legacy_keeps_full_source_ids(self):
        from matplotlib.figure import Figure
        shape = (7, 7, 7)
        labels = np.ones(shape, dtype=np.int32)
        labels[:, 3, 3] = 20
        labels[:, 1, 1] = 21
        candidates = np.isin(labels, (20, 21))
        roi = np.zeros(shape, dtype=bool)
        roi[2:5, 2:5, 2:5] = True
        selected = candidates & roi
        arrays = {'act': labels, 'roi': roi, 'selected': selected, 'candidates': candidates,
                  'attenuation_cm_inverse': np.where(candidates, .25, .15)}
        saved = deepcopy(arrays)
        resolved = {'crop_low_ijk': (0, 0, 0), 'crop_high_ijk_exclusive': (7, 7, 7),
                    'slice_ijk': (3, 3, 3), 'spacing_ijk_mm': (1, 1, 1)}
        config = {'study': {'schema_version': 'fakect.recipe-study/1', 'name': 'ROI lineage'},
                  'selection': {'tissue': 'artery', 'source_ids': []},
                  'train': {'target_scope': 'selected_lineage'},
                  'preview': {'overlay_opacity': .22}, 'roi': {'coordinate_reviewed': True}}
        captured = {}
        save = Figure.savefig

        def record(figure, *args, **kwargs):
            captured['title'] = figure._suptitle.get_text()
            captured['caption'] = figure.texts[-1].get_text()
            captured['coronal_target'] = np.asarray(figure.axes[4].images[0].get_array()).copy()
            return save(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, 'savefig', record):
                metadata = render_training_target(arrays, resolved, config, directory)
            self.assertTrue((Path(directory)/'training-target.png').read_bytes().startswith(b'\x89PNG'))
            self.assertEqual(metadata['target_scope'], 'selected_lineage')
            self.assertNotIn('target_source_ids', metadata)
            self.assertEqual(metadata['target_voxels_in_crop'], 3)
            self.assertEqual(metadata['target_voxels_outside_roi'], 0)
            self.assertIn('grown descendants', metadata['target_definition'])
            self.assertIn('Unselected anatomy stays negative', metadata['scope'])
            self.assertTrue(metadata['source_image_only'])
            np.testing.assert_array_equal(captured['coronal_target'], selected[:, 3, :])
            self.assertIn('ROI-selected artery lineage', captured['title'])
            self.assertNotIn('Target source IDs', captured['title'])
            self.assertIn('not a complete anatomical aorta annotation', captured['caption'])

            config['study']['schema_version'] = 'fakect.study/1'
            config['train'] = {'target_source_ids': (20,)}
            legacy = render_training_target(arrays, resolved, config, directory)
            self.assertEqual(legacy['target_source_ids'], [20])
            self.assertEqual(legacy['target_voxels_in_crop'], 7)
            self.assertEqual(legacy['target_voxels_outside_roi'], 4)
            self.assertNotIn('target_scope', legacy)
        for key, values in saved.items():
            np.testing.assert_array_equal(arrays[key], values)


class RecipeTrainingReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output/'training-target.png').write_bytes(PNG)
        self.report = {
            'config': {'study': {'schema_version': 'fakect.recipe-study/1', 'name': 'Lineage cohort'},
                       'selection': {'tissue': 'artery', 'source_ids': []},
                       'recipe': {'steps': ['grow', 'narrow']},
                       'edits': {'grow': {'operation': 'dilation', 'roi': 'main', 'path_percent': [15, 55],
                                          'distance_mm': 9, 'iterations': 1},
                                 'narrow': {'operation': 'erosion', 'roi': 'main', 'path_percent': [65, 75],
                                            'distance_mm': 10, 'iterations': 1}}},
            'training_plan': {'schema_version': 'fakect.recipe-study-plan/1',
                              'target_scope': 'selected_lineage', 'target_definition': 'Original selected tissue and its descendants.',
                              'target_preview': {'target_voxels_in_crop': 100, 'target_voxels_in_roi': 100,
                                                 'target_voxels_outside_roi': 0, 'target_components_6': 2,
                                                 'scope': 'ROI-defined source mask'},
                              'recipe_steps': ['grow', 'narrow'],
                              'variant_count': 2, 'sweeps': {'grow': {'distance_mm': [5, 9]},
                                                           'narrow': {'distance_mm': [4, 10]}},
                              'variants': [{'variant_id': 'baseline', 'operation': 'none', 'parameters': {}, 'split': 'train'},
                                           {'variant_id': 'recipe_001', 'operation': 'recipe',
                                            'parameters': {'grow': {'distance_mm': 5}, 'narrow': {'distance_mm': 4}},
                                            'split': 'test'}],
                              'validation_scope': 'Metadata only; native preflight still required.',
                              'image_warning': 'Attenuation copy proxy', 'split_warning': 'One anatomy family',
                              'model': {'architecture': 'unet2d'},
                              'dataset_directory': '/future/pairs', 'preflight_directory': '/future/preflight',
                              'model_directory': '/future/model'}}

    def write(self):
        metadata = write_preview_report(self.output, self.report, '[train]\ntarget_scope = selected_lineage\n')
        return metadata, (self.output/'report.html').read_text()

    def test_order_sweeps_variant_parameters_and_native_preflight_are_explicit(self):
        metadata, document = self.write()
        self.assertIn('training-target.png', metadata['embedded_assets'])
        self.assertIn('Binary target: selected lineage', document)
        self.assertIn('Unselected same-category anatomy stays negative', document)
        self.assertIn('not a complete or anatomically exclusive aorta annotation', document)
        self.assertIn('<td>1</td><td>grow</td><td>dilation</td>', document)
        self.assertIn('<td>2</td><td>narrow</td><td>erosion</td>', document)
        self.assertIn('<td>grow</td><td>distance_mm</td><td>5, 9</td>', document)
        self.assertIn('<code>grow.distance_mm</code> = 5<br><code>narrow.distance_mm</code> = 4', document)
        self.assertIn('Original phantom; no edits', document)
        self.assertIn('this preview has not executed every listed variant', document)
        self.assertIn('stage = preflight', document)
        self.assertIn('baseline-equivalent masks stay in training', document)
        self.assertIn('[edit.NAME]', document)
        self.assertIn('[sweep.NAME]', document)
        self.assertNotIn('original IDs <code>Not recorded', document)
        self.assertNotIn('<th scope="col">Shape k</th>', document)
        self.assertNotIn('Unchanged target outside ROI', document)
        self.assertNotIn('[edit] operation = erosion', document)

    def test_recipe_plan_fields_cannot_insert_markup(self):
        payload = '<script>window.BAD=true</script><img src=x onerror=bad()>'
        plan = self.report['training_plan']
        plan['target_definition'] = payload
        plan['sweeps'][payload] = {payload: [payload]}
        plan['variants'][1]['parameters'] = {payload: {payload: payload}}
        plan['validation_scope'] = payload
        _, document = self.write()
        self.assertNotIn(payload, document)
        self.assertIn('&lt;script&gt;window.BAD=true&lt;/script&gt;', document)

        class Scripts(HTMLParser):
            count = 0
            def handle_starttag(self, tag, attrs):
                if tag == 'script':
                    self.count += 1

        parsed = Scripts()
        parsed.feed(document)
        self.assertEqual(parsed.count, 1)


if __name__ == '__main__':
    unittest.main()
