"""Portable centerline views preserve geometry and suppress untrusted arrows."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_frame_preview import render_centerline_frames, _arrow_lines, _target_context
from fakect_preview_report import _AssetReferences


class FramePreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        angle = np.linspace(0, np.pi/2, 31)
        xyz = np.column_stack((10+6*np.cos(angle), 10+6*np.sin(angle), np.full(len(angle), 6)))
        tangent = np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(len(angle))))
        normal = np.column_stack((-np.cos(angle), -np.sin(angle), np.zeros(len(angle))))
        reliable = np.ones(len(angle), dtype=bool)
        reliable[[0, 15, 30]] = False
        self.frame = {'xyz_mm': xyz, 'tangent': tangent, 'normal': normal,
                      'binormal': np.cross(tangent, normal), 'reference_normal': normal.copy(),
                      'reference_binormal': np.cross(tangent, normal),
                      'parent_fraction': angle/(np.pi/2), 'parent_distance_mm': angle*6,
                      'spline_distance_mm': angle*6, 'curvature_per_mm': np.full(len(angle), 1/6),
                      'reliable': reliable, 'normal_status': ['reliable' if value else 'unreliable' for value in reliable],
                      'endpoint': np.isin(np.arange(len(angle)), [0, 30]),
                      'inflection_candidate': np.zeros(len(angle), dtype=bool),
                      'metadata': {'fit_rms_displacement_mm': .25, 'min_curvature_per_mm': .002}}
        spacing = np.array([1, 1, 2])
        self.config = {'roi': {'shape': 'tube', 'center_ijk': (xyz[[0, 8, 15, 23, 30]]/spacing).tolist(),
                               'radius_mm': [2.5]*5}, 'preview': {'volume_stride': 2},
                       'centerline': {'smoothing_mm': 1, 'sample_step_mm': 1, 'min_curvature_per_mm': .002}}
        k, j, i = np.indices((8, 24, 24))
        self.arrays = {'candidates': (np.abs(np.sqrt((i-10)**2+(j-10)**2)-6) < 1.5) &
                                     (np.abs(k-3) < 2) & (i >= 10) & (j >= 10)}
        self.resolved = {'crop_low_ijk': (0, 0, 0), 'spacing_ijk_mm': spacing.tolist()}

    def test_portable_views_preserve_frame_and_only_show_trusted_normal_arrows(self):
        original = copy.deepcopy(self.frame)
        target = self.arrays['candidates'].copy()
        payload = '<script>window.BAD=true</script>'
        self.frame['normal_status'][1] = payload
        result, = render_centerline_frames({'main': self.frame}, self.arrays, self.resolved, self.config, self.output)
        self.assertEqual(result['parent_roi'], 'main')
        self.assertEqual(result['sample_count'], 31)
        self.assertEqual(result['unreliable_samples'], 3)
        for key in ('inner', 'outer', 'binormal'):
            indices = result['arrow_sample_indices'][key]
            self.assertTrue(indices)
            self.assertTrue(self.frame['reliable'][indices].all())
        self.assertEqual(result['arrow_sample_indices']['inner'], result['arrow_sample_indices']['outer'])
        self.assertIn(0, result['arrow_sample_indices']['tangent'])
        self.assertIn(30, result['arrow_sample_indices']['tangent'])
        saved = json.loads((self.output/result['frame_artifact']).read_text())
        for key in ('xyz_mm', 'tangent', 'normal', 'binormal', 'reliable', 'reference_normal'):
            np.testing.assert_array_equal(saved['frame'][key], original[key])
            np.testing.assert_array_equal(self.frame[key], original[key])
        np.testing.assert_array_equal(self.arrays['candidates'], target)
        np.testing.assert_array_equal(np.asarray(saved['original_roi']['center_ijk'])*[1, 1, 2],
                                      original['xyz_mm'][[0, 8, 15, 23, 30]])
        self.assertFalse(saved['source_arrays_modified'])
        for key, artifact in result['artifacts'].items():
            raw = (self.output/artifact['path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), artifact['sha256'])
            self.assertEqual(len(raw), artifact['bytes'])
            if key in ('figure', 'curvature_figure'):
                self.assertTrue(raw.startswith(b'\x89PNG\r\n'))
        document = (self.output/result['html']).read_text()
        parser = _AssetReferences()
        parser.feed(document)
        self.assertEqual(parser.references, [])
        self.assertNotIn(payload, document)
        self.assertIn('Plotly.newPlot', document)
        self.assertIn('the smoothed spline is a direction reference', document)
        self.assertIn('this is not the edit displacement', document)
        self.assertTrue('Curvature (1\\u002fmm)' in document, 'Interactive curvature axis retains its physical unit')
        with self.assertRaises(FileExistsError):
            render_centerline_frames({'main': self.frame}, self.arrays, self.resolved, self.config, self.output)

    def test_unreliable_reference_never_gets_inner_outer_or_binormal_arrows(self):
        self.frame['reliable'][:] = False
        self.frame['normal_status'] = ['straight; transported display frame']*31
        self.frame['curvature_per_mm'][:] = 0
        result, = render_centerline_frames({'main': self.frame}, self.arrays, self.resolved, self.config, self.output)
        for key in ('inner', 'outer', 'binormal'):
            self.assertEqual(result['arrow_sample_indices'][key], [])
        self.assertTrue(result['arrow_sample_indices']['tangent'])
        self.assertEqual(result['reliable_samples'], 0)
        self.assertTrue(any('31 of 31' in warning for warning in result['warnings']))
        document = (self.output/result['html']).read_text()
        self.assertNotIn('Inner N (trusted curvature)', document)
        self.assertNotIn('Outer -N (trusted curvature)', document)

    def test_invalid_frames_fail_before_writing_artifacts(self):
        for key, value in (('xyz_mm', np.ones((4, 2))), ('normal', np.full((31, 3), np.nan)),
                           ('reliable', np.ones(31, dtype=int)),
                           ('parent_fraction', np.linspace(1, 0, 31))):
            with self.subTest(key=key):
                frame = dict(self.frame, **{key: value})
                with self.assertRaises(ValueError):
                    render_centerline_frames({'main': frame}, self.arrays, self.resolved, self.config, self.output)
                self.assertEqual(list(self.output.iterdir()), [])

    def test_arrow_shafts_keep_signed_direction_and_context_sampling_is_bounded(self):
        points = np.array([[10, 20, 30], [20, 30, 40]])
        lines = _arrow_lines(points, [[1, 0, 0], [-1, 0, 0]], 4)
        np.testing.assert_array_equal(np.asarray(lines[:2], dtype=float), [[10, 20, 30], [14, 20, 30]])
        np.testing.assert_array_equal(np.asarray(lines[9:11], dtype=float), [[20, 30, 40], [16, 30, 40]])
        self.config['preview']['volume_stride'] = 1
        vertices, faces, stride, warnings = _target_context({'candidates': np.ones((64, 64, 64), dtype=bool)},
                                                            self.resolved, self.config)
        self.assertGreater(stride, 1)
        self.assertLessEqual(len(faces), 100_000)
        self.assertTrue(warnings)
        np.testing.assert_allclose(vertices.min(axis=0), [-.5, -.5, -1])
        np.testing.assert_allclose(vertices.max(axis=0), [63.5, 63.5, 127])


if __name__ == '__main__':
    unittest.main()
