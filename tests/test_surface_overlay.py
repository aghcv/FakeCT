"""Native edit preservation, registration, and controls for surface overlays."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_surface_overlay import render_surface_overlay


class SurfaceOverlayTests(unittest.TestCase):
    def setUp(self):
        self.before = np.zeros((6, 8, 10), dtype=bool)
        self.before[1:5, 2:6, 3:7] = True
        self.after = self.before.copy()
        self.after[2, 3, 7] = True
        self.after[2, 3, 3] = False
        self.kwargs = dict(crop_origin_ijk=(10, 20, 30), spacing_ijk_mm=(1, 2, 3), volume_stride=3)

    def render(self, output, before=None, after=None, **kwargs):
        return render_surface_overlay(self.before if before is None else before,
                                      self.after if after is None else after,
                                      output_dir=output, **(self.kwargs | kwargs))

    def test_registered_native_surfaces_preserve_one_voxel_edits_and_inputs(self):
        original_before, original_after = self.before.copy(), self.after.copy()
        self.before.setflags(write=False)
        self.after.setflags(write=False)
        with tempfile.TemporaryDirectory() as output:
            result = self.render(output)
            self.assertEqual(result['requested_volume_stride'], 3)
            self.assertEqual(result['volume_stride'], 1)
            self.assertEqual(result['counts'], {'before_voxels': 64, 'after_voxels': 64,
                             'added_voxels': 1, 'removed_voxels': 1,
                             'unchanged_target_voxels': 63, 'net_delta_voxels': 0})
            before, after = result['surfaces']
            self.assertNotEqual(before['mask_sha256'], after['mask_sha256'])
            # Coordinates are i,j,k (not the array's k,j,i), with anisotropic spacing.
            np.testing.assert_allclose(before['bounds_ijk_relative_mm'],
                                       [[12.5, 16.5], [43., 51.], [91.5, 103.5]])
            np.testing.assert_allclose(after['bounds_ijk_relative_mm'],
                                       [[12.5, 17.5], [43., 51.], [91.5, 103.5]])
            self.assertGreater(before['surface_triangles'], 0)
            self.assertGreater(after['surface_triangles'], before['surface_triangles'])
            self.assertEqual(result, json.loads(Path(result['metadata_path']).read_text()))
            self.assertGreater(Path(result['png_path']).stat().st_size, 10_000)
            from PIL import Image
            with Image.open(result['png_path']) as preview:
                self.assertEqual(preview.size, (1920, 1600))
            with self.assertRaises(FileExistsError):
                self.render(output)
        np.testing.assert_array_equal(self.before, original_before)
        np.testing.assert_array_equal(self.after, original_after)

    def test_standalone_overlay_has_independent_visibility_and_three_opacity_choices(self):
        with tempfile.TemporaryDirectory() as output:
            result = self.render(output)
            html = Path(result['html_path']).read_text()
            self.assertIn('plotly.js', html)
            self.assertNotIn('<script src=', html)
            plot_start = re.search(r'Plotly\.newPlot\(\s*"surface-overlay-plot"\s*,\s*', html)
            self.assertIsNotNone(plot_start)
            traces, _ = json.JSONDecoder().raw_decode(html[plot_start.end():])
            self.assertEqual([trace['type'] for trace in traces], ['mesh3d', 'mesh3d'])
            self.assertIn('id="surface-overlay-plot"', html)
            for key in ('before', 'after'):
                self.assertIn(f'id="overlay-{key}-visible" type="checkbox" checked', html)
                self.assertEqual(html.count(f'name="{key}-opacity"'), 3)
                for percentage in (15, 45, 80):
                    self.assertIn(f'id="overlay-{key}-opacity-{percentage}"', html)
            # Each handler mutates only its own property on its own trace.
            self.assertIn('Plotly.restyle(plot, {visible: visible.checked}, [index])', html)
            self.assertIn('Plotly.restyle(plot, {opacity: Number(radio.value)}, [index])', html)
            self.assertIn('plotly_restyle', html)
            self.assertIn('surface-overlay-camera', html)
            self.assertEqual(result['opacity_presets'], [.15, .45, .8])
            self.assertEqual(result['initial_opacity'], {'before': .15, 'after': .45})

    def test_identical_and_absent_geometry_are_explicit(self):
        for mask in (self.before, np.zeros_like(self.before)):
            with self.subTest(empty=not mask.any()), tempfile.TemporaryDirectory() as output:
                result = self.render(output, before=mask, after=mask)
                self.assertEqual(result['counts']['added_voxels'], 0)
                self.assertEqual(result['counts']['removed_voxels'], 0)
                self.assertEqual(result['surfaces'][0]['mask_sha256'], result['surfaces'][1]['mask_sha256'])
                self.assertTrue(any('masks are identical' in item for item in result['warnings']))
                if not mask.any():
                    self.assertTrue(any('no surfaces' in item for item in result['warnings']))
                    self.assertEqual(result['surfaces'][0]['surface_triangles'], 0)
                    self.assertIsNone(result['surfaces'][1]['bounds_ijk_relative_mm'])

    def test_empty_after_keeps_before_and_exact_crop_boundary(self):
        before = np.ones((5, 4, 7), dtype=bool)
        with tempfile.TemporaryDirectory() as output:
            result = self.render(output, before=before, after=np.zeros_like(before))
            expected_bounds = [[9.5, 16.5], [39., 47.], [88.5, 103.5]]
            np.testing.assert_allclose(result['crop_bounds_ijk_relative_mm'], expected_bounds)
            np.testing.assert_allclose(result['surfaces'][0]['bounds_ijk_relative_mm'], expected_bounds)
            self.assertEqual(result['surfaces'][1]['surface_triangles'], 0)
            self.assertEqual(result['counts']['removed_voxels'], before.size)
            self.assertTrue(any('only the before surface' in item for item in result['warnings']))

    def test_invalid_inputs_fail_without_publishing_outputs(self):
        invalid = [{'before': self.before.astype(np.uint8)}, {'after': self.after[:, :, :-1]},
                   {'before': np.ones((1, 4, 4), dtype=bool), 'after': np.ones((1, 4, 4), dtype=bool)},
                   {'crop_origin_ijk': (-1, 0, 0)}, {'crop_origin_ijk': (.1, 0, 0)},
                   {'crop_origin_ijk': (0, 0, float('nan'))}, {'crop_origin_ijk': (0, 0)},
                   {'spacing_ijk_mm': (1, 0, 1)}, {'spacing_ijk_mm': (1, float('inf'), 1)},
                   {'volume_stride': 0}, {'volume_stride': True}, {'volume_stride': 1.5}]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'must-not-exist'
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.render(output, **kwargs)
                self.assertFalse(output.exists())

    def test_resource_caps_reject_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'must-not-exist'
            with patch('fakect_surface_overlay.MAX_SOURCE_VOXELS', 100), self.assertRaisesRegex(ValueError, 'native voxels'):
                self.render(output)
            with patch('fakect_surface_overlay.MAX_SURFACE_TRIANGLES', 10), self.assertRaisesRegex(ValueError, 'too complex'):
                self.render(output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
