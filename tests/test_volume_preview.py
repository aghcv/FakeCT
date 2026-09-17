"""Coordinate and label-preservation checks for the bounded 3D renderer."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_volume_preview import occupancy_grid, render_volume_preview, _surface


class VolumePreviewTests(unittest.TestCase):
    def test_max_pool_preserves_thin_label_and_axis_coordinates(self):
        mask = np.zeros((5, 4, 7), dtype=bool)
        mask[4, 3, 6] = True
        original = mask.copy()
        grid, axes = occupancy_grid(mask, 2, (10, 20, 30), (1, 2, 3))
        self.assertEqual(grid.shape, (3, 2, 4))
        self.assertEqual(int(grid.sum()), 1)
        self.assertTrue(grid[2, 1, 3])
        np.testing.assert_array_equal(axes[0], [10.5, 12.5, 14.5, 16])
        np.testing.assert_array_equal(axes[1], [41, 45])
        np.testing.assert_array_equal(axes[2], [91.5, 97.5, 102])
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
        at = np.flatnonzero(grid.ravel())[0]
        self.assertEqual((x.ravel()[at], y.ravel()[at], z.ravel()[at]), (16, 45, 102))
        np.testing.assert_array_equal(mask, original)

    def test_surface_preserves_crop_edges_for_partial_blocks(self):
        mask = np.ones((5, 4, 7), dtype=bool)
        origin, spacing = np.array((10, 20, 30)), np.array((1, 2, 3))
        grid, axes = occupancy_grid(mask, 2, origin, spacing)
        bounds = np.column_stack(((origin - .5) * spacing,
                                  (origin + np.array(mask.shape[::-1]) - .5) * spacing))
        vertices, faces = _surface(grid, axes, bounds)
        np.testing.assert_allclose(vertices.min(axis=0), bounds[:, 0])
        np.testing.assert_allclose(vertices.max(axis=0), bounds[:, 1])
        self.assertGreater(len(faces), 0)

    def test_standalone_artifacts_and_provenance(self):
        source = np.zeros((6, 8, 10), dtype=np.float32)
        source[1:5, 2:6, 3:7] = -123
        tissues = np.zeros(source.shape, dtype=np.uint8)
        tissues[source == -123] = 5
        tissues[:, 0, :] = 2
        catalog = {'categories': [{'id': 0, 'name': 'background'}, {'id': 2, 'name': 'bone'},
                                  {'id': 5, 'name': 'artery'}],
                   'records': [{'original_id': -123, 'original_name': 'test artery'}]}
        before = source.copy(), tissues.copy()
        kwargs = dict(crop_origin_ijk=(10, 20, 30), spacing_ijk_mm=(1, 2, 3),
                      roi_center_ijk=(15, 24, 33), roi_radius_mm=4,
                      context_tissues=['bone'], volume_stride=2,
                      volume_opacity=.3, context_opacity=.08)
        with tempfile.TemporaryDirectory() as temporary:
            result = render_volume_preview(source, tissues, catalog, source == -123,
                                           output_dir=temporary, **kwargs)
            html = Path(result['html_path']).read_text()
            self.assertIn('plotly.js', html)
            self.assertNotIn('<script src=', html)
            self.assertIn('"type":"volume"', html)
            self.assertEqual(result['display_shape_kji'], [3, 4, 5])
            self.assertEqual(result['original_selected_labels'][0]['original_id'], -123)
            self.assertEqual(result['original_selected_labels'][0]['voxel_count'], 64)
            self.assertEqual(len(result['static_panels']), 2)
            self.assertEqual(result['traces'][1]['static_surface_opacity'], .08)
            self.assertTrue(Path(result['png_path']).stat().st_size > 10_000)
            from PIL import Image
            with Image.open(result['png_path']) as preview:
                self.assertEqual(preview.size, (2880, 1440))
            self.assertEqual(json.loads(Path(result['metadata_path']).read_text()), result)
            with self.assertRaises(FileExistsError):
                render_volume_preview(source, tissues, catalog, source == -123,
                                      output_dir=temporary, **kwargs)
        np.testing.assert_array_equal(source, before[0])
        np.testing.assert_array_equal(tissues, before[1])

    def test_absent_selection_and_zero_opacity_still_render_roi_for_relocation(self):
        source = np.full((5, 5, 5), 42, dtype=np.int32)
        tissues = np.full(source.shape, 2, dtype=np.uint8)
        catalog = {'categories': [{'id': 2, 'name': 'bone'}],
                   'records': [{'original_id': 42, 'original_name': 'test bone'}]}
        with tempfile.TemporaryDirectory() as temporary:
            result = render_volume_preview(source, tissues, catalog, source == 999,
                       crop_origin_ijk=(0, 0, 0), spacing_ijk_mm=(1, 1, 1),
                       roi_center_ijk=(2, 2, 2), roi_radius_mm=1,
                       context_tissues=['bone'], volume_stride=2,
                       volume_opacity=0, context_opacity=0, output_dir=temporary)
            self.assertEqual(result['original_selected_labels'], [])
            self.assertEqual(result['traces'][0]['name'], 'Selected: absent from crop')
            self.assertEqual(result['traces'][0]['surface_triangles'], 0)
            self.assertEqual(result['traces'][1]['source_voxels'], 125)
            self.assertTrue(result['warnings'])
            for trace in result['traces']:
                self.assertEqual(trace['opacity'], 0)
                self.assertEqual(trace['static_surface_opacity'], 0)
            html = Path(result['html_path']).read_text()
            self.assertIn('Editable ROI sphere', html)
            self.assertIn('Selected: absent from crop', html)
            self.assertTrue(Path(result['png_path']).stat().st_size > 10_000)


if __name__ == '__main__':
    unittest.main()
