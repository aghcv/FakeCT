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

    def test_tube_overlay_uses_native_mask_and_physical_control_points(self):
        source = np.zeros((7, 8, 9), dtype=np.int32)
        source[:, 3, 4] = 123
        source[:, 3, 6] = 124  # An adjacent artery remains outside the selected tube.
        tissues = np.where(source != 0, 5, 0).astype(np.uint8)
        roi = np.zeros(source.shape, dtype=bool)
        roi[1:6, 2:5, 3:6] = True
        selected = (tissues == 5) & roi
        catalog = {'categories': [{'id': 0, 'name': 'background'}, {'id': 5, 'name': 'artery'}],
                   'records': [{'original_id': 123, 'original_name': 'target artery'},
                               {'original_id': 124, 'original_name': 'nearby artery'}]}
        before = source.copy(), roi.copy()
        with tempfile.TemporaryDirectory() as temporary:
            result = render_volume_preview(source, tissues, catalog, selected,
                      crop_origin_ijk=(10, 20, 30), spacing_ijk_mm=(1, 2, 3),
                      roi_shape='tube', roi_mask=roi,
                      roi_nodes_ijk=((14, 23, 31), (14, 23, 35)), roi_radii_mm=(2, 3),
                      context_tissues=['artery'], volume_stride=3,
                      volume_opacity=.25, context_opacity=.08, output_dir=temporary)
            self.assertEqual(result['roi_shape'], 'tube')
            self.assertEqual(result['roi_nodes_ijk'], [[14., 23., 31.], [14., 23., 35.]])
            self.assertEqual(result['roi_radii_mm'], [2., 3.])
            self.assertEqual(result['roi_native_voxels'], 45)
            self.assertEqual(result['original_selected_labels'],
                             [{'original_id': 123, 'original_name': 'target artery', 'voxel_count': 5}])
            self.assertEqual(result['traces'][1]['source_voxels'], 9)
            # Native voxel boundaries: independently calculated before pooling.
            np.testing.assert_allclose(result['roi_surface_bounds_ijk_relative_mm'],
                                       [[12.5, 15.5], [43., 49.], [91.5, 106.5]])
            self.assertEqual(result['display_shape_kji'], [3, 3, 3])
            html = Path(result['html_path']).read_text()
            self.assertIn('Editable ROI tube', html)
            self.assertIn('Ordered tube centerline', html)
            self.assertNotIn('Editable ROI sphere', html)
            self.assertIsNone(result['roi_center_ijk'])
        np.testing.assert_array_equal(source, before[0])
        np.testing.assert_array_equal(roi, before[1])

    def test_tube_rejects_missing_or_sampled_mask_and_bad_nodes(self):
        source = np.zeros((5, 5, 5), dtype=np.int32)
        kwargs = dict(crop_origin_ijk=(0, 0, 0), spacing_ijk_mm=(1, 1, 1),
                      roi_shape='tube', roi_mask=source == 0,
                      roi_nodes_ijk=((2, 2, 1), (2, 2, 3)), roi_radii_mm=(1, 2),
                      context_tissues=[], volume_stride=2, volume_opacity=.2, context_opacity=.1)
        invalid = [{'roi_mask': None}, {'roi_mask': np.zeros((3, 3, 3), dtype=bool)},
                   {'roi_mask': np.zeros(source.shape, dtype=np.uint8)},
                   {'roi_nodes_ijk': ((2, 2, 1), (2, 2, 1))},
                   {'roi_nodes_ijk': ((2, 2, 1),)}, {'roi_radii_mm': (1,)},
                   {'roi_radii_mm': (1, float('inf'))}]
        with tempfile.TemporaryDirectory() as temporary:
            for override in invalid:
                with self.subTest(override=override), self.assertRaises(ValueError):
                    render_volume_preview(source, source, {'categories': [], 'records': []},
                                          source == 1, output_dir=temporary, **(kwargs | override))

    def test_subvoxel_tube_with_empty_native_mask_retains_centerline(self):
        source = np.zeros((4, 4, 4), dtype=np.int32)
        with tempfile.TemporaryDirectory() as temporary:
            result = render_volume_preview(source, source, {'categories': [], 'records': []},
                      source == 1, crop_origin_ijk=(0, 0, 0), spacing_ijk_mm=(1, 1, 1),
                      roi_shape='tube', roi_mask=source == 1,
                      roi_nodes_ijk=((1.5, 1.5, 1), (1.5, 1.5, 2)), roi_radii_mm=(.1, .1),
                      context_tissues=[], volume_stride=2, volume_opacity=.25, context_opacity=.1,
                      output_dir=temporary)
            self.assertEqual(result['roi_native_voxels'], 0)
            self.assertEqual(result['roi_surface_triangles'], 0)
            self.assertIsNone(result['roi_surface_bounds_ijk_relative_mm'])
            self.assertTrue(any('no voxel centers' in warning for warning in result['warnings']))
            self.assertIn('Ordered tube centerline', Path(result['html_path']).read_text())


if __name__ == '__main__':
    unittest.main()
