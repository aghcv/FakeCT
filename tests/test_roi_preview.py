"""Native-coordinate and bounded-reader checks on small signed-label phantoms."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_roi import (crop_bounds, prepare_crop, read_crop, resolve_preview,
                        selection_diagnostics, sphere_mask, tube_crop_bounds, tube_mask)


class RoiPreviewTests(unittest.TestCase):
    def test_anisotropic_sphere_native_coordinates(self):
        mask = sphere_mask((7, 7, 7), (10, 20, 30), (13, 23, 33), (1, 2, 3), 3)
        self.assertTrue(mask[3, 3, 6])  # Three i voxels = 3 mm.
        self.assertTrue(mask[4, 3, 3])  # One k voxel = 3 mm.
        self.assertFalse(mask[3, 5, 3])  # Two j voxels = 4 mm.
        self.assertFalse(mask[4, 3, 4])
        expected = np.zeros_like(mask)
        for k, j, i in np.ndindex(mask.shape):
            expected[k, j, i] = (i-3)**2 + ((j-3)*2)**2 + ((k-3)*3)**2 <= 9
        np.testing.assert_array_equal(mask, expected)

    def test_crop_bounds_clipping_spacing_and_limits(self):
        self.assertEqual(crop_bounds((0, 5, 6), 4, (20, 20, 20), (1, 2, 4)), ((0, 3, 5), (5, 8, 8)))
        with self.assertRaises(ValueError):
            crop_bounds((20, 1, 1), 4, (20, 20, 20), (1, 1, 1))
        with self.assertRaises(ValueError):
            crop_bounds((300, 300, 300), 300, (700, 700, 700), (1, 1, 1))

    def test_anisotropic_tube_round_caps_and_continuous_segments(self):
        nodes = ((5, 4, 3), (5, 4, 7))
        mask = tube_mask((11, 11, 11), (0, 0, 0), nodes, (1, 2, 3), (2, 2))
        # Thin j direction and thick k spacing make index-space radii incorrect.
        self.assertTrue(mask[5, 5, 5])
        self.assertFalse(mask[5, 6, 5])
        self.assertTrue(mask[3, 4, 7])
        self.assertFalse(mask[2, 4, 5])
        # Independent constant-radius capsule distance: clamp the k coordinate.
        expected = np.zeros_like(mask)
        for k, j, i in np.ndindex(mask.shape):
            axial_distance = 3 * max(3-k, 0, k-7)
            expected[k, j, i] = (i-5)**2 + (2*(j-4))**2 + axial_distance**2 <= 4
        np.testing.assert_array_equal(mask, expected)

    def test_taper_is_union_of_interpolated_balls_not_radius_at_nearest_center(self):
        mask = tube_mask((12, 11, 11), (0, 0, 0), ((5, 5, 3), (5, 5, 7)), (1, 2, 1), (1, 3))
        # At k=5 the centerline radius is 2 mm, but the swept balls include this
        # point at sqrt(5) radial distance. Using perpendicular distance alone
        # would incorrectly exclude it: a center nearer the wide end contains it.
        self.assertTrue(mask[5, 6, 6])
        self.assertFalse(mask[5, 6, 7])
        self.assertTrue(mask[10, 5, 5])  # Wide endpoint round cap.
        self.assertFalse(mask[11, 5, 5])
        self.assertTrue(mask[2, 5, 5])  # Narrow endpoint round cap.
        self.assertFalse(mask[1, 5, 5])

    def test_steep_taper_quadratic_and_linear_minima_use_endpoints(self):
        for end_k, radii in [(4, (1, 4)), (5, (1, 3))]:
            actual = tube_mask((10, 10, 10), (0, 0, 0), ((4, 4, 3), (4, 4, end_k)), (1, 1, 1), radii)
            expected = sphere_mask((10, 10, 10), (0, 0, 0), (4, 4, end_k), (1, 1, 1), radii[-1])
            np.testing.assert_array_equal(actual, expected)

    def test_tube_preserves_order_and_is_invariant_under_path_reversal(self):
        nodes = ((2, 2, 2), (8, 8, 2), (2, 8, 2))
        radii = (.4, .8, .5)
        forward = tube_mask((11, 11, 11), (0, 0, 0), nodes, (1, 1, 2), radii)
        reverse = tube_mask((11, 11, 11), (0, 0, 0), nodes[::-1], (1, 1, 2), radii[::-1])
        reordered = tube_mask((11, 11, 11), (0, 0, 0), (nodes[0], nodes[2], nodes[1]),
                              (1, 1, 2), (radii[0], radii[2], radii[1]))
        np.testing.assert_array_equal(forward, reverse)
        self.assertTrue(forward[2, 5, 5])
        self.assertFalse(reordered[2, 5, 5])

    def test_tube_fractional_controls_crop_envelope_and_rejections(self):
        nodes = ((.5, 5.25, 6.5), (7.25, 5.25, 9.5))
        self.assertEqual(tube_crop_bounds(nodes, 4, (20, 20, 20), (1, 2, 4)),
                         ((0, 3, 5), (13, 9, 12)))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            tube_mask((5, 5, 5), (0, 0, 0), ((1, 1, 1), (1, 1, 1)), (1, 1, 1), (1, 1))
        for radii in [(1,), (1, 0), (1, float('nan'))]:
            with self.assertRaisesRegex(ValueError, 'one positive finite'):
                tube_mask((5, 5, 5), (0, 0, 0), ((1, 1, 1), (2, 2, 2)), (1, 1, 1), radii)
        with self.assertRaisesRegex(ValueError, 'outside source'):
            tube_crop_bounds(((1, 1, 1), (20, 2, 2)), 1, (20, 20, 20), (1, 1, 1))

    def test_crop_reads_exact_native_signed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.bin'
            original = (np.arange(7*8*9, dtype=np.float32).reshape(7, 8, 9) - 300)
            original.tofile(path)
            before = path.read_bytes()
            crop, meta = read_crop(path, original.shape, (2, 3, 1), (6, 7, 5))
            np.testing.assert_array_equal(crop, original[1:5, 3:7, 2:6])
            self.assertEqual(meta['source_bytes_read'], 4*8*9*4)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                read_crop(path, original.shape, (0, 0, 0), (10, 8, 7))

    def fixture(self, root):
        directory = root / '001'
        directory.mkdir()
        label = np.zeros((9, 9, 9), dtype='<f4')
        label[:, 4, 4] = -7
        label.tofile(directory / '001_act_1.bin')
        (label * 0 + .018).astype('<f4').tofile(directory / '001_atn_1.bin')
        par, log = root/'001.par', directory/'001_log'
        par.write_text('test parameter metadata')
        log.write_text('test log metadata')
        catalog = {'categories': [{'id': 0, 'name': 'background'}, {'id': 5, 'name': 'artery'}, {'id': 255, 'name': 'unknown'}],
                   'records': [{'original_id': key, 'original_name': name,
                                'classification': {'tissue_id': code, 'tissue_name': group}}
                               for key, name, code, group in [(0, 'background', 0, 'background'), (-7, 'test_artery', 5, 'artery')]]}
        catalog_path = root/'catalog.json'
        catalog_path.write_text(json.dumps(catalog))
        audit = {'cases': [{'case_id': '001', 'directory': str(directory), 'shape_kji': [9, 9, 9],
                            'spacing_ijk_mm': [1, 1, 2], 'par_path': str(par), 'log_path': str(log)}],
                 'source_metadata_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [par, log]}}
        audit_path = root/'audit.json'
        audit_path.write_text(json.dumps(audit))
        config = {'input': {'root': root, 'case_id': '001', 'frame': 1, 'audit': audit_path, 'catalog': catalog_path},
                  'selection': {'tissue': 'artery', 'source_ids': (-7,)},
                  'roi': {'center_ijk': (4, 4, 4), 'radius_mm': 2, 'crop_half_width_mm': 3},
                  'preview': {'slice_ijk': None, 'volume_stride': 1, 'context_tissues': ()}}
        return config, label

    def test_resolution_selection_and_crop_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            config, source = self.fixture(Path(directory))
            resolved = resolve_preview(config)
            arrays, _ = prepare_crop(resolved, config)
            self.assertEqual(resolved['source_ids'], (-7,))
            self.assertEqual(int((arrays['selected'] & arrays['roi']).sum()), 3)
            self.assertEqual(int(arrays['selected'].sum()), 3)
            self.assertEqual(int(arrays['candidates'].sum()), 5)
            self.assertTrue(np.all(arrays['tissue'][arrays['selected']] == 5))
            np.testing.assert_array_equal(arrays['act'], source[2:7, 1:8, 1:8])
            np.testing.assert_allclose(arrays['attenuation_cm_inverse'], .18, rtol=1e-6)
            config['selection']['source_ids'] = ()
            self.assertEqual(resolve_preview(config)['source_ids'], (-7,))

    def test_tissue_only_tube_excludes_nearby_artery_with_same_source_id(self):
        with tempfile.TemporaryDirectory() as directory:
            config, source = self.fixture(Path(directory))
            source[:, 4, 6] = -7
            path = Path(directory)/'001'/'001_act_1.bin'
            source.tofile(path)
            before = path.read_bytes()
            config['selection']['source_ids'] = ()
            config['roi'] = {'shape': 'tube', 'center_ijk': ((4, 4, 1), (4, 4, 7)),
                             'radius_mm': (.75, .75), 'crop_half_width_mm': 3}
            resolved = resolve_preview(config)
            self.assertEqual(resolved['roi_nodes_ijk'], ((4., 4., 1.), (4., 4., 7.)))
            self.assertEqual(resolved['focus_ijk'], (4, 4, 7))
            arrays, _ = prepare_crop(resolved, config)
            self.assertEqual(int(arrays['candidates'].sum()), 18)
            self.assertEqual(int(arrays['selected'].sum()), 7)
            self.assertFalse(np.any(arrays['selected'][:, :, 6-resolved['crop_low_ijk'][0]]))
            self.assertEqual(path.read_bytes(), before)
            diag = selection_diagnostics(arrays, resolved)
            self.assertEqual(diag['component_count_6'], 1)
            self.assertEqual(diag['selected_original_id_counts'], {'-7': 7})
            self.assertEqual(diag['warnings'], [])
            self.assertEqual(diag['selected_original_names'], {'-7': 'test_artery'})
            # A broad tube can contain both vessels; report, never silently pick.
            config['roi']['radius_mm'] = (3, 3)
            arrays, _ = prepare_crop(resolve_preview(config), config)
            diag = selection_diagnostics(arrays)
            self.assertEqual(diag['component_count_6'], 2)
            self.assertEqual(diag['components_voxels_6'], [9, 9])
            self.assertIn('disconnected', diag['warnings'][0])

    def test_multiple_source_ids_in_one_connected_selection_are_not_multiple_vessels(self):
        act = np.zeros((5, 5, 5), dtype=np.float32)
        act[:2, 2, 2], act[2:, 2, 2] = -7, 8
        candidates = act != 0
        arrays = {'act': act, 'candidates': candidates, 'roi': np.ones_like(candidates), 'selected': candidates}
        diag = selection_diagnostics(arrays)
        self.assertEqual(diag['component_count_6'], 1)
        self.assertEqual(diag['selected_original_ids'], [-7, 8])
        self.assertEqual(diag['selected_original_id_counts'], {'-7': 2, '8': 3})
        self.assertEqual(diag['warnings'], [])
        arrays['selected'] = np.zeros_like(candidates)
        self.assertEqual(selection_diagnostics(arrays)['component_count_6'], 0)
        self.assertIn('no voxels', selection_diagnostics(arrays)['warnings'][0])

    def test_fractional_sphere_and_tube_focus_crop_and_radius_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(Path(directory))
            config['roi']['center_ijk'] = (4.25, 4, 4.75)
            resolved = resolve_preview(config)
            self.assertEqual(resolved['focus_ijk'], (4, 4, 5))
            self.assertEqual(resolved['roi_nodes_ijk'], ((4.25, 4., 4.75),))
            arrays, _ = prepare_crop(resolved, config)
            self.assertEqual(int(arrays['selected'].sum()), 2)
            config['roi']['radius_mm'] = 4
            with self.assertRaisesRegex(ValueError, 'largest ROI radius'):
                resolve_preview(config)

    def test_reject_wrong_group_and_changed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(Path(directory))
            config['selection']['source_ids'] = (0,)
            with self.assertRaisesRegex(ValueError, 'selected tissue'):
                resolve_preview(config)
            config['selection']['source_ids'] = (-7,)
            (Path(directory)/'001.par').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed audited metadata'):
                resolve_preview(config)

    def test_catalog_provenance_matches_loaded_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(Path(directory))
            resolved = resolve_preview(config)
            old_bytes = config['input']['catalog'].read_bytes()
            config['input']['catalog'].write_text('{}')
            self.assertEqual(resolved['catalog_bytes'], old_bytes)
            self.assertEqual(resolved['catalog_sha256'], hashlib.sha256(old_bytes).hexdigest())
            self.assertEqual(resolved['catalog']['records'][1]['original_name'], 'test_artery')

    def test_reject_slice_outside_crop_and_too_coarse_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(Path(directory))
            config['preview']['slice_ijk'] = (0, 0, 0)
            with self.assertRaisesRegex(ValueError, 'outside the preview crop'):
                resolve_preview(config)
            config['preview']['slice_ijk'] = None
            config['preview']['volume_stride'] = 10
            with self.assertRaisesRegex(ValueError, 'fewer than two blocks'):
                resolve_preview(config)


if __name__ == '__main__':
    unittest.main()
