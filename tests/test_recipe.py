"""Ordered recipes retain current-state semantics and a bounded spatial domain."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_recipe import apply_recipe, recipe_masks, validate_recipe
from fakect_roi import sphere_mask
from fakect_tissues import coarse_labels


class RecipeTests(unittest.TestCase):
    def fixture(self, spacing=(1., 1., 1.)):
        shape = (31, 31, 31)
        catalog = {'categories': [{'id': code, 'name': name} for code, name in
                    [(0, 'background'), (1, 'soft_tissue'), (2, 'bone'), (5, 'artery'), (255, 'unknown')]],
                   'records': [{'original_id': value, 'original_name': name,
                                'classification': {'tissue_id': code, 'tissue_name': name}}
                               for value, code, name in [(0, 0, 'background'), (20, 1, 'soft_tissue'),
                                                         (30, 2, 'bone'), (-7, 5, 'artery')]]}
        labels = np.full(shape, 20, dtype=np.int32)
        labels[15, 15, 15] = -7
        outer = sphere_mask(shape, (0, 0, 0), (15, 15, 15), spacing, 7)
        arrays = {'act': labels, 'atn': (np.arange(labels.size).reshape(shape) / 1e6 + .01).astype(np.float32),
                  'roi': outer}
        resolved = {'crop_low_ijk': (0, 0, 0), 'crop_high_ijk_exclusive': (31, 31, 31),
                    'spacing_ijk_mm': spacing, 'roi_kind': 'sphere', 'shape_kji': shape,
                    'roi_nodes_ijk': ((15, 15, 15),), 'roi_radii_mm': (7,),
                    'catalog': catalog, 'source_ids': (-7,), 'slice_ijk': (15, 15, 15)}
        roi = {'shape': 'sphere', 'center_ijk': (15., 15., 15.), 'radius_mm': 5.,
               'coordinate_reviewed': False}
        edit = {'roi': 'center', 'iterations': 1, 'operation': 'dilation', 'distance_mm': 1.,
                'profile': 'uniform', 'profile_axis': 'k', 'shape_k': 10., 'shape_window': (0., 1.)}
        config = {'selection': {'tissue': 'artery', 'source_ids': (-7,)},
                  'roi': {**roi, 'radius_mm': 7., 'crop_half_width_mm': 15.},
                  'rois': {'center': roi}, 'edits': {'grow': edit, 'shrink': {**edit, 'operation': 'erosion'}},
                  'recipe': {'steps': ('grow',), 'overlap': 'sequential'},
                  'reassignment': {'allowed_tissues': ('soft_tissue',), 'max_distance_mm': 3., 'unresolved': 'preserve'}}
        self.refresh(arrays, resolved)
        return arrays, resolved, config

    def refresh(self, arrays, resolved):
        arrays['tissue'] = coarse_labels(arrays['act'], resolved['catalog'])
        arrays['candidates'] = np.isin(arrays['act'], resolved['source_ids'])
        arrays['selected'] = arrays['candidates'] & arrays['roi']

    def test_repeated_passes_use_updated_target_and_proxy(self):
        arrays, resolved, config = self.fixture()
        config['edits']['grow']['iterations'] = 2
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['result']['summary']['counts']['after'], 7)
        self.assertEqual(events[1]['result']['summary']['counts']['before'], 7)
        self.assertEqual(result['summary']['counts']['after'], 25)
        self.assertEqual(result['summary']['activity_counts']['changed'], 24)
        np.testing.assert_array_equal(events[1]['before_arrays']['act'], events[0]['result']['edited_labels'])
        np.testing.assert_array_equal(events[1]['before_arrays']['atn'], events[0]['result']['attenuation_proxy_per_pixel'])
        self.assertEqual(events[1]['index'], 2)
        self.assertEqual(events[1]['iteration'], 2)
        self.assertEqual(events[1]['result']['summary']['blocked_input_labels'], [])
        self.assertTrue(np.all(result['attenuation_proxy_per_pixel'][result['added_mask']] == arrays['atn'][15, 15, 15]))

    def test_order_changes_result_and_empty_current_target_is_reported_noop(self):
        arrays, resolved, config = self.fixture()
        config['recipe']['steps'] = ('grow', 'shrink')
        grow_first = apply_recipe(arrays, resolved, config)
        config['recipe']['steps'] = ('shrink', 'grow')
        shrink_first = apply_recipe(arrays, resolved, config)
        self.assertEqual(grow_first['summary']['counts']['after'], 1)
        self.assertEqual(shrink_first['summary']['counts']['after'], 0)
        self.assertEqual(shrink_first['summary']['steps'][1]['status'], 'empty_target')
        self.assertEqual(shrink_first['summary']['steps'][1]['requested_operation'], 'dilation')
        self.assertIn('pass skipped', shrink_first['summary']['steps'][1]['no_op_reason'])

    def test_net_restoration_differs_from_activity_and_scalar_changes(self):
        arrays, resolved, config = self.fixture()
        config['recipe']['steps'] = ('grow', 'shrink')
        result = apply_recipe(arrays, resolved, config)
        self.assertEqual(result['summary']['counts']['changed'], 0)
        self.assertEqual(result['summary']['counts']['ever_changed'], 6)
        self.assertEqual(result['summary']['activity_counts']['changed'], 12)
        self.assertGreater(result['summary']['counts']['scalar_changed'], 0)
        self.assertFalse(result['changed_mask'].any())
        self.assertTrue(result['scalar_changed_mask'].any())
        np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
        self.assertEqual(result['summary']['transitions'], [])

    def test_sources_immutable_and_results_deterministic(self):
        arrays, resolved, config = self.fixture()
        config['recipe']['steps'] = ('grow', 'shrink')
        saved = {name: value.copy() for name, value in arrays.items()}
        first = apply_recipe(arrays, resolved, config)
        second = apply_recipe(arrays, resolved, config)
        for name, value in saved.items():
            np.testing.assert_array_equal(arrays[name], value)
        for name in ('edited_labels', 'attenuation_proxy_per_pixel', 'ever_changed_mask', 'scalar_changed_mask'):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(first['summary'], second['summary'])

    def test_fixed_outer_boundary_clips_named_masks_and_preserves_protected_tissue(self):
        arrays, resolved, config = self.fixture()
        config['rois']['center']['radius_mm'] = 9.
        config['edits']['grow']['iterations'] = 3
        arrays['roi'] = sphere_mask(arrays['act'].shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), 2)
        arrays['act'][15, 15, 16] = 30
        arrays['act'][15, 14, 15] = 999  # Uncatalogued IDs remain protected.
        self.refresh(arrays, resolved)
        result = apply_recipe(arrays, resolved, config)
        self.assertGreater(result['summary']['roi_coverage']['center']['clipped_by_outer_roi_voxels'], 0)
        self.assertFalse(np.any(result['changed_mask'] & ~arrays['roi']))
        self.assertFalse(np.any(result['scalar_changed_mask'] & ~arrays['roi']))
        for value in (30, 999):
            np.testing.assert_array_equal(result['edited_labels'][arrays['act'] == value], arrays['act'][arrays['act'] == value])

    def test_overlap_error_checks_distinct_active_names_before_any_callback(self):
        arrays, resolved, config = self.fixture()
        config['rois']['other'] = {**config['rois']['center'], 'center_ijk': (16, 15, 15)}
        config['edits']['shrink']['roi'] = 'other'
        config['recipe'] = {'steps': ('grow', 'shrink'), 'overlap': 'error'}
        events = []
        with self.assertRaisesRegex(ValueError, 'overlap at'):
            apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(events, [])
        config['recipe']['overlap'] = 'sequential'
        result = apply_recipe(arrays, resolved, config)
        self.assertGreater(result['summary']['roi_overlaps'][0]['effective_overlap_voxels'], 0)
        config['edits']['shrink']['roi'] = 'center'
        config['recipe']['overlap'] = 'error'
        result = apply_recipe(arrays, resolved, config)
        self.assertEqual(result['summary']['roi_overlaps'], [])  # Same named ROI is intentional.

    def test_sphere_and_tube_masks_use_native_anisotropic_spacing(self):
        arrays, resolved, config = self.fixture(spacing=(1., 2., 3.))
        config['rois']['center']['radius_mm'] = 2.1
        config['rois']['tube'] = {'shape': 'tube', 'center_ijk': ((13, 15, 15), (17, 15, 15)),
                                  'radius_mm': (2.1, 2.1), 'coordinate_reviewed': False}
        masks = recipe_masks(arrays, resolved, config)
        self.assertTrue(masks['center'][15, 16, 15])
        self.assertFalse(masks['center'][16, 15, 15])
        self.assertTrue(masks['tube'][15, 16, 17])
        self.assertFalse(masks['tube'][16, 15, 17])
        self.assertTrue(masks['tube'][15, 15, 19])

    def test_metadata_plan_validates_named_halos_and_pass_budget_without_masks(self):
        arrays, resolved, config = self.fixture()
        with patch('fakect_recipe._native_masks', side_effect=AssertionError('Voxel work in metadata validation')):
            plan = validate_recipe(config, resolved)
        self.assertEqual(plan['total_passes'], 1)
        self.assertEqual(plan['steps'][0]['geometry']['roi_kind'], 'sphere')
        config['rois']['center']['radius_mm'] = 12
        with self.assertRaisesRegex(ValueError, r'\[edit.grow\] using \[roi.center\].*halo'):
            validate_recipe(config, resolved)
        config['rois']['center']['radius_mm'] = 5
        config['edits']['grow']['iterations'] = 10
        config['edits']['shrink']['iterations'] = 10
        config['edits']['again'] = {**config['edits']['grow'], 'iterations': 1}
        config['recipe']['steps'] = ('grow', 'shrink', 'again')
        with self.assertRaisesRegex(ValueError, '20 total passes'):
            validate_recipe(config, resolved)

    def test_empty_effective_active_roi_fails_before_any_pass(self):
        arrays, resolved, config = self.fixture()
        arrays['roi'][:] = False
        self.refresh(arrays, resolved)
        events = []
        with self.assertRaisesRegex(ValueError, 'no voxels inside the outer ROI'):
            apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(events, [])

    def test_invalid_metadata_and_input_contracts_fail(self):
        arrays, resolved, config = self.fixture()
        for replacement in (0, 11, 1.5, True):
            bad = deepcopy(config)
            bad['edits']['grow']['iterations'] = replacement
            with self.assertRaisesRegex(ValueError, 'iterations'):
                validate_recipe(bad, resolved)
        bad = deepcopy(config)
        bad['edits']['grow']['roi'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'missing ROI'):
            validate_recipe(bad, resolved)
        bad = deepcopy(config)
        bad['recipe']['steps'] = ('grow', 'grow')
        with self.assertRaisesRegex(ValueError, 'distinct'):
            validate_recipe(bad, resolved)
        arrays['candidates'][:] = True
        with self.assertRaisesRegex(ValueError, 'candidates disagrees'):
            apply_recipe(arrays, resolved, config)

    def test_gaussian_tube_step_uses_named_path_geometry(self):
        arrays, resolved, config = self.fixture()
        config['rois']['center'] = {'shape': 'tube', 'center_ijk': ((15, 15, 11), (15, 15, 19)),
                                    'radius_mm': (3., 3.), 'coordinate_reviewed': False}
        config['edits']['grow'].update(profile='gaussian', profile_axis='tube', distance_mm=1.5)
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(events[0]['resolved']['roi_kind'], 'tube')
        self.assertGreater(result['added_mask'].sum(), 0)
        self.assertAlmostEqual(result['strength_mm'][15, 15, 15], 1.5)
        self.assertEqual(result['strength_mm'][11, 15, 15], 0)

    def test_full_crop_components_are_separate_from_roi_clipped_components(self):
        arrays, resolved, config = self.fixture()
        arrays['act'][2, 15, 15] = -7
        self.refresh(arrays, resolved)
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(result['summary']['components_before'], 1)
        self.assertEqual(result['summary']['full_target_components_before'], 2)
        self.assertEqual(result['summary']['full_target_components_after'], 2)
        self.assertEqual(events[0]['result']['summary']['full_target_components_before'], 2)
        self.assertEqual(result['summary']['steps'][0]['full_target_components_after'], 2)

    def test_blocked_input_labels_preserve_signed_ids_and_catalog_names(self):
        arrays, resolved, config = self.fixture()
        resolved['catalog']['records'].extend([
            {'original_id': -30, 'original_name': 'signed_bone',
             'classification': {'tissue_id': 2, 'tissue_name': 'bone'}},
            {'original_id': -11, 'original_name': 'other_artery',
             'classification': {'tissue_id': 5, 'tissue_name': 'artery'}}])
        arrays['act'][15, 15, 14] = -30
        arrays['act'][15, 15, 16] = -11
        arrays['act'][15, 14, 15] = 999
        self.refresh(arrays, resolved)
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        expected = [
            {'original_id': -30, 'original_name': 'signed_bone', 'tissue_name': 'bone', 'count': 1},
            {'original_id': -11, 'original_name': 'other_artery', 'tissue_name': 'artery', 'count': 1},
            {'original_id': 999, 'original_name': 'Unmapped original ID 999', 'tissue_name': 'unknown', 'count': 1}]
        self.assertEqual(events[0]['result']['summary']['blocked_input_labels'], expected)
        self.assertEqual(result['summary']['steps'][0]['blocked_input_labels'], expected)
        self.assertEqual(sum(row['count'] for row in expected), result['summary']['counts']['blocked'])


if __name__ == '__main__':
    unittest.main()
