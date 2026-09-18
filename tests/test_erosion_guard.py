"""Erosion safeguards use a fixed local reference and discard rejected trials."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_directional_morphology
import test_morphology
import test_recipe
from fakect_direction import direction_weight_field
from fakect_erosion_guard import (apply_guarded_erosion, create_erosion_guard,
                                  evaluate_erosion_guard)
from fakect_morphology import apply_morphology, strength_field_mm
from fakect_recipe import apply_recipe
from fakect_released import RELEASED_LABEL_ID
from fakect_roi import sphere_mask, tube_mask


class ErosionGuardTests(unittest.TestCase):
    def fixture(self, distance=1.):
        arrays, resolved, config = test_morphology.MorphologyTests().fixture(
            operation='erosion', distance=distance)
        config['edit'].update(min_volume_ratio=.2, preserve_connectivity=False,
                              backoff_factor=.5, max_backoff_steps=8)
        return arrays, resolved, config

    def next_state(self, arrays, result):
        return {**arrays, 'act': result['edited_labels'],
                'atn': result['attenuation_proxy_per_pixel'],
                'tissue': result['edited_tissue_labels'],
                'candidates': np.isin(result['edited_labels'], (-7, 8)),
                'selected': result['target_mask_after']}

    def assert_unchanged(self, arrays, snapshot):
        for key, value in snapshot.items():
            np.testing.assert_array_equal(arrays[key], value, err_msg=key)

    def test_local_longitudinal_volume_floor_cannot_be_diluted_by_untouched_parent(self):
        arrays, resolved, config = self.fixture(distance=3.)
        nodes = ((15., 15., 10.), (15., 15., 20.))
        resolved.update(roi_kind='tube', roi_nodes_ijk=nodes, roi_radii_mm=(2.5, 2.5))
        arrays['roi'] = tube_mask(arrays['act'].shape, (0, 0, 0), nodes, (1., 1., 1.), (2.5, 2.5))
        _, j, i = np.indices(arrays['act'].shape)
        arrays['act'][(i-15)**2 + (j-15)**2 <= 4] = -7
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        config['edit'].update(profile='gaussian', profile_axis='tube', shape_k=6.,
                              shape_window=(.3, .7), min_volume_ratio=.75,
                              assign_surrounding_tissue=False)
        reference = create_erosion_guard(arrays, resolved, config)
        expected_scope = arrays['selected'] & (strength_field_mm(arrays['act'].shape, resolved, config) > 0)
        np.testing.assert_array_equal(reference['scope_mask'], expected_scope)
        self.assertEqual(reference['baseline_target_voxels'], 39)
        ordinary = deepcopy(config)
        ordinary['edit'].update(min_volume_ratio=0., preserve_connectivity=False)
        unrestricted = apply_morphology(arrays, resolved, ordinary)
        local_fraction = np.count_nonzero(unrestricted['target_mask_after'] & expected_scope) / 39
        whole_fraction = unrestricted['target_mask_after'].sum() / arrays['selected'].sum()
        self.assertLess(local_fraction, .75)
        self.assertGreater(whole_fraction, .75)
        result = apply_guarded_erosion(arrays, resolved, config, reference)
        self.assertGreater(result['removed_mask'].sum(), 0)
        self.assertLess(result['removed_mask'].sum(), unrestricted['removed_mask'].sum())
        retained = result['target_mask_after'] & expected_scope
        self.assertGreaterEqual(retained.sum() / 39, .75)
        np.testing.assert_array_equal(result['erosion_guard_scope_mask'], expected_scope)
        np.testing.assert_array_equal(result['erosion_guard_retained_mask'], retained)

    def test_reference_includes_opposite_angular_side_not_just_editable_sector(self):
        arrays, resolved, config = test_directional_morphology.DirectionalMorphologyTests().fixture(
            direction='inner', operation='erosion')
        config['edit'].update(min_volume_ratio=.5, preserve_connectivity=False)
        reference = create_erosion_guard(arrays, resolved, config)
        directional = direction_weight_field(arrays['act'].shape, resolved, config, arrays['roi'])
        np.testing.assert_array_equal(reference['scope_mask'], arrays['selected'])
        self.assertGreater(np.count_nonzero(reference['scope_mask'] & (directional['direction_weight'] == 0)), 0)

    def test_preexisting_disconnected_components_can_each_shrink_without_splitting(self):
        arrays, resolved, config = self.fixture()
        arrays['act'].fill(20)
        arrays['act'][14:17, 14:17, 11:14] = -7
        arrays['act'][14:17, 14:17, 17:20] = 8
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        config['edit'].update(min_volume_ratio=0., preserve_connectivity=True)
        reference = create_erosion_guard(arrays, resolved, config)
        self.assertEqual(len(reference['affected_component_ids']), 2)
        result = apply_guarded_erosion(arrays, resolved, config, reference)
        self.assertEqual(result['summary']['erosion_safeguard']['accepted_distance_mm'], 1.)
        self.assertEqual(int(result['target_mask_after'].sum()), 2)
        evaluation = evaluate_erosion_guard(reference, np.isin(result['edited_labels'], (-7, 8)))
        self.assertTrue(evaluation['accepted'])
        self.assertTrue(evaluation['connectivity_ok'])
        self.assertEqual([row['surviving_components'] for row in evaluation['component_details']], [1, 1])

    def test_one_split_cannot_be_hidden_by_loss_of_another_component(self):
        arrays, resolved, config = self.fixture()
        arrays['act'].fill(20)
        arrays['act'][14:17, 14:17, 11:14] = -7
        arrays['act'][14:17, 14:17, 17:20] = -7
        arrays['act'][15, 15, 14:17] = -7
        arrays['act'][15, 19, 15] = 8
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        config['edit'].update(min_volume_ratio=0., preserve_connectivity=True)
        reference = create_erosion_guard(arrays, resolved, config)
        survivors = arrays['candidates'].copy()
        survivors[15, 15, 14:17] = False
        survivors[15, 19, 15] = False
        self.assertEqual(ndimage.label(arrays['candidates'])[1], 2)
        self.assertEqual(ndimage.label(survivors)[1], 2)
        evaluation = evaluate_erosion_guard(reference, survivors)
        self.assertFalse(evaluation['accepted'])
        self.assertFalse(evaluation['connectivity_ok'])
        self.assertEqual(sorted(row['surviving_components'] for row in evaluation['component_details']), [0, 2])
        self.assertTrue(evaluation['reasons'])

    def test_reference_volume_floor_stays_fixed_across_iterations(self):
        arrays, resolved, config = self.fixture()
        arrays['act'][12:19, 12:19, 12:19] = -7
        arrays['roi'] = sphere_mask(arrays['act'].shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), 6.)
        resolved['roi_radii_mm'] = (6.,)
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        config['edit'].update(min_volume_ratio=.15, assign_surrounding_tissue=False)
        reference = create_erosion_guard(arrays, resolved, config)
        self.assertEqual(reference['baseline_target_voxels'], 343)
        first = apply_guarded_erosion(arrays, resolved, config, reference)
        self.assertEqual(int(first['target_mask_after'].sum()), 125)
        next_arrays = self.next_state(arrays, first)
        second = apply_guarded_erosion(next_arrays, resolved, config, reference)
        self.assertEqual(int(second['target_mask_after'].sum()), 125)
        self.assertLess(second['summary']['erosion_safeguard']['accepted_distance_mm'], 1.)
        reset_reference = create_erosion_guard(next_arrays, resolved, config)
        incorrectly_reset = apply_guarded_erosion(next_arrays, resolved, config, reset_reference)
        self.assertEqual(int(incorrectly_reset['target_mask_after'].sum()), 27)
        self.assertLess(27 / 343, .15)

    def test_exhausted_backoff_returns_identity_without_rejected_diagnostic_or_scalar_leak(self):
        arrays, resolved, config = self.fixture(distance=2.)
        config['recipe'] = {'roi_role': 'selection'}
        resolved['source_ids'] = (-7, 8)
        arrays['edit_region'] = arrays['selected'].copy()
        config['edit'].update(min_volume_ratio=1., preserve_connectivity=True,
                              backoff_factor=.9, max_backoff_steps=1,
                              assign_surrounding_tissue=False)
        snapshot, resolved_before, config_before = deepcopy(arrays), deepcopy(resolved), deepcopy(config)
        result = apply_guarded_erosion(arrays, resolved, config)
        metadata = result['summary']['erosion_safeguard']
        self.assertEqual(metadata['accepted_distance_mm'], 0.)
        self.assertEqual(metadata['status'], 'skipped')
        self.assertEqual(len(metadata['attempts']), 3)
        self.assertFalse(result['changed_mask'].any())
        self.assertFalse(result['diagnostic_released_mask'].any())
        self.assertFalse(result['released_mask'].any())
        self.assertFalse(result['attenuation_unassigned_mask'].any())
        self.assertTrue(np.all(result['target_seed_index'] == -1))
        np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])
        self.assert_unchanged(arrays, snapshot)
        self.assertEqual(resolved, resolved_before)
        self.assertEqual(config, config_before)

    def test_disabled_safeguards_preserve_exact_ordinary_result(self):
        arrays, resolved, config = self.fixture()
        ordinary = deepcopy(config)
        for key in ('min_volume_ratio', 'preserve_connectivity', 'backoff_factor', 'max_backoff_steps'):
            ordinary['edit'].pop(key)
        expected = apply_morphology(arrays, resolved, ordinary)
        config['edit'].update(min_volume_ratio=0., preserve_connectivity=False)
        self.assertIsNone(create_erosion_guard(arrays, resolved, config))
        result = apply_guarded_erosion(arrays, resolved, config)
        self.assertEqual(set(result), set(expected))
        self.assertEqual(result['summary'], expected['summary'])
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(result[key], value)

    def test_underflowing_distance_reduction_safely_keeps_input(self):
        arrays, resolved, config = test_morphology.MorphologyTests().fixture(
            spacing=(.25, .25, .25), operation='erosion', distance=.5)
        resolved['roi_radii_mm'] = (.5,)
        config['reassignment']['max_distance_mm'] = .5
        config['edit'].update(min_volume_ratio=1., assign_surrounding_tissue=False,
                              backoff_factor=5e-324, max_backoff_steps=1)
        result = apply_guarded_erosion(arrays, resolved, config)
        self.assertEqual(result['summary']['erosion_safeguard']['status'], 'skipped')
        self.assertEqual(result['summary']['erosion_safeguard']['accepted_distance_mm'], 0.)
        np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
        self.assertFalse(result['changed_mask'].any())

    def test_empty_reference_skips_erosion_with_vacuous_connectivity(self):
        arrays, resolved, config = self.fixture()
        arrays['act'].fill(20)
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        config['edit']['preserve_connectivity'] = True
        reference = create_erosion_guard(arrays, resolved, config)
        self.assertEqual(reference['baseline_target_voxels'], 0)
        evaluation = evaluate_erosion_guard(reference, arrays['candidates'])
        self.assertTrue(evaluation['accepted'])
        self.assertTrue(evaluation['connectivity_ok'])
        self.assertEqual(evaluation['retained_volume_ratio'], 1.)
        result = apply_guarded_erosion(arrays, resolved, config, reference)
        self.assertEqual(result['summary']['erosion_safeguard']['status'], 'skipped')
        self.assertFalse(result['changed_mask'].any())

    def test_recipe_keeps_pre_step_reference_over_multiple_iterations(self):
        arrays, resolved, config = test_recipe.RecipeTests().fixture()
        arrays['act'][12:19, 12:19, 12:19] = -7
        config['rois']['center']['radius_mm'] = 6.
        config['recipe'].update(steps=('shrink',), roi_role='selection')
        config['edits']['shrink'].update(iterations=2, min_volume_ratio=.15,
                                        preserve_connectivity=False, assign_surrounding_tissue=False)
        test_recipe.RecipeTests().refresh(arrays, resolved)
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['result']['summary']['counts']['after'], 125)
        self.assertEqual(events[1]['result']['summary']['counts']['after'], 125)
        self.assertEqual(result['summary']['counts']['after'], 125)
        self.assertEqual(np.count_nonzero(result['edited_labels'] == RELEASED_LABEL_ID), 218)
        np.testing.assert_array_equal(events[0]['result']['erosion_guard_scope_mask'],
                                      events[1]['result']['erosion_guard_scope_mask'])

    def test_recipe_rejected_trials_cannot_erase_target_ancestry(self):
        arrays, resolved, config = test_recipe.RecipeTests().fixture()
        config['recipe'].update(steps=('shrink',), roi_role='selection')
        config['edits']['shrink'].update(distance_mm=2., min_volume_ratio=1.,
                                        preserve_connectivity=True, backoff_factor=.9,
                                        max_backoff_steps=1, assign_surrounding_tissue=False)
        snapshot = deepcopy(arrays)
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        selected = arrays['selected']
        expected_origins = np.full(selected.shape, -1, dtype=np.int32)
        expected_origins[selected] = np.flatnonzero(selected)
        self.assertEqual(events[0]['result']['summary']['erosion_safeguard']['accepted_distance_mm'], 0.)
        np.testing.assert_array_equal(result['target_origin_index'], expected_origins)
        np.testing.assert_array_equal(result['target_mask_after'], selected)
        np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])
        self.assertFalse(result['ever_changed_mask'].any())
        self.assertFalse(result['diagnostic_released_mask'].any())
        self.assertFalse(result['released_mask'].any())
        self.assert_unchanged(arrays, snapshot)


if __name__ == '__main__':
    unittest.main()
