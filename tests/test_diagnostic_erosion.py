"""Geometry-only erosion records released labels without inventing CT values."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_directional_morphology
import test_growth_morphology
import test_morphology
from fakect_morphology import apply_morphology, validate_edit_geometry
from fakect_released import RELEASED_LABEL_ID, RELEASED_TISSUE_ID


class DiagnosticErosionTests(unittest.TestCase):
    def fixture(self, cube=False):
        arrays, resolved, config = test_morphology.MorphologyTests().fixture(
            operation='erosion', distance=1.1)
        # This surrounding label is coarse soft_tissue but has fine skin
        # identity, reproducing the recipient-reach barrier independently of
        # the original XCAT IDs or an anatomical interpretation of the fixture.
        record = next(row for row in resolved['catalog']['records'] if row['original_id'] == 20)
        record['classification']['structure_type'] = 'skin'
        if cube:
            arrays['act'][14:17, 14:17, 14:17] = -7
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        config['reassignment'] = {
            'mode': 'stiffness', 'allowed_tissues': (), 'max_distance_mm': 3.,
            'unresolved': 'preserve', 'stiffness': {
                'default': .05, 'tissues': {'bone': 1., 'skin': .95, 'artery': .05}, 'labels': {}}}
        return arrays, resolved, config

    def next_state(self, arrays, result):
        return {**arrays, 'act': result['edited_labels'],
                'atn': result['attenuation_proxy_per_pixel'],
                'tissue': result['edited_tissue_labels'],
                'candidates': np.isin(result['edited_labels'], (-7, 8)),
                'selected': result['target_mask_after']}

    def test_default_and_explicit_true_keep_original_assignment_and_output(self):
        arrays, resolved, config = self.fixture()
        implicit = apply_morphology(arrays, resolved, config)
        config['edit']['assign_surrounding_tissue'] = True
        explicit = apply_morphology(arrays, resolved, config)
        self.assertEqual(set(implicit), set(explicit))
        self.assertEqual(implicit['summary'], explicit['summary'])
        for key, value in implicit.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, explicit[key])
        self.assertEqual(implicit['summary']['counts']['unresolved'], 1)
        self.assertEqual(implicit['summary']['counts']['removed'], 0)
        self.assertNotIn('released_mask', implicit)
        self.assertNotIn('release_assignment', implicit['summary'])

    def test_diagnostic_bypasses_recipient_barrier_preserves_attenuation_and_inputs(self):
        arrays, resolved, config = self.fixture()
        config['edit']['assign_surrounding_tissue'] = False
        # No surrounding assignment takes place, so unresolved=error does not
        # reject an otherwise valid geometry-only release.
        config['reassignment']['unresolved'] = 'error'
        saved_arrays = {key: value.copy() for key, value in arrays.items()}
        saved_resolved, saved_config = deepcopy(resolved), deepcopy(config)
        result = apply_morphology(arrays, resolved, config)
        counts = result['summary']['counts']
        self.assertEqual(counts['proposed_removed'], 1)
        self.assertEqual(counts['removed'], 1)
        self.assertEqual(counts['unresolved'], 0)
        self.assertEqual(counts['released'], 1)
        np.testing.assert_array_equal(result['diagnostic_released_mask'], result['removed_mask'])
        np.testing.assert_array_equal(result['released_mask'], result['removed_mask'])
        np.testing.assert_array_equal(result['attenuation_unassigned_mask'], result['released_mask'])
        self.assertTrue(np.all(result['edited_labels'][result['removed_mask']] == RELEASED_LABEL_ID))
        self.assertTrue(np.all(result['edited_tissue_labels'][result['removed_mask']] == RELEASED_TISSUE_ID))
        self.assertFalse(result['target_mask_after'].any())
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])
        np.testing.assert_array_equal(result['edited_labels'][~result['removed_mask']],
                                      arrays['act'][~result['removed_mask']])
        assignment = result['summary']['release_assignment']
        self.assertEqual(assignment['mode'], 'diagnostic_label')
        self.assertEqual(assignment['label_id'], RELEASED_LABEL_ID)
        self.assertEqual(assignment['tissue_id'], RELEASED_TISSUE_ID)
        self.assertEqual(assignment['newly_released_voxels'], 1)
        self.assertEqual(assignment['current_released_voxels'], 1)
        self.assertTrue(assignment['surrounding_labels'])
        self.assertIn('unassigned', result['summary']['scalar_status'])
        for key, value in saved_arrays.items():
            np.testing.assert_array_equal(arrays[key], value)
        self.assertEqual(resolved, saved_resolved)
        self.assertEqual(config, saved_config)

    def test_target_stiffness_still_suppresses_release(self):
        arrays, resolved, config = self.fixture()
        config['edit']['assign_surrounding_tissue'] = False
        config['reassignment']['stiffness']['tissues']['artery'] = .2
        result = apply_morphology(arrays, resolved, config)
        counts = result['summary']['counts']
        self.assertEqual(counts['requested_removed_before_stiffness'], 1)
        self.assertEqual(counts['release_suppressed_by_target_stiffness'], 1)
        self.assertEqual(counts['proposed_removed'], 0)
        self.assertEqual(counts['removed'], 0)
        self.assertFalse(result['released_mask'].any())

    def test_repeated_releases_keep_distinct_new_and_cumulative_masks(self):
        arrays, resolved, config = self.fixture(cube=True)
        config['edit']['assign_surrounding_tissue'] = False
        first = apply_morphology(arrays, resolved, config)
        self.assertEqual(first['summary']['counts']['removed'], 26)
        second_arrays = self.next_state(arrays, first)
        second = apply_morphology(second_arrays, resolved, config)
        self.assertEqual(second['summary']['counts']['removed'], 1)
        self.assertEqual(second['summary']['counts']['diagnostic_released'], 1)
        self.assertEqual(second['summary']['counts']['released'], 27)
        self.assertFalse(np.any(second['diagnostic_released_mask'] & first['released_mask']))
        self.assertEqual(second['summary']['release_assignment']['newly_released_voxels'], 1)
        self.assertEqual(second['summary']['release_assignment']['current_released_voxels'], 27)
        np.testing.assert_array_equal(second['attenuation_proxy_per_pixel'], arrays['atn'])
        # The recipe can internally turn a now-empty requested pass into none.
        config['edit']['operation'] = 'none'
        third = apply_morphology(self.next_state(second_arrays, second), resolved, config)
        self.assertFalse(third['diagnostic_released_mask'].any())
        self.assertEqual(third['summary']['counts']['released'], 27)
        np.testing.assert_array_equal(third['edited_labels'], second['edited_labels'])

    def test_incoming_markers_are_never_dilation_donors_or_erosion_recipients(self):
        arrays, resolved, config = self.fixture(cube=True)
        config['edit']['assign_surrounding_tissue'] = False
        first = apply_morphology(arrays, resolved, config)
        state = self.next_state(arrays, first)
        for operation in ('dilation', 'erosion'):
            for mode in ('allowlist', 'permissive_except_bone_skin', 'stiffness'):
                with self.subTest(operation=operation, mode=mode):
                    trial = deepcopy(config)
                    trial['edit'].update(operation=operation, assign_surrounding_tissue=True)
                    trial['reassignment']['mode'] = mode
                    trial['reassignment']['allowed_tissues'] = ('soft_tissue',) if mode == 'allowlist' else ()
                    result = apply_morphology(state, resolved, trial)
                    self.assertFalse(result['changed_mask'].any())
                    self.assertFalse(result['diagnostic_released_mask'].any())
                    np.testing.assert_array_equal(result['released_mask'], first['released_mask'])
                    np.testing.assert_array_equal(result['attenuation_unassigned_mask'], first['released_mask'])
                    np.testing.assert_array_equal(result['edited_labels'], first['edited_labels'])
                    np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])
                    if operation == 'erosion':
                        self.assertEqual(result['summary']['counts']['unresolved'], 1)
                    else:
                        self.assertEqual(result['summary']['counts']['blocked'], 6)

    def test_gaussian_direction_and_roi_still_bound_diagnostic_release(self):
        arrays, resolved, config = test_directional_morphology.DirectionalMorphologyTests().fixture(
            direction='inner', operation='erosion')
        config['edit'].update(assign_surrounding_tissue=False, profile='gaussian', profile_axis='tube', shape_k=6.)
        result = apply_morphology(arrays, resolved, config)
        removed = result['removed_mask']
        self.assertGreater(removed.sum(), 0)
        self.assertFalse(np.any(removed & ~arrays['roi']))
        self.assertFalse(np.any(removed & ~result['direction_reliable_mask']))
        self.assertTrue(np.all(result['direction_weight'][removed] > 0))
        self.assertTrue(result['changed_mask'][15, 15, 20])
        self.assertFalse(result['changed_mask'][15, 15, 22])
        self.assertEqual(np.count_nonzero(result['unresolved_mask']), 0)
        np.testing.assert_array_equal(removed, result['proposed_removed_mask'])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])

    def test_selection_role_releases_tracked_growth_without_erosion_at_roi_cuts(self):
        arrays, resolved, config = test_growth_morphology.GrowthMorphologyTests().fixture(operation='erosion')
        _, j, i = np.indices(arrays['act'].shape)
        arrays['act'][(i-15)**2 + (j-15)**2 <= 9] = -7
        arrays['edit_region'].fill(False)
        arrays['edit_region'][15, 10:21, 10:21] = True
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        arrays['selected'] = arrays['candidates'] & arrays['edit_region']
        config['edit']['assign_surrounding_tissue'] = False
        result = apply_morphology(arrays, resolved, config)
        self.assertFalse(result['removed_mask'][15, 15, 15])
        self.assertTrue(result['removed_mask'][15, 15, 18])
        self.assertGreater(result['summary']['counts']['removed_outside_selection_roi'], 0)
        self.assertFalse(result['changed_mask'][arrays['candidates'] & ~arrays['selected']].any())
        self.assertFalse(result['changed_mask'][~arrays['edit_region']].any())
        self.assertTrue(np.all(result['target_seed_index'] == -1))
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])

    def test_invalid_flag_and_dilation_are_rejected_before_execution(self):
        arrays, resolved, config = self.fixture()
        for value in ('false', None, 0, 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'assign_surrounding_tissue'):
                config['edit']['assign_surrounding_tissue'] = value
                apply_morphology(arrays, resolved, config)
        config['edit'].update(operation='dilation', assign_surrounding_tissue=False)
        with self.assertRaisesRegex(ValueError, 'only supported for erosion'):
            validate_edit_geometry(resolved, config)


if __name__ == '__main__':
    unittest.main()
