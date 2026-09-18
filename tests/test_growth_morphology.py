"""Selection-only growth retains target ownership outside fixed ROI outlines."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_morphology
import test_directional_morphology
from fakect_morphology import apply_morphology, validate_edit_geometry
from fakect_roi import sphere_mask, tube_mask
from fakect_tissues import coarse_labels


class GrowthMorphologyTests(unittest.TestCase):
    def fixture(self, distance=1., operation='dilation'):
        arrays, resolved, config = test_morphology.MorphologyTests().fixture(distance=distance, operation=operation)
        arrays['roi'] = sphere_mask(arrays['act'].shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), .25)
        arrays['edit_region'] = sphere_mask(arrays['act'].shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), 6.)
        resolved['roi_radii_mm'] = (.25,)
        resolved['source_ids'] = (-7, 8)
        config['recipe'] = {'roi_role': 'selection'}
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        return arrays, resolved, config

    def next_state(self, arrays, result, resolved):
        return {**arrays, 'act': result['edited_labels'], 'atn': result['attenuation_proxy_per_pixel'],
                'tissue': result['edited_tissue_labels'],
                'candidates': np.isin(result['edited_labels'], resolved['source_ids']),
                'selected': result['target_mask_after']}

    def test_repeated_growth_exits_selection_roi_and_maps_each_addition_to_current_seed(self):
        arrays, resolved, config = self.fixture()
        original = {key: value.copy() for key, value in arrays.items()}
        first = apply_morphology(arrays, resolved, config)
        self.assertEqual(first['summary']['counts']['added'], 6)
        self.assertEqual(first['summary']['counts']['added_outside_selection_roi'], 6)
        self.assertTrue(first['target_mask_after'][15, 15, 16])
        second_state = self.next_state(arrays, first, resolved)
        second = apply_morphology(second_state, {**resolved, 'growth_reach_mm': 1.}, config)
        self.assertTrue(second['target_mask_after'][15, 15, 17])
        self.assertGreater(second['summary']['counts']['added_outside_selection_roi'], 0)
        self.assertFalse(np.array_equal(second_state['selected'], second_state['candidates'] & arrays['roi']))
        for state, result in ((arrays, first), (second_state, second)):
            owners, added = result['target_seed_index'], result['added_mask']
            self.assertEqual(owners.dtype, np.dtype('int32'))
            self.assertTrue(np.all(owners[~added] == -1))
            self.assertTrue(np.all(state['selected'].ravel()[owners[added]]))
            np.testing.assert_array_equal(result['edited_labels'][added], state['act'].ravel()[owners[added]])
            np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][added], state['atn'].ravel()[owners[added]])
            np.testing.assert_array_equal(result['selection_roi_mask'], arrays['roi'])
            np.testing.assert_array_equal(result['edit_region_mask'], arrays['edit_region'])
            self.assertFalse(np.any(result['changed_mask'] & ~arrays['edit_region']))
        for name, value in original.items():
            np.testing.assert_array_equal(arrays[name], value)

    def test_same_id_neighbors_never_seed_or_bridge_growth_but_contact_is_reported(self):
        arrays, resolved, config = self.fixture(distance=6.)
        arrays['act'].fill(20)
        arrays['act'][15, 15, [12, 14, 20]] = -7
        arrays['roi'] = sphere_mask(arrays['act'].shape, (0, 0, 0), (12, 15, 15), (1, 1, 1), .25)
        arrays['edit_region'].fill(False)
        arrays['edit_region'][15, 15, 12:23] = True
        resolved['roi_nodes_ijk'] = ((12, 15, 15),)
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertTrue(result['added_mask'][15, 15, 13])
        self.assertFalse(result['added_mask'][15, 15, 15:].any())
        # Neither the proposal metric nor accepted ownership can traverse the
        # unselected same-ID target at x=14 or start from the one at x=20.
        self.assertFalse(result['proposed_added_mask'][15, 15, 15:].any())
        self.assertEqual(result['summary']['counts']['unselected_target_contact_voxels'], 1)
        self.assertEqual(result['target_seed_index'][15, 15, 13], np.ravel_multi_index((15, 15, 12), arrays['act'].shape))
        untouched = arrays['candidates'] & ~arrays['selected']
        np.testing.assert_array_equal(result['edited_labels'][untouched], arrays['act'][untouched])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][untouched], arrays['atn'][untouched])
        self.assertFalse(result['target_mask_after'][untouched].any())

    def test_erosion_uses_full_vessel_boundaries_and_removes_tracked_voxels_outside_roi(self):
        arrays, resolved, config = self.fixture(operation='erosion')
        _, j, i = np.indices(arrays['act'].shape)
        arrays['act'][(i - 15)**2 + (j - 15)**2 <= 9] = -7
        arrays['edit_region'].fill(False)
        arrays['edit_region'][15, 10:21, 10:21] = True
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        arrays['selected'] = arrays['candidates'] & arrays['edit_region']
        result = apply_morphology(arrays, resolved, config)
        self.assertFalse(result['proposed_removed_mask'][15, 15, 15])
        self.assertTrue(result['removed_mask'][15, 15, 18])
        self.assertGreater(result['summary']['counts']['removed_outside_selection_roi'], 0)
        self.assertTrue(np.all(result['target_seed_index'] == -1))
        unselected = arrays['candidates'] & ~arrays['selected']
        self.assertFalse(result['changed_mask'][unselected].any())
        np.testing.assert_array_equal(result['edited_labels'][unselected], arrays['act'][unselected])

    def test_direction_and_stiffness_apply_to_new_support_beyond_original_tube(self):
        arrays, resolved, config = test_directional_morphology.DirectionalMorphologyTests().fixture()
        arrays['edit_region'] = arrays['roi'].copy()
        parent = resolved['roi_nodes_ijk']
        arrays['roi'] = tube_mask(arrays['act'].shape, (0, 0, 0), parent, (1, 1, 1), (1.25,) * len(parent))
        arrays['act'][15, 15, 24] = 30  # Rigid bone within the new outward search support.
        resolved['roi_radii_mm'] = (1.25,) * len(parent)
        resolved['source_ids'] = (-7, 8)
        config['recipe'] = {'roi_role': 'selection'}
        config['edit']['distance_mm'] = 2.1
        config['reassignment'] = {'mode': 'stiffness', 'allowed_tissues': (), 'max_distance_mm': 1.,
                                  'unresolved': 'preserve', 'stiffness': {
                                      'default': .25, 'tissues': {'bone': 1., 'skin': .95}, 'labels': {}}}
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertFalse(arrays['roi'][15, 15, 23])
        self.assertTrue(result['added_mask'][15, 15, 23])
        self.assertFalse(result['changed_mask'][15, 15, 19])
        self.assertEqual(result['edited_labels'][15, 15, 24], 30)
        self.assertEqual(result['attenuation_proxy_per_pixel'][15, 15, 24], arrays['atn'][15, 15, 24])
        self.assertGreater(result['summary']['counts']['changed_outside_selection_roi'], 0)
        self.assertFalse(np.any(result['changed_mask'] & ((result['direction_weight'] <= 0) | ~result['direction_reliable_mask'])))
        np.testing.assert_allclose(result['strength_mm'], 2.1 * result['direction_weight'])
        np.testing.assert_allclose(result['effective_strength_mm'], result['strength_mm'] * (1 - result['stiffness_field']))
        self.assertEqual(result['summary']['direction']['roi_voxels'], int(arrays['edit_region'].sum()))

    def test_malformed_domains_selection_and_incomplete_candidates_are_rejected(self):
        arrays, resolved, config = self.fixture()
        for value in (None, np.ones((2, 2, 2), dtype=bool), arrays['edit_region'].astype(np.uint8)):
            with self.subTest(edit_region=value is None), self.assertRaisesRegex(ValueError, 'edit_region'):
                apply_morphology({**arrays, 'edit_region': value}, resolved, config)
        selected = arrays['selected'].copy()
        selected[15, 15, 16] = True  # A soft-tissue voxel cannot masquerade as a tracked target.
        with self.assertRaisesRegex(ValueError, 'subset'):
            apply_morphology({**arrays, 'selected': selected}, resolved, config)
        domain = arrays['edit_region'].copy()
        domain[15, 15, 15] = False
        with self.assertRaisesRegex(ValueError, 'subset'):
            apply_morphology({**arrays, 'edit_region': domain}, resolved, config)
        arrays['act'][15, 15, 20] = -7
        arrays['tissue'] = coarse_labels(arrays['act'], resolved['catalog'])
        with self.assertRaisesRegex(ValueError, 'full current candidate mask'):
            apply_morphology(arrays, resolved, config)
        with patch('fakect_morphology.MAX_MORPHOLOGY_VOXELS', 100), self.assertRaisesRegex(ValueError, 'limited'):
            apply_morphology(arrays, resolved, config)

    def test_lineage_guard_rejects_an_unselected_winning_seed(self):
        arrays, resolved, config = self.fixture()
        arrays['act'][15, 15, 20] = -7
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        unselected_index = np.ravel_multi_index((15, 15, 20), arrays['act'].shape)
        owners = np.full(arrays['act'].shape, unselected_index, dtype=np.int32)
        with patch('fakect_morphology._owned_paths', return_value=(np.zeros_like(arrays['act']), owners)), \
                self.assertRaisesRegex(RuntimeError, 'input-state selected target seeds'):
            apply_morphology(arrays, resolved, config)

    def test_growth_reach_adds_to_halo_and_rejects_invalid_or_missing_context(self):
        _, resolved, config = self.fixture(distance=2.)
        ordinary = validate_edit_geometry(resolved, config)
        grown = validate_edit_geometry({**resolved, 'growth_reach_mm': 3.}, config)
        self.assertEqual(grown['required_halo_mm'], ordinary['required_halo_mm'] + 3.)
        self.assertEqual(grown['growth_reach_mm'], 3.)
        self.assertNotIn('growth_reach_mm', ordinary)
        with self.assertRaisesRegex(ValueError, 'inherited growth'):
            validate_edit_geometry({**resolved, 'growth_reach_mm': 10.}, config)
        for value in (-1, float('nan'), float('inf'), None, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'growth_reach_mm'):
                validate_edit_geometry({**resolved, 'growth_reach_mm': value}, config)

    def test_boundary_mode_is_unchanged_and_none_keeps_selection_provenance_empty(self):
        arrays, resolved, config = test_morphology.MorphologyTests().fixture()
        arrays['edit_region'] = np.ones_like(arrays['roi'])  # Ignored by legacy boundary mode.
        legacy = apply_morphology(arrays, resolved, config)
        explicit = apply_morphology(arrays, resolved, {**config, 'recipe': {'roi_role': 'boundary'}})
        self.assertEqual(legacy['summary'], explicit['summary'])
        self.assertEqual(set(legacy), set(explicit))
        self.assertNotIn('target_seed_index', legacy)
        for key, value in legacy.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, explicit[key])
        arrays, resolved, config = self.fixture(operation='none')
        saved = deepcopy(arrays)
        result = apply_morphology(arrays, resolved, config)
        self.assertEqual(result['summary']['counts']['changed_outside_selection_roi'], 0)
        self.assertTrue(np.all(result['target_seed_index'] == -1))
        np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
        for key, value in saved.items():
            np.testing.assert_array_equal(arrays[key], value)


if __name__ == '__main__':
    unittest.main()
