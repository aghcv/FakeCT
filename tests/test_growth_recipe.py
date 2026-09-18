"""Selection-only recipes retain original ancestors when growth leaves the ROI."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_recipe
from fakect_recipe import apply_recipe, validate_recipe
from fakect_roi import sphere_mask


class GrowthRecipeTests(unittest.TestCase):
    def fixture(self):
        arrays, resolved, config = test_recipe.RecipeTests().fixture()
        roi = {'shape': 'sphere', 'center_ijk': (15., 15., 15.), 'radius_mm': .4,
               'coordinate_reviewed': True}
        config['roi'] = {**roi, 'crop_half_width_mm': 15.}
        config['rois'] = {}
        config['edits'] = {'grow': {**config['edits']['grow'], 'roi': 'main'}}
        config['recipe']['roi_role'] = 'selection'
        resolved['roi_radii_mm'] = (.4,)
        arrays['roi'] = sphere_mask(arrays['act'].shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), .4)
        test_recipe.RecipeTests().refresh(arrays, resolved)
        return arrays, resolved, config

    def set_outer(self, arrays, resolved, config, radius):
        config['roi']['radius_mm'] = radius
        resolved['roi_radii_mm'] = (radius,)
        arrays['roi'] = sphere_mask(arrays['act'].shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), radius)
        test_recipe.RecipeTests().refresh(arrays, resolved)

    def region(self, i):
        return {'shape': 'sphere', 'center_ijk': (float(i), 15., 15.), 'radius_mm': .4,
                'coordinate_reviewed': True}

    def test_repeated_growth_uses_offspring_seeds_and_keeps_original_ancestor(self):
        arrays, resolved, config = self.fixture()
        config['edits']['grow']['iterations'] = 2
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        root = np.ravel_multi_index((15, 15, 15), arrays['act'].shape)
        self.assertEqual([int(event['before_arrays']['selected'].sum()) for event in events], [1, 7])
        self.assertEqual(result['summary']['counts']['after'], 25)
        self.assertEqual(result['summary']['counts']['added'], 24)
        self.assertEqual(result['summary']['counts']['changed_outside_selection_roi'], 24)
        self.assertEqual(result['summary']['counts']['added_outside_selection_roi'], 24)
        self.assertEqual(result['summary']['full_target_after'], 25)
        self.assertEqual(result['target_origin_index'].dtype, np.dtype('int32'))
        self.assertTrue(np.all(result['target_origin_index'][result['target_mask_after']] == root))
        self.assertTrue(np.all(result['target_origin_index'][~result['target_mask_after']] == -1))
        # The second pass uses a first-pass offspring as its immediate seed,
        # while the persistent origin remains the single ORIGINAL center.
        second = events[1]['result']
        self.assertEqual(second['target_seed_index'][15, 15, 17],
                         np.ravel_multi_index((15, 15, 16), arrays['act'].shape))
        self.assertEqual(second['target_origin_index'][15, 15, 17], root)
        self.assertTrue(np.all(second['target_seed_index'][~second['added_mask']] == -1))
        np.testing.assert_array_equal(second['target_origin_before_index'], events[0]['result']['target_origin_index'])
        self.assertFalse(np.any(result['changed_mask'] & ~result['edit_region_mask']))
        np.testing.assert_array_equal(result['selection_roi_mask'], arrays['roi'])

    def test_later_overlapping_erosion_tracks_prior_growth_outside_original_roi(self):
        arrays, resolved, config = self.fixture()
        config['edits']['grow']['iterations'] = 2
        config['edits']['shrink'] = {**config['edits']['grow'], 'iterations': 1, 'operation': 'erosion'}
        config['recipe']['steps'] = ('grow', 'shrink')
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        shrink = events[2]
        self.assertEqual(shrink['before_arrays']['selected'].sum(), 25)
        self.assertEqual(shrink['before_arrays']['roi'].sum(), 1)
        self.assertTrue(shrink['result']['removed_mask'][15, 15, 17])
        self.assertFalse(arrays['roi'][15, 15, 17])
        self.assertEqual(shrink['result']['target_origin_index'][15, 15, 17], -1)
        self.assertEqual(result['summary']['counts']['after'], 7)
        self.assertEqual(result['summary']['counts']['added'], 6)
        self.assertEqual(result['summary']['counts']['changed_outside_selection_roi'], 6)
        self.assertEqual(result['summary']['planned_steps'][1]['inherited_growth_mm'], 2.)
        np.testing.assert_array_equal(shrink['before_arrays']['act'], events[1]['result']['edited_labels'])
        np.testing.assert_array_equal(shrink['before_arrays']['atn'], events[1]['result']['attenuation_proxy_per_pixel'])

    def test_same_id_neighbor_is_preserved_and_never_adopted_as_a_growth_seed(self):
        arrays, resolved, config = self.fixture()
        arrays['act'][15, 15, 18] = -7
        arrays['atn'][15, 15, 18] = .75
        test_recipe.RecipeTests().refresh(arrays, resolved)
        config['edits']['grow'].update(distance_mm=2., iterations=2)
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        other = np.ravel_multi_index((15, 15, 18), arrays['act'].shape)
        self.assertEqual(result['edited_labels'][15, 15, 18], -7)
        self.assertEqual(result['attenuation_proxy_per_pixel'][15, 15, 18], .75)
        self.assertEqual(result['target_origin_index'][15, 15, 18], other)
        self.assertFalse(result['target_mask_after'][15, 15, 18])
        # A false seed at i=18 would grow into i=19. A permitted two-edge
        # path from the tracked frontier cannot cross the unrelated target.
        self.assertEqual(result['edited_labels'][15, 15, 19], 20)
        for event in events:
            self.assertFalse(event['before_arrays']['selected'][15, 15, 18])
            self.assertFalse(np.any(event['result']['target_seed_index'] == other))
        self.assertTrue(np.all(result['attenuation_proxy_per_pixel'][result['added_mask']] == arrays['atn'][15, 15, 15]))
        self.assertGreater(sum(event['result']['summary']['counts']['unselected_target_contact_voxels']
                               for event in events), 0)

    def test_empty_original_selection_cannot_adopt_growth_that_later_enters_it(self):
        arrays, resolved, config = self.fixture()
        self.set_outer(arrays, resolved, config, 4.)
        config['rois'] = {'source': self.region(15), 'empty': self.region(17)}
        config['edits']['grow'].update(roi='source', iterations=2)
        config['edits']['adopt'] = {**config['edits']['grow'], 'roi': 'empty', 'iterations': 1}
        config['recipe']['steps'] = ('grow', 'adopt')
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        last = events[-1]
        self.assertEqual(last['before_arrays']['act'][15, 15, 17], -7)
        self.assertTrue(last['before_arrays']['roi'][15, 15, 17])
        self.assertFalse(last['before_arrays']['selected'].any())
        self.assertFalse(last['result']['changed_mask'].any())
        self.assertEqual(result['summary']['steps'][-1]['status'], 'empty_target')
        self.assertEqual(result['summary']['steps'][-1]['original_selection_voxels'], 0)
        self.assertEqual(result['summary']['counts']['after'], 25)

    def test_partial_original_selector_tracks_only_its_winning_root_offspring(self):
        arrays, resolved, config = self.fixture()
        arrays['act'].fill(20)
        arrays['act'][15, 15, 14] = arrays['act'][15, 15, 16] = -7
        self.set_outer(arrays, resolved, config, 1.1)
        config['rois']['narrow'] = {**self.region(16), 'radius_mm': 1.4}
        config['edits']['grow']['iterations'] = 2
        config['edits']['shrink'] = {**config['edits']['grow'], 'roi': 'narrow',
                                   'iterations': 1, 'operation': 'erosion'}
        config['recipe']['steps'] = ('grow', 'shrink')
        snapshots = {key: value.copy() for key, value in arrays.items()}
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        root_left, root_right = (np.ravel_multi_index((15, 15, i), arrays['act'].shape)
                                 for i in (14, 16))
        shrink = events[-1]
        before, edited = shrink['before_arrays'], shrink['result']
        origins = edited['target_origin_before_index']
        expected = origins == root_right
        other_offspring = origins == root_left
        # Both roots grew in the first edit. The later original selector
        # includes root_right, but spatially also contains root_left offspring.
        self.assertEqual(result['summary']['steps'][0]['original_selection_voxels'], 2)
        self.assertEqual(result['summary']['steps'][-1]['original_selection_voxels'], 1)
        self.assertEqual(origins[15, 15, 15], root_left)
        self.assertTrue(before['roi'][15, 15, 15])
        self.assertFalse(before['selected'][15, 15, 15])
        self.assertGreater(np.count_nonzero(other_offspring & before['roi']), 0)
        self.assertGreater(np.count_nonzero(expected & ~before['roi']), 0)
        np.testing.assert_array_equal(before['selected'], expected)
        self.assertGreater(edited['removed_mask'].sum(), 0)
        self.assertFalse(np.any(edited['changed_mask'] & other_offspring))
        np.testing.assert_array_equal(edited['edited_labels'][other_offspring], before['act'][other_offspring])
        np.testing.assert_array_equal(edited['attenuation_proxy_per_pixel'][other_offspring], before['atn'][other_offspring])
        np.testing.assert_array_equal(edited['target_origin_index'][other_offspring], origins[other_offspring])
        for key, value in snapshots.items():
            np.testing.assert_array_equal(arrays[key], value)

    def test_original_arrays_config_and_frozen_pass_states_are_immutable_and_deterministic(self):
        arrays, resolved, config = self.fixture()
        config['edits']['grow']['iterations'] = 2
        saved_arrays = {key: value.copy() for key, value in arrays.items()}
        saved_config, saved_resolved = deepcopy(config), deepcopy(resolved)
        snapshots, events = [], []
        def capture(event):
            events.append(event)
            snapshots.append({key: value.copy() for key, value in event['before_arrays'].items()
                              if isinstance(value, np.ndarray)})
        first = apply_recipe(arrays, resolved, config, on_step=capture)
        second = apply_recipe(arrays, resolved, config)
        for key, value in saved_arrays.items():
            np.testing.assert_array_equal(arrays[key], value)
        self.assertEqual(config, saved_config)
        self.assertEqual(resolved, saved_resolved)
        for event, snapshot in zip(events, snapshots):
            for key, value in snapshot.items():
                np.testing.assert_array_equal(event['before_arrays'][key], value)
        self.assertEqual(first['summary'], second['summary'])
        for key in ('edited_labels', 'attenuation_proxy_per_pixel', 'target_origin_index'):
            np.testing.assert_array_equal(first[key], second[key])

    def test_cumulative_halo_rejects_repeats_and_later_overlapping_growth_before_editing(self):
        _, resolved, config = self.fixture()
        resolved.update(crop_low_ijk=(8, 8, 8), crop_high_ijk_exclusive=(23, 23, 23))
        config['edits']['grow']['distance_mm'] = 2.
        validate_recipe(config, resolved)  # 6 mm fits the 6.6 mm margin.
        config['edits']['grow']['iterations'] = 3
        with self.assertRaisesRegex(ValueError, 'halo.*inherited growth'):
            validate_recipe(config, resolved)
        config['edits']['grow']['iterations'] = 1
        config['edits']['again'] = dict(config['edits']['grow'])
        config['recipe']['steps'] = ('grow', 'again')
        with self.assertRaisesRegex(ValueError, 'edit.again.*halo.*inherited growth'):
            validate_recipe(config, resolved)

    def test_disjoint_original_selectors_do_not_inherit_another_region_growth_halo(self):
        arrays, resolved, config = self.fixture()
        self.set_outer(arrays, resolved, config, 10.)
        config['rois'] = {'left': self.region(12), 'right': self.region(23)}
        config['edits']['grow'].update(roi='left', distance_mm=2., iterations=2)
        config['edits']['later'] = {**config['edits']['grow'], 'roi': 'right', 'distance_mm': 1., 'iterations': 1}
        config['recipe']['steps'] = ('grow', 'later')
        plan = validate_recipe(config, resolved)
        self.assertEqual(plan['steps'][0]['halo']['required_halo_mm'], 8.)
        self.assertEqual(plan['steps'][1]['inherited_growth_mm'], 0.)
        self.assertEqual(plan['steps'][1]['halo']['required_halo_mm'], 5.)
        # Right has only 6.6 mm context. Incorrectly inheriting left's 4 mm
        # would reject it even though their original selections are disjoint.

    def test_growth_envelope_overlap_errors_before_callback_for_disjoint_selectors(self):
        arrays, resolved, config = self.fixture()
        self.set_outer(arrays, resolved, config, 5.)
        arrays['act'].fill(20)
        arrays['act'][15, 15, 13] = arrays['act'][15, 15, 17] = -7
        test_recipe.RecipeTests().refresh(arrays, resolved)
        config['rois'] = {'left': self.region(13), 'right': self.region(17)}
        config['edits']['grow'].update(roi='left', distance_mm=3.)
        config['edits']['other'] = {**config['edits']['grow'], 'roi': 'right'}
        config['recipe'].update(steps=('grow', 'other'), overlap='error')
        events = []
        with self.assertRaisesRegex(ValueError, 'Growth regions.*may overlap'):
            apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
