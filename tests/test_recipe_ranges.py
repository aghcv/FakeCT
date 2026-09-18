"""Native edits obey inherited tube intervals, local profiles and recipe order."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_recipe
from fakect_recipe import apply_recipe, recipe_masks, recipe_regions
from fakect_roi import tube_mask


class RecipeRangeTests(unittest.TestCase):
    def fixture(self, spacing=(1., 1., 1.), nodes=None):
        arrays, resolved, config = test_recipe.RecipeTests().fixture(spacing)
        nodes = nodes or ((15., 15., 7.), (15., 15., 9.),
                          (15., 15., 15.), (15., 15., 23.))
        radii = (3.,) * len(nodes)
        config['roi'] = {'shape': 'tube', 'center_ijk': nodes, 'radius_mm': radii,
                         'coordinate_reviewed': True, 'crop_half_width_mm': 15.}
        config['rois'] = {}
        config['edits'] = {'grow': {**config['edits']['grow'], 'roi': 'main'}}
        config['reassignment']['max_distance_mm'] = 1.
        resolved.update(roi_kind='tube', roi_nodes_ijk=nodes, roi_radii_mm=radii)
        shape = arrays['act'].shape
        arrays['roi'] = tube_mask(shape, (0, 0, 0), nodes, spacing, radii)
        arrays['act'].fill(20)
        arrays['act'][tube_mask(shape, (0, 0, 0), nodes, spacing, (.75,) * len(nodes))] = -7
        test_recipe.RecipeTests().refresh(arrays, resolved)
        return arrays, resolved, config

    def test_uniform_dilation_and_erosion_cannot_leak_past_range_round_caps(self):
        for operation in ('dilation', 'erosion'):
            with self.subTest(operation=operation):
                arrays, resolved, config = self.fixture()
                config['edits']['grow'].update(operation=operation, path_percent=(30., 75.))
                regions = recipe_regions(config, resolved)
                subpath = regions['edit:grow']
                rounded = tube_mask(arrays['act'].shape, (0, 0, 0), subpath['center_ijk'],
                                    (1, 1, 1), subpath['radius_mm'])
                mask = recipe_masks(arrays, resolved, config)['edit:grow']
                # The selected centerline spans k=11.8..19. Its standalone
                # rounded tube would include k=11 and 20, outside 30..75%.
                self.assertTrue(rounded[11, 15, 15])
                self.assertTrue(rounded[20, 15, 15])
                self.assertFalse(mask[11, 15, 15])
                self.assertFalse(mask[20, 15, 15])
                self.assertTrue(mask[12, 15, 15])
                self.assertTrue(mask[19, 15, 15])
                result = apply_recipe(arrays, resolved, config)
                changed = np.argwhere(result['changed_mask'])
                self.assertGreater(len(changed), 0)
                self.assertGreaterEqual(changed[:, 0].min(), 12)
                self.assertLessEqual(changed[:, 0].max(), 19)
                self.assertFalse(np.any(result['changed_mask'] & ~mask))
                np.testing.assert_array_equal(result['edited_labels'][~mask], arrays['act'][~mask])
                np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][~mask], arrays['atn'][~mask])

    def test_disjoint_ranges_share_main_under_overlap_error(self):
        arrays, resolved, config = self.fixture()
        config['edits']['grow']['path_percent'] = (10., 40.)
        config['edits']['later'] = {**config['edits']['grow'], 'path_percent': (60., 90.)}
        config['recipe'] = {'steps': ('grow', 'later'), 'overlap': 'error'}
        masks = recipe_masks(arrays, resolved, config)
        self.assertFalse(np.any(masks['edit:grow'] & masks['edit:later']))
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event['result']['changed_mask'].any() for event in events))
        self.assertEqual(result['summary']['roi_overlaps'][0]['effective_overlap_voxels'], 0)
        self.assertEqual(set(result['summary']['roi_coverage']), {'edit:grow', 'edit:later'})

    def test_overlapping_ranges_reject_early_or_chain_full_state_sequentially(self):
        arrays, resolved, config = self.fixture()
        config['edits']['grow']['path_percent'] = (10., 60.)
        config['edits']['shrink'] = {**config['edits']['grow'], 'operation': 'erosion',
                                   'path_percent': (30., 75.)}
        config['recipe'] = {'steps': ('grow', 'shrink'), 'overlap': 'error'}
        events = []
        with self.assertRaisesRegex(ValueError, 'overlap at'):
            apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(events, [])
        config['recipe']['overlap'] = 'sequential'
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertGreater(result['summary']['roi_overlaps'][0]['effective_overlap_voxels'], 0)
        self.assertTrue(all(event['result']['changed_mask'].any() for event in events))
        np.testing.assert_array_equal(events[1]['before_arrays']['act'], events[0]['result']['edited_labels'])
        np.testing.assert_array_equal(events[1]['before_arrays']['atn'],
                                      events[0]['result']['attenuation_proxy_per_pixel'])
        initial_target = arrays['candidates'] & events[1]['before_arrays']['roi']
        self.assertGreater(events[1]['before_arrays']['selected'].sum(), initial_target.sum())
        self.assertEqual(result['summary']['steps'][0]['labels_after_sha256'],
                         result['summary']['steps'][1]['labels_before_sha256'])

    def test_point_and_percent_intervals_agree_for_uneven_anisotropic_bent_path(self):
        # Three physical segments are 2, 6 and 12 mm despite native lengths
        # 2, 3 and 4. Controls 2..3 therefore mean 10..40%, not 1/3..2/3.
        nodes = ((10., 12., 12.), (12., 12., 12.), (12., 15., 12.), (12., 15., 16.))
        arrays, resolved, config = self.fixture(spacing=(1., 2., 3.), nodes=nodes)
        config['edits']['grow']['point_range'] = (2, 3)
        percent = deepcopy(config)
        percent['edits']['grow'].pop('point_range')
        percent['edits']['grow']['path_percent'] = (10., 40.)
        point_region = recipe_regions(config, resolved)['edit:grow']
        percent_region = recipe_regions(percent, resolved)['edit:grow']
        self.assertEqual(point_region['center_ijk'], nodes[1:3])
        self.assertEqual(point_region['center_ijk'], percent_region['center_ijk'])
        self.assertEqual(point_region['range_metadata']['start_distance_mm'], 2.)
        self.assertEqual(point_region['range_metadata']['end_distance_mm'], 8.)
        np.testing.assert_array_equal(recipe_masks(arrays, resolved, config)['edit:grow'],
                                      recipe_masks(arrays, resolved, percent)['edit:grow'])
        first, second = apply_recipe(arrays, resolved, config), apply_recipe(arrays, resolved, percent)
        self.assertGreater(first['changed_mask'].sum(), 0)
        for name in ('edited_labels', 'attenuation_proxy_per_pixel', 'changed_mask', 'strength_mm'):
            np.testing.assert_array_equal(first[name], second[name])

    def test_gaussian_peak_and_taper_are_local_to_selected_interval(self):
        arrays, resolved, config = self.fixture()
        config['edits']['grow'].update(profile='gaussian', profile_axis='tube', distance_mm=2.,
                                      path_percent=(25., 50.))
        result = apply_recipe(arrays, resolved, config)
        self.assertGreater(result['added_mask'].sum(), 0)
        # Selected k=11..15 has its own midpoint at k=13.
        self.assertAlmostEqual(result['strength_mm'][13, 15, 15], 2.)
        self.assertEqual(result['strength_mm'][11, 15, 15], 0.)
        self.assertEqual(result['strength_mm'][15, 15, 15], 0.)
        self.assertGreater(result['strength_mm'][12, 15, 15], 0.)
        self.assertLess(result['strength_mm'][12, 15, 15], 2.)
        config['edits']['grow'].pop('path_percent')
        whole = apply_recipe(arrays, resolved, config)
        self.assertAlmostEqual(whole['strength_mm'][15, 15, 15], 2.)
        self.assertLess(whole['strength_mm'][13, 15, 15], 2.)
        self.assertGreater(whole['strength_mm'][11, 15, 15], 0.)

    def test_main_without_range_matches_legacy_whole_named_roi(self):
        for profile in ('uniform', 'gaussian'):
            with self.subTest(profile=profile):
                arrays, resolved, config = self.fixture()
                config['edits']['grow'].update(profile=profile, profile_axis='tube')
                legacy = deepcopy(config)
                legacy['rois']['whole'] = {key: value for key, value in config['roi'].items()
                                            if key != 'crop_half_width_mm'}
                legacy['edits']['grow']['roi'] = 'whole'
                np.testing.assert_array_equal(recipe_masks(arrays, resolved, config)['edit:grow'],
                                              recipe_masks(arrays, resolved, legacy)['whole'])
                first, second = apply_recipe(arrays, resolved, config), apply_recipe(arrays, resolved, legacy)
                for name in ('edited_labels', 'attenuation_proxy_per_pixel', 'strength_mm'):
                    np.testing.assert_array_equal(first[name], second[name])

    def test_named_parent_subrange_uses_its_own_path_and_main_boundary(self):
        arrays, resolved, config = self.fixture()
        config['rois']['short'] = {'shape': 'tube', 'center_ijk': ((15., 15., 9.), (15., 15., 17.)),
                                  'radius_mm': (2., 2.), 'coordinate_reviewed': False}
        config['edits']['grow'].update(roi='short', path_percent=(25., 75.))
        arrays['roi'][14, 15, 15] = False
        test_recipe.RecipeTests().refresh(arrays, resolved)
        regions = recipe_regions(config, resolved)
        self.assertEqual(set(regions), {'short', 'edit:grow'})
        self.assertEqual(regions['edit:grow']['center_ijk'], ((15., 15., 11.), (15., 15., 15.)))
        masks = recipe_masks(arrays, resolved, config)
        self.assertFalse(masks['edit:grow'][14, 15, 15])
        self.assertTrue(masks['edit:grow'][12, 15, 17])
        self.assertFalse(masks['edit:grow'][12, 15, 18])
        self.assertFalse(np.any(masks['edit:grow'] & ~masks['short']))
        result = apply_recipe(arrays, resolved, config)
        changed = np.argwhere(result['changed_mask'])
        self.assertGreater(len(changed), 0)
        self.assertGreaterEqual(changed[:, 0].min(), 11)
        self.assertLessEqual(changed[:, 0].max(), 15)
        self.assertEqual(result['edited_labels'][14, 15, 15], arrays['act'][14, 15, 15])

    def test_range_resolution_and_execution_preserve_config_geometry_and_source_arrays(self):
        arrays, resolved, config = self.fixture()
        config['edits']['grow']['point_range'] = (2, 3)
        original_config, original_resolved = deepcopy(config), deepcopy(resolved)
        original_arrays = {name: value.copy() for name, value in arrays.items()}
        recipe_regions(config, resolved)
        recipe_masks(arrays, resolved, config)
        first = apply_recipe(arrays, resolved, config)
        second = apply_recipe(arrays, resolved, config)
        self.assertEqual(config, original_config)
        self.assertEqual(resolved, original_resolved)
        for name, value in original_arrays.items():
            np.testing.assert_array_equal(arrays[name], value)
        self.assertEqual(first['summary'], second['summary'])
        for name in ('edited_labels', 'attenuation_proxy_per_pixel', 'strength_mm'):
            np.testing.assert_array_equal(first[name], second[name])


if __name__ == '__main__':
    unittest.main()
