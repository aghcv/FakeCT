"""Curvature-side budgets constrain native edits without changing source labels."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_morphology
from fakect_direction import direction_weight_field, directional_spec
from fakect_morphology import apply_morphology, tube_position_mm
from fakect_roi import tube_mask
from fakect_tube_range import resolve_tube_range


class DirectionalMorphologyTests(unittest.TestCase):
    def fixture(self, direction='outer', operation='dilation', straight=False):
        arrays, resolved, config = test_morphology.MorphologyTests().fixture(
            distance=1.5, operation=operation)
        if straight:
            nodes = ((15., 10., 15.), (15., 13., 15.), (15., 17., 15.), (15., 20., 15.))
        else:
            angles = np.linspace(-np.pi * 7 / 18, np.pi * 7 / 18, 21)
            nodes = tuple((15. + 6. * np.cos(angle), 15. + 6. * np.sin(angle), 15.)
                          for angle in angles)
        shape = arrays['act'].shape
        resolved.update(roi_kind='tube', roi_nodes_ijk=nodes, roi_radii_mm=(3.,) * len(nodes))
        arrays['roi'] = tube_mask(shape, (0, 0, 0), nodes, (1, 1, 1), (3.,) * len(nodes))
        arrays['act'].fill(20)
        arrays['act'][tube_mask(shape, (0, 0, 0), nodes, (1, 1, 1), (1.25,) * len(nodes))] = -7
        config['edit'].update(direction=direction, angular_width_deg=180.)
        config['reassignment']['max_distance_mm'] = 1.
        config['centerline'] = {'smoothing_mm': 0., 'sample_step_mm': .2, 'min_curvature_per_mm': .002}
        test_morphology.MorphologyTests().refresh(arrays, resolved)
        return arrays, resolved, config

    def test_circle_inner_and_outer_point_to_opposite_known_radial_sides(self):
        arrays, resolved, config = self.fixture()
        outer = direction_weight_field(arrays['act'].shape, resolved, config, arrays['roi'])
        config['edit']['direction'] = 'inner'
        inner = direction_weight_field(arrays['act'].shape, resolved, config, arrays['roi'])
        # At the circle's rightmost point (21,15,15), the curvature center
        # is to the left: inner=-i, outer=+i. Array indexing is k,j,i.
        self.assertGreater(outer['direction_weight'][15, 15, 23], .98)
        self.assertEqual(inner['direction_weight'][15, 15, 23], 0.)
        self.assertGreater(inner['direction_weight'][15, 15, 19], .98)
        self.assertEqual(outer['direction_weight'][15, 15, 19], 0.)
        self.assertFalse(np.any((inner['direction_weight'] > 0) & (outer['direction_weight'] > 0)))
        np.testing.assert_array_equal(inner['direction_reliable_mask'], outer['direction_reliable_mask'])
        np.testing.assert_allclose(inner['direction_cosine'], -outer['direction_cosine'])

    def test_narrower_sector_has_smaller_support_and_raised_cosine_taper(self):
        arrays, resolved, config = self.fixture()
        wide = direction_weight_field(arrays['act'].shape, resolved, config, arrays['roi'])
        config['edit']['angular_width_deg'] = 60.
        narrow = direction_weight_field(arrays['act'].shape, resolved, config, arrays['roi'])
        # Offset (+2,0,+2) is approximately 45 degrees from outward: weight
        # 1/2 in a 180-degree sector, but outside a 60-degree sector.
        self.assertAlmostEqual(wide['direction_weight'][17, 15, 23], .5, delta=.015)
        self.assertEqual(narrow['direction_weight'][17, 15, 23], 0.)
        self.assertGreater(narrow['direction_weight'][15, 15, 23], .98)
        support = narrow['direction_weight'] > 0
        self.assertLess(support.sum(), (wide['direction_weight'] > 0).sum())
        self.assertTrue(np.all(narrow['direction_cosine'][support] > np.cos(np.pi / 6) - 1e-7))
        self.assertTrue(np.all(narrow['direction_weight'] <= wide['direction_weight'] + 1e-12))
        self.assertTrue(np.all(narrow['direction_weight'][~arrays['roi']] == 0))

    def test_straight_parent_skips_directional_edit_instead_of_inventing_an_inner_side(self):
        arrays, resolved, config = self.fixture(straight=True)
        field = direction_weight_field(arrays['act'].shape, resolved, config, arrays['roi'])
        self.assertFalse(field['direction_reliable_mask'].any())
        self.assertFalse(field['direction_weight'].any())
        self.assertEqual(field['summary']['angular_supported_roi_voxels'], 0)
        result = apply_morphology(arrays, resolved, config)
        self.assertFalse(result['changed_mask'].any())
        self.assertEqual(result['summary']['direction']['angular_supported_roi_voxels'], 0)
        np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])

    def test_two_node_subrange_uses_the_original_curved_parent_frame(self):
        arrays, resolved, config = self.fixture()
        parent = resolved['roi_nodes_ijk']
        selected = resolve_tube_range(parent, resolved['roi_radii_mm'], (1, 1, 1), point_range=(10, 11))
        self.assertEqual(len(selected['nodes_ijk']), 2)
        lower, upper = (selected['metadata'][key] for key in ('start_distance_mm', 'end_distance_mm'))
        arc = tube_position_mm(arrays['act'].shape, (0, 0, 0), parent, (1, 1, 1))
        region = arrays['roi'] & (arc >= lower - 1e-9) & (arc <= upper + 1e-9)
        ranged = {**resolved, 'roi_nodes_ijk': selected['nodes_ijk'], 'roi_radii_mm': selected['radii_mm'],
                  'range_parent_nodes_ijk': parent, 'range_interval_mm': (lower, upper)}
        expected = direction_weight_field(arrays['act'].shape, resolved, config, region)
        actual = direction_weight_field(arrays['act'].shape, ranged, config, region)
        self.assertGreater(actual['direction_weight'].sum(), 0.)
        for key in ('direction_weight', 'direction_reliable_mask', 'direction_cosine'):
            np.testing.assert_array_equal(actual[key], expected[key])
        self.assertEqual(actual['summary']['frame_metadata']['input_node_count'], len(parent))

    def test_actual_dilation_and_erosion_change_only_the_requested_supported_side(self):
        for operation in ('dilation', 'erosion'):
            for direction in ('inner', 'outer'):
                with self.subTest(operation=operation, direction=direction):
                    arrays, resolved, config = self.fixture(direction=direction, operation=operation)
                    result = apply_morphology(arrays, resolved, config)
                    supported = result['direction_weight'] > 0
                    self.assertGreater(result['changed_mask'].sum(), 0)
                    self.assertFalse(np.any(result['changed_mask'] & ~supported))
                    self.assertTrue(np.all(result['direction_reliable_mask'][result['changed_mask']]))
                    self.assertTrue(np.all(result['direction_cosine'][result['changed_mask']] > 0))
                    # True source interfaces at i=20 and22 erode; i=19 and23
                    # are neighboring soft tissue that can be borrowed.
                    inner_i, outer_i = (19, 23) if operation == 'dilation' else (20, 22)
                    edited_i, preserved_i = (inner_i, outer_i) if direction == 'inner' else (outer_i, inner_i)
                    self.assertTrue(result['changed_mask'][15, 15, edited_i])
                    self.assertFalse(result['changed_mask'][15, 15, preserved_i])
                    np.testing.assert_array_equal(result['edited_labels'][~supported], arrays['act'][~supported])
                    np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][~supported], arrays['atn'][~supported])
                    np.testing.assert_allclose(result['strength_mm'], 1.5 * result['direction_weight'])

    def test_angular_budget_is_multiplied_before_tissue_resistance(self):
        arrays, resolved, config = self.fixture()
        config['reassignment'] = {'mode': 'stiffness', 'allowed_tissues': (), 'max_distance_mm': 1.,
                                  'unresolved': 'preserve', 'stiffness': {
                                      'default': .25, 'tissues': {'bone': 1., 'skin': .95}, 'labels': {}}}
        result = apply_morphology(arrays, resolved, config)
        np.testing.assert_allclose(result['strength_mm'], 1.5 * result['direction_weight'])
        np.testing.assert_allclose(result['effective_strength_mm'],
                                   1.5 * result['direction_weight'] * (1 - result['stiffness_field']))
        self.assertFalse(np.any(result['changed_mask'] & (result['direction_weight'] == 0)))

    def test_directional_execution_preserves_config_geometry_and_original_arrays(self):
        arrays, resolved, config = self.fixture()
        saved_arrays = {key: value.copy() for key, value in arrays.items()}
        saved_resolved, saved_config = deepcopy(resolved), deepcopy(config)
        first = apply_morphology(arrays, resolved, config)
        second = apply_morphology(arrays, resolved, config)
        for key, value in saved_arrays.items():
            np.testing.assert_array_equal(arrays[key], value)
        self.assertEqual(resolved, saved_resolved)
        self.assertEqual(config, saved_config)
        self.assertEqual(first['summary'], second['summary'])
        np.testing.assert_array_equal(first['edited_labels'], second['edited_labels'])
        np.testing.assert_array_equal(first['attenuation_proxy_per_pixel'], second['attenuation_proxy_per_pixel'])

    def test_missing_direction_and_explicit_all_keep_ordinary_bilateral_edit(self):
        arrays, resolved, config = self.fixture()
        config['edit'].pop('direction')
        config['edit'].pop('angular_width_deg')
        config.pop('centerline')
        legacy = apply_morphology(arrays, resolved, config)
        config['edit']['direction'] = 'all'
        explicit = apply_morphology(arrays, resolved, config)
        self.assertTrue(legacy['changed_mask'][15, 15, 19])
        self.assertTrue(legacy['changed_mask'][15, 15, 23])
        self.assertEqual(legacy['summary'], explicit['summary'])
        for key in ('edited_labels', 'attenuation_proxy_per_pixel', 'strength_mm', 'changed_mask'):
            np.testing.assert_array_equal(legacy[key], explicit[key])

    def test_programmatic_directional_width_rejects_beyond_halfplane_and_nonfinite_inputs(self):
        _, resolved, config = self.fixture()
        for value in (0, -1, 180.01, float('nan'), float('inf')):
            with self.subTest(width=value), self.assertRaisesRegex(ValueError, 'angular_width_deg'):
                config['edit']['angular_width_deg'] = value
                directional_spec(resolved, config)


if __name__ == '__main__':
    unittest.main()
