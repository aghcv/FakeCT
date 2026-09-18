"""Arc profiles conserve native volume and retain physical parent coordinates."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_arc_profile import build_arc_profile, render_arc_profile


SERIES = {'before': 'target_mask_before', 'after': 'target_mask_after',
          'added': 'added_mask', 'removed': 'removed_mask',
          'blocked': 'blocked_mask', 'unresolved': 'unresolved_mask'}


class ArcProfileTests(unittest.TestCase):
    def fixture(self, shape=(3, 3, 13), nodes=((0., 1., 1.), (12., 1., 1.)),
                spacing=(1., 1., 1.), origin=(0, 0, 0)):
        edit = {key: np.zeros(shape, dtype=bool) for key in SERIES.values()}
        edit['strength_mm'] = np.zeros(shape, dtype=np.float64)
        resolved = {'roi_kind': 'tube', 'roi_nodes_ijk': nodes,
                    'roi_radii_mm': (2.,) * len(nodes), 'spacing_ijk_mm': spacing,
                    'crop_low_ijk': origin,
                    'crop_high_ijk_exclusive': tuple(np.asarray(origin) + shape[::-1])}
        config = {'edit': {'operation': 'dilation', 'roi': 'main'},
                  'roi': {'shape': 'tube', 'center_ijk': nodes, 'radius_mm': (2.,) * len(nodes)}}
        return edit, resolved, config, np.ones(shape, dtype=bool)

    def test_all_mask_series_conserve_volume_interior_plus_endpoint_bins(self):
        edit, resolved, config, region = self.fixture(
            shape=(7, 5, 9), nodes=((2., 2., 1.), (6., 2., 5.)), spacing=(1., 2., 3.))
        points = [(0, 2, 0), (1, 2, 2), (3, 2, 4), (5, 2, 6), (6, 2, 8)]
        for point in points:
            edit['target_mask_before'][point] = True
        edit['target_mask_after'][:] = edit['target_mask_before']
        edit['target_mask_after'][points[0]] = False
        edit['target_mask_after'][2, 2, 3] = True
        edit['added_mask'] = edit['target_mask_after'] & ~edit['target_mask_before']
        edit['removed_mask'] = edit['target_mask_before'] & ~edit['target_mask_after']
        edit['blocked_mask'][1, 3, 2] = True
        edit['blocked_mask'][4, 2, 5] = True
        edit['unresolved_mask'][points[-1]] = True
        result = build_arc_profile(edit, resolved, config, region)
        self.assertEqual(result['profile_coordinate'], 'centerline_arc_length_mm')
        self.assertAlmostEqual(result['profile_length_mm'], np.sqrt(160.))
        self.assertIn('profile_roi_name', result)
        self.assertIn('profile_area_semantics', result)
        profile = result['arc_profile']
        widths = np.asarray(profile['widths_mm'])
        self.assertAlmostEqual(widths.sum(), result['profile_length_mm'])
        for name, mask_key in SERIES.items():
            with self.subTest(series=name):
                volume = np.asarray(profile['volume_mm3'][name])
                endpoints = np.asarray(profile['endpoint_volume_mm3'][name])
                density = np.asarray(profile['volume_per_length_mm2'][name])
                self.assertEqual(endpoints.shape, (2,))
                self.assertAlmostEqual(volume.sum() + endpoints.sum(), edit[mask_key].sum() * 6.)
                np.testing.assert_allclose(density * widths, volume)
        np.testing.assert_allclose(profile['endpoint_volume_mm3']['before'], [12., 12.])

    def test_anisotropic_kji_arrays_and_nonzero_ijk_origin_use_physical_arc_length(self):
        edit, resolved, config, region = self.fixture(
            shape=(7, 3, 3), nodes=((11., 21., 31.), (11., 21., 35.)),
            spacing=(2., 3., 4.), origin=(10, 20, 30))
        for k in (0, 2, 4, 6):
            edit['target_mask_before'][k, 1, 1] = True
        result = build_arc_profile(edit, resolved, config, region)
        profile = result['arc_profile']
        self.assertEqual(result['profile_length_mm'], 16.)
        np.testing.assert_allclose(profile['edges_mm'], [0., 8., 16.])
        np.testing.assert_allclose(profile['centers_mm'], [4., 12.])
        np.testing.assert_allclose(profile['volume_mm3']['before'], [24., 24.])
        np.testing.assert_allclose(profile['volume_per_length_mm2']['before'], [3., 3.])
        np.testing.assert_allclose(profile['endpoint_volume_mm3']['before'], [24., 24.])

    def test_curved_path_separates_locations_with_the_same_native_k(self):
        nodes = ((1., 1., 1.), (1., 1., 5.), (7., 1., 5.), (7., 1., 1.))
        edit, resolved, config, region = self.fixture(shape=(7, 3, 9), nodes=nodes)
        edit['target_mask_before'][3, 1, 1] = edit['removed_mask'][3, 1, 1] = True
        edit['target_mask_after'][3, 1, 7] = edit['added_mask'][3, 1, 7] = True
        profile = build_arc_profile(edit, resolved, config, region)['arc_profile']
        # Both positions have k=3, but are at arc distances 2 and 12 mm on
        # opposite arms of the ordered U-shaped parent.
        np.testing.assert_array_equal(np.flatnonzero(profile['volume_mm3']['before']), [1])
        np.testing.assert_array_equal(np.flatnonzero(profile['volume_mm3']['after']), [6])
        np.testing.assert_array_equal(profile['volume_mm3']['before'], profile['volume_mm3']['removed'])
        np.testing.assert_array_equal(profile['volume_mm3']['after'], profile['volume_mm3']['added'])

    def test_subrange_retains_full_parent_origin_length_and_interval(self):
        parent = ((0., 1., 1.), (4., 1., 1.), (8., 1., 1.), (12., 1., 1.))
        edit, resolved, config, region = self.fixture(nodes=parent[1:3])
        resolved.update(range_parent_nodes_ijk=parent, range_interval_mm=(4., 8.))
        edit['target_mask_before'][1, 1, 5] = True
        edit['target_mask_after'][1, 1, 7] = True
        result = build_arc_profile(edit, resolved, config, region)
        profile = result['arc_profile']
        self.assertEqual(result['profile_length_mm'], 12.)
        np.testing.assert_allclose(profile['range_interval_mm'], [4., 8.])
        np.testing.assert_allclose(profile['edges_mm'], np.arange(0., 13., 2.))
        np.testing.assert_array_equal(np.flatnonzero(profile['volume_mm3']['before']), [2])
        np.testing.assert_array_equal(np.flatnonzero(profile['volume_mm3']['after']), [3])

    def test_growth_is_binned_at_current_nearest_arc_position_not_ancestor_position(self):
        edit, resolved, config, region = self.fixture()
        edit['target_mask_before'][1, 1, 2] = True
        edit['target_mask_after'][1, 1, 2] = edit['target_mask_after'][1, 1, 9] = True
        edit['added_mask'][1, 1, 9] = True
        ancestry = np.full(region.shape, -1, dtype=np.int32)
        ancestry[edit['target_mask_after']] = np.ravel_multi_index((1, 1, 2), region.shape)
        edit['target_origin_index'] = ancestry
        profile = build_arc_profile(edit, resolved, config, region)['arc_profile']
        np.testing.assert_array_equal(np.flatnonzero(profile['volume_mm3']['before']), [1])
        np.testing.assert_array_equal(np.flatnonzero(profile['volume_mm3']['added']), [4])
        np.testing.assert_array_equal(np.flatnonzero(profile['volume_mm3']['after']), [1, 4])

    def test_maximum_strength_ignores_values_outside_edit_region(self):
        edit, resolved, config, region = self.fixture()
        region[:] = False
        region[1, 1, 4] = region[1, 1, 7] = True
        edit['strength_mm'][1, 1, 4] = 3.
        edit['strength_mm'][1, 1, 5] = 99.  # Same arc bin, outside permitted region.
        edit['strength_mm'][1, 1, 7] = 2.
        region[1, 1, 0] = True
        edit['strength_mm'][1, 1, 0] = 11.  # Endpoint request stays outside interior bins.
        profile = build_arc_profile(edit, resolved, config, region)['arc_profile']
        np.testing.assert_allclose(profile['maximum_strength_mm'], [0., 0., 3., 2., 0., 0.])
        np.testing.assert_allclose(profile['endpoint_maximum_strength_mm'], [11., 0.])

    def test_profile_name_uses_explicit_reference_then_parent_then_main(self):
        edit, resolved, config, region = self.fixture()
        self.assertEqual(build_arc_profile(edit, resolved, config, region)['profile_roi_name'], 'main')
        config['roi']['parent_roi'] = 'ascending'
        self.assertEqual(build_arc_profile(edit, resolved, config, region)['profile_roi_name'], 'ascending')
        config['profile_roi_name'] = 'named_reference'
        self.assertEqual(build_arc_profile(edit, resolved, config, region)['profile_roi_name'], 'named_reference')

    def test_bin_count_is_bounded_for_short_and_long_parent_paths(self):
        for length, count in ((.5, 1), (2000., 512)):
            with self.subTest(length=length):
                edit, resolved, config, region = self.fixture(
                    shape=(3, 3, 3), nodes=((0., 1., 1.), (length, 1., 1.)))
                profile = build_arc_profile(edit, resolved, config, region)['arc_profile']
                self.assertEqual(len(profile['centers_mm']), count)
                self.assertEqual(len(profile['edges_mm']), count + 1)
                self.assertAlmostEqual(np.sum(profile['widths_mm']), length)
                np.testing.assert_allclose(profile['range_interval_mm'], [0., length])

    def test_input_masks_strength_geometry_and_config_are_unchanged(self):
        edit, resolved, config, region = self.fixture()
        edit['target_mask_before'][1, 1, 3] = True
        edit['target_mask_after'][1, 1, 3:5] = True
        edit['strength_mm'][region] = 2.5
        saved = {key: value.copy() for key, value in edit.items()}
        saved_resolved, saved_config, saved_region = deepcopy(resolved), deepcopy(config), region.copy()
        build_arc_profile(edit, resolved, config, region)
        for key, value in saved.items():
            np.testing.assert_array_equal(edit[key], value)
        np.testing.assert_array_equal(region, saved_region)
        self.assertEqual(resolved, saved_resolved)
        self.assertEqual(config, saved_config)

    def test_non_tube_uses_fallback_without_requiring_tube_inputs(self):
        self.assertIsNone(build_arc_profile({}, {'roi_kind': 'sphere'}, {}, None))

    def test_guarded_strength_plot_reports_accepted_field_and_original_requested_distance(self):
        from matplotlib.figure import Figure
        edit, resolved, config, region = self.fixture()
        config['edit'].update(operation='erosion', distance_mm=4.)
        edit['strength_mm'][region] = 2.
        guard = {'enabled': True, 'status': 'reduced', 'requested_distance_mm': 4.,
                 'accepted_distance_mm': 2., 'retained_volume_ratio': .75, 'min_volume_ratio': .6}
        edit['summary'] = {'erosion_safeguard': guard}
        metadata = build_arc_profile(edit, resolved, config, region)
        self.assertIn('accepted-trial spatial distance', metadata['arc_profile']['strength_semantics'])
        self.assertIn('not the original INI request or achieved displacement',
                      metadata['arc_profile']['strength_semantics'])
        self.assertEqual(metadata['arc_profile']['erosion_safeguard'], guard)
        self.assertIsNot(metadata['arc_profile']['erosion_safeguard'], guard)
        snapshot = {}

        def capture(figure, *args, **kwargs):
            snapshot['title'] = figure._suptitle.get_text()
            snapshot['caption'] = figure.texts[-1].get_text()
            snapshot['axis'] = figure.axes[2].get_ylabel()
            snapshot['strength'] = figure.axes[2].lines[0].get_ydata().copy()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, 'savefig', capture):
                result = render_arc_profile(metadata, config, {}, {}, directory)
            header = (Path(directory)/result['profile_data']).read_text().splitlines()[0]
        self.assertIn('maximum_accepted_trial_distance_mm', header)
        self.assertNotIn('maximum_request_mm', header)
        self.assertIn('accepted trial profile', snapshot['title'])
        self.assertIn('Maximum accepted-trial distance', snapshot['axis'])
        self.assertIn('requested 4 mm; accepted 2 mm; local volume retained 75% (minimum 60%)', snapshot['caption'])
        np.testing.assert_allclose(snapshot['strength'], 2.)

        # Disabled metadata follows the legacy request-field convention exactly.
        guard['enabled'] = False
        legacy = build_arc_profile(edit, resolved, config, region)
        self.assertNotIn('erosion_safeguard', legacy['arc_profile'])
        self.assertIn('Maximum requested spatial distance', legacy['arc_profile']['strength_semantics'])


if __name__ == '__main__':
    unittest.main()
