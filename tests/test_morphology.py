"""Small signed-label phantoms exercise locality, barriers, ownership and metrics."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_morphology import apply_morphology, strength_field_mm, validate_edit_geometry
from fakect_roi import sphere_mask, tube_mask
from fakect_tissues import coarse_labels


class MorphologyTests(unittest.TestCase):
    def fixture(self, spacing=(1, 1, 1), distance=1, operation='dilation'):
        shape = (31, 31, 31)
        catalog = {'categories': [{'id': code, 'name': name} for code, name in
                    [(0, 'background'), (1, 'soft_tissue'), (2, 'bone'), (5, 'artery'),
                     (6, 'vein'), (255, 'unknown')]],
                   'records': [{'original_id': value, 'original_name': name,
                                'classification': {'tissue_id': code, 'tissue_name': group}}
                               for value, name, code, group in
                               [(0, 'background', 0, 'background'), (-7, 'artery_a', 5, 'artery'),
                                (8, 'artery_b', 5, 'artery'), (20, 'soft_a', 1, 'soft_tissue'),
                                (21, 'soft_b', 1, 'soft_tissue'), (30, 'bone', 2, 'bone'),
                                (40, 'vein', 6, 'vein'), (999, 'unreviewed', 255, 'unknown')]]}
        labels = np.full(shape, 20, dtype=np.float32)
        labels[15, 15, 15] = -7
        roi = sphere_mask(shape, (0, 0, 0), (15, 15, 15), spacing, 5)
        arrays = {'act': labels, 'atn': (np.arange(labels.size).reshape(shape) / 1e6 + .01).astype(np.float32),
                  'roi': roi}
        resolved = {'crop_low_ijk': (0, 0, 0), 'crop_high_ijk_exclusive': (31, 31, 31),
                    'spacing_ijk_mm': spacing, 'roi_kind': 'sphere',
                    'roi_nodes_ijk': ((15, 15, 15),), 'roi_radii_mm': (5,), 'catalog': catalog}
        config = {'selection': {'tissue': 'artery', 'source_ids': ()},
                  'edit': {'operation': operation, 'distance_mm': distance, 'profile': 'uniform',
                           'profile_axis': 'k', 'shape_k': 10, 'shape_window': (0, 1)},
                  'reassignment': {'allowed_tissues': ('soft_tissue',), 'max_distance_mm': 3,
                                   'unresolved': 'preserve'}}
        self.refresh(arrays, resolved)
        return arrays, resolved, config

    def refresh(self, arrays, resolved):
        arrays['tissue'] = coarse_labels(arrays['act'], resolved['catalog'])
        arrays['candidates'] = arrays['tissue'] == 5
        arrays['selected'] = arrays['candidates'] & arrays['roi']

    def test_anisotropic_weighted_path_distance_not_isotropic_iterations(self):
        arrays, resolved, config = self.fixture((1, 2, 3), distance=2.1)
        result = apply_morphology(arrays, resolved, config)
        mask = result['target_mask_after']
        self.assertTrue(mask[15, 15, 17])  # Two 1-mm edges.
        self.assertTrue(mask[15, 16, 15])  # One 2-mm edge.
        self.assertFalse(mask[16, 15, 15])  # One 3-mm edge exceeds budget.
        self.assertFalse(mask[15, 16, 16])  # Weighted six-edge path is 3 mm.
        self.assertEqual(result['summary']['counts']['added'], 6)
        self.assertEqual(result['summary']['volume_mm3']['added'], 36)

    def test_barrier_blocks_accepted_growth_and_unknown_labels_are_preserved(self):
        arrays, resolved, config = self.fixture(distance=4)
        arrays['act'][15, 15, 15] = 20
        arrays['act'][15, 15, 14] = -7
        arrays['act'][:, :, 16] = 30
        arrays['act'][15, 14, 14] = 999
        self.refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertTrue(result['proposed_added_mask'][15, 15, 17])
        self.assertTrue(result['blocked_mask'][15, 15, 17])
        self.assertFalse(np.any(result['added_mask'][:, :, 16:]))
        self.assertFalse(np.any(result['changed_mask'][arrays['act'] == 30]))
        self.assertFalse(np.any(result['changed_mask'][arrays['act'] == 999]))
        self.assertEqual(result['summary']['counts']['proposed_added'],
                         result['summary']['counts']['added'] + result['summary']['counts']['blocked'])

    def test_original_signed_arrays_outside_roi_and_source_attenuation_are_unchanged(self):
        arrays, resolved, config = self.fixture(distance=2)
        snapshots = {name: value.copy() for name, value in arrays.items()}
        result = apply_morphology(arrays, resolved, config)
        for name, original in snapshots.items():
            np.testing.assert_array_equal(arrays[name], original)
        np.testing.assert_array_equal(result['edited_labels'][~arrays['roi']], arrays['act'][~arrays['roi']])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][~result['changed_mask']],
                                      arrays['atn'][~result['changed_mask']])
        self.assertEqual(result['edited_labels'].dtype, np.dtype('int32'))
        self.assertTrue(np.all(result['edited_labels'][result['added_mask']] == -7))
        self.assertTrue(np.all(result['attenuation_proxy_per_pixel'][result['added_mask']] == arrays['atn'][15, 15, 15]))
        self.assertEqual(result['summary']['transitions'],
                         [{'original_id': 20, 'new_id': -7, 'count': int(result['added_mask'].sum())}])

    def test_dilation_ties_use_signed_original_id_then_original_source_position(self):
        arrays, resolved, config = self.fixture()
        arrays['act'][15, 15, 15] = 20
        arrays['act'][15, 15, 14], arrays['act'][15, 15, 16] = 8, -7
        self.refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertEqual(result['edited_labels'][15, 15, 15], -7)
        self.assertEqual(result['attenuation_proxy_per_pixel'][15, 15, 15], arrays['atn'][15, 15, 16])
        arrays['act'][15, 15, 14] = -7
        self.refresh(arrays, resolved)
        first = apply_morphology(arrays, resolved, config)
        second = apply_morphology(arrays, resolved, config)
        self.assertEqual(first['attenuation_proxy_per_pixel'][15, 15, 15], arrays['atn'][15, 15, 14])
        np.testing.assert_array_equal(first['edited_labels'], second['edited_labels'])
        np.testing.assert_array_equal(first['attenuation_proxy_per_pixel'], second['attenuation_proxy_per_pixel'])

    def test_erosion_uses_full_candidate_context_not_artificial_roi_cut_faces(self):
        arrays, resolved, config = self.fixture(operation='erosion')
        k, j, i = np.indices(arrays['act'].shape)
        cylinder = (i-15)**2 + (j-15)**2 <= 9
        arrays['act'][cylinder] = -7
        self.refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertTrue(arrays['selected'][10, 15, 15])  # Sphere endpoint cuts thick vessel.
        self.assertFalse(result['proposed_removed_mask'][10, 15, 15])
        self.assertEqual(result['edited_labels'][10, 15, 15], -7)
        self.assertTrue(result['removed_mask'][15, 15, 18])  # True tissue interface.
        self.assertTrue(np.any(result['removed_mask']))
        np.testing.assert_array_equal(result['edited_labels'][~arrays['roi']], arrays['act'][~arrays['roi']])

    def test_erosion_recipient_barrier_preserve_and_error_modes(self):
        arrays, resolved, config = self.fixture(operation='erosion')
        arrays['act'][14:17, 14:17, 14:17] = 30
        arrays['act'][15, 15, 15] = -7
        self.refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertTrue(result['proposed_removed_mask'][15, 15, 15])
        self.assertTrue(result['unresolved_mask'][15, 15, 15])
        self.assertFalse(np.any(result['changed_mask']))
        self.assertFalse(np.any(result['blocked_mask']))
        config['reassignment']['unresolved'] = 'error'
        with self.assertRaisesRegex(ValueError, 'no eligible recipient'):
            apply_morphology(arrays, resolved, config)

    def test_erosion_recipient_ties_and_search_radius(self):
        arrays, resolved, config = self.fixture(operation='erosion')
        arrays['act'][14:17, 14:17, 14:17] = 30
        arrays['act'][15, 15, 15] = -7
        arrays['act'][15, 15, 14], arrays['act'][15, 15, 16] = 21, 20
        self.refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertEqual(result['edited_labels'][15, 15, 15], 20)
        self.assertEqual(result['attenuation_proxy_per_pixel'][15, 15, 15], arrays['atn'][15, 15, 16])
        config['reassignment']['max_distance_mm'] = .5
        result = apply_morphology(arrays, resolved, config)
        self.assertTrue(result['unresolved_mask'][15, 15, 15])

    def test_thick_release_shell_requested_equals_assigned_plus_unresolved(self):
        arrays, resolved, config = self.fixture(operation='erosion', distance=3)
        target = sphere_mask(arrays['act'].shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), 4)
        arrays['act'][target] = -7
        self.refresh(arrays, resolved)
        config['reassignment']['max_distance_mm'] = 1
        result = apply_morphology(arrays, resolved, config)
        counts = result['summary']['counts']
        self.assertGreater(counts['removed'], 0)
        self.assertGreater(counts['unresolved'], 0)
        self.assertEqual(counts['proposed_removed'], counts['removed'] + counts['unresolved'])
        self.assertEqual(counts['before'] - counts['after'], counts['removed'])
        self.assertFalse(np.any(result['unresolved_mask'] & result['changed_mask']))

    def test_empty_allowlist_means_no_change_and_unknown_target_or_donor_is_rejected(self):
        for operation in ('erosion', 'dilation'):
            arrays, resolved, config = self.fixture(operation=operation)
            config['reassignment']['allowed_tissues'] = ()
            result = apply_morphology(arrays, resolved, config)
            np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
            self.assertEqual(result['summary']['counts']['changed'], 0)
        for name in ('unknown', 'artery', 'not_a_tissue'):
            config['reassignment']['allowed_tissues'] = (name,)
            with self.assertRaises(ValueError):
                validate_edit_geometry(resolved, config)
        config['reassignment']['allowed_tissues'] = ()
        config['selection']['tissue'] = 'unknown'
        with self.assertRaises(ValueError):
            validate_edit_geometry(resolved, config)

    def test_background_requires_explicit_allowlist_and_unmapped_id_stays_unknown(self):
        arrays, resolved, config = self.fixture()
        arrays['act'][15, 15, 14] = 0
        arrays['act'][15, 15, 16] = 123456
        self.refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertTrue(result['blocked_mask'][15, 15, 14])
        self.assertTrue(result['blocked_mask'][15, 15, 16])
        config['reassignment']['allowed_tissues'] = ('background',)
        result = apply_morphology(arrays, resolved, config)
        self.assertTrue(result['added_mask'][15, 15, 14])
        self.assertEqual(result['edited_labels'][15, 15, 16], 123456)

    def test_none_identity_accepts_empty_target_and_enabled_zero_distance_is_rejected(self):
        arrays, resolved, config = self.fixture(operation='none')
        arrays['act'][15, 15, 15] = 20
        self.refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertFalse(np.any(result['strength_mm']))
        np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])
        self.assertEqual(result['summary']['counts']['changed'], 0)
        config['edit']['operation'] = 'dilation'
        with self.assertRaisesRegex(ValueError, 'nonempty selected'):
            apply_morphology(arrays, resolved, config)
        arrays['act'][15, 15, 15] = -7
        self.refresh(arrays, resolved)
        config['edit']['distance_mm'] = 0
        with self.assertRaisesRegex(ValueError, 'positive edit.distance_mm'):
            apply_morphology(arrays, resolved, config)
        config['edit']['operation'] = 'none'
        result = apply_morphology(arrays, resolved, config)
        np.testing.assert_array_equal(result['edited_labels'], arrays['act'])
        self.assertFalse(np.any(result['changed_mask']))

    def test_spatial_gaussian_tube_profile_has_zero_ends_and_center_peak(self):
        arrays, resolved, config = self.fixture(distance=2)
        resolved.update(roi_kind='tube', roi_nodes_ijk=((15, 15, 8), (15, 15, 15), (15, 15, 22)),
                        roi_radii_mm=(3, 3, 3))
        config['edit'].update(profile='gaussian', profile_axis='tube', shape_window=(0, 1))
        config['reassignment']['max_distance_mm'] = 1
        strength = strength_field_mm(arrays['act'].shape, resolved, config)
        self.assertEqual(strength[8, 15, 15], 0)
        self.assertEqual(strength[22, 15, 15], 0)
        self.assertAlmostEqual(strength[15, 15, 15], 2)
        self.assertAlmostEqual(strength[12, 15, 15], strength[18, 15, 15])
        arrays['act'][:, 15, 15] = -7
        arrays['roi'] = tube_mask(arrays['act'].shape, (0, 0, 0), resolved['roi_nodes_ijk'],
                                  (1, 1, 1), resolved['roi_radii_mm'])
        self.refresh(arrays, resolved)
        result = apply_morphology(arrays, resolved, config)
        self.assertTrue(result['added_mask'][15, 15, 17])
        self.assertFalse(np.any(result['added_mask'][8]))
        self.assertFalse(np.any(result['added_mask'][22]))
        config['edit']['shape_window'] = (.25, .75)
        strength = strength_field_mm(arrays['act'].shape, resolved, config)
        self.assertEqual(strength[11, 15, 15], 0)
        self.assertAlmostEqual(strength[15, 15, 15], 2)

    def test_gaussian_axis_extent_and_nearest_tube_segment_ties(self):
        arrays, resolved, config = self.fixture(distance=2)
        config['edit'].update(profile='gaussian', profile_axis='i')
        strength = strength_field_mm(arrays['act'].shape, resolved, config)
        self.assertEqual(strength[15, 15, 10], 0)
        self.assertEqual(strength[15, 15, 20], 0)
        self.assertAlmostEqual(strength[15, 15, 15], 2)
        # Repeated nonconsecutive node: both terminal points are equally close.
        # Supplied-order first segment gives arc position zero at the shared end.
        resolved.update(roi_kind='tube', roi_nodes_ijk=((10, 10, 15), (20, 10, 15), (10, 10, 15)),
                        roi_radii_mm=(2, 2, 2))
        config['edit'].update(profile_axis='tube', shape_window=(0, .75))
        strength = strength_field_mm(arrays['act'].shape, resolved, config)
        self.assertEqual(strength[15, 10, 10], 0)
        self.assertGreater(strength[15, 10, 15], 0)

    def test_halo_geometry_and_invalid_labels_fail_before_editing(self):
        arrays, resolved, config = self.fixture()
        resolved['roi_nodes_ijk'] = ((6, 15, 15),)
        with self.assertRaisesRegex(ValueError, 'halo'):
            validate_edit_geometry(resolved, config)
        resolved['roi_nodes_ijk'] = ((15, 15, 15),)
        arrays['act'][15, 15, 15] = 1.25
        with self.assertRaisesRegex(ValueError, 'integral'):
            apply_morphology(arrays, resolved, config)

    def test_actual_crop_bounds_and_profile_axis_contract(self):
        arrays, resolved, config = self.fixture()
        resolved['crop_high_ijk_exclusive'] = (32, 31, 31)
        with self.assertRaisesRegex(ValueError, 'actual label array shape'):
            apply_morphology(arrays, resolved, config)
        resolved['crop_high_ijk_exclusive'] = (31, 31, 31)
        config['edit']['profile_axis'] = 'tube'
        validate_edit_geometry(resolved, config)  # Uniform profile ignores its axis.
        config['edit']['profile'] = 'gaussian'
        with self.assertRaisesRegex(ValueError, 'requires a tube ROI'):
            validate_edit_geometry(resolved, config)

    def test_crop_and_full_execution_are_identical_with_valid_physical_halo(self):
        region = (slice(5, 26),) * 3
        for operation in ('erosion', 'dilation'):
            with self.subTest(operation=operation):
                arrays, resolved, config = self.fixture(operation=operation, distance=2)
                _, j, i = np.indices(arrays['act'].shape)
                arrays['act'][(i-15)**2 + (j-15)**2 <= 4] = -7
                # Adjacent target IDs and varying source scalar values also
                # exercise stable ownership when the crop changes flat indices.
                arrays['act'][15:, 15, 15] = 8
                self.refresh(arrays, resolved)
                config['reassignment']['max_distance_mm'] = 1
                full = apply_morphology(arrays, resolved, config)
                cropped = {key: value[region].copy() for key, value in arrays.items()}
                cropped_geometry = dict(resolved, crop_low_ijk=(5, 5, 5), crop_high_ijk_exclusive=(26, 26, 26))
                part = apply_morphology(cropped, cropped_geometry, config)
                for name, value in part.items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(value, full[name][region], err_msg=f'{operation}: {name}')
                for name in ('counts', 'transitions', 'volume_mm3', 'components_before', 'components_after'):
                    self.assertEqual(part['summary'][name], full['summary'][name])

    def test_accepted_changes_are_monotone_for_fixed_roi_profile_and_policy(self):
        for operation in ('erosion', 'dilation'):
            for profile in ('uniform', 'gaussian'):
                with self.subTest(operation=operation, profile=profile):
                    arrays, resolved, config = self.fixture(operation=operation)
                    _, j, i = np.indices(arrays['act'].shape)
                    arrays['act'][(i-15)**2 + (j-15)**2 <= 9] = -7
                    self.refresh(arrays, resolved)
                    config['edit']['profile'] = profile
                    previous = np.zeros_like(arrays['roi'])
                    counts = []
                    for distance in (.5, 1, 2):
                        config['edit']['distance_mm'] = distance
                        result = apply_morphology(arrays, resolved, config)
                        accepted = result['changed_mask']
                        self.assertFalse(np.any(previous & ~accepted))
                        counts.append(int(accepted.sum()))
                        previous = accepted
                    self.assertEqual(counts, sorted(counts))
                    self.assertEqual(counts[0], 0)
                    self.assertGreater(counts[-1], counts[0])


if __name__ == '__main__':
    unittest.main()
