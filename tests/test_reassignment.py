"""Reassignment uses original anatomical identities rather than coarse groups alone."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_reassignment import reassignment_masks, stiffness_field, validate_reassignment_policy
from fakect_morphology import _owned_paths, apply_morphology
from fakect_recipe import apply_recipe
from fakect_roi import sphere_mask
from fakect_tissues import coarse_labels


def catalog_fixture():
    groups = [(0, 'background'), (1, 'soft_tissue'), (2, 'bone'), (4, 'muscle'),
              (5, 'artery'), (6, 'vein'), (255, 'unknown')]
    codes = {name: value for value, name in groups}
    definitions = [
        (0, 'background', 'background', {}),
        (1, 'chest_surface', 'soft_tissue', {'hierarchy_path': '11_Integumentary/Skin'}),
        (2, 'rib', 'bone', {}),
        (-3, 'rib_inner', 'unknown', {'structure_type': 'bone'}),
        (4, 'dermal_region', 'unknown', {'structure_type': 'skin'}),
        (5, 'outer_region', 'unknown', {'canonical_name': 'skin'}),
        (6, 'skin_42', 'soft_tissue', {}),
        (7, 'dias_pericardium', 'unknown', {'structure_type': 'heart'}),
        (8, 'aorta', 'artery', {}),
        (9, 'rca', 'artery', {}),
        (10, 'vein', 'vein', {}),
        (11, 'rfoot_musc', 'muscle', {'structure_type': 'muscle',
                                    'matched_rules': ['region_surface_skin'],
                                    'atlas_rows': [{'canonical_name': 'skin'}]}),
        (12, 'soft', 'soft_tissue', {})]
    return {'categories': [{'id': value, 'name': name} for value, name in groups],
            'records': [{'original_id': value, 'original_name': name,
                         'classification': {'tissue_id': codes[group], 'tissue_name': group, **extra}}
                        for value, name, group, extra in definitions],
            'sources': {'atlas': {'sha256': 'frozen-test-catalog'}}}


class ReassignmentTests(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog_fixture()
        self.labels = np.array([record['original_id'] for record in self.catalog['records']] + [999], dtype=np.int32)
        self.candidates = self.labels == 8
        self.policy = {'mode': 'permissive_except_bone_skin', 'allowed_tissues': ()}

    def test_permissive_uses_fine_bone_skin_identity_and_allows_other_vessels(self):
        eligible, protected, metadata = reassignment_masks(self.labels, self.candidates, self.catalog, self.policy)
        self.assertEqual(set(self.labels[eligible]), {7, 9, 10, 11, 12})
        self.assertEqual(set(self.labels[protected]), {0, 1, 2, -3, 4, 5, 6, 999})
        self.assertFalse(np.any(eligible & self.candidates))
        reasons = {row['original_id']: row['reasons'] for row in metadata['protected_input_labels']}
        self.assertEqual(reasons[-3], ['bone'])
        self.assertEqual(reasons[1], ['skin'])
        self.assertEqual(reasons[999], ['unmapped_original_id'])
        self.assertEqual(metadata['catalog_sources'], self.catalog['sources'])

    def test_metadata_rejects_protected_targets_even_when_no_voxels_read(self):
        for value in (0, 1, 2, -3, 4, 5, 6, 999):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'protected target ID'):
                validate_reassignment_policy(self.catalog, self.policy, target_source_ids=(value,))
        validate_reassignment_policy(self.catalog, self.policy, target_source_ids=(8,))

    def test_legacy_allowlist_remains_explicit_and_does_not_add_skin_exclusions(self):
        eligible, protected, metadata = reassignment_masks(self.labels, self.candidates, self.catalog,
                                                           {'allowed_tissues': ('soft_tissue',)})
        self.assertEqual(set(self.labels[eligible]), {1, 6, 12})
        self.assertEqual(metadata['mode'], 'allowlist')
        self.assertTrue(protected[self.labels == 7].all())
        with self.assertRaisesRegex(ValueError, 'unknown can never'):
            validate_reassignment_policy(self.catalog, {'allowed_tissues': ('unknown',)})
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_reassignment_policy(self.catalog, {'allowed_tissues': ('artery',)}, target_tissue='artery')

    def test_permissive_requires_blank_allowlist_and_does_not_mutate_catalog(self):
        original = deepcopy(self.catalog)
        reassignment_masks(self.labels, self.candidates, self.catalog, self.policy)
        self.assertEqual(self.catalog, original)
        with self.assertRaisesRegex(ValueError, 'empty'):
            validate_reassignment_policy(self.catalog, {**self.policy, 'allowed_tissues': ('soft_tissue',)})

    def stiffness_policy(self, default=0.):
        return {'mode': 'stiffness', 'allowed_tissues': (), 'max_distance_mm': 3., 'unresolved': 'preserve',
                'stiffness': {'default': default, 'tissues': {'bone': 1., 'skin': .95}, 'labels': {}}}

    def morphology_fixture(self, operation='dilation', distance=2.):
        shape = (31, 31, 31)
        labels = np.full(shape, 12, dtype=np.int32)
        labels[15, 15, 15] = 8
        roi = sphere_mask(shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), 5)
        arrays = {'act': labels, 'atn': (np.arange(labels.size).reshape(shape) / 1e6 + .01).astype(np.float32),
                  'roi': roi, 'tissue': coarse_labels(labels, self.catalog), 'candidates': labels == 8,
                  'selected': (labels == 8) & roi}
        resolved = {'crop_low_ijk': (0, 0, 0), 'crop_high_ijk_exclusive': (31, 31, 31),
                    'spacing_ijk_mm': (1, 1, 1), 'roi_kind': 'sphere', 'shape_kji': shape,
                    'roi_nodes_ijk': ((15, 15, 15),), 'roi_radii_mm': (5,), 'catalog': self.catalog,
                    'source_ids': (8,), 'slice_ijk': (15, 15, 15)}
        config = {'selection': {'tissue': 'artery', 'source_ids': (8,)},
                  'roi': {'shape': 'sphere', 'center_ijk': (15, 15, 15), 'radius_mm': 5., 'crop_half_width_mm': 15},
                  'edit': {'operation': operation, 'distance_mm': distance, 'profile': 'uniform',
                           'profile_axis': 'k', 'shape_k': 6., 'shape_window': (0., 1.)},
                  'reassignment': self.stiffness_policy()}
        return arrays, resolved, config

    def refresh(self, arrays):
        arrays['tissue'] = coarse_labels(arrays['act'], self.catalog)
        arrays['candidates'] = arrays['act'] == 8
        arrays['selected'] = arrays['candidates'] & arrays['roi']

    def test_stiffness_uses_semantic_skin_bone_and_editable_signed_id_override(self):
        policy = self.stiffness_policy(.05)
        field = stiffness_field(self.labels, self.catalog, policy)
        self.assertEqual(field[self.labels == -3][0], 1.)
        self.assertEqual(field[self.labels == 1][0], .95)
        self.assertEqual(field[self.labels == 7][0], .05)
        self.assertEqual(field[self.labels == 11][0], .05)  # Ignore historical skin rule.
        self.assertEqual(field[self.labels == 999][0], 1.)
        policy['stiffness']['labels'][-3] = .2
        eligible, protected, metadata = reassignment_masks(self.labels, self.candidates, self.catalog, policy)
        self.assertTrue(eligible[self.labels == -3][0])
        self.assertFalse(protected[self.labels == -3][0])
        row = next(row for row in metadata['effective_input_labels'] if row['original_id'] == -3)
        self.assertEqual(row['stiffness'], .2)
        self.assertEqual(row['stiffness_basis'], 'original_id_override')

    def test_dilation_distance_budget_decreases_with_stiffness_and_one_is_rigid(self):
        arrays, resolved, config = self.morphology_fixture()
        sizes = []
        for stiffness in (0., .5, .95, 1.):
            config['reassignment']['stiffness']['tissues']['soft_tissue'] = stiffness
            result = apply_morphology(arrays, resolved, config)
            sizes.append(int(result['added_mask'].sum()))
            self.assertTrue(np.all(result['stiffness_field'][arrays['act'] == 12] == stiffness))
        self.assertEqual(sizes, [24, 6, 0, 0])

    def test_zero_resistance_matches_permissive_mode_for_same_eligible_id_set(self):
        arrays, resolved, config = self.morphology_fixture()
        for operation in ('dilation', 'erosion'):
            config['edit']['operation'] = operation
            config['reassignment'] = self.stiffness_policy()
            weighted = apply_morphology(arrays, resolved, config)
            config['reassignment'] = {**self.policy, 'max_distance_mm': 3., 'unresolved': 'preserve'}
            permissive = apply_morphology(arrays, resolved, config)
            np.testing.assert_array_equal(weighted['edited_labels'], permissive['edited_labels'])
            np.testing.assert_array_equal(weighted['attenuation_proxy_per_pixel'], permissive['attenuation_proxy_per_pixel'])

    def test_erosion_scales_target_release_and_recipient_reach_separately(self):
        arrays, resolved, config = self.morphology_fixture('erosion', 1.)
        config['reassignment']['stiffness']['tissues']['artery'] = .5
        result = apply_morphology(arrays, resolved, config)
        self.assertEqual(result['summary']['counts']['requested_removed_before_stiffness'], 1)
        self.assertEqual(result['summary']['counts']['release_suppressed_by_target_stiffness'], 1)
        self.assertEqual(result['summary']['counts']['removed'], 0)
        config['reassignment']['stiffness']['tissues']['artery'] = 0.
        arrays['act'][arrays['act'] == 12] = 1  # Fine skin ID lives in soft_tissue.
        self.refresh(arrays)
        result = apply_morphology(arrays, resolved, config)
        self.assertEqual(result['summary']['counts']['proposed_removed'], 1)
        self.assertEqual(result['summary']['counts']['unresolved'], 1)  # .15-mm recipient reach < one voxel.
        config['reassignment']['stiffness']['labels'][1] = 0.
        result = apply_morphology(arrays, resolved, config)
        self.assertEqual(result['summary']['counts']['removed'], 1)
        self.assertEqual(result['edited_labels'][15, 15, 15], 1)

    def test_recipient_offsets_keep_farther_soft_owner_when_nearer_stiff_owner_cannot_reach(self):
        shape = (1, 1, 6)
        domain = np.ones(shape, dtype=bool)
        seeds = np.zeros(shape, dtype=bool)
        seeds[0, 0, [0, 2]] = True
        labels = np.full(shape, 8, dtype=np.int32)
        labels[0, 0, 0], labels[0, 0, 2] = 12, 1
        initial = np.zeros(shape)
        initial[0, 0, 2] = 4.  # The closer recipient has only one mm of its five-mm cap left.
        costs, owners = _owned_paths(domain, seeds, (1., 1., 1.), 5., labels, initial=initial)
        self.assertEqual(owners[0, 0, 4], 0)
        self.assertEqual(costs[0, 0, 4], 4.)
        self.assertEqual(owners[0, 0, 5], 0)
        self.assertEqual(costs[0, 0, 5], 5.)

    def test_repeated_recipe_preserves_rigid_bone_and_background_but_borrows_unknown_vessel_ids(self):
        arrays, resolved, config = self.morphology_fixture()
        arrays['act'][15, 15, 14] = -3  # Unknown coarse group, rigid bone anatomy.
        arrays['act'][15, 15, 16] = 9   # Different artery may be borrowed.
        arrays['act'][15, 14, 15] = 7   # Catalogued unknown pericardium may be borrowed.
        arrays['act'][15, 16, 15] = 1   # Skin .95: too stiff at this edit distance.
        arrays['act'][14, 15, 15] = 999 # Unmapped stays categorically protected.
        arrays['act'][16, 15, 15] = 0   # Background stays categorically protected.
        self.refresh(arrays)
        original = {name: value.copy() for name, value in arrays.items()}
        config['rois'] = {'local': {**config['roi'], 'coordinate_reviewed': False}}
        config['edits'] = {'grow': {**config['edit'], 'roi': 'local', 'iterations': 2},
                           'shrink': {**config['edit'], 'roi': 'local', 'iterations': 2, 'operation': 'erosion'}}
        config['recipe'] = {'steps': ('grow', 'shrink'), 'overlap': 'sequential'}
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        protected = np.isin(arrays['act'], (-3, 1, 999, 0))
        for event in events:
            np.testing.assert_array_equal(event['result']['edited_labels'][protected], arrays['act'][protected])
            np.testing.assert_array_equal(event['result']['attenuation_proxy_per_pixel'][protected], arrays['atn'][protected])
        self.assertEqual(events[0]['result']['edited_labels'][15, 15, 16], 8)
        self.assertEqual(events[0]['result']['edited_labels'][15, 14, 15], 8)
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], original[name])
        self.assertIn('final_stiffness_field', result)

    def test_invalid_stiffness_and_unmapped_overrides_fail_metadata_validation(self):
        for value in (-.1, 1.01, float('nan'), float('inf')):
            policy = self.stiffness_policy()
            policy['stiffness']['default'] = value
            with self.assertRaisesRegex(ValueError, 'finite numbers'):
                validate_reassignment_policy(self.catalog, policy)
        policy = self.stiffness_policy()
        policy['stiffness']['labels'][999] = 0.
        with self.assertRaisesRegex(ValueError, 'catalogued signed original ID'):
            validate_reassignment_policy(self.catalog, policy)


if __name__ == '__main__':
    unittest.main()
