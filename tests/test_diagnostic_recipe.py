"""Deferred material assignment remains explicit across serial recipe passes."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fakect_recipe import apply_recipe
from fakect_released import RELEASED_LABEL_ID, RELEASED_TISSUE_ID, released_catalog
from fakect_training_data import validate_study_plan
import test_recipe


class DiagnosticRecipeTests(unittest.TestCase):
    def test_derived_catalog_preserves_source_and_rejects_identity_collisions(self):
        _, resolved, _ = test_recipe.RecipeTests().fixture()
        original = deepcopy(resolved['catalog'])
        derived = released_catalog(original)
        self.assertEqual(derived['records'][:-1], original['records'])
        self.assertEqual(derived['categories'][:-1], original['categories'])
        self.assertEqual(released_catalog(derived), derived)
        self.assertEqual(original, resolved['catalog'])
        for kind in ('id', 'category'):
            bad = deepcopy(original)
            if kind == 'id':
                bad['records'][1]['original_id'] = RELEASED_LABEL_ID
            else:
                bad['categories'].append({'id': RELEASED_TISSUE_ID, 'name': 'existing_material'})
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'collides'):
                released_catalog(bad)

    def test_release_after_growth_persists_and_cannot_seed_or_supply_later_growth(self):
        arrays, resolved, config = test_recipe.RecipeTests().fixture()
        config['recipe'].update(steps=('grow', 'shrink', 'regrow'), roi_role='selection')
        config['edits']['shrink']['assign_surrounding_tissue'] = False
        config['edits']['regrow'] = deepcopy(config['edits']['grow'])
        original = deepcopy(arrays)
        original_catalog = deepcopy(resolved['catalog'])
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        released = result['released_mask']
        self.assertEqual(int(released.sum()), 6)
        self.assertEqual(events[2]['result']['summary']['counts']['added'], 0)
        self.assertEqual(events[2]['result']['summary']['counts']['blocked'], 6)
        self.assertEqual(result['summary']['counts']['after'], 1)
        self.assertTrue(np.all(result['edited_labels'][released] == RELEASED_LABEL_ID))
        self.assertTrue(np.all(result['edited_tissue_labels'][released] == RELEASED_TISSUE_ID))
        self.assertTrue(np.all(result['target_origin_index'][released] == -1))
        np.testing.assert_array_equal(result['attenuation_unassigned_mask'], released)
        np.testing.assert_array_equal(result['diagnostic_released_mask'], released)
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][released], events[1]['before_arrays']['atn'][released])
        metadata = result['summary']['release_assignment']
        self.assertEqual(metadata['newly_released_voxels'], 6)
        self.assertEqual(metadata['current_released_voxels'], 6)
        self.assertIn('placeholder', result['summary']['scalar_status'])
        self.assertIn('Final-state'.lower(), metadata['surrounding_semantics'].lower())
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], original[key])
        self.assertEqual(resolved['catalog'], original_catalog)

    def test_repeated_diagnostic_erosion_handles_exhausted_target_without_reassigning_marker(self):
        arrays, resolved, config = test_recipe.RecipeTests().fixture()
        config['recipe']['steps'] = ('shrink',)
        config['edits']['shrink'].update(iterations=2, assign_surrounding_tissue=False)
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual(result['summary']['counts']['after'], 0)
        self.assertEqual(int(result['released_mask'].sum()), 1)
        self.assertEqual(events[1]['pass_summary']['status'], 'empty_target')
        self.assertEqual(events[1]['result']['summary']['counts']['removed'], 0)
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'], arrays['atn'])

    def test_unassigned_attenuation_cannot_be_prepared_as_training_pairs(self):
        with self.assertRaisesRegex(ValueError, 'cannot form training pairs'):
            validate_study_plan({'edit': {'assign_surrounding_tissue': False}}, {})


if __name__ == '__main__':
    unittest.main()
