"""Fold locator isolation and real tiny TensorFlow save/refit contracts."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import fakect_coa_rotation_locator as rotation
from test_multiscale_experiment_review import MiniVolume, fixture_rows
from test_multiscale_v2_integration import ShiftedValidationVolume, overview
from test_multiscale_v3_integration import config


def originals():
    return deepcopy([row for row in fixture_rows() if row['baseline']])


class RotationLocatorIsolationTests(unittest.TestCase):
    def test_rejects_nonoriginal_evaluation_and_invalid_refit_before_reading(self):
        for mutation in ('test', 'heldout', 'edited', 'role', 'duplicate', 'family', 'refit_validation', 'selection_without_validation'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                rows, fixed = originals(), False
                if mutation in ('test', 'heldout'):
                    rows[-1]['split'] = mutation
                elif mutation == 'edited':
                    rows[-1]['baseline'] = False
                elif mutation == 'role':
                    rows[-1]['pilot_role'] = 'higher_positive'
                elif mutation == 'duplicate':
                    rows.append(deepcopy(rows[0]))
                elif mutation == 'family':
                    rows[-1]['anatomy_family'] = rows[0]['anatomy_family']
                elif mutation == 'refit_validation':
                    fixed = True
                else:
                    rows = rows[:1]
                forbidden = Mock(side_effect=AssertionError('Invalid fold must fail before payload reading'))
                with patch.object(rotation, 'ContextVolume', forbidden):
                    with self.assertRaises(ValueError):
                        rotation.fit_locator(None, config(directory), rows, Path(directory) / 'fit',
                                             epochs=1, fixed_budget=fixed)
                    forbidden.assert_not_called()

    def test_cross_partition_attenuation_and_duplicate_reference_rejected(self):
        for identity in ('source', 'mask'):
            with self.subTest(identity=identity):
                rows = originals()
                for row in rows:
                    if identity == 'source':
                        row['source_atn'] = {'path': '/tmp/shared-original-atn.bin'}
                    else:
                        row['original_reference_mask_sha256'] = 'same-reference'
                        row['spacing_ijk_mm'] = [1., 1., 1.]
                with self.assertRaisesRegex(ValueError, 'crosses'):
                    rotation.validate_rows(rows)

    def test_secondary_overlap_requires_explicit_flag_and_retains_exact_identity_guards(self):
        rows = originals()
        rows[-1]['anatomy_family'] = rows[0]['anatomy_family']
        with self.assertRaisesRegex(ValueError, 'family crosses'):
            rotation.validate_rows(rows)
        selected, train, validation = rotation.validate_rows(rows, allow_provisional_family_overlap=True)
        self.assertEqual(selected, rows)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)
        for field in ('source', 'mask'):
            with self.subTest(identity=field):
                repeated = deepcopy(rows)
                for row in repeated:
                    if field == 'source':
                        row['source_atn'] = {'path': '/tmp/shared-secondary-atn.bin'}
                    else:
                        row['original_reference_mask_sha256'] = 'same-secondary-anatomy'
                        row['spacing_ijk_mm'] = [1., 1., 1.]
                with self.assertRaisesRegex(ValueError, 'crosses'):
                    rotation.validate_rows(repeated, allow_provisional_family_overlap=True)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            rotation.validate_rows(rows, allow_provisional_family_overlap='true')

    def test_fixed_refit_accepts_train_only_and_epoch_budget_is_checked_before_reading(self):
        rows = originals()[:1]
        selected, train, validation = rotation.validate_rows(rows, fixed_budget=True)
        self.assertEqual(selected, rows)
        self.assertEqual(train, rows)
        self.assertEqual(validation, [])
        for value in (True, 0, -1, 501, 1.5):
            with self.subTest(epochs=value), tempfile.TemporaryDirectory() as directory:
                forbidden = Mock(side_effect=AssertionError('Invalid budget cannot read images'))
                with patch.object(rotation, 'ContextVolume', forbidden):
                    with self.assertRaisesRegex(ValueError, '[Ee]poch'):
                        rotation.fit_locator(None, config(directory), rows, Path(directory) / 'fit',
                                             epochs=value, fixed_budget=True)
                    forbidden.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class RotationLocatorTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_multiscale_network_v3 as network
        cls.network = network
        cls.tf = network._tensorflow()

    def setUp(self):
        self.tf.keras.backend.clear_session()
        MiniVolume.accessed = []

    def fit_fixture(self, directory, rows, *, epochs, fixed_budget):
        c = config(directory)
        class FixtureInputs:
            def __init__(self, runtime):
                self.runtime = runtime
            def coarse(self, row):
                ov = overview(row, self.runtime)
                image = rotation.geometry.normalized_input(ov['image'], ov['valid_fraction'], self.runtime)[None]
                return ov, image
        fake_legacy = SimpleNamespace(**vars(rotation.legacy))
        fake_legacy.Inputs = FixtureInputs
        build = Mock(wraps=self.network.build_locator)
        load = Mock(wraps=self.network.load_model)
        with patch.object(rotation, 'legacy', fake_legacy), \
             patch.object(rotation, 'ContextVolume', ShiftedValidationVolume), \
             patch.object(self.network, 'build_locator', build), \
             patch.object(self.network, 'load_model', load):
            result = rotation.fit_locator(self.tf, c, rows, directory,
                                          epochs=epochs, fixed_budget=fixed_budget)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(load.call_count, 1)  # Reload only after fresh fitting.
        return result, build.call_args.kwargs['prior_box']

    def test_inner_locator_prior_uses_train_only_and_reloads_selected_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = originals()
            result, prior = self.fit_fixture(directory, rows, epochs=1, fixed_budget=False)
            train_volume = ShiftedValidationVolume(rows[0])
            expected = rotation.geometry.bbox_target(train_volume, overview(rows[0], config(directory)))
            np.testing.assert_array_equal(prior, expected)
            validation_volume = ShiftedValidationVolume(rows[-1])
            different = rotation.geometry.bbox_target(validation_volume, overview(rows[-1], config(directory)))
            self.assertFalse(np.array_equal(prior, (expected + different) / 2))
            self.assertEqual(result['prior_training_sample_ids'], [rows[0]['sample_id']])
            self.assertEqual(result['validation_sample_ids'], [rows[-1]['sample_id']])
            self.assertEqual(result['epochs_completed'], 1)
            self.assertEqual(result['selected_epoch'], 1)
            self.assertEqual(result['optimizer_steps'], 2)
            self.assertEqual(result['traces'], {'train': 1, 'predict': 1})
            self.assertTrue(result['fresh_initialization'])
            self.assertFalse(result['outer_evaluation_used'])
            self.assertTrue(result['reload_parity']['passed'])
            self.assertNotEqual(result['initial_weights_sha256'], result['final_weights_sha256'])
            self.assertTrue(Path(result['checkpoint']).exists())
            record = result['history'][0]
            self.assertAlmostEqual(record['validation_loss'],
                                   .5 * record['validation_identity_loss'] + .5 * record['validation_translation_loss'])
            self.assertEqual(set(MiniVolume.accessed), {row['sample_id'] for row in rows})

    def test_fixed_budget_refit_saves_final_epoch_and_never_opens_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = originals()[:1]
            result, _ = self.fit_fixture(directory, rows, epochs=2, fixed_budget=True)
            self.assertEqual(result['selected_epoch'], 2)
            self.assertEqual(result['epochs_completed'], 2)
            self.assertEqual(result['optimizer_steps'], 4)
            self.assertEqual(result['validation_sample_ids'], [])
            self.assertTrue(all(record['validation_loss'] is None and record['validation_geometry'] is None
                                for record in result['history']))
            self.assertTrue(result['reload_parity']['passed'])
            self.assertEqual(set(MiniVolume.accessed), {rows[0]['sample_id']})


if __name__ == '__main__':
    unittest.main()
