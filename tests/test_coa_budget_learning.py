"""Fixed-budget instrumentation parity on small real TensorFlow fixtures.

Only generated 16-cubed arrays are read. No cohort images or held-out samples
are opened. Two epochs test the unchanged optimizer/data order, not accuracy.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import fakect_coa_budget_learning as budget
import fakect_coa_rotation_learning as learning
from fakect_tensorflow_steps import StableSteps
from test_coa_rotation_classifier import TinyFineInputs, fixture_rows
from test_multiscale_v3_integration import config


class BudgetInputTests(unittest.TestCase):
    # Purpose: Reject invalid fixed budgets before fitting or image access.
    # Inputs: Zero, negative, boolean or float epochs/intervals and incompatible checkpoint intervals.
    # Outputs: ValueError for every budget; fitter and reader never called.
    def test_invalid_budgets_rejected_before_fitter_or_reader(self):
        rows = [row for row in fixture_rows() if row['split'] == 'train']
        invalid = [(0, 1), (-1, 1), (True, 1), (2.0, 1), (2, 0),
                   (2, -1), (2, True), (2, 1.0), (2, 3), (3, 2)]
        with patch.object(learning, 'fit_classifier') as fitter, \
             patch.object(learning, 'StageFineInputs') as reader:
            for epochs, interval in invalid:
                with self.subTest(epochs=epochs, interval=interval), self.assertRaises(ValueError):
                    budget.fit_classifier(None, {}, rows, {}, '/tmp/not-created', {},
                                          epochs=epochs, checkpoint_interval=interval)
            fitter.assert_not_called()
            reader.assert_not_called()

    # Purpose: Reject initialization mismatches before opening image payloads.
    # Inputs: Mock classifier fingerprint a repeated 64 times versus expected b repeated 64 times.
    # Outputs: Initialization ValueError before reader use and original builder restored.
    def test_initialization_mismatch_rejected_before_image_access(self):
        import fakect_coa_paired_steps as network
        original = network.build_classifier
        rows = [row for row in fixture_rows() if row['split'] == 'train']

        def fitter(*args, **kwargs):
            network.build_classifier([16, 16, 16], seed=42)
            learning.StageFineInputs({}, 1, rows)

        with patch.object(network, 'build_classifier', return_value=Mock()) as build, \
             patch.object(learning, 'weight_fingerprint', return_value='a' * 64), \
             patch.object(learning, 'fit_classifier', side_effect=fitter), \
             patch.object(learning, 'StageFineInputs') as reader:
            with self.assertRaisesRegex(ValueError, 'initialization differs'):
                budget.fit_classifier(None, {}, rows, {}, '/tmp/not-created', {},
                                      epochs=2, checkpoint_interval=1,
                                      expected_initial_weights_sha256='b' * 64)
            build.assert_called_once()
            reader.assert_not_called()
        self.assertIs(network.build_classifier, original)

    # Purpose: Reject held-out roles and nonfixed classifier budgets.
    # Inputs: Training rows mutated to validation, final_test or reserved_test; fixed_budget false.
    # Outputs: ValueError before fitter or reader invocation.
    def test_validation_and_test_roles_rejected_before_fitter_or_reader(self):
        rows = [row for row in fixture_rows() if row['split'] == 'train']
        with patch.object(learning, 'fit_classifier') as fitter, \
             patch.object(learning, 'StageFineInputs') as reader:
            for role in ('validation', 'final_test', 'reserved_test'):
                changed = deepcopy(rows)
                changed[-1]['split'] = role
                with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'held-out'):
                    budget.fit_classifier(None, {}, changed, {}, '/tmp/not-created', {},
                                          epochs=2, checkpoint_interval=1)
            with self.assertRaises(ValueError):
                budget.fit_classifier(None, {}, rows, {}, '/tmp/not-created', {},
                                      epochs=2, checkpoint_interval=1, fixed_budget=False)
            fitter.assert_not_called()
            reader.assert_not_called()

    # Purpose: Verify builder patches restore after fitter failure.
    # Inputs: Valid two-epoch budget and fitter raising a fixture RuntimeError.
    # Outputs: RuntimeError propagates and original classifier builder restored.
    def test_builder_patch_restored_on_fitter_failure(self):
        import fakect_coa_paired_steps as network
        original = network.build_classifier
        rows = [row for row in fixture_rows() if row['split'] == 'train']
        with patch.object(learning, 'fit_classifier', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                budget.fit_classifier(None, {}, rows, {}, '/tmp/not-created', {},
                                      epochs=2, checkpoint_interval=1)
        self.assertIs(network.build_classifier, original)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class BudgetTensorFlowParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_coa_paired_steps as network
        cls.network = network
        cls.tf = network.focused._tensorflow()

    # Purpose: Verify the real fitter checks initialization before its first payload.
    # Inputs: 16-cube synthetic runtime and mismatching all-zero expected weight checksum.
    # Outputs: Initialization rejected, no samples accessed and no initialization JSON written.
    def test_actual_fitter_checks_initialization_before_first_payload(self):
        self.tf.keras.backend.clear_session()
        TinyFineInputs.accessed = []
        with tempfile.TemporaryDirectory() as directory:
            runtime = config(directory)
            runtime['input.detail']['shape_kji'] = [16, 16, 16]
            rows = [row for row in fixture_rows() if row['split'] == 'train']
            with patch.object(learning, 'StageFineInputs', TinyFineInputs):
                with self.assertRaisesRegex(ValueError, 'initialization differs'):
                    budget.fit_classifier(self.tf, runtime, rows, {}, Path(directory) / 'mismatch',
                                          {'training': {'cache_limit_mb': 1}}, epochs=2,
                                          checkpoint_interval=1,
                                          expected_initial_weights_sha256='0' * 64)
            self.assertEqual(TinyFineInputs.accessed, [])
            self.assertFalse((Path(directory) / 'mismatch' / 'initialization.json').exists())

    def fit_fixture(self, output, *, wrapped):
        self.tf.keras.backend.clear_session()
        TinyFineInputs.accessed = []
        runtime = config(str(output.parent))
        runtime['input.detail']['shape_kji'] = [16, 16, 16]
        runtime['model']['seed'] = 42
        rows = [row for row in fixture_rows() if row['split'] == 'train']
        proposals = {row['sample_id']: {'origins_kji': [[0, 0, 0], [0, 0, 0]]} for row in rows}
        c = {'training': {'cache_limit_mb': 1, 'classifier_learning_rate': .001,
                          'regression_weight': 1.}}
        actual_make = self.network.make_steps
        updates = []

        def make(model, *args, **kwargs):
            actual = actual_make(model, *args, **kwargs)

            def train(images, targets):
                before = int(np.asarray(actual.optimizer.iterations))
                result = actual.train(images, targets)
                after = int(np.asarray(actual.optimizer.iterations))
                self.assertEqual(after, before + 1)
                updates.append(after)
                return result

            train.experimental_get_tracing_count = actual.train.experimental_get_tracing_count
            return StableSteps(train, actual.predict, actual.optimizer)

        fitter = budget.fit_classifier if wrapped else learning.fit_classifier
        kwargs = {'epochs': 2, 'fixed_budget': True}
        if wrapped:
            kwargs['checkpoint_interval'] = 1
            reference = self.network.build_classifier([16, 16, 16], num_proposals=2,
                                                       base_filters=2, seed=42)
            kwargs['expected_initial_weights_sha256'] = learning.weight_fingerprint(reference)
            del reference
        original_build = self.network.build_classifier
        with patch.object(learning, 'StageFineInputs', TinyFineInputs), \
             patch.object(self.network, 'make_steps', new=make):
            receipt = fitter(self.tf, runtime, rows, proposals, output, c, **kwargs)
        self.assertIs(self.network.build_classifier, original_build)
        self.assertEqual(set(TinyFineInputs.accessed), {row['sample_id'] for row in rows})
        return receipt, updates

    # Purpose: Verify archive instrumentation preserves full training and optimizer parity.
    # Inputs: Raw and wrapped two-epoch fits with identical seeded synthetic inputs.
    # Outputs: Same 12 updates, initial weights, histories and final predictions; verified milestone checkpoints.
    def test_archive_wrapper_preserves_initialization_training_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, baseline_updates = self.fit_fixture(root / 'baseline', wrapped=False)
            wrapped, wrapped_updates = self.fit_fixture(root / 'wrapped', wrapped=True)
            self.assertEqual(baseline_updates, list(range(1, 13)))
            self.assertEqual(wrapped_updates, baseline_updates)
            for key in ('fresh_initial_weights_sha256', 'training_sample_ids',
                        'validation_sample_ids', 'epochs_completed', 'selected_epoch',
                        'optimizer_updates', 'validation'):
                self.assertEqual(wrapped[key], baseline[key])
            self.assertEqual(wrapped['selected_epoch'], 2)
            self.assertEqual(wrapped['model_save_calls'], 2)
            self.assertFalse(wrapped['milestones_selection_allowed'])
            self.assertFalse(wrapped['parent_classifier_checkpoint_reused'])
            self.assertTrue(wrapped['checkpoint_parity']['passed'])
            self.assertEqual(wrapped['validation_sample_ids'], [])
            self.assertIsNone(wrapped['validation'])
            self.assertEqual(wrapped['fresh_initial_weights_sha256'],
                             learning.geometry.checked_json(root / 'wrapped' / 'initialization.json')['weights_sha256'])

            first = learning.geometry.checked_json(root / 'baseline' / 'initialization.json')
            second = learning.geometry.checked_json(root / 'wrapped' / 'initialization.json')
            self.assertEqual(first, second)
            histories = [learning.geometry.checked_json(root / name / 'history.json')['epochs']
                         for name in ('baseline', 'wrapped')]
            for before, after in zip(*histories):
                for key in ('epoch', 'training_pair_order', 'optimizer_updates',
                            'validation', 'selection_bce'):
                    self.assertEqual(before[key], after[key])
                for key in ('train_loss', 'gradient_norm_mean'):
                    self.assertAlmostEqual(before[key], after[key], places=7)
                for key in before['training']:
                    self.assertAlmostEqual(before['training'][key], after['training'][key], places=7)

            self.assertEqual([r['epoch'] for r in wrapped['budget_checkpoints']], [1, 2])
            self.assertEqual([r['optimizer_updates'] for r in wrapped['budget_checkpoints']], [6, 12])
            milestones = [self.network.load_model(record['checkpoint'])
                          for record in wrapped['budget_checkpoints']]
            self.assertNotEqual(learning.weight_fingerprint(milestones[0]),
                                learning.weight_fingerprint(milestones[1]))
            for record in wrapped['budget_checkpoints']:
                self.assertTrue(Path(record['checkpoint']).is_file())
                self.assertEqual(record['checkpoint_sha256'], learning.geometry.sha256(record['checkpoint']))
            self.assertEqual(wrapped['budget_checkpoints'][-1]['checkpoint_sha256'],
                             wrapped['checkpoint_sha256'])
            models = [self.network.load_model(root / name / 'classifier.keras')
                      for name in ('baseline', 'wrapped')]
            for before, after in zip(models[0].weights, models[1].weights):
                np.testing.assert_allclose(np.asarray(before), np.asarray(after), atol=1e-7, rtol=1e-6)
            inputs = TinyFineInputs(config(str(root)), 1, [])
            inputs.runtime['input.detail']['shape_kji'] = [16, 16, 16]
            row = {**fixture_rows()[0]}
            inputs.allowed[row['sample_id']] = 'train'
            values = inputs.classifier(row, {})
            before = models[0](values, training=False)
            after = models[1](values, training=False)
            for key in before:
                np.testing.assert_allclose(np.asarray(before[key]), np.asarray(after[key]), atol=1e-7, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
