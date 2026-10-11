"""Loss isolation and frozen-fitter parity using synthetic, small 3D inputs."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import fakect_coa_classification_learning as control
import fakect_coa_budget_learning as budget
from test_coa_rotation_classifier import TinyFineInputs, fixture_rows
from test_multiscale_v3_integration import config


class ClassificationMetricTests(unittest.TestCase):
    def test_only_classification_metrics_match_frozen_definitions(self):
        records = [dict(label=y, probability=p, score_prediction=s, score_label=t)
                   for y, p, s, t in [(0, .2, .9, .1), (1, .8, .1, .7), (1, 0., .2, .4)]]
        reference = control.learning.legacy.metrics(records)
        reference.pop('score_mae')
        self.assertEqual(control.classification_metrics(records), reference)
        without_severity = [{k: v for k, v in row.items() if not k.startswith('score_')}
                            for row in records]
        self.assertEqual(control.classification_metrics(without_severity), reference)
        self.assertIsNone(control.classification_metrics([]))

    def test_exports_withhold_auxiliary_prediction_without_mutation(self):
        raw = [{'sample_id': 'sample', 'label': 1, 'probability': .8,
                'score_label': .4, 'score_prediction': .93}]
        original = deepcopy(raw)
        exported = control.sanitize_predictions(raw)
        self.assertEqual(raw, original)
        self.assertIsNone(exported[0]['score_prediction'])
        self.assertFalse(exported[0]['severity_trained'])
        self.assertEqual(exported[0]['score_label'], .4)
        self.assertEqual(control.sanitize_predictions(exported), exported)
        self.assertNotIn('score_mae', control.classification_metrics(exported))

    def test_nonzero_or_missing_weight_rejected_before_fitter(self):
        with patch.object(budget, 'fit_classifier') as fitter:
            for value in (None, 1, .001, -1, True, False, '0', float('nan')):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'regression_weight'):
                    control.fit_classifier(None, {}, [], {}, '/tmp/not-created',
                                           {'training': {'regression_weight': value}})
            fitter.assert_not_called()

    def test_metrics_patch_restored_on_fitter_failure(self):
        before = control.learning.legacy.metrics
        with patch.object(budget, 'fit_classifier', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                control.fit_classifier(None, {}, [], {}, '/tmp/not-created',
                                       {'training': {'regression_weight': 0}})
        self.assertIs(control.learning.legacy.metrics, before)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class ClassificationTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_coa_paired_steps as network
        cls.network = network
        cls.tf = network.focused._tensorflow()

    def test_zero_weight_updates_are_independent_of_severity_targets(self):
        self.tf.keras.backend.clear_session()
        shape = [16, 16, 16]
        first = self.network.build_classifier(shape, base_filters=2, seed=42)
        second = self.network.build_classifier(shape, base_filters=2, seed=42)
        for a, b in zip(first.weights, second.weights):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        self.assertEqual(first.losses, [])
        values = {f'detail_{i}': self.tf.random.stateless_uniform((2, *shape, 2), seed=(42, i))
                  for i in range(2)}
        severity_before = [np.array(v) for v in first.get_layer('local_volume_loss').weights]
        all_before = [np.array(v) for v in first.weights]
        steps = [self.network.make_steps(model, shape, regression_weight=0)
                 for model in (first, second)]
        for step, severity in zip(steps, ([[.1], [.4]], [[.9], [.2]])):
            targets = {'coa_logit': self.tf.constant([[0.], [1.]]),
                       'local_volume_loss': self.tf.constant(severity)}
            for _ in range(2):
                result = step.train(values, targets)
                self.assertEqual(float(result['loss']), float(result['binary_crossentropy']))
            self.assertEqual(int(step.optimizer.iterations.numpy()), 2)
            step.predict(values)
            self.assertEqual(step.assert_stable_traces(), {'train': 1, 'predict': 1})
        for before, after in zip(severity_before, first.get_layer('local_volume_loss').weights):
            np.testing.assert_array_equal(before, np.asarray(after))
        self.assertTrue(any(not np.array_equal(a, np.asarray(b))
                            for a, b in zip(all_before, first.weights)))
        for a, b in zip(first.weights, second.weights):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-7, rtol=1e-6)
        a, b = steps[0].predict(values), steps[1].predict(values)
        np.testing.assert_allclose(np.asarray(a['coa_logit']), np.asarray(b['coa_logit']),
                                   atol=1e-7, rtol=1e-6)

    def _fit(self, root, *, instrumented):
        self.tf.keras.backend.clear_session()
        TinyFineInputs.accessed = []
        runtime = config(str(root.parent))
        runtime['input.detail']['shape_kji'] = [16, 16, 16]
        runtime['model']['seed'] = 42
        rows = [r for r in fixture_rows() if r['split'] == 'train']
        proposals = {r['sample_id']: {'origins_kji': [[0, 0, 0], [0, 0, 0]]} for r in rows}
        c = {'training': {'cache_limit_mb': 1, 'classifier_learning_rate': .001,
                          'regression_weight': 0.}}
        reference = self.network.build_classifier([16, 16, 16], base_filters=2, seed=42)
        fingerprint = control.learning.weight_fingerprint(reference)
        fitter = control.fit_classifier if instrumented else budget.fit_classifier
        console = StringIO()
        with patch.object(control.learning, 'StageFineInputs', TinyFineInputs), redirect_stdout(console):
            receipt = fitter(self.tf, runtime, rows, proposals, root, c, epochs=2,
                             checkpoint_interval=1, expected_initial_weights_sha256=fingerprint)
        self.assertEqual(set(TinyFineInputs.accessed), {r['sample_id'] for r in rows})
        return receipt, console.getvalue()

    def test_wrapper_preserves_both_output_initialization_and_all_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_metrics = control.learning.legacy.metrics
            raw, _ = self._fit(root / 'raw', instrumented=False)
            wrapped, console = self._fit(root / 'wrapped', instrumented=True)
            self.assertIs(control.learning.legacy.metrics, original_metrics)
            self.assertNotIn('score_mae', console)
            self.assertNotIn('score_mae', wrapped['training'])
            self.assertFalse(wrapped['severity_trained'])
            self.assertFalse(wrapped['severity_performance_evaluated'])
            self.assertTrue(wrapped['checkpoint_parity']['passed'])
            self.assertEqual(wrapped['selected_epoch'], 2)
            self.assertEqual(wrapped['optimizer_updates'], 12)
            self.assertEqual([r['epoch'] for r in wrapped['budget_checkpoints']], [1, 2])
            self.assertEqual(wrapped['budget_checkpoints'][-1]['checkpoint_sha256'],
                             wrapped['checkpoint_sha256'])
            self.assertEqual(wrapped['fresh_initial_weights_sha256'], raw['fresh_initial_weights_sha256'])
            initial = [control.learning.geometry.checked_json(root / folder / 'initialization.json')
                       for folder in ('raw', 'wrapped')]
            self.assertEqual(initial[0]['predictions'], initial[1]['predictions'])
            self.assertTrue(initial[1]['raw_auxiliary_predictions_for_parity_only'])
            self.assertTrue(all(isinstance(r['score_prediction'], float)
                                for r in initial[1]['predictions']))
            history = [control.learning.geometry.checked_json(root / folder / 'history.json')
                       for folder in ('raw', 'wrapped')]
            for a, b in zip(history[0]['epochs'], history[1]['epochs']):
                for key in ('epoch', 'training_pair_order', 'optimizer_updates', 'validation', 'selection_bce'):
                    self.assertEqual(a[key], b[key])
                for key in ('train_loss', 'gradient_norm_mean'):
                    self.assertAlmostEqual(a[key], b[key], places=7)
                expected = {k: v for k, v in a['training'].items() if k != 'score_mae'}
                self.assertEqual(expected, b['training'])
            models = [self.network.load_model(root / folder / 'classifier.keras')
                      for folder in ('raw', 'wrapped')]
            for a, b in zip(models[0].weights, models[1].weights):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-7, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
