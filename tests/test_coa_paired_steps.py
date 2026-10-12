"""One-update matched-sibling optimization and unchanged graph integration."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_coa_focused_network as focused
import fakect_coa_paired_steps as paired


class PairedContractTests(unittest.TestCase):
    # Purpose: Verify paired training reuses the established classifier graph.
    # Inputs: Focused and paired build, load, and feature-probe functions.
    # Outputs: Each paired export is the identical original function.
    def test_graph_build_load_and_probe_are_the_original_functions(self):
        self.assertIs(paired.build_classifier, focused.build_classifier)
        self.assertIs(paired.load_model, focused.load_model)
        self.assertIs(paired.feature_probe, focused.feature_probe)

    # Purpose: Require the exact paired microbatch size.
    # Inputs: Batch sizes 1,3,True,2.0 with no model.
    # Outputs: Step creation raises ValueError before TensorFlow.
    def test_microbatch_configuration_rejected_before_tensorflow(self):
        for batch in (1, 3, True, 2.):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                paired.make_steps(None, (16, 16, 16), batch_size=batch)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class PairedTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf

    def setUp(self):
        self.tf.keras.backend.clear_session()

    def toy_model(self):
        """Zero-logit linear readout with a large common background feature."""
        layers = self.tf.keras.layers
        inputs = {name: layers.Input((8, 8, 8, 2), name=name) for name in ('detail_0', 'detail_1')}
        features = layers.Concatenate()([layers.GlobalAveragePooling3D()(image) for image in inputs.values()])
        outputs = {'coa_logit': layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros',
                                            name='coa_logit')(features),
                   'local_volume_loss': layers.Dense(1, activation='sigmoid', kernel_initializer='zeros',
                                                    bias_initializer='zeros', name='local_volume_loss')(features)}
        return self.tf.keras.Model(inputs, outputs)

    def toy_inputs(self, difference=0.):
        image = np.ones((2, 8, 8, 8, 2), np.float32)
        image[..., 0] = 10.
        image[1, ..., 0] += difference
        return {'detail_0': image.copy(), 'detail_1': image.copy()}

    def targets(self):
        return {'coa_logit': np.array([[0.], [1.]], np.float32),
                'local_volume_loss': np.full((2, 1), .5, np.float32)}

    # Purpose: Verify opposing sibling gradients cancel before clipping.
    # Inputs: Zero-logit toy model, identical background inputs, opposite binary labels.
    # Outputs: Large individual gradients sum to zero; one optimizer step leaves every weight unchanged.
    def test_large_opposing_background_gradients_cancel_before_clipping(self):
        model = self.toy_model()
        inputs, targets = self.toy_inputs(), self.targets()
        variables = model.get_layer('coa_logit').trainable_variables
        individual = []
        for row in range(2):
            with self.tf.GradientTape() as tape:
                output = model({key: value[row:row + 1] for key, value in inputs.items()})['coa_logit']
                bce = self.tf.reduce_mean(self.tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets['coa_logit'][row:row + 1], logits=output))
            individual.append(tape.gradient(bce, variables))
        self.assertGreater(float(self.tf.linalg.global_norm(individual[0])), 7.)
        self.assertGreater(float(self.tf.linalg.global_norm(individual[1])), 7.)
        for negative, positive in zip(*individual):
            np.testing.assert_array_equal(np.asarray(negative) + np.asarray(positive), 0.)
        steps = paired.make_steps(model, (8, 8, 8), regression_weight=0., clipnorm=1.)
        before = [np.asarray(variable).copy() for variable in model.trainable_variables]
        result = steps.train(inputs, targets)
        self.assertEqual(float(result['gradient_global_norm']), 0.)
        self.assertAlmostEqual(float(result['binary_crossentropy']), float(np.log(2)), places=6)
        for old, variable in zip(before, model.trainable_variables):
            np.testing.assert_array_equal(old, np.asarray(variable))
        self.assertEqual(int(np.asarray(steps.optimizer.iterations)), 1)

    # Purpose: Preserve a tiny sibling signal in the combined gradient.
    # Inputs: Toy pair differing by 0.01 attenuation with opposite labels.
    # Outputs: Gradient norm matches direct differentiation between 0.003 and 0.004; positive logit increases.
    def test_combined_gradient_retains_tiny_sibling_signal(self):
        model = self.toy_model()
        inputs, targets = self.toy_inputs(difference=.01), self.targets()
        with self.tf.GradientTape() as tape:
            output = model(inputs)['coa_logit']
            bce = self.tf.reduce_mean(self.tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets['coa_logit'], logits=output))
        direct = tape.gradient(bce, model.get_layer('coa_logit').trainable_variables)
        expected = float(self.tf.linalg.global_norm(direct))
        self.assertGreater(expected, .003)
        self.assertLess(expected, .004)
        steps = paired.make_steps(model, (8, 8, 8), regression_weight=0.)
        result = steps.train(inputs, targets)
        self.assertAlmostEqual(float(result['gradient_global_norm']), expected, places=6)
        output = steps.predict(inputs)['coa_logit'].numpy()
        self.assertGreater(float(output[1, 0] - output[0, 0]), 0.)
        self.assertEqual(int(np.asarray(steps.optimizer.iterations)), 1)

    # Purpose: Verify paired sparse learning and safe checkpoint reload.
    # Inputs: 16-cube sibling pair trained for 60 alternating-order updates.
    # Outputs: BCE falls below 60%, class probabilities separate, traces stay single, and reload outputs match.
    def test_real_paired_sparse_learning_reload_parity_and_one_trace(self):
        shape = (16, 16, 16)
        model = paired.build_classifier(shape, base_filters=2, seed=42)
        values = {}
        for name in ('detail_0', 'detail_1'):
            image = np.ones((2,) + shape + (2,), np.float32)
            image[..., 0] = .2
            values[name] = image
        values['detail_0'][1, 6:10, 6:10, 6:10, 0] = .8
        targets = {'coa_logit': np.array([[0.], [1.]], np.float32),
                   'local_volume_loss': np.array([[0.], [.5]], np.float32)}
        steps = paired.make_steps(model, shape, learning_rate=.003)

        def score():
            logits = steps.predict(values)['coa_logit']
            bce = self.tf.reduce_mean(self.tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets['coa_logit'], logits=logits))
            return self.tf.sigmoid(logits).numpy().ravel(), float(bce)

        _, initial = score()
        for iteration in range(60):
            order = [0, 1] if iteration % 2 == 0 else [1, 0]
            result = steps.train({key: value[order] for key, value in values.items()},
                                 {key: value[order] for key, value in targets.items()})
            self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
        probability, bce = score()
        self.assertLess(bce, initial * .6)
        self.assertLess(probability[0], .3)
        self.assertGreater(probability[1], .7)
        self.assertEqual(int(np.asarray(steps.optimizer.iterations)), 60)
        self.assertEqual(steps.assert_stable_traces(), {'train': 1, 'predict': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paired.keras'
            model.save(path)
            restored = paired.load_model(path)
            before, after = steps.predict(values), restored(values, training=False)
            for key in before:
                np.testing.assert_allclose(before[key], after[key], rtol=1e-6, atol=1e-6)
            maps = paired.feature_probe(restored)(values, training=False)
            self.assertEqual(tuple(maps['detail_0_down4'].shape), (2, 4, 4, 4, 1))

    # Purpose: Enforce the fixed paired input signature.
    # Inputs: Single-row detail inputs and an extra overview input.
    # Outputs: Prediction raises TypeError or ValueError.
    def test_wrong_batch_and_extra_input_are_rejected_by_fixed_signature(self):
        steps = paired.make_steps(self.toy_model(), (8, 8, 8))
        values = {key: value[:1] for key, value in self.toy_inputs().items()}
        with self.assertRaises((TypeError, ValueError)):
            steps.predict(values)
        values = self.toy_inputs()
        values['overview'] = values['detail_0']
        with self.assertRaises((TypeError, ValueError)):
            steps.predict(values)

    # Purpose: Reject unbalanced or nonbinary sibling labels without updating.
    # Inputs: Toy pairs labeled [0,0], [1,1], or [0.5,0.5].
    # Outputs: InvalidArgumentError leaves optimizer iterations zero and model weights unchanged.
    def test_unbalanced_or_nonbinary_targets_fail_without_update(self):
        model = self.toy_model()
        steps = paired.make_steps(model, (8, 8, 8))
        before = [np.asarray(variable).copy() for variable in model.trainable_variables]
        for labels in ([[0.], [0.]], [[1.], [1.]], [[.5], [.5]]):
            targets = self.targets()
            targets['coa_logit'] = np.array(labels, np.float32)
            with self.subTest(labels=labels), self.assertRaises(self.tf.errors.InvalidArgumentError):
                steps.train(self.toy_inputs(), targets)
            self.assertEqual(int(np.asarray(steps.optimizer.iterations)), 0)
        for old, variable in zip(before, model.trainable_variables):
            np.testing.assert_array_equal(old, np.asarray(variable))


if __name__ == '__main__':
    unittest.main()
