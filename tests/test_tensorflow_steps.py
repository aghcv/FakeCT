import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_tensorflow_steps import make_locator_steps, make_classifier_steps, StableSteps

HAS_TF = importlib.util.find_spec('tensorflow') is not None


class StepContractTests(unittest.TestCase):
    def test_configuration_rejected_before_tensorflow_import(self):
        for settings in [dict(overview_shape=(0, 4, 4)), dict(overview_shape=(4, 4, 4), clipnorm=0),
                         dict(overview_shape=(4, 4, 4), learning_rate=float('nan'))]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_locator_steps(None, **settings)
        with self.assertRaises(ValueError):
            make_classifier_steps(None, (4, 4, 4), (4, 4, 4), num_proposals=0)
        with self.assertRaises(ValueError):
            make_classifier_steps(None, (4, 4, 4), (4, 4, 4), regression_weight=-1)

    def test_trace_audit_detects_growth(self):
        class Trace:
            def __init__(self, n): self.n = n
            def experimental_get_tracing_count(self): return self.n
        steps = StableSteps(Trace(0), Trace(0), None)
        self.assertEqual(steps.assert_stable_traces(), {'train': 0, 'predict': 0})
        steps.train.n = 2
        with self.assertRaisesRegex(RuntimeError, 'retraced'):
            steps.assert_stable_traces()


@unittest.skipUnless(HAS_TF, 'TensorFlow is optional outside the training container')
class TensorFlowStepTests(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        self.tf = tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        self.shape = (4, 4, 4)

    def locator(self, *, auxiliary=False):
        tf = self.tf
        x = tf.keras.Input(self.shape + (2,), name='overview')
        features = tf.keras.layers.GlobalAveragePooling3D()(x)
        logits = tf.keras.layers.Dense(6, kernel_initializer='zeros', bias_initializer='zeros')(features)
        box = tf.keras.layers.Activation('sigmoid')(logits)
        return tf.keras.Model({'overview': x}, {'box': box, 'box_logits': logits} if auxiliary else box)

    def classifier(self):
        tf = self.tf
        inputs = {'overview': tf.keras.Input(self.shape + (2,), name='overview'),
                  'proposal_boxes': tf.keras.Input((2, 6), name='proposal_boxes')}
        features = [tf.keras.layers.GlobalAveragePooling3D()(inputs['overview']),
                    tf.keras.layers.Flatten()(inputs['proposal_boxes'])]
        for i in range(2):
            inputs['detail_' + str(i)] = tf.keras.Input(self.shape + (2,), name='detail_' + str(i))
            features.append(tf.keras.layers.GlobalAveragePooling3D()(inputs['detail_' + str(i)]))
        merged = tf.keras.layers.Concatenate()(features)
        logit = tf.keras.layers.Dense(1)(merged)
        score = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
        return tf.keras.Model(inputs, {'coa_logit': logit, 'local_volume_loss': score})

    def test_locator_learns_and_uses_one_trace_for_changing_tensor_values(self):
        tf = self.tf
        model = self.locator()
        loss = lambda y, p: tf.reduce_mean((y - p) ** 2)
        steps = make_locator_steps(model, self.shape, learning_rate=.05, loss_function=loss)
        optimizer_variables_before = len(steps.optimizer.variables)
        target = tf.constant([[.65, .3, .4, .2, .1, .15]], tf.float32)
        losses = []
        for i in range(20):
            image = tf.ones((1,) + self.shape + (2,), tf.float32) * (.2 + (i % 2) * .001)
            result = steps.train(image, target)
            losses.append(float(result['loss']))
            self.assertTrue(np.isfinite(float(result['gradient_global_norm'])))
            self.assertEqual(tuple(steps.predict(image).shape), (1, 6))
        self.assertLess(losses[-1], .5 * losses[0])
        self.assertEqual(steps.assert_stable_traces(), {'train': 1, 'predict': 1})
        self.assertEqual(int(steps.optimizer.iterations.numpy()), 20)
        self.assertEqual(len(steps.optimizer.variables), optimizer_variables_before)

    def test_auxiliary_output_loss_and_prediction_structure(self):
        tf = self.tf
        model = self.locator(auxiliary=True)
        loss = lambda y, output: tf.reduce_mean((output['box'] - y) ** 2) + .01 * tf.reduce_mean(output['box_logits'] ** 2)
        steps = make_locator_steps(model, self.shape, loss_function=loss)
        image = tf.ones((1,) + self.shape + (2,), tf.float32)
        result = steps.train(image, tf.ones((1, 6), tf.float32) * .2)
        self.assertTrue(np.isfinite(float(result['loss'])))
        self.assertEqual(set(steps.predict(image)), {'box', 'box_logits'})
        self.assertEqual(steps.assert_stable_traces(), {'train': 1, 'predict': 1})

    def test_nonfinite_and_wrong_shape_inputs_cannot_update_model(self):
        tf = self.tf
        model = self.locator()
        steps = make_locator_steps(model, self.shape)
        target = tf.ones((1, 6), tf.float32) * .2
        before = [v.numpy().copy() for v in model.trainable_variables]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            steps.train(tf.fill((1,) + self.shape + (2,), float('nan')), target)
        self.assertEqual(int(steps.optimizer.iterations.numpy()), 0)
        for variable, initial in zip(model.trainable_variables, before):
            np.testing.assert_array_equal(variable.numpy(), initial)
        with self.assertRaises((TypeError, ValueError)):
            steps.train(tf.ones((2,) + self.shape + (2,), tf.float32), target)
        self.assertEqual(int(steps.optimizer.iterations.numpy()), 0)

    def test_classifier_matches_configured_losses_and_roundtrips(self):
        tf = self.tf
        model = self.classifier()
        steps = make_classifier_steps(model, self.shape, self.shape, regression_weight=1.7)
        inputs = {'overview': tf.ones((1,) + self.shape + (2,), tf.float32) * .2,
                  'proposal_boxes': tf.ones((1, 2, 6), tf.float32) * .3}
        inputs['detail_0'], inputs['detail_1'] = inputs['overview'], inputs['overview'] * 2
        targets = {'coa_logit': tf.constant([[1.]], tf.float32),
                   'local_volume_loss': tf.constant([[.35]], tf.float32)}
        initial = model(inputs, training=False)
        expected = tf.keras.losses.BinaryCrossentropy(from_logits=True)(targets['coa_logit'], initial['coa_logit'])
        expected += 1.7 * tf.keras.losses.Huber(delta=.1)(targets['local_volume_loss'], initial['local_volume_loss'])
        first = steps.train(inputs, targets)
        self.assertAlmostEqual(float(first['loss']), float(expected), places=6)
        for _ in range(5):
            steps.train(inputs, targets)
        before = steps.predict(inputs)
        self.assertEqual(steps.assert_stable_traces(), {'train': 1, 'predict': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.keras'
            model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            after = loaded(inputs, training=False)
            for key in before:
                np.testing.assert_allclose(before[key], after[key], rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
