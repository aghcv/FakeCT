"""Regression tests for the r1 locator saturation failure and v2 correction."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_multiscale_network_v2 as network


class PriorContractTests(unittest.TestCase):
    # Purpose: Verify network imports stay lazy and priors invert the bounded-box parameterization.
    # Inputs: Fresh subprocess import and explicit six-value prior box.
    # Outputs: No TensorFlow import and reconstructed center/extent matches the prior.
    def test_import_stays_lazy_and_prior_is_explicit_data(self):
        code = "import sys; sys.path.insert(0,'src'); import fakect_multiscale_network_v2; assert 'tensorflow' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True)
        prior = np.array([.67, .46, .49, .12, .13, .08])
        logits = network.prior_parameters(prior)
        probability = 1 / (1 + np.exp(-logits))
        extent = .001 + .999 * probability[3:]
        center = extent / 2 + (1 - extent) * probability[:3]
        np.testing.assert_allclose(np.concatenate([center, extent]), prior, atol=1e-7)

    # Purpose: Verify malformed box priors fail before model construction.
    # Inputs: Five-value prior, zero extent, out-of-bounds center, and NaN prior.
    # Outputs: ValueError for every invalid prior.
    def test_invalid_prior_fails_before_tensorflow(self):
        for prior in ([.5] * 5, [.5, .5, .5, 0, .1, .1], [.95, .5, .5, .2, .2, .2],
                      [np.nan, .5, .5, .2, .2, .2]):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                network.build_locator((16, 16, 16), prior_box=prior)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is unavailable')
class StableLocatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tf = network._tensorflow()
        except ImportError as error:
            raise unittest.SkipTest(str(error))

    def setUp(self):
        self.tf.keras.backend.clear_session()

    # Purpose: Verify TensorFlow classifiers reuse reloaded singleton-list encoders.
    # Inputs: Locator encoder wrapped in one-output list model and tiny classifier input.
    # Outputs: Same frozen encoder, finite heads, exact reload predictions, and multiple-output rejection.
    def test_classifier_accepts_singleton_list_encoder_without_replacing_weights(self):
        tf = self.tf
        original = network.build_locator((16, 16, 16), base_filters=2)
        encoder = original.get_layer('overview_encoder')
        # Explicitly reproduce the structure returned by TF 2.17 / Keras 3.4
        # nested models after reload; newer local versions flatten it instead.
        listed = tf.keras.Model(encoder.input, [encoder.output], name='overview_encoder')
        locator = SimpleNamespace(get_layer=lambda name: listed)
        model = network.build_classifier((16, 16, 16), (16, 16, 16), base_filters=2, locator=locator)
        self.assertIs(model.get_layer('overview_encoder'), listed)
        self.assertFalse(listed.trainable)
        image = np.ones((1, 16, 16, 16, 2), np.float32)
        batch = {'overview': image, 'detail_0': image, 'detail_1': image,
                 'proposal_boxes': np.array([[[.4, .5, .5, .5, .5, .5],
                                              [.7, .5, .5, .5, .5, .5]]], np.float32)}
        expected = model(batch, training=False)
        for value in expected.values():
            self.assertEqual(tuple(value.shape), (1, 1))
            self.assertTrue(np.isfinite(np.asarray(value)).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'classifier.keras'
            model.save(path)
            loaded = network.load_model(path)
            actual = loaded(batch, training=False)
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            network._single_feature_output([encoder.output, encoder.output], 'overview_encoder')

    # Purpose: Verify locator logit loss retains corrective gradients at saturation.
    # Inputs: Wrong center logits at magnitudes 20 and 1000 with bounded-box target.
    # Outputs: Finite loss and correctly signed center gradients above magnitude 0.08.
    def test_raw_loss_has_corrective_gradient_when_wrong_sigmoid_saturates(self):
        tf = self.tf
        BoundedBox, _, _ = network.base._components()
        truth = tf.constant([[.5, .5, .5, .15, .15, .15]])
        for magnitude in (20., 1000.):
            logits = tf.Variable([[magnitude, -magnitude, magnitude, -2., -2., -2.]])
            with tf.GradientTape() as tape:
                prediction = BoundedBox()(logits)
                loss = network.locator_loss(truth, prediction, logits=logits)
            gradient = tape.gradient(loss, logits).numpy()
            self.assertTrue(np.isfinite(loss.numpy()).all())
            self.assertGreater(gradient[0, 0], .08)
            self.assertLess(gradient[0, 1], -.08)
            self.assertGreater(gradient[0, 2], .08)

    # Purpose: Verify locator raw-logit loss agrees with box loss for ordinary predictions.
    # Inputs: Benign logits and exact prior-derived logits for target box.
    # Outputs: Matching losses and near-zero loss for exact target.
    def test_raw_and_box_loss_agree_at_benign_logits_and_correct_box_has_zero_loss(self):
        tf = self.tf
        BoundedBox, _, _ = network.base._components()
        truth = tf.constant([[.66, .47, .49, .12, .13, .08]])
        logits = tf.constant([[.3, -.4, .1, -1.7, -1.8, -2.2]])
        prediction = BoundedBox()(logits)
        np.testing.assert_allclose(network.locator_loss(truth, prediction, logits=logits),
                                   network.locator_loss(truth, prediction), atol=1e-6)
        exact_logits = tf.constant(network.prior_parameters(truth.numpy()[0])[None])
        exact_prediction = BoundedBox()(exact_logits)
        np.testing.assert_allclose(network.locator_loss(truth, exact_prediction, logits=exact_logits), 0, atol=1e-7)

    # Purpose: Verify corrected locators start near priors and train/save without retracing.
    # Inputs: 16-cubed seeded image, explicit prior, six Adam updates, and saved locator.
    # Outputs: Reduced finite loss, one trace, reload parity, and frozen reused classifier encoder.
    def test_model_starts_near_prior_trains_with_single_trace_and_reloads(self):
        tf = self.tf
        prior = np.array([.66, .47, .49, .12, .13, .08], np.float32)
        model = network.build_locator((16, 16, 16), base_filters=2, prior_box=prior)
        view = network.locator_outputs(model)
        rng = np.random.default_rng(11)
        images = np.ones((1, 16, 16, 16, 2), np.float32)
        images[..., 0] = rng.random(images.shape[:-1])
        prediction = network.predict_boxes(model, images)
        np.testing.assert_allclose(prediction[0], prior, atol=.002)
        optimizer = tf.keras.optimizers.Adam(.001, clipnorm=1.)
        optimizer.build(model.trainable_variables)
        @tf.function(input_signature=[tf.TensorSpec((1, 16, 16, 16, 2), tf.float32), tf.TensorSpec((1, 6), tf.float32)])
        def step(image, target):
            with tf.GradientTape() as tape:
                output = view({'overview': image}, training=True)
                loss = tf.reduce_mean(network.locator_loss(target, output['box'], logits=output['box_logits']))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        target = np.array([[.69, .48, .51, .10, .12, .07]], np.float32)
        first = float(step(images, target))
        for _ in range(5):
            last = float(step(images, target))
        self.assertTrue(np.isfinite(last))
        self.assertLess(last, first)
        self.assertEqual(step.experimental_get_tracing_count(), 1)
        expected = network.predict_boxes(model, images)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'locator.keras'
            model.save(path)
            loaded = network.load_model(path)
            np.testing.assert_allclose(network.predict_boxes(loaded, images), expected, atol=1e-6)
            self.assertEqual(set(network.locator_outputs(loaded)({'overview': images})), {'box', 'box_logits'})
            classifier = network.build_classifier((16, 16, 16), (16, 16, 16), base_filters=2, locator=loaded)
            self.assertIs(classifier.get_layer('overview_encoder'), loaded.get_layer('overview_encoder'))
            self.assertFalse(classifier.get_layer('overview_encoder').trainable)


if __name__ == '__main__':
    unittest.main()
