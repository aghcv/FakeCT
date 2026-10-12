"""Image-only multiscale graph, physical alignment, gradients and safe checkpoints."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_multiscale_network as network


class NetworkContractTests(unittest.TestCase):
    # Purpose: Verify plain multiscale-network imports remain TensorFlow lazy.
    # Inputs: Fresh Python subprocess importing fakect_multiscale_network.
    # Outputs: Import succeeds with tensorflow absent from loaded modules.
    def test_plain_import_does_not_import_tensorflow(self):
        code = "import sys; sys.path.insert(0, 'src'); import fakect_multiscale_network; assert 'tensorflow' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True)

    # Purpose: Verify TensorFlow helper leaves device configuration to the runner.
    # Inputs: Mock tensorflow module supplied through sys.modules.
    # Outputs: Same module returned without device enumeration or memory/logical-device changes.
    def test_tensorflow_helper_leaves_device_configuration_to_runner(self):
        fake_tf = Mock()
        with patch.dict(sys.modules, {'tensorflow': fake_tf}):
            self.assertIs(network._tensorflow(), fake_tf)
        fake_tf.config.list_physical_devices.assert_not_called()
        fake_tf.config.experimental.set_memory_growth.assert_not_called()
        fake_tf.config.set_logical_device_configuration.assert_not_called()

    # Purpose: Verify feature-grid centers match same-padding arithmetic.
    # Inputs: 160x96x96, odd 17-cube and unpooled 8-cube shapes.
    # Outputs: Expected feature shapes, center offsets and strides 16 or 1.
    def test_feature_centers_follow_actual_same_padding(self):
        self.assertEqual(network.feature_grid((160, 96, 96)), ((10, 6, 6), (15., 15., 15.), 16))
        self.assertEqual(network.feature_grid((17, 17, 17)), ((2, 2, 2), (0., 0., 0.), 16))
        self.assertEqual(network.feature_grid((8, 8, 8), levels=0), ((8, 8, 8), (0., 0., 0.), 1))

    # Purpose: Reject invalid locator geometry before TensorFlow use.
    # Inputs: Undersized, malformed, fractional or oversized shapes and base_filters 1.
    # Outputs: ValueError rejects each graph construction request.
    def test_invalid_shapes_fail_before_tensorflow(self):
        for shape in ((2, 8, 8), (8, 8), (8.5, 8, 8), (512, 512, 512)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                network.build_locator(shape)
        with self.assertRaises(ValueError):
            network.build_locator((8, 8, 8), base_filters=1)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is unavailable')
class TensorFlowNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tf = network._tensorflow()
        except ImportError as error:
            raise unittest.SkipTest(f'TensorFlow runtime unavailable: {error}')

    def setUp(self):
        self.tf.keras.backend.clear_session()

    def images(self, shape=(16, 16, 16), seed=41):
        rng = np.random.default_rng(seed)
        result = np.ones((1,) + shape + (2,), np.float32)
        result[..., 0] = rng.uniform(0, 1, result.shape[:-1])
        return result

    # Purpose: Verify bounded boxes and disjoint-target loss have corrective gradients.
    # Inputs: Extreme raw box parameters and disjoint predicted/truth boxes.
    # Outputs: Positive extents inside FOV, finite loss, corrective center gradients and zero exact-match loss.
    def test_parameterized_boxes_are_bounded_and_disjoint_loss_has_gradient(self):
        tf = self.tf
        BoundedBox, _, _ = network._components()
        box = BoundedBox()(tf.constant([[-100., 0., 100., -100., 100., 0.]]))
        center, extent = box.numpy()[:, :3], box.numpy()[:, 3:]
        self.assertTrue(np.all(extent > 0))
        self.assertTrue(np.all(center - extent / 2 >= -1e-7))
        self.assertTrue(np.all(center + extent / 2 <= 1 + 1e-7))
        prediction = tf.Variable([[.8, .8, .8, .1, .1, .1]])
        truth = tf.constant([[.2, .2, .2, .1, .1, .1]])
        with tf.GradientTape() as tape:
            loss = network.locator_loss(truth, prediction)
        gradient = tape.gradient(loss, prediction).numpy()
        self.assertTrue(np.isfinite(loss.numpy()).all())
        self.assertTrue(np.all(gradient[0, :3] > 0))
        np.testing.assert_allclose(network.locator_loss(truth, truth).numpy(), 0, atol=1e-6)

    # Purpose: Verify aligned trilinear feature samples follow physical coordinates.
    # Inputs: 8-cube linear feature field, central box and an outside-FOV box.
    # Outputs: Analytic samples and box gradients 216,432,648; finite feature gradients and outside samples zero.
    def test_aligned_trilinear_samples_match_physical_linear_field(self):
        tf = self.tf
        _, Sample, _ = network._components()
        grid = np.indices((8, 8, 8)).astype(np.float32)
        features = tf.Variable((grid[0] + 2 * grid[1] + 3 * grid[2])[None, ..., None])
        boxes = tf.Variable([[[.5, .5, .5, .25, .25, .25]]])
        layer = Sample((8, 8, 8), levels=0)
        with tf.GradientTape(persistent=True) as tape:
            actual = layer([features, boxes])
            loss = tf.reduce_sum(actual)
        points = np.linspace(2.5, 4.5, 3)
        k, j, i = np.meshgrid(points, points, points, indexing='ij')
        expected = (k + 2 * j + 3 * i).reshape(1, 1, 27)
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
        self.assertTrue(np.isfinite(tape.gradient(loss, features).numpy()).all())
        np.testing.assert_allclose(tape.gradient(loss, boxes).numpy()[0, 0, :3], [216, 432, 648], atol=1e-5)
        outside = layer([features, tf.constant([[[-2., -2., -2., .1, .1, .1]]])])
        np.testing.assert_array_equal(outside.numpy(), 0)

    # Purpose: Verify locator training updates predictions and safely reloads compiled state.
    # Inputs: 16-cube seeded image, known target box and one training batch.
    # Outputs: Finite learning, changed predictions, matching reload and further training succeeds.
    def test_locator_training_and_compiled_safe_reload(self):
        model = network.compile_locator(network.build_locator((16, 16, 16), base_filters=2))
        image = self.images()
        truth = np.array([[.4, .5, .6, .3, .4, .3]], np.float32)
        before = network.predict_boxes(model, image)
        loss = model.train_on_batch({'overview': image}, truth)
        self.assertTrue(np.isfinite(loss))
        after = network.predict_boxes(model, image)
        self.assertFalse(np.array_equal(before, after))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'locator.keras'
            model.save(path)
            loaded = network.load_model(path, compile=True)
            np.testing.assert_allclose(network.predict_boxes(loaded, image), after, atol=1e-6)
            self.assertTrue(np.isfinite(loaded.train_on_batch({'overview': image}, truth)))

    # Purpose: Verify classifier trains shared fine features while freezing locator features.
    # Inputs: Locator encoder reused by classifier with two fine proposals and class/severity targets.
    # Outputs: Overview weights unchanged, fine weights update, bounded severity and matching safe reload.
    def test_classifier_reuses_frozen_encoder_and_shared_fine_encoder(self):
        tf = self.tf
        locator = network.build_locator((16, 16, 16), base_filters=2)
        model = network.compile_classifier(network.build_classifier(
            (16, 16, 16), (16, 16, 16), base_filters=2, locator=locator))
        self.assertIsInstance(model.get_layer('detail_mean'), tf.keras.layers.GlobalAveragePooling3D)
        self.assertIsInstance(model.get_layer('detail_max'), tf.keras.layers.GlobalMaxPooling3D)
        encoder = locator.get_layer('overview_encoder')
        self.assertIs(model.get_layer('overview_encoder'), encoder)
        self.assertFalse(encoder.trainable)
        before = [v.numpy().copy() for v in encoder.weights]
        fine_before = [v.numpy().copy() for v in model.get_layer('detail_encoder').weights]
        data = {'overview': self.images(), 'detail_0': self.images(seed=10), 'detail_1': self.images(seed=11),
                'proposal_boxes': np.array([[[.4, .5, .5, .5, .5, .5], [.7, .5, .5, .5, .5, .5]]], np.float32)}
        targets = {'coa_logit': np.array([[1]], np.float32), 'local_volume_loss': np.array([[.35]], np.float32)}
        loss = model.train_on_batch(data, targets, return_dict=True)
        self.assertTrue(all(np.isfinite(value) for value in loss.values()))
        for old, new in zip(before, encoder.weights):
            np.testing.assert_array_equal(old, new.numpy())
        self.assertTrue(any(not np.array_equal(old, new.numpy()) for old, new in
                            zip(fine_before, model.get_layer('detail_encoder').weights)))
        prediction = {key: np.asarray(value) for key, value in model(data, training=False).items()}
        self.assertEqual(set(prediction), {'coa_logit', 'local_volume_loss'})
        self.assertTrue(0 <= prediction['local_volume_loss'][0, 0] <= 1)
        self.assertFalse(any(isinstance(layer, (tf.keras.layers.Lambda, tf.keras.layers.BatchNormalization))
                             for layer in model.layers))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'classifier.keras'
            model.save(path)
            loaded = network.load_model(path, compile=True)
            actual = loaded(data, training=False)
            for key in prediction:
                np.testing.assert_allclose(actual[key].numpy(), prediction[key], atol=1e-6)
            self.assertFalse(loaded.get_layer('overview_encoder').trainable)
            self.assertTrue(all(np.isfinite(value) for value in loaded.train_on_batch(data, targets)))

    # Purpose: Verify wrong saturated logits retain corrective BCE gradients.
    # Inputs: Logits -1000 and 1000 with opposite binary labels.
    # Outputs: Finite loss and gradients -0.5 and 0.5.
    def test_wrong_saturated_classifier_logits_retain_corrective_gradients(self):
        tf = self.tf
        logits = tf.Variable([[-1000.], [1000.]])
        truth = tf.constant([[1.], [0.]])
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            loss = loss_function(truth, logits)
        self.assertTrue(np.isfinite(float(loss)))
        np.testing.assert_allclose(tape.gradient(loss, logits).numpy(), [[-.5], [.5]], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
