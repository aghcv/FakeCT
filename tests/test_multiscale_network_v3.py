"""Spatial geometry, stable supervision, image response and TF checkpoint tests."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_multiscale_network_v3 as network


class SpatialContractTests(unittest.TestCase):
    # Purpose: Verify lazy TensorFlow import and stride-four heatmap coordinates.
    # Inputs: Subprocess module import, 160x96x96 geometry, and invalid dimension 17.
    # Outputs: Import loads no TensorFlow; grid edges match voxel centers and nondivisible geometry fails.
    def test_lazy_import_and_stride_four_voxel_edge_coordinates(self):
        code = "import sys; sys.path.insert(0,'src'); import fakect_multiscale_network_v3; assert 'tensorflow' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True)
        grid = network.heatmap_grid((160, 96, 96))
        self.assertEqual(grid.shape, (40, 24, 24, 3))
        np.testing.assert_allclose(grid[0, 0, 0], np.array([3.5/160, 3.5/96, 3.5/96]), atol=1e-7)
        np.testing.assert_allclose(grid[-1, -1, -1], np.array([159.5/160, 95.5/96, 95.5/96]), atol=1e-7)
        with self.assertRaisesRegex(ValueError, 'divisible by 4'):
            network.build_locator((17, 16, 16))


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is unavailable')
class SpatialLocatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tf = network._tensorflow()
        except ImportError as error:
            raise unittest.SkipTest(str(error))

    def setUp(self):
        self.tf.keras.backend.clear_session()

    def images(self, centers):
        grid = np.indices((32, 32, 32), dtype=np.float32)
        image = np.ones((len(centers), 32, 32, 32, 2), np.float32)
        for index, center in enumerate(centers):
            radius_squared = np.sum((grid - np.array(center)[:, None, None, None]) ** 2, axis=0)
            image[index, ..., 0] = np.exp(-radius_squared / 8.)
        return image

    # Purpose: Verify heatmap coordinates match actual stride-two convolution sampling.
    # Inputs: 32-cube coordinate ramp passed through two centered stride-two convolutions.
    # Outputs: Convolved coordinates match heatmap_grid within 1e-7.
    def test_grid_matches_two_actual_same_stride_convolutions(self):
        tf = self.tf
        grid = np.indices((32, 32, 32), dtype=np.float32)
        ramp = ((grid[0] + .5)/32)[None, ..., None]
        x = tf.constant(ramp)
        for _ in range(2):
            convolution = tf.keras.layers.Conv3D(1, 3, strides=2, padding='same', use_bias=False)
            convolution.build(x.shape)
            kernel = np.zeros((3, 3, 3, 1, 1), np.float32)
            kernel[1, 1, 1, 0, 0] = 1
            convolution.set_weights([kernel])
            x = convolution(x)
        np.testing.assert_allclose(x.numpy()[0, ..., 0], network.heatmap_grid((32, 32, 32))[..., 0], atol=1e-7)

    # Purpose: Verify wrong heatmaps retain corrective gradients while ignoring padding.
    # Inputs: Far-away high logit, invalid first heatmap slice, and a fixed target box.
    # Outputs: Finite loss and corrective peak gradients ignore padding; target heatmap sums to one.
    def test_distant_wrong_heatmap_retains_corrective_gradient_and_ignores_padding(self):
        tf = self.tf
        components = network._components()
        value = np.zeros((1, 8, 8, 8, 1), np.float32)
        value[0, 5, 3, 3, 0] = 50
        logits = tf.Variable(value)
        valid = np.ones_like(value)
        valid[:, 0] = 0
        truth = tf.constant([[11.5/32, 15.5/32, 15.5/32, .25, .25, .25]])
        extent_logits = tf.constant([[-1.1, -1.1, -1.1]])
        with tf.GradientTape() as tape:
            masked_logits = components['MaskHeatmap']()([logits, tf.constant(valid)])
            center = components['SpatialExpectation']((32, 32, 32))(masked_logits)
            box = components['SpatialBox']()([center, extent_logits])
            outputs = {'box': box, 'heatmap_logits': masked_logits,
                       'heatmap_valid': valid, 'extent_logits': extent_logits}
            loss = network.locator_loss(truth, outputs)
        gradient = tape.gradient(loss, logits).numpy()
        self.assertTrue(np.isfinite(loss.numpy()).all())
        self.assertGreater(gradient[0, 5, 3, 3, 0], .9)
        self.assertLess(gradient[0, 2, 3, 3, 0], -.1)
        np.testing.assert_array_equal(gradient[:, 0], 0)
        target = network.target_heatmap(truth, (32, 32, 32), valid).numpy().reshape(1, 8, 8, 8, 1)
        np.testing.assert_array_equal(target[:, 0], 0)
        np.testing.assert_allclose(target.sum(), 1, atol=1e-6)

    # Purpose: Verify locator centers depend on images rather than prior coordinates.
    # Inputs: Same image with different center priors and a legacy locator.
    # Outputs: Predicted centers match; no flattened head exists and incompatible legacy reuse raises ValueError.
    def test_center_does_not_use_prior_and_no_flattened_coordinate_head_exists(self):
        image = self.images([(15, 15, 15)])
        first = network.build_locator((32, 32, 32), base_filters=2,
                                      prior_box=[.3, .4, .5, .2, .2, .2])
        first_box = network.predict_boxes(first, image)
        second = network.build_locator((32, 32, 32), base_filters=2,
                                       prior_box=[.7, .6, .5, .2, .2, .2])
        np.testing.assert_allclose(network.predict_boxes(second, image), first_box, atol=1e-7)
        self.assertFalse(any(isinstance(layer, self.tf.keras.layers.Flatten) for layer in first.layers))
        dense_layers = [layer for layer in first.layers if isinstance(layer, self.tf.keras.layers.Dense)]
        self.assertEqual([(layer.name, layer.units) for layer in dense_layers], [('locator_extent_logits', 3)])
        old_locator = network.previous.build_locator((32, 32, 32), base_filters=2)
        with self.assertRaisesRegex(ValueError, 'stride-four'):
            network.build_classifier((32, 32, 32), (16, 16, 16), base_filters=2, locator=old_locator)

    # Purpose: Verify translation learning and safe reuse of a saved locator.
    # Inputs: Four translated Gaussian images, 60 TensorFlow updates, and saved locator/classifier.
    # Outputs: Loss halves with one trace; translations track targets and both checkpoint reloads match.
    def test_synthetic_translation_learning_and_safe_classifier_reuse(self):
        tf = self.tf
        centers = [(7, 15, 15), (11, 15, 15), (19, 15, 15), (23, 15, 15)]
        image = self.images(centers)
        truth = np.array([[(k+.5)/32, (j+.5)/32, (i+.5)/32, .25, .25, .25]
                          for k, j, i in centers], np.float32)
        model = network.build_locator((32, 32, 32), base_filters=2,
                                      prior_box=[.5, .5, .5, .25, .25, .25])
        view = network.locator_outputs(model)
        optimizer = tf.keras.optimizers.Adam(.003, clipnorm=1.)
        optimizer.build(model.trainable_variables)
        @tf.function(input_signature=[tf.TensorSpec((4, 32, 32, 32, 2), tf.float32), tf.TensorSpec((4, 6), tf.float32)])
        def step(x, target):
            with tf.GradientTape() as tape:
                output = view({'overview': x}, training=True)
                loss = tf.reduce_mean(network.locator_loss(target, output))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        first = float(step(image, truth))
        for _ in range(59):
            last = float(step(image, truth))
        self.assertLess(last, first * .5)
        self.assertEqual(step.experimental_get_tracing_count(), 1)
        translated = self.images([(13, 15, 15), (21, 15, 15)])
        prediction = network.predict_boxes(model, translated)
        # The shifted Gaussian is new; this is a toy spatial-response check,
        # not a claim of held-out anatomical or clinical generalization.
        self.assertGreater(prediction[1, 0] - prediction[0, 0], .15)
        np.testing.assert_allclose(prediction[:, :3], np.array([[13.5, 15.5, 15.5], [21.5, 15.5, 15.5]])/32, atol=.08)
        with tempfile.TemporaryDirectory() as directory:
            locator_path = Path(directory) / 'locator.keras'
            model.save(locator_path)
            loaded = network.load_model(locator_path)
            np.testing.assert_allclose(network.predict_boxes(loaded, translated), prediction, atol=1e-6)
            restored_outputs = network.locator_outputs(loaded)({'overview': translated})
            self.assertEqual(set(restored_outputs), {'box', 'heatmap_logits', 'heatmap_valid', 'extent_logits'})
            classifier = network.build_classifier((32, 32, 32), (16, 16, 16), base_filters=2, locator=loaded)
            self.assertIs(classifier.get_layer('overview_encoder'), loaded.get_layer('overview_encoder'))
            self.assertFalse(classifier.get_layer('overview_encoder').trainable)
            self.assertEqual(classifier.get_layer('aligned_coarse_samples').get_config()['levels'], 2)
            detail = np.ones((1, 16, 16, 16, 2), np.float32)
            batch = {'overview': translated[:1], 'detail_0': detail, 'detail_1': detail,
                     'proposal_boxes': np.array([[[.4, .5, .5, .5, .5, .5],
                                                  [.7, .5, .5, .5, .5, .5]]], np.float32)}
            expected = classifier(batch, training=False)
            classifier_path = Path(directory) / 'classifier.keras'
            classifier.save(classifier_path)
            actual = network.load_model(classifier_path)(batch, training=False)
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
