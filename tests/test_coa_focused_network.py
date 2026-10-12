"""Focused high-resolution graph, validity, real learning and reload checks.

Only tiny synthetic attenuation fixtures are read. No cohort/test payloads.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_coa_focused_network as network


class FocusedGeometryTests(unittest.TestCase):
    # Purpose: Verify production encoder preserves useful spatial resolution.
    # Inputs: Input geometry [216,192,192].
    # Outputs: Stride is 4, feature grid is [54,48,48], and six-bin windows are [9,8,8].
    def test_production_features_retain_stride_four_and_ordered_spatial_grid(self):
        geometry = network.feature_geometry([216, 192, 192])
        self.assertEqual(geometry['encoder_stride'], 4)
        self.assertEqual(geometry['feature_shape_kji'], [54, 48, 48])
        self.assertEqual(geometry['spatial_bin_shape_kji'], [6, 6, 6])
        self.assertEqual(geometry['spatial_bin_window_kji'], [9, 8, 8])

    # Purpose: Reject unsupported geometry and resource settings before TensorFlow.
    # Inputs: Malformed/oversized shapes, invalid proposal counts, filters, and seed.
    # Outputs: Geometry or classifier construction raises ValueError.
    def test_invalid_geometry_and_resource_parameters_rejected_before_tensorflow(self):
        for shape in ((7, 8, 8), (12, 9, 12), (True, 16, 16), (512, 512, 512)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                network.feature_geometry(shape)
        for settings in ({'num_proposals': True}, {'num_proposals': 5},
                         {'base_filters': 32}, {'base_filters': True}, {'seed': -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                network.build_classifier((16, 16, 16), **settings)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class FocusedTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf

    def setUp(self):
        self.tf.keras.backend.clear_session()
        self.shape = (16, 16, 16)
        self.model = network.build_classifier(self.shape, base_filters=2, seed=42)

    def image(self, value=.2):
        image = np.zeros((1,) + self.shape + (2,), np.float32)
        image[..., 0] = value
        image[..., 1] = 1.
        return image

    def inputs(self, *, positive=False):
        image = self.image()
        if positive:
            image[:, 6:10, 6:10, 6:10, 0] = .8
        return {'detail_0': image, 'detail_1': self.image()}

    # Purpose: Verify focused model input channels, heads, and downsampling.
    # Inputs: 16-cube TensorFlow classifier with two detail blocks.
    # Outputs: Only detail inputs and two heads exist; encoder has two stride-2 layers and no LayerNormalization.
    def test_inputs_are_only_attenuation_validity_detail_blocks(self):
        self.assertEqual(set(self.model.input), {'detail_0', 'detail_1'})
        self.assertEqual(set(self.model.output), {'coa_logit', 'local_volume_loss'})
        self.assertEqual(len(self.model.get_layer('focused_detail_encoder').input_shape), 5)
        encoder = self.model.get_layer('focused_detail_encoder')
        self.assertFalse(any(isinstance(layer, self.tf.keras.layers.LayerNormalization)
                             for layer in encoder.layers))
        strided = [layer for layer in encoder.layers if isinstance(layer, self.tf.keras.layers.Conv3D)
                   and layer.strides != (1, 1, 1)]
        self.assertEqual([layer.strides for layer in strided], [(2, 2, 2), (2, 2, 2)])

    # Purpose: Verify a sparse detail signal influences all features and logits.
    # Inputs: A 4-cube attenuation increase from 0.2 to 0.8 in one detail input.
    # Outputs: Each feature stage and logit changes; the signal region has nonzero input gradients.
    def test_sparse_signal_changes_fine_features_evidence_and_prediction(self):
        probe = network.feature_probe(self.model)
        a = probe(self.inputs(), training=False)
        b = probe(self.inputs(positive=True), training=False)
        for stage in ('amplitude', 'down2', 'down4', 'context', 'local_logit', 'spatial_grid', 'pooled'):
            key = 'detail_0_' + stage
            difference = np.abs(np.asarray(a[key]) - np.asarray(b[key]))
            self.assertGreater(float(difference.max()), 1e-7, key)
        self.assertEqual(tuple(a['detail_0_down4'].shape[1:4]), (4, 4, 4))
        self.assertEqual(tuple(a['detail_0_context'].shape[1:4]), (4, 4, 4))
        output_a = self.model(self.inputs(), training=False)
        output_b = self.model(self.inputs(positive=True), training=False)
        self.assertGreater(abs(float(output_a['coa_logit'][0, 0]) - float(output_b['coa_logit'][0, 0])), 1e-7)
        # A changed attenuation voxel has a differentiable route to the scalar
        # decision rather than only a visually changing diagnostic map.
        image = self.tf.convert_to_tensor(self.inputs(positive=True)['detail_0'])
        with self.tf.GradientTape() as tape:
            tape.watch(image)
            scalar = self.model({'detail_0': image, 'detail_1': self.tf.convert_to_tensor(self.image())},
                                training=False)['coa_logit']
        gradient = tape.gradient(scalar, image).numpy()
        self.assertGreater(float(np.abs(gradient[:, 6:10, 6:10, 6:10, 0]).max()), 1e-8)

    # Purpose: Verify invalid padding cannot affect predictions or gradients.
    # Inputs: First four slices invalid with attenuation replaced by 10000.
    # Outputs: Outputs stay identical, padding gradients/attention are zero, and training stays finite.
    def test_invalid_padding_attenuation_has_no_effect_or_gradient(self):
        image = self.image()
        image[:, :4, :, :, 1] = 0.
        altered = image.copy()
        altered[:, :4, :, :, 0] = 10000.
        a = self.model({'detail_0': image, 'detail_1': image}, training=False)
        b = self.model({'detail_0': altered, 'detail_1': altered}, training=False)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0)
        watched = self.tf.convert_to_tensor(altered)
        with self.tf.GradientTape() as tape:
            tape.watch(watched)
            logit = self.model({'detail_0': watched, 'detail_1': self.tf.convert_to_tensor(image)},
                               training=False)['coa_logit']
        gradient = tape.gradient(logit, watched).numpy()
        np.testing.assert_array_equal(gradient[:, :4, :, :, 0], 0.)
        probe = network.feature_probe(self.model)({'detail_0': altered, 'detail_1': image}, training=False)
        np.testing.assert_array_equal(np.asarray(probe['detail_0_attention'])[:, :1], 0.)
        self.assertAlmostEqual(float(np.asarray(probe['detail_0_attention']).sum()), 1., places=6)
        steps = network.make_steps(self.model, self.shape)
        targets = {'coa_logit': np.array([[0]], np.float32),
                   'local_volume_loss': np.array([[0]], np.float32)}
        result = steps.train({'detail_0': altered, 'detail_1': image}, targets)
        self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))

    # Purpose: Verify spatial pooling ignores invalid values and empty bins.
    # Inputs: Features -2 with invalid values 1000000, then fully invalid detail images.
    # Outputs: Invalid bins, attention, and pooled features are zero; all model outputs stay finite.
    def test_masked_spatial_bins_ignore_large_invalid_values_and_empty_bins(self):
        components = network._components()
        layer = components['SpatialBins']((4, 4, 4), bins=2)
        features = np.full((1, 4, 4, 4, 1), -2., np.float32)
        valid = np.ones_like(features)
        valid[:, :2] = 0.
        features[:, :2] = 1e6
        bins = np.asarray(layer([features, valid]))
        np.testing.assert_array_equal(bins[:, :1], 0.)
        np.testing.assert_array_equal(bins[:, 1:], -2.)
        all_invalid = self.image(1000.)
        all_invalid[..., 1] = 0.
        output = self.model({'detail_0': all_invalid, 'detail_1': all_invalid}, training=False)
        self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in output.values()))
        probe = network.feature_probe(self.model)({'detail_0': all_invalid, 'detail_1': all_invalid}, training=False)
        np.testing.assert_array_equal(np.asarray(probe['detail_0_attention']), 0.)
        np.testing.assert_array_equal(np.asarray(probe['detail_0_pooled']), 0.)

    # Purpose: Verify sparse classifier learning, stable traces, and checkpoint parity.
    # Inputs: Two sparse-signal fixtures trained for 60 updates at learning rate 0.003.
    # Outputs: BCE falls below 60%, probabilities separate, traces stay single, and reload predictions match.
    def test_real_sparse_training_stable_traces_and_safe_reload(self):
        steps = network.make_steps(self.model, self.shape, learning_rate=.003)
        fixtures = [(self.inputs(), {'coa_logit': np.array([[0]], np.float32),
                                     'local_volume_loss': np.array([[0]], np.float32)}),
                    (self.inputs(positive=True), {'coa_logit': np.array([[1]], np.float32),
                                                 'local_volume_loss': np.array([[.5]], np.float32)})]

        def scores():
            probabilities, losses = [], []
            for inputs, target in fixtures:
                output = steps.predict(inputs)
                logit = output['coa_logit']
                probabilities.append(float(self.tf.sigmoid(logit)[0, 0]))
                losses.append(float(self.tf.reduce_mean(self.tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=target['coa_logit'], logits=logit))))
            return probabilities, float(np.mean(losses))

        initial_probability, initial_bce = scores()
        for _ in range(30):
            for inputs, targets in fixtures:
                result = steps.train(inputs, targets)
                self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
        probability, bce = scores()
        self.assertLess(bce, initial_bce * .6)
        self.assertLess(probability[0], .3)
        self.assertGreater(probability[1], .7)
        self.assertEqual(steps.assert_stable_traces(), {'train': 1, 'predict': 1})
        self.assertEqual(int(np.asarray(steps.optimizer.iterations)), 60)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'focused.keras'
            self.model.save(path)
            restored = network.load_model(path)
            for inputs, _ in fixtures:
                before = steps.predict(inputs)
                after = restored(inputs, training=False)
                for key in before:
                    np.testing.assert_allclose(before[key], after[key], rtol=1e-6, atol=1e-6)
            restored_probe = network.feature_probe(restored)
            self.assertIn('detail_0_spatial_grid', restored_probe.output)

    # Purpose: Reject extra inputs and unsupported inference batch sizes.
    # Inputs: Overview added to detail inputs, then detail batch size 2.
    # Outputs: Fixed prediction signature raises TypeError or ValueError.
    def test_training_signature_rejects_extra_overview_and_wrong_batch(self):
        steps = network.make_steps(self.model, self.shape)
        values = self.inputs()
        values['overview'] = self.image()
        with self.assertRaises((TypeError, ValueError)):
            steps.predict(values)
        values = {name: np.repeat(image, 2, axis=0) for name, image in self.inputs().items()}
        with self.assertRaises((TypeError, ValueError)):
            steps.predict(values)


if __name__ == '__main__':
    unittest.main()
