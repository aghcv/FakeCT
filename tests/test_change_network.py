"""Net-change supervision, padding, actual learning and safe model reload.

These tests use small synthetic attenuation patches only, never held-out cohort
images or models. Geometry and numerical contracts match the production graph.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_change_network as network


class ChangeSettingsTests(unittest.TestCase):
    def test_import_remains_tensorflow_lazy(self):
        process = subprocess.run([sys.executable, '-c',
            'import sys; sys.path.insert(0, "src"); import fakect_change_network; '
            'assert "tensorflow" not in sys.modules'], cwd=ROOT, text=True,
            capture_output=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_geometry_and_loss_settings_fail_before_tensorflow_import(self):
        for shape in ((4, 8, 8), (12, 16, 16), (True, 16, 16), (104, 104, 104)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                network.build_model(shape)
        for setting in ({'base_filters': True}, {'base_filters': 32}, {'seed': -1}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                network.build_model((8, 8, 8), **setting)
        for change_weight, dice_weight in ((0., 1.), (10000., 1.), (8., -1.), (8., 100.)):
            with self.subTest(weights=(change_weight, dice_weight)), self.assertRaises(ValueError):
                network.loss_components(None, None, None, change_weight, dice_weight)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class ChangeLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf

    def test_known_three_class_targets_penalize_swapped_added_removed(self):
        y = np.array([[[[0, 1, 2]]]], np.int32)
        valid = np.ones_like(y, np.float32)
        logits = np.full(y.shape + (3,), -8., np.float32)
        for i, cls in enumerate((0, 1, 2)):
            logits[0, 0, 0, i, cls] = 8.
        good = network.loss_components(logits, y, valid)
        swapped = network.loss_components(logits[..., [0, 2, 1]], y, valid)
        self.assertLess(float(good['loss']), 1e-5)
        self.assertGreater(float(swapped['loss']), 10.)
        self.assertAlmostEqual(float(good['changed_voxels']), 2.)
        self.assertAlmostEqual(float(good['valid_voxels']), 3.)

    def test_weighted_ce_uses_weighted_valid_mass(self):
        y = np.array([[[[0, 1, 2]]]], np.int32)
        validity = np.array([[[[1., .5, 0.]]]], np.float32)
        logits = np.array([[[[[2., 0., 0.], [0., 1., 0.], [100., -100., 0.]]]]], np.float32)
        result = network.loss_components(logits, y, validity, change_weight=8., dice_weight=0.)
        ce0 = np.log(np.exp(2.) + 2.) - 2.
        ce1 = np.log(np.exp(1.) + 2.) - 1.
        self.assertAlmostEqual(float(result['cross_entropy']), (ce0 + 4. * ce1) / 5., places=6)
        self.assertAlmostEqual(float(result['loss']), float(result['cross_entropy']), places=6)

    def test_padding_cannot_change_loss_or_gradients_even_with_nan_and_sentinel(self):
        tf = self.tf
        y = np.array([[[[1, -1]]]], np.int32)
        valid = np.array([[[[1., 0.]]]], np.float32)
        base = np.zeros(y.shape + (3,), np.float32)
        poisoned = base.copy()
        poisoned[..., 1, :] = np.nan
        a = network.loss_components(base, y, valid)
        watched = tf.convert_to_tensor(poisoned)
        with tf.GradientTape() as tape:
            tape.watch(watched)
            b = network.loss_components(watched, y, valid)
        gradient = tape.gradient(b['loss'], watched).numpy()
        for name in a:
            self.assertAlmostEqual(float(a[name]), float(b[name]), places=7)
        np.testing.assert_array_equal(gradient[..., 1, :], 0.)
        self.assertLess(float(gradient[0, 0, 0, 0, 1]), 0.)
        self.assertTrue(np.isfinite(gradient).all())

    def test_unchanged_control_has_foreground_false_positive_penalty(self):
        y = np.zeros((1, 2, 2, 2), np.int32)
        valid = np.ones_like(y, np.float32)
        good = np.zeros(y.shape + (3,), np.float32)
        good[..., 0] = 12.
        bad = good[..., [1, 0, 2]]
        low = network.loss_components(good, y, valid)
        high = network.loss_components(bad, y, valid)
        self.assertLess(float(low['foreground_dice_loss']), 1e-5)
        self.assertGreater(float(high['foreground_dice_loss']), .49)
        self.assertGreater(float(high['cross_entropy']), 11.)
        self.assertEqual(float(high['changed_voxels']), 0.)

    def test_invalid_only_patch_has_zero_loss_and_zero_gradient(self):
        tf = self.tf
        watched = tf.ones((1, 2, 2, 2, 3), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(watched)
            result = network.loss_components(watched, tf.fill((1, 2, 2, 2), -1),
                                              tf.zeros((1, 2, 2, 2), tf.float32))
        self.assertTrue(all(float(v) == 0. for v in result.values()))
        np.testing.assert_array_equal(tape.gradient(result['loss'], watched).numpy(), 0.)

    def test_invalid_validity_classes_and_float_supervision_rejected(self):
        logits = np.zeros((1, 1, 1, 1, 3), np.float32)
        y = np.zeros((1, 1, 1, 1), np.int32)
        valid = np.ones(y.shape, np.float32)
        with self.assertRaises(ValueError):
            network.loss_components(logits, y.astype(np.float32), valid)
        for labels, validity in ((y + 3, valid), (y - 1, valid), (y, valid * 1.1),
                                 (y, valid * -.1), (y, valid * np.nan)):
            with self.subTest(labels=labels, validity=validity), self.assertRaises(
                    (ValueError, self.tf.errors.InvalidArgumentError)):
                network.loss_components(logits, labels, validity)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class ChangeNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf

    def setUp(self):
        self.tf.keras.backend.clear_session()
        self.shape = (16, 16, 16)
        self.model = network.build_model(self.shape, base_filters=2)

    def fixture(self, changes=True):
        x = np.zeros((1,) + self.shape + (2,), np.float32)
        x[..., 0] = .2
        x[..., 1] = 1.
        y = np.zeros((1,) + self.shape, np.int32)
        if changes:
            x[:, 3:7, 3:7, 3:7, 0] = .9
            y[:, 3:7, 3:7, 3:7] = 1
            x[:, 9:13, 9:13, 9:13, 0] = -.9
            y[:, 9:13, 9:13, 9:13] = 2
        return x, y, x[..., 1].copy()

    def test_graph_restores_voxel_resolution_and_exclusive_three_classes(self):
        self.assertEqual(self.model.input_shape, (None, 16, 16, 16, 2))
        self.assertEqual(self.model.output_shape, (None, 16, 16, 16, 3))
        self.assertEqual(self.model.get_layer('bottleneck_conv0').filters, 16)
        self.assertEqual(len([layer for layer in self.model.layers
                              if isinstance(layer, self.tf.keras.layers.MaxPool3D)]), 3)
        x, _, _ = self.fixture()
        probabilities = self.tf.nn.softmax(self.model(x), -1).numpy()
        np.testing.assert_allclose(probabilities.sum(-1), 1., atol=1e-6)
        classes = np.argmax(probabilities, -1)
        self.assertFalse(np.any((classes == 1) & (classes == 2)))

    def test_invalid_attenuation_has_no_effect_and_no_gradient(self):
        x, _, _ = self.fixture()
        x[:, :4, ..., 1] = 0.
        dirty = x.copy()
        dirty[:, :4, ..., 0] = np.nan
        np.testing.assert_allclose(self.model(x), self.model(dirty), rtol=0., atol=0.)
        watched = self.tf.convert_to_tensor(dirty)
        with self.tf.GradientTape() as tape:
            tape.watch(watched)
            value = self.tf.reduce_sum(self.model(watched))
        gradient = tape.gradient(value, watched).numpy()
        np.testing.assert_array_equal(gradient[:, :4, ..., 0], 0.)

    def test_real_training_reduces_sparse_change_loss_and_safe_reload(self):
        steps = network.make_steps(self.model, learning_rate=.003)
        x, y, valid = self.fixture()
        initial = network.loss_components(steps.predict(x), y, valid)
        before_probability = self.tf.nn.softmax(steps.predict(x), -1).numpy()
        for _ in range(35):
            values = steps.train(x, y, valid)
            self.assertTrue(all(np.isfinite(float(value)) for value in values.values()))
        after_probability = self.tf.nn.softmax(steps.predict(x), -1).numpy()
        final = network.loss_components(steps.predict(x), y, valid)
        self.assertLess(float(final['loss']), float(initial['loss']) * .6)
        for cls in (1, 2):
            target = y == cls
            self.assertGreater(float(after_probability[..., cls][target].mean()),
                               float(before_probability[..., cls][target].mean()) + .1)
        self.assertEqual(steps.assert_stable_traces(), {'train': 1, 'predict': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'change.keras'
            self.model.save(path)
            restored = network.load_model(path)
            np.testing.assert_allclose(restored(x), steps.predict(x), rtol=1e-6, atol=1e-6)

    def test_invalid_only_training_is_noop_even_after_adam_momentum_exists(self):
        steps = network.make_steps(self.model)
        x, y, valid = self.fixture()
        steps.train(x, y, valid)
        before = [v.numpy().copy() for v in self.model.variables]
        optimizer_before = [v.numpy().copy() for v in steps.optimizer.variables]
        invalid = np.zeros_like(x)
        invalid[..., 0] = np.nan
        result = steps.train(invalid, np.full_like(y, -1), np.zeros_like(valid))
        self.assertTrue(all(float(value) == 0. for value in result.values()))
        for old, new in zip(before, self.model.variables):
            np.testing.assert_array_equal(old, new.numpy())
        for old, new in zip(optimizer_before, steps.optimizer.variables):
            np.testing.assert_array_equal(old, new.numpy())
        self.assertEqual(int(steps.optimizer.iterations.numpy()), 1)

    def test_fixed_signature_and_supervision_validity_contract(self):
        steps = network.make_steps(self.model)
        x, y, valid = self.fixture()
        with self.assertRaises((TypeError, ValueError)):
            steps.predict(np.repeat(x, 2, 0))
        with self.assertRaises((TypeError, ValueError)):
            steps.train(x, self.tf.convert_to_tensor(y, dtype=self.tf.float32), valid)
        with self.assertRaises(self.tf.errors.InvalidArgumentError):
            steps.train(x, y, np.zeros_like(valid))


if __name__ == '__main__':
    unittest.main()
