"""Physical context, complete inference coverage, and diagnostic isolation."""
import configparser
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import fakect_context_learning as learning


class FakeVolume:
    def __init__(self, shape=(9, 11, 10)):
        self.shape = shape
        self.mask = np.zeros(shape, np.uint8)
        self.mask[3:6, 4:7, 4:7] = 1
        self.image = self.mask.astype(np.float32)*.5
        self.origins = []
        self.closed = False

    def model_shape(self, spacing):
        return self.shape

    def foreground_points(self, spacing, max_points=100_000, seed=42):
        return np.argwhere(self.mask)

    def read_block(self, origin, shape, spacing):
        self.origins.append(tuple(origin))
        image, mask, valid = (np.zeros(shape, dtype) for dtype in (np.float32, np.uint8, np.uint8))
        sizes = tuple(min(p, n-o) for n, p, o in zip(self.shape, shape, origin))
        src = tuple(slice(o, o+n) for o, n in zip(origin, sizes))
        dst = tuple(slice(0, n) for n in sizes)
        image[dst], mask[dst], valid[dst] = self.image[src], self.mask[src], 1
        return dict(image=image, mask=mask, valid=valid)

    def close(self):
        self.closed = True


class IdentityModel:
    def __call__(self, image, training=False):
        return image


def identity_inference(model):
    def predict(image, training=False):
        probability = model(image, training=training)
        logits = np.where(probability >= .5, 40., -40.).astype(np.float32)
        return probability, logits
    return predict


class ContextLearningTests(unittest.TestCase):
    def settings(self, **kwargs):
        return learning.settings_for({'model': dict(patch_size_kji=[8, 8, 8], depth=1, base_filters=2, **kwargs)})

    def rows(self):
        return [dict(sample_id=f'{part}:sample', case_id=f'{part}:case',
                     anatomy_family=f'{part}:family', split=part)
                for part in ('train', 'validation', 'test')]

    # Purpose: Verify typed context settings load from ConfigParser.
    # Inputs: 128x96x96 patch, 1/1/2 mm spacing, false flips, three epochs, and blank diagnostic ID.
    # Outputs: Parsed numeric vectors, boolean false, epoch integer, and blank diagnostic selector.
    def test_configparser_types_and_larger_physical_default_context(self):
        parser = configparser.ConfigParser()
        parser.read_string('[model]\npatch_size_kji=128,96,96\ntarget_spacing_mm=1,1,2\n'
                           'augmentation_flip=false\ndiagnostic_sample_id=\nepochs=3\n')
        settings = learning.settings_for(parser)
        self.assertEqual(settings['target_spacing_mm'], [1., 1., 2.])
        self.assertEqual(settings['patch_size_kji'], [128, 96, 96])
        self.assertFalse(settings['augmentation_flip'])
        self.assertEqual(settings['epochs'], 3)
        self.assertEqual(settings['diagnostic_sample_id'], '')

    # Purpose: Verify GPU allocator audits handle unsupported devices.
    # Inputs: CPU plus two GPUs, second GPU memory query raising ValueError.
    # Outputs: Only GPUs queried, byte counts retained, and unavailable reason recorded.
    def test_gpu_memory_snapshot_records_bytes_and_tolerates_unsupported_devices(self):
        devices = [SimpleNamespace(name='/device:CPU:0', device_type='CPU'),
                   SimpleNamespace(name='/device:GPU:0', device_type='GPU'),
                   SimpleNamespace(name='/device:GPU:1', device_type='GPU')]
        queried = []
        def memory_info(name):
            queried.append(name)
            if name == 'GPU:1':
                raise ValueError('Allocator does not expose statistics')
            return dict(current=1024, peak=8192)
        tf = SimpleNamespace(config=SimpleNamespace(list_logical_devices=lambda: devices,
            experimental=SimpleNamespace(get_memory_info=memory_info)))
        values = learning.gpu_allocator_memory(tf)
        self.assertEqual(queried, ['GPU:0', 'GPU:1'])
        self.assertEqual(values[0]['current_bytes'], 1024)
        self.assertEqual(values[0]['peak_bytes'], 8192)
        self.assertEqual(values[1]['status'], 'unavailable')
        self.assertIn('Allocator', values[1]['reason'])

    # Purpose: Verify context models expose one predictor and one output.
    # Inputs: Single input/output model and duplicate input/output variants.
    # Outputs: Single tensors returned and ValueError for multiple tensors.
    def test_model_boundary_requires_single_image_and_probability_tensor(self):
        image, probability = object(), object()
        model = SimpleNamespace(inputs=[image], outputs=[probability])
        self.assertEqual(learning._single_io(model), (image, probability))
        for inputs, outputs in (([image, image], [probability]), ([image], [probability, probability])):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                learning._single_io(SimpleNamespace(inputs=inputs, outputs=outputs))

    # Purpose: Verify prediction validation distinguishes tensor, numeric, and probability failures.
    # Inputs: Wrong shapes, NaN probability, infinite logits, and probabilities outside [0,1].
    # Outputs: Specific ValueError messages for every invalid prediction pair.
    def test_prediction_guard_distinguishes_structure_numeric_and_range_errors(self):
        image = np.zeros((1, 2, 2, 2, 1), np.float32)
        cases = [(image[None], image, 'shape mismatch'),
                 (image, image[None], 'shape mismatch'),
                 (image+np.nan, image, 'probability has .* nonfinite'),
                 (image, image+np.inf, 'logits have .* nonfinite'),
                 (image-1e-7, image, 'outside'),
                 (image+1.000001, image, 'outside')]
        for probability, logits, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                learning._predict_pair(lambda *args, **kwargs: (probability, logits), image)

    # Purpose: Verify model settings enforce supported architecture and sampling contracts.
    # Inputs: Invalid depth/batch/spacing/patch/budgets/rates/ranges/booleans and unknown fields.
    # Outputs: ValueError for every unsupported or malformed setting.
    def test_invalid_or_unsupported_settings_fail(self):
        cases = [dict(depth=5), dict(batch_size=2), dict(threshold=1),
                 dict(target_spacing_mm='1,nan,1'), dict(target_spacing_mm='1,0,1'),
                 dict(patch_size_kji='130,96,96'), dict(patch_size_kji='256,256,256'),
                 dict(epochs=True), dict(epochs='1.5'), dict(learning_rate=0),
                 dict(clip_min=.5, clip_max=.5), dict(augmentation_flip='maybe'),
                 dict(save_predictions=True), dict(foreground_fraction=1),
                 dict(intensity_scale_range='1.2,.8'), dict(unknown=1)]
        for model in cases:
            with self.subTest(model=model), self.assertRaises(ValueError):
                learning.settings_for({'model': model})

    # Purpose: Verify inference tiles cover irregular volume edges exactly once.
    # Inputs: 13x7x19 volume tiled with 8-cubed blocks.
    # Outputs: All voxels counted once, tile count agrees, and edge origin 8/0/16 included.
    def test_nonoverlapping_tiles_cover_each_voxel_once_including_edges(self):
        shape, patch_size = (13, 7, 19), (8, 8, 8)
        coverage = np.zeros(shape, np.uint8)
        origins = list(learning.tile_origins(shape, patch_size))
        self.assertEqual(len(origins), learning.tile_count(shape, patch_size))
        for origin in origins:
            coverage[tuple(slice(o, o+p) for o, p in zip(origin, patch_size))] += 1
        np.testing.assert_array_equal(coverage, 1)
        self.assertIn((8, 0, 16), origins)

    # Purpose: Verify sampling mixes target-centered and uniform physical context reproducibly.
    # Inputs: 100-cubed shape, 16-cubed patch, center point, four draws, and seed 12.
    # Outputs: Identical offsets, exactly two foreground-containing patches, and distant uniform context.
    def test_sampling_is_reproducible_and_keeps_foreground_and_uniform_context(self):
        points = np.asarray([[50, 50, 50]], np.int64)
        args = ((100, 100, 100), (16, 16, 16), points, 4)
        a = learning.sample_origins(*args, np.random.default_rng(12))
        b = learning.sample_origins(*args, np.random.default_rng(12))
        self.assertEqual(a, b)
        includes = [all(o <= c < o+16 for o, c in zip(origin, points[0])) for origin in a]
        self.assertEqual(sum(includes), 2)
        self.assertTrue(any(max(origin) > 50 for origin in a))

    # Purpose: Verify foreground sampling rejects missing or invalid target coordinates.
    # Inputs: Empty points, out-of-bounds integer point, and floating-point point array.
    # Outputs: ValueError for each invalid target geometry.
    def test_sampling_requires_target_and_valid_geometry(self):
        for points in (np.empty((0, 3), int), np.asarray([[100, 1, 1]]), np.asarray([[1., 1., 1.]])):
            with self.assertRaises(ValueError):
                learning.sample_origins((10, 10, 10), (8, 8, 8), points, 4, np.random.default_rng(1))

    # Purpose: Verify invalid padding contributes neither metrics nor training loss.
    # Inputs: Two correct valid predictions plus two incorrect padded predictions.
    # Outputs: Dice 1, zero false positives, two valid voxels, and same loss as valid-only data.
    def test_padding_does_not_contribute_counts_or_loss(self):
        truth = np.asarray([1, 0, 1, 0])
        probability = np.asarray([.9, .1, 0., 1.])
        valid = np.asarray([1, 1, 0, 0])
        metrics = learning.metrics_from_counts(learning.counts_for(truth, probability >= .5, valid), 2)
        self.assertEqual(metrics['dice'], 1)
        self.assertEqual(metrics['false_positive'], 0)
        self.assertEqual(metrics['valid_voxels'], 2)
        a = learning.loss_from_totals(learning.loss_totals(truth, probability, valid))
        b = learning.loss_from_totals(learning.loss_totals(truth[:2], probability[:2], valid[:2]))
        self.assertAlmostEqual(a, b)

    # Purpose: Verify loss aggregation uses global sufficient statistics.
    # Inputs: Eight truth/probability values split into unequal target/background tiles.
    # Outputs: Summed totals match whole loss and differ from the mean tile loss.
    def test_loss_totals_are_additive_and_global_dice_is_not_mean_tile_dice(self):
        truth = np.asarray([1, 1, 1, 0, 0, 0, 0, 0])
        probability = np.asarray([.8, .7, .4, .2, .3, .6, .9, .1])
        valid = np.ones_like(truth)
        first = learning.loss_totals(truth[:3], probability[:3], valid[:3])
        second = learning.loss_totals(truth[3:], probability[3:], valid[3:])
        combined = {key: first[key]+second[key] for key in first}
        whole = learning.loss_totals(truth, probability, valid)
        self.assertAlmostEqual(learning.loss_from_totals(combined), learning.loss_from_totals(whole))
        self.assertNotAlmostEqual(learning.loss_from_totals(whole),
            (learning.loss_from_totals(first)+learning.loss_from_totals(second))/2)

    # Purpose: Verify logit BCE preserves extreme error magnitude and ignores padding.
    # Inputs: Wrong logits -1000/+1000 plus padded third value and NaN variant.
    # Outputs: BCE sum 2000, uncapped loss above 1000, and finite-logit validation.
    def test_stable_logit_score_does_not_cap_extreme_wrong_predictions(self):
        truth, probability, valid = np.asarray([1, 0, 1]), np.asarray([0., 1., 0.]), np.asarray([1, 1, 0])
        totals = learning.loss_totals(truth, probability, valid, logits=np.asarray([-1000., 1000., -1000.]))
        self.assertEqual(totals['bce_sum'], 2000.)
        clipped = learning.loss_totals(truth, probability, valid)
        self.assertLess(clipped['bce_sum'], 28.)
        self.assertGreater(learning.loss_from_totals(totals), 1000.)
        with self.assertRaisesRegex(ValueError, 'finite logits'):
            learning.loss_totals(truth, probability, valid, logits=np.asarray([0., 0., np.nan]))

    # Purpose: Verify paired augmentation aligns image, mask, and validity.
    # Inputs: 8-cubed mask/image with invalid slab, seed 3, and unit intensity scaling.
    # Outputs: Aligned predictors/targets, contiguous buffers, and 1x8x8x8x1 shape.
    def test_paired_augmentation_retains_alignment_and_excludes_padding(self):
        settings = self.settings(intensity_scale_range=[1., 1.])
        mask = np.zeros((8, 8, 8), np.uint8)
        mask[1:4, 2:5, 5:7] = 1
        image, valid = mask.astype(np.float32)*.5, np.ones_like(mask)
        valid[-1] = 0
        image[-1] = .4
        x, y, v = learning.prepare_block(image, mask, valid, settings, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(x, y)
        self.assertTrue(x.flags.c_contiguous)
        self.assertTrue(v.flags.c_contiguous)
        self.assertEqual(x.shape, (1, 8, 8, 8, 1))

    # Purpose: Verify complete-volume inference detects distant false positives.
    # Inputs: FakeVolume with extra positive signal at the far edge and identity predictor.
    # Outputs: 27 true positives, one false positive, eight unique tiles, and all native voxels scored.
    def test_full_volume_score_finds_false_positive_outside_target_region(self):
        volume = FakeVolume()
        volume.image[8, 10, 9] = .5
        with patch.object(learning, '_inference_model', side_effect=identity_inference):
            result = learning.score_volume(IdentityModel(), volume, self.settings())
        self.assertEqual(result['metrics']['valid_voxels'], np.prod(volume.shape))
        self.assertEqual(result['metrics']['true_positive'], 27)
        self.assertEqual(result['metrics']['false_positive'], 1)
        self.assertEqual(result['tiles'], 8)
        self.assertEqual(len(set(volume.origins)), 8)

    # Purpose: Verify whole-volume scoring requires nonempty targets.
    # Inputs: FakeVolume with zeroed target mask and identity predictor.
    # Outputs: ValueError identifying an empty target.
    def test_full_volume_score_rejects_empty_target(self):
        volume = FakeVolume()
        volume.mask[:] = 0
        with patch.object(learning, '_inference_model', side_effect=identity_inference), \
                self.assertRaisesRegex(ValueError, 'target is empty'):
            learning.score_volume(IdentityModel(), volume, self.settings())

    # Purpose: Verify metadata validation rejects cross-partition case/family identities.
    # Inputs: Three split rows with validation case or family copied from TRAIN.
    # Outputs: ValueError for leakage and three rows accepted for the disjoint fixture.
    def test_family_and_case_split_leakage_rejected_without_loading_data(self):
        for key in ('case_id', 'anatomy_family'):
            rows = self.rows()
            rows[1][key] = rows[0][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'crosses partitions'):
                learning.inspect_rows({'samples': rows})
        self.assertEqual(len(learning.inspect_rows({'samples': self.rows()})), 3)

    # Purpose: Verify diagnostics open and close only the selected TRAIN volume.
    # Inputs: Three split rows and mock FakeVolume factory.
    # Outputs: One TRAIN access, closed reader, nonempty target, and TRAIN diagnostic metadata.
    def test_diagnostic_selection_opens_only_training_sample(self):
        settings = self.settings()
        loaded, volume = [], FakeVolume()
        def create(row):
            loaded.append(row['split'])
            return volume
        with patch.object(learning, '_volume', side_effect=create):
            row, arrays, info = learning._diagnostic_selection(self.rows(), settings)
        self.assertEqual(loaded, ['train'])
        self.assertEqual(row['split'], 'train')
        self.assertTrue(volume.closed)
        self.assertTrue(arrays[1].any())
        self.assertEqual(info['split'], 'train')

    # Purpose: Verify explicit diagnostic selectors cannot request test data.
    # Inputs: Diagnostic sample ID test:sample and mocked volume loader.
    # Outputs: ValueError for nontraining selection before any loader invocation.
    def test_diagnostic_cannot_request_heldout_sample(self):
        with patch.object(learning, '_volume') as loader:
            with self.assertRaisesRegex(ValueError, 'training partition'):
                learning._diagnostic_selection(self.rows(), self.settings(diagnostic_sample_id='test:sample'))
            loader.assert_not_called()

    # Purpose: Verify family summaries avoid weighting by sibling count.
    # Inputs: Nine perfect siblings in family one and one failed sample in family two.
    # Outputs: Sample Dice 0.9 and equal-family Dice 0.5.
    def test_family_summary_does_not_weight_families_by_sibling_count(self):
        def record(case, family, dice):
            return dict(case_id=case, anatomy_family=family,
                metrics=dict(dice=dice, precision=dice, recall=dice,
                             false_positive_mm3=1-dice, false_negative_mm3=1-dice))
        result = learning.summarize([record('a', 'one', 1.)]*9+[record('b', 'two', 0.)])
        self.assertAlmostEqual(result['sample_macro']['dice'], .9)
        self.assertAlmostEqual(result['family_macro']['dice'], .5)

    # Purpose: Verify mocked diagnostic orchestration preserves TRAIN-only provenance.
    # Inputs: Two-epoch identity model, fake TensorFlow/Keras, and one sample per split.
    # Outputs: TRAIN-only fit/evaluation, saved prediction, and rejection of scope/history provenance drift.
    def test_diagnostic_fit_and_evaluation_preserve_train_only_scope(self):
        # Exercise the orchestration without TensorFlow; real tensor-gradient
        # smoke is opt-in below and is also run in the cluster environment.
        settings = self.settings(epochs=2, diagnostic_steps_per_epoch=2)
        config, lock = dict(model=settings), dict(samples=self.rows())
        model = IdentityModel()
        model.input_shape = model.output_shape = (None, 8, 8, 8, 1)
        model.inputs = [SimpleNamespace(shape=model.input_shape)]
        model.outputs = [SimpleNamespace(shape=model.output_shape)]
        model.count_params = lambda: 1
        model.save = lambda path: Path(path).write_bytes(b'test model')
        step = lambda *args: .1
        step.experimental_get_tracing_count = lambda: 1
        tf = SimpleNamespace(__version__='test',
            keras=SimpleNamespace(utils=SimpleNamespace(set_random_seed=lambda x: None),
                                  models=SimpleNamespace(load_model=lambda *args, **kwargs: model)),
            config=SimpleNamespace(experimental=SimpleNamespace(enable_op_determinism=lambda: None),
                                   list_logical_devices=lambda: []))
        loaded = []
        def volume(row):
            loaded.append(row['split'])
            return FakeVolume()
        real_package_version = learning.package_version
        def fixture_package_version(name):
            # The fake TensorFlow/Keras runtime above has no installed Keras
            # distribution in the core environment. Keep real CPU dependency
            # metadata and supply only the explicitly mocked backend version.
            return 'test' if name == 'keras' else real_package_version(name)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(learning, '_tensorflow', return_value=tf), \
                patch.object(learning, 'package_version', side_effect=fixture_package_version), \
                patch.object(learning, 'build_model', return_value=model), \
                patch.object(learning, '_training_step', return_value=step), \
                patch.object(learning, '_inference_model', side_effect=identity_inference), \
                patch.object(learning, '_volume', side_effect=volume):
            directory = Path(tmp)/'model'
            metadata = learning.fit(config, lock, directory, diagnostic=True)
            self.assertEqual(metadata['keras_version'], 'test')
            self.assertFalse(metadata['test_payloads_loaded'])
            self.assertEqual(metadata['epochs_completed'], 2)
            self.assertEqual(metadata['loaded_samples'], ['train:sample'])
            self.assertEqual(metadata['diagnostic_before']['metrics']['dice'], 1.)
            self.assertTrue((directory/'diagnostic-prediction.npz').is_file())
            result = learning.evaluate(config, lock, directory, Path(tmp)/'evaluation', diagnostic=True)
            self.assertEqual(result['split'], 'train')
            self.assertIn('not generalization', result['scope'])
            self.assertFalse(result['test_payloads_loaded'])
            self.assertEqual(loaded, ['train', 'train'])
            with self.assertRaisesRegex(ValueError, 'provenance'):
                learning.load_model(directory, config, lock, diagnostic=False)
            (directory/'history.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                learning.load_model(directory, config, lock, diagnostic=True)

    # Purpose: Verify opt-in TensorFlow loss agrees with NumPy and survives checkpoint reload.
    # Inputs: 8-cubed FakeVolume patch and real context model/training step.
    # Outputs: Five-place loss agreement, correct model shape, and equal saved/reloaded predictions.
    @unittest.skipUnless(os.environ.get('FAKECT_TEST_TENSORFLOW') == '1', 'Opt-in TensorFlow runtime smoke')
    def test_real_tensorflow_training_step_matches_numpy_loss_and_saves(self):
        settings = self.settings()
        tf = learning._tensorflow()
        tf.keras.utils.set_random_seed(42)
        model = learning.build_model(settings)
        volume = FakeVolume()
        arrays = tuple(volume.read_block((0, 0, 0), (8, 8, 8), [1., 1., 1.])[key]
                       for key in ('image', 'mask', 'valid'))
        x, y, valid = learning.prepare_block(*arrays, settings)
        probability, logits = learning._predict_pair(learning._inference_model(model), x)
        expected = learning.loss_from_totals(learning.loss_totals(y[0, ..., 0], probability, valid[0, ..., 0], logits=logits))
        step = learning._training_step(model, settings)
        actual = float(step(x, y, valid))
        self.assertAlmostEqual(expected, actual, places=5)
        self.assertEqual(tuple(model.output_shape), (None, 8, 8, 8, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.keras'
            model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(learning._predict(model, x), learning._predict(loaded, x))
            # Keras 3 / TF 2.17 may reload model.output as a one-element list.
            # The auxiliary view must not add another leading array dimension.
            reloaded_probability, reloaded_logits = learning._predict_pair(learning._inference_model(loaded), x)
            self.assertEqual(reloaded_probability.shape, (8, 8, 8))
            self.assertEqual(reloaded_logits.shape, (8, 8, 8))
            np.testing.assert_allclose(reloaded_probability, learning._predict(loaded, x))

    # Purpose: Verify opt-in logit loss retains gradients for saturated incorrect predictions.
    # Inputs: Wrong logits of magnitude 40/1000 with two valid and two padded entries.
    # Outputs: Corrective gradients -0.5/+0.5 on valid entries, zero padding gradients, and finite loss.
    @unittest.skipUnless(os.environ.get('FAKECT_TEST_TENSORFLOW') == '1', 'Opt-in TensorFlow runtime smoke')
    def test_saturated_wrong_logits_keep_corrective_gradients_and_ignore_padding(self):
        tf = learning._tensorflow()
        for magnitude in (40., 1000.):
            logits = tf.Variable([-magnitude, magnitude, -magnitude, magnitude], dtype=tf.float32)
            truth, valid = tf.constant([1., 0., 1., 0.]), tf.constant([1., 1., 0., 0.])
            with tf.GradientTape(persistent=True) as tape:
                probability = tf.math.sigmoid(logits)
                clipped = tf.clip_by_value(probability, 1e-6, 1-1e-6)
                old = tf.reduce_sum(-(truth*tf.math.log(clipped)+(1-truth)*tf.math.log1p(-clipped))*valid)/2
                corrected = learning._logit_loss(tf, truth, logits, valid)
            np.testing.assert_array_equal(tape.gradient(old, logits).numpy(), [0., 0., 0., 0.])
            np.testing.assert_allclose(tape.gradient(corrected, logits).numpy(), [-.5, .5, 0., 0.], atol=1e-6)
            self.assertTrue(np.isfinite(float(corrected)))

    # Purpose: Verify opt-in training updates a saturated negative output head.
    # Inputs: Real model with zero head kernel, bias -40, and positive FakeVolume target.
    # Outputs: Finite loss and head bias increases above -40.
    @unittest.skipUnless(os.environ.get('FAKECT_TEST_TENSORFLOW') == '1', 'Opt-in TensorFlow runtime smoke')
    def test_actual_training_step_recovers_gradient_from_saturated_head(self):
        settings = self.settings()
        tf = learning._tensorflow()
        model = learning.build_model(settings)
        head = model.get_layer('aorta_logits')
        head.kernel.assign(tf.zeros_like(head.kernel))
        head.bias.assign([-40.])
        volume = FakeVolume()
        block = volume.read_block((0, 0, 0), (8, 8, 8), [1., 1., 1.])
        arrays = tuple(block[key] for key in ('image', 'mask', 'valid'))
        x, truth, valid = learning.prepare_block(*arrays, settings)
        self.assertLess(float(learning._predict(model, x).max()), 1e-6)
        step = learning._training_step(model, settings)
        loss = float(step(x, truth, valid))
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(float(head.bias.numpy()[0]), -40.)

    # Purpose: Verify opt-in real diagnostic fitting reloads into consistent evaluation.
    # Inputs: One epoch/step with FakeVolume and real TensorFlow model.
    # Outputs: Complete fit/evaluation, no INCOMPLETE marker, TRAIN-only access, and matching diagnostic Dice.
    @unittest.skipUnless(os.environ.get('FAKECT_TEST_TENSORFLOW') == '1', 'Opt-in TensorFlow runtime smoke')
    def test_real_fit_then_reload_and_evaluate_diagnostic_end_to_end(self):
        settings = self.settings(epochs=1, diagnostic_steps_per_epoch=1)
        config, lock = dict(model=settings), dict(samples=self.rows())
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(learning, '_volume', side_effect=lambda row: FakeVolume()):
            directory = Path(tmp)/'model'
            metadata = learning.fit(config, lock, directory, diagnostic=True)
            self.assertFalse((directory/'INCOMPLETE').exists())
            self.assertTrue(metadata['complete'])
            result = learning.evaluate(config, lock, directory, Path(tmp)/'evaluation', diagnostic=True)
            self.assertTrue(result['complete'])
            self.assertEqual(result['loaded_samples'], ['train:sample'])
            self.assertAlmostEqual(result['records'][0]['metrics']['dice'],
                                   metadata['diagnostic_after']['metrics']['dice'], places=6)


if __name__ == '__main__':
    unittest.main()
