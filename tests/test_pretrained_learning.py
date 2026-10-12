"""Complete pretrained heads, partial-label loss and consistent tiled inference."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import fakect_pretrained_learning as learning

try:
    import torch
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
except ImportError:
    torch = None


class NormalizationTests(unittest.TestCase):
    # Purpose: Verify foreground channel resolution uses exact vendor names.
    # Inputs: Nineteen-label ontology, cardiac/aortic groups, and unsupported or absent names.
    # Outputs: Aorta maps to channel 2, supported group maps exactly, and invalid names raise ValueError.
    def test_resolve_group_uses_exact_vendor_names(self):
        labels = {f'label_{i}': i for i in range(19)}
        for channel, name in zip((2, 4, 5, 6, 7, 8), learning.CARDIAC_AORTIC_NAMES):
            del labels[f'label_{channel}']; labels[name] = channel
        self.assertEqual(learning.resolve_foreground_channels(labels), (2,))
        self.assertEqual(learning.resolve_foreground_channels(labels, learning.CARDIAC_AORTIC_NAMES),
                         (2, 4, 5, 6, 7, 8))
        self.assertEqual(learning.resolve_foreground_channels(labels, 'aorta'), (2,))
        for names in [(), ('aorta', 'aorta'), ('aorta', 'coronary_arteries'), ('aorta', 'pulmonary_vein'),
                      ('aorta', 'subclavian_left')]:
            with self.assertRaises(ValueError):
                learning.resolve_foreground_channels(labels, names)
        with self.assertRaisesRegex(ValueError, 'absent'):
            learning.resolve_foreground_channels({f'label_{i}': i for i in range(19) if i != 2} | {'aorta':2},
                                                 ('aorta', 'brachiocephalic_trunk'))

    # Purpose: Require valid foreground channels while preserving source ontology.
    # Inputs: Empty/duplicate/background/out-of-range/noninteger channels and unsorted valid channels.
    # Outputs: Invalid groups fail; valid channels sort and default extraction remains vendor aorta channel 2.
    def test_foreground_rejects_invalid_channels_and_keeps_source_ontology(self):
        for channels in [(), (2, 2), (0, 2), (19,), (-1,), (True, 2), (2.0,), None, '2']:
            with self.assertRaises(ValueError):
                learning.foreground_metadata(channels)
        self.assertEqual(learning.foreground_metadata((8, 2, 4))['foreground_channels'], [2, 4, 8])
        self.assertEqual(learning.foreground_metadata()['extraction'],
                         '19-way argmax == local aorta channel 2')

    # Purpose: Verify CT normalization follows frozen int32 conversion and scaling.
    # Inputs: HU values spanning clipping bounds and fractional values.
    # Outputs: Result exactly matches clipped int32 normalization as float32 and preserves the input.
    def test_int32_after_resampling_then_original_ct_scaling(self):
        image = np.array([-1000., -101.9, -100.9, 0., 142.9, 813.9, 2000.], dtype=np.float32)
        original = image.copy()
        result = learning.normalize_hu(image)
        expected = (np.clip(image.astype(np.int32), -101, 813).astype(np.float32)
                    - np.float32(142.1977119916391)) / np.float32(154.74032651622053)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(result.dtype, np.float32)

    # Purpose: Reject altered normalization properties and invalid HU values.
    # Inputs: Changed mean, NaN, infinity, and value 1e12.
    # Outputs: Normalization raises ValueError for frozen properties or invalid numeric range.
    def test_reject_changed_normalization_nonfinite_and_overflow(self):
        changed = dict(learning.CT_PROPERTIES, mean=0)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            learning.normalize_hu(np.zeros(1), changed)
        for value in [np.nan, np.inf, 1e12]:
            with self.assertRaises(ValueError):
                learning.normalize_hu(np.array([value]))


@unittest.skipIf(torch is None, 'Requires isolated PyTorch/nnU-Net environment')
class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    # Purpose: Verify binary aorta logits preserve original softmax probability.
    # Inputs: Random 19-channel logits and a highest aorta logit below majority probability.
    # Outputs: Sigmoid equals channel-2 softmax; vendor argmax and binary 0.5 threshold can differ.
    def test_marginal_probability_exactly_preserves_original_aorta_head(self):
        z = torch.randn(2, 19, 4, 3, 5)
        q = learning.aorta_logit(z)
        torch.testing.assert_close(q.sigmoid(), z.softmax(1)[:, 2:3], atol=1e-7, rtol=1e-6)
        # Original aorta argmax can be true while binary probability is < 0.5.
        equal = torch.zeros(1, 19, 1, 1, 1); equal[:, 2] = 1
        self.assertEqual(int(equal.argmax(1)), 2)
        self.assertLess(float(learning.aorta_logit(equal).sigmoid()), .5)

    # Purpose: Verify grouped logits sum selected foreground probabilities.
    # Inputs: Six cardiac/aortic channels, default singleton, and selected-channel permutation.
    # Outputs: Grouped sigmoid equals softmax sum; default and explicit channel 2 match exactly.
    def test_grouped_logit_equals_summed_foreground_softmax_and_default_is_identical(self):
        z = torch.randn(2, 19, 4, 3, 5)
        channels = (2, 4, 5, 6, 7, 8)
        grouped = learning.aorta_logit(z, foreground_channels=channels)
        torch.testing.assert_close(grouped.sigmoid(), z.softmax(1)[:, channels].sum(1, keepdim=True),
                                   atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(learning.aorta_logit(z), learning.aorta_logit(z, foreground_channels=(2,)),
                                   atol=0, rtol=0)
        permutation = list(range(19)); permutation[2], permutation[8] = 8, 2
        torch.testing.assert_close(learning.aorta_logit(z[:, permutation], foreground_channels=channels),
                                   grouped, atol=1e-6, rtol=1e-6)

    # Purpose: Verify grouped loss trains all foreground heads and ignores excluded voxels.
    # Inputs: Zero 19-channel logits, positive truth, and one invalid voxel.
    # Outputs: Selected heads receive negative gradients, background positive gradients, and ignored cell none.
    def test_grouped_loss_trains_all_selected_heads_and_preserves_ignore_mask(self):
        z = torch.zeros(1, 19, 2, 2, 2, requires_grad=True)
        truth = torch.ones(2, 2, 2); valid = torch.ones_like(truth); valid[0, 0, 0] = 0
        channels = (2, 4, 5, 6, 7, 8)
        result = learning.aorta_loss(z, truth, valid, foreground_channels=channels)
        result['loss'].backward()
        self.assertEqual(int(torch.count_nonzero(z.grad[:, :, 0, 0, 0])), 0)
        self.assertTrue(torch.all(z.grad[0, channels, 1, 1, 1] < 0))
        self.assertGreater(float(z.grad[0, 3, 1, 1, 1]), 0)  # Pulmonary vein remains complement.

    # Purpose: Verify ignored voxels and nonforeground permutations leave loss invariant.
    # Inputs: Random logits with invalid last slice, huge ignored values, and swapped nonforeground classes.
    # Outputs: Ignored gradients are zero and all loss values match within tolerance.
    def test_ignored_voxels_have_no_loss_or_gradient_and_other_classes_are_marginalized(self):
        z = torch.randn(1, 19, 4, 4, 4, requires_grad=True)
        truth = torch.zeros(1, 1, 4, 4, 4); truth[:, :, :2] = 1
        valid = torch.ones_like(truth); valid[:, :, :, :, -1] = 0
        truth[:, :, :, :, -1] = 19  # Unknown annotation, not a background class.
        first = learning.aorta_loss(z, truth, valid)
        first['loss'].backward()
        self.assertEqual(int(torch.count_nonzero(z.grad[:, :, :, :, -1])), 0)
        shifted = z.detach().clone(); shifted[:, :, :, :, -1] += 1e4
        second = learning.aorta_loss(shifted, truth, valid)
        torch.testing.assert_close(first['loss'], second['loss'], atol=0, rtol=0)
        # Permuting other cardiac channels cannot change binary supervision.
        permutation = list(range(19)); permutation[0], permutation[7] = 7, 0
        third = learning.aorta_loss(z.detach()[:, permutation], truth, valid)
        torch.testing.assert_close(first['loss'], third['loss'], atol=1e-6, rtol=1e-6)

    # Purpose: Verify coarse supervision excludes mixed annotation support.
    # Inputs: Full/coarse logits with one invalid fine voxel and equal level weights.
    # Outputs: Both levels train but the corresponding coarse cell has zero gradient.
    def test_coarse_supervision_ignores_entire_mixed_annotation_support(self):
        full = torch.randn(1, 19, 4, 4, 4, requires_grad=True)
        coarse = torch.randn(1, 19, 2, 2, 2, requires_grad=True)
        truth = torch.ones(1, 1, 4, 4, 4)
        valid = torch.ones_like(truth); valid[:, :, 0, 0, 0] = 0
        result = learning.aorta_loss([full, coarse], truth, valid, [1., 1.])
        self.assertEqual(result['active_levels'], 2)
        result['loss'].backward()
        self.assertEqual(int(torch.count_nonzero(coarse.grad[:, :, 0, 0, 0])), 0)
        self.assertGreater(int(torch.count_nonzero(coarse.grad[:, :, 1, 1, 1])), 0)

    # Purpose: Verify default deep supervision excludes the lowest-resolution head.
    # Inputs: Full and coarse logits with fully valid positive truth.
    # Outputs: Only one level is active and the coarse head has no gradient.
    def test_default_supervision_does_not_train_lowest_resolution_head(self):
        full = torch.randn(1, 19, 4, 4, 4, requires_grad=True)
        coarse = torch.randn(1, 19, 2, 2, 2, requires_grad=True)
        result = learning.aorta_loss([full, coarse], torch.ones(4, 4, 4), torch.ones(4, 4, 4))
        result['loss'].backward()
        self.assertEqual(result['active_levels'], 1)
        self.assertIsNone(coarse.grad)

    # Purpose: Reject invalid loss supervision contracts.
    # Inputs: No annotated voxels, nonbinary truth, or negative level weights.
    # Outputs: Loss raises ValueError for annotations, binary labels, or weights.
    def test_invalid_supervision_is_rejected(self):
        z = torch.zeros(1, 19, 4, 4, 4)
        with self.assertRaisesRegex(ValueError, 'annotated'):
            learning.aorta_loss(z, torch.zeros(4, 4, 4), torch.zeros(4, 4, 4))
        with self.assertRaisesRegex(ValueError, 'binary'):
            learning.aorta_loss(z, torch.full((4, 4, 4), 2.), torch.ones(4, 4, 4))
        with self.assertRaisesRegex(ValueError, 'weights'):
            learning.aorta_loss(z, torch.zeros(4, 4, 4), torch.ones(4, 4, 4), [-1.])


def small_official_checkpoint(path):
    """Small official PlainConvUNet fixture with the same 19-label head contract."""
    architecture = {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
        'arch_kwargs': {'n_stages': 3, 'features_per_stage': [2, 4, 8],
            'conv_op': 'torch.nn.modules.conv.Conv3d', 'kernel_sizes': [[3, 3, 3]] * 3,
            'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]], 'n_conv_per_stage': [2, 2, 2],
            'n_conv_per_stage_decoder': [2, 2], 'conv_bias': True,
            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True}, 'dropout_op': None,
            'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU',
            'nonlin_kwargs': {'inplace': True}},
        '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}
    plans = {'dataset_name': 'Fixture', 'plans_name': 'fixture',
        'configurations': {'3d_fullres': {'architecture': architecture, 'spacing': [1.5] * 3,
            'patch_size': [8] * 3, 'normalization_schemes': ['CTNormalization']}},
        'foreground_intensity_properties_per_channel': {'0': dict(learning.CT_PROPERTIES)}}
    labels = {f'label_{i}': i for i in range(19)}
    for channel, name in zip((2, 4, 5, 6, 7, 8), learning.CARDIAC_AORTIC_NAMES):
        del labels[f'label_{channel}']; labels[name] = channel
    model = get_network_from_plans(architecture['network_class_name'], architecture['arch_kwargs'],
        architecture['_kw_requires_import'], 1, 19, allow_init=True, deep_supervision=True)
    # Give output heads known nonzero weights that a skip-head loader would lose.
    with torch.no_grad():
        for index, head in enumerate(model.decoder.seg_layers):
            head.bias.copy_(torch.arange(19) * .03 + index * .07)
    torch.save({'network_weights': model.state_dict(), 'optimizer_state': {'old': 'must not restore'},
        'current_epoch': 1001, 'init_args': {'plans': plans, 'configuration': '3d_fullres',
                                          'dataset_json': {'labels': labels}}}, path)
    return model


@unittest.skipIf(torch is None, 'Requires isolated PyTorch/nnU-Net environment')
class PretrainedModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / 'original.pth'
        self.reference = small_official_checkpoint(self.original)

    # Purpose: Verify official checkpoint loading preserves heads and resets optimizer state.
    # Inputs: Small official PlainConvUNet checkpoint with 19 labels and old optimizer metadata.
    # Outputs: All weights/biases match, optimizer is fresh at 0.0001, and metadata pins source checkpoint.
    def test_official_graph_load_preserves_all_output_heads_and_optimizer_is_fresh(self):
        loaded = learning.build_model(self.original)
        self.assertEqual(learning.weights_sha256(loaded), learning.weights_sha256(self.reference))
        for a, b in zip(loaded.decoder.seg_layers, self.reference.decoder.seg_layers):
            torch.testing.assert_close(a.weight, b.weight, atol=0, rtol=0)
            torch.testing.assert_close(a.bias, b.bias, atol=0, rtol=0)
        optimizer = learning.make_optimizer(loaded)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        metadata = learning.model_metadata(loaded)
        self.assertEqual(metadata['output_channels'], 19)
        self.assertEqual(metadata['aorta_channel'], 2)
        self.assertEqual(len(metadata['head_shapes']), 4)
        self.assertEqual(metadata['loaded_checkpoint_sha256'], learning.file_sha256(self.original))

    # Purpose: Reject incomplete checkpoint weights or unsupported source spacing.
    # Inputs: Missing decoder head weight or spacing changed from 1.5 to 3 mm.
    # Outputs: Loading raises RuntimeError for missing key or ValueError for spacing.
    def test_missing_head_wrong_label_or_changed_spacing_cannot_load(self):
        saved = torch.load(self.original, weights_only=False)
        missing = copy.deepcopy(saved)
        del missing['network_weights']['decoder.seg_layers.0.weight']
        path = self.root / 'missing.pth'; torch.save(missing, path)
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            learning.build_model(path)
        changed = copy.deepcopy(saved)
        changed['init_args']['plans']['configurations']['3d_fullres']['spacing'] = [3.] * 3
        path = self.root / 'changed.pth'; torch.save(changed, path)
        with self.assertRaisesRegex(ValueError, '1.5-mm'):
            learning.build_model(path)

    # Purpose: Verify real gradient updates and immutable checkpoint round-trips.
    # Inputs: Official small model, one 8-cube training step, saved and restored weights.
    # Outputs: Loss/gradient are finite, weights change, reload predictions match, and overwrite is rejected.
    def test_real_gradient_step_saved_reload_and_immutable_checkpoint(self):
        model = learning.build_model(self.original)
        initial = learning.weights_sha256(model)
        optimizer = learning.make_optimizer(model)
        image = np.random.default_rng(42).normal(size=(8, 8, 8)).astype(np.float32)
        truth = np.zeros_like(image); truth[2:6, 2:6, 2:6] = 1
        receipt = learning.train_step(model, optimizer, image, truth, np.ones_like(image))
        self.assertTrue(np.isfinite(receipt['loss']))
        self.assertGreater(receipt['gradient_norm'], 0)
        self.assertNotEqual(initial, learning.weights_sha256(model))
        first = learning.predict_tiled(model, image, patch_size=(8, 8, 8), return_probabilities=True)
        archived = learning.save_checkpoint(self.root / 'trained.pth', model, {'epoch': 1})
        restored = learning.build_model(archived['path'])
        self.assertEqual(learning.weights_sha256(model), learning.weights_sha256(restored))
        second = learning.predict_tiled(restored, image, patch_size=(8, 8, 8), return_probabilities=True)
        np.testing.assert_array_equal(first['mask'], second['mask'])
        np.testing.assert_array_equal(first['probability'], second['probability'])
        self.assertEqual(len(learning.make_optimizer(restored).state), 0)
        reinitialized = learning.build_model(self.original)
        receipt = learning.load_weights(reinitialized, archived['path'])
        self.assertEqual(receipt['weights_sha256'], learning.weights_sha256(model))
        self.assertEqual(learning.model_metadata(reinitialized)['loaded_checkpoint_sha256'], archived['sha256'])
        with self.assertRaises(FileExistsError):
            learning.save_checkpoint(self.root / 'trained.pth', model)

    # Purpose: Verify grouped training preserves vendor labels in saved archives.
    # Inputs: Six-channel foreground group, one gradient step, and grouped checkpoint reload.
    # Outputs: Every selected head changes; source ontology and target definition persist with exact prediction parity.
    def test_grouped_gradient_archive_preserves_vendor_labels_and_reload_predictions(self):
        model = learning.build_model(self.original)
        channels = learning.resolve_foreground_channels(model._fakect_pretrained_metadata['dataset_json']['labels'],
                                                         learning.CARDIAC_AORTIC_NAMES)
        optimizer = learning.make_optimizer(model)
        image = np.random.default_rng(7).normal(size=(8, 8, 8)).astype(np.float32)
        truth = np.zeros_like(image); truth[2:6, 2:6, 2:6] = 1
        # Decoder heads are stored coarse-to-fine; supervision outputs are the
        # reverse order, so the final stored head is the active full-grid head.
        initial = model.decoder.seg_layers[-1].bias.detach().clone()
        step = learning.train_step(model, optimizer, image, truth, np.ones_like(image),
                                   foreground_channels=channels)
        self.assertGreater(step['gradient_norm'], 0)
        self.assertTrue(torch.all(model.decoder.seg_layers[-1].bias.detach()[list(channels)] != initial[list(channels)]))
        original_labels = dict(model._fakect_pretrained_metadata['dataset_json']['labels'])
        metadata = learning.model_metadata(model, foreground_channels=channels)
        self.assertEqual(metadata['source_aorta_channel'], 2)
        self.assertEqual(metadata['foreground_channels'], list(channels))
        self.assertEqual(metadata['source_labels'], original_labels)
        saved = learning.save_checkpoint(self.root / 'grouped.pth', model, foreground_channels=channels)
        archive = torch.load(saved['path'], weights_only=False)
        self.assertEqual(archive['init_args']['dataset_json']['labels'], original_labels)
        self.assertEqual(archive['metadata']['target_definition']['foreground_channels'], list(channels))
        restored = learning.build_model(saved['path'])
        a = learning.predict_tiled(model, image, patch_size=(8, 8, 8), return_probabilities=True,
                                   foreground_channels=channels)
        b = learning.predict_tiled(restored, image, patch_size=(8, 8, 8), return_probabilities=True,
                                   foreground_channels=channels)
        np.testing.assert_array_equal(a['mask'], b['mask'])
        np.testing.assert_array_equal(a['probability'], b['probability'])

    # Purpose: Verify grouped production smoke propagates foreground semantics through reload.
    # Inputs: Small CPU checkpoint, six foreground channels, and two observed inference calls.
    # Outputs: Smoke passes, both calls use the same channels, and receipt records the target group.
    def test_grouped_production_smoke_propagates_target_through_reload(self):
        channels = (2, 4, 5, 6, 7, 8)
        image = np.random.default_rng(8).normal(size=(8, 8, 8)).astype(np.float32)
        truth = np.zeros_like(image); truth[2:6, 2:6, 2:6] = 1
        original_predict = learning.predict_tiled
        def small_predict(*args, **kwargs):
            self.assertEqual(kwargs['foreground_channels'], channels)
            return original_predict(*args, patch_size=(8, 8, 8), **kwargs)
        with mock.patch.object(learning, 'PATCH_SIZE', (8, 8, 8)), \
                mock.patch.object(learning, 'predict_tiled', side_effect=small_predict) as inference:
            receipt = learning.production_smoke(self.original, image, truth, np.ones_like(image),
                self.root / 'smoke', device='cpu', steps=1, foreground_channels=channels)
        self.assertTrue(receipt['passed'])
        self.assertEqual(inference.call_count, 2)
        self.assertEqual(receipt['target_definition']['foreground_channels'], list(channels))


@unittest.skipIf(torch is None, 'Requires isolated PyTorch/nnU-Net environment')
class TilingTests(unittest.TestCase):
    # Purpose: Verify tiled grouping follows every selected vendor argmax winner.
    # Inputs: Analytic 19-head model and images cycling through all label values.
    # Outputs: Mask matches selected channels, excludes other vessels, and probabilities equal grouped softmax.
    def test_grouped_tiling_includes_every_foreground_winner_and_excludes_other_vessels(self):
        torch.set_num_threads(2)
        model = torch.nn.Conv3d(1, 19, 1)
        with torch.no_grad():
            centers = torch.arange(19, dtype=torch.float32)
            model.weight[:, 0, 0, 0, 0] = 2 * centers
            model.bias.copy_(-centers.square())
        image = (np.arange(8**3) % 19).astype(np.float32).reshape(8, 8, 8)
        channels = (2, 4, 5, 6, 7, 8)
        actual = learning.predict_tiled(model, image, patch_size=(8, 8, 8), return_probabilities=True,
                                        foreground_channels=channels)
        np.testing.assert_array_equal(actual['mask'], np.isin(image, channels))
        self.assertFalse(actual['mask'][image == 3].any())  # Pulmonary vein.
        self.assertFalse(actual['mask'][image == 9].any())  # Brachiocephalic vein.
        logits = (2 * np.arange(19)[:, None, None, None] * image
                  - np.arange(19)[:, None, None, None]**2)
        exponential = np.exp(logits - logits.max(axis=0, keepdims=True))
        np.testing.assert_allclose(actual['probability'], exponential[list(channels)].sum(0) / exponential.sum(0),
                                   atol=3e-7, rtol=1e-5)
        default = learning.predict_tiled(model, image, patch_size=(8, 8, 8), return_probabilities=True)
        explicit = learning.predict_tiled(model, image, patch_size=(8, 8, 8), return_probabilities=True,
                                          foreground_channels=(2,))
        np.testing.assert_array_equal(default['mask'], image == 2)
        np.testing.assert_array_equal(default['mask'], explicit['mask'])
        np.testing.assert_array_equal(default['probability'], explicit['probability'])
        self.assertEqual(actual['metadata']['foreground_channels'], list(channels))

    # Purpose: Verify overlapping tiled inference supports padding and disk accumulators.
    # Inputs: 5x7x13 image, 8-cube patches, analytic model, and tiny accumulator memory limit.
    # Outputs: Mask/probabilities match argmax; disk-backed tiles clean up temporary files and restore model mode.
    def test_overlap_padding_disk_accumulation_and_original_argmax(self):
        torch.set_num_threads(2)
        model = torch.nn.Conv3d(1, 19, 1)
        with torch.no_grad():
            model.weight.zero_(); model.bias.zero_()
            model.weight[2] = 1.
        image = np.linspace(-1, 1, 5 * 7 * 13, dtype=np.float32).reshape(5, 7, 13)
        model.train()
        with tempfile.TemporaryDirectory() as temporary:
            actual = learning.predict_tiled(model, image, patch_size=(8, 8, 8),
                return_probabilities=True, accumulator_memory_mb=.001, temporary_directory=temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        self.assertTrue(model.training)
        np.testing.assert_array_equal(actual['mask'], image > 0)
        np.testing.assert_allclose(actual['probability'], np.exp(image) / (18 + np.exp(image)), atol=2e-7)
        self.assertTrue(actual['metadata']['accumulator_disk_backed'])
        self.assertGreater(actual['metadata']['tiles'], 1)

    # Purpose: Reject inference models with an incompatible output head.
    # Inputs: Two-channel Conv3d model on an 8-cube image.
    # Outputs: Tiled prediction raises ValueError requiring 19 channels.
    def test_inference_rejects_mismatched_head_count(self):
        model = torch.nn.Conv3d(1, 2, 1)
        with self.assertRaisesRegex(ValueError, '19-channel'):
            learning.predict_tiled(model, np.zeros((8, 8, 8), np.float32), patch_size=(8, 8, 8))


if __name__ == '__main__':
    unittest.main()
