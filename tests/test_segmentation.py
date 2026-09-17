"""Paired patches, split integrity, fixed inference preprocessing and CPU smoke."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_segmentation import (PatchDataset, inspect_manifest, model_settings,
                                 normalize_image, predict_axial_slice, validate_clip_bounds)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def make_manifest(directory, shape=(3, 5, 7)):
    directory = Path(directory)
    (directory / 'variants').mkdir(parents=True)
    samples = []
    images, masks = {}, {}
    for index, split in enumerate(('train', 'train', 'validation', 'test')):
        variant = f'variant-{index}'
        mask = np.zeros(shape, np.uint8)
        if index != 1:  # Keep a completely empty foreground training variant.
            mask.ravel()[index:index+3] = 1
        image = (.1 + .4 * mask).astype(np.float32)
        image.ravel()[-1] = -.1
        path = directory / f'variants/{variant}.npz'
        np.savez_compressed(path, image=image, mask=mask)
        samples.append({'variant_id': variant, 'split': split, 'path': f'variants/{variant}.npz',
                        'sha256': digest(path), 'image_sha256': array_digest(image),
                        'mask_sha256': array_digest(mask), 'geometry_group': f'geometry-{index}'})
        images[variant], masks[variant] = image, mask
    manifest = {'schema_version': 'fakect.training-dataset/1', 'split_mode': 'scenario_only',
                'anatomy_family': 'synthetic-fixture-family', 'case_id': 'unit-fixture', 'frame': 1,
                'image_method': 'attenuation_copy_proxy', 'image_units': 'cm^-1',
                'geometry': {'array_order': 'kji', 'crop_shape_kji': list(shape),
                             'crop_origin_ijk': [0, 0, 0], 'spacing_ijk_mm': [1, 1, 2]},
                'samples': samples}
    path = directory / 'dataset-manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest, images, masks


class SegmentationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path, self.manifest, self.images, self.masks = make_manifest(self.directory)
        self.settings = model_settings({'model': {'patch_size': [4, 4], 'batch_size': 3, 'seed': 17}})

    def rewrite(self, manifest):
        self.path.write_text(json.dumps(manifest))

    def test_metadata_index_keeps_every_voxel_and_empty_foreground(self):
        dataset = PatchDataset(self.path, self.settings)
        self.assertEqual(dataset.loaded_variants, set())
        self.assertEqual(len(dataset.patches('train')), 24)
        self.assertEqual(len(dataset.patches('validation')), 12)
        totals = {'variant-0': 0, 'variant-1': 0}
        foreground = {'variant-0': 0, 'variant-1': 0}
        for record in dataset.patches('train'):
            images, masks, valid = dataset.batch([record])
            self.assertEqual(images.shape, (1, 4, 4, 1))
            self.assertEqual(images.dtype, np.float32)
            self.assertTrue(np.all((masks == 0) | (masks == 1)))
            self.assertTrue(np.all(images[valid == 0] == 0))
            self.assertTrue(np.all(masks[valid == 0] == 0))
            variant = dataset.manifest['samples'][record[0]]['variant_id']
            totals[variant] += int(valid.sum())
            foreground[variant] += int(masks.sum())
            source = self.images[variant]
            _, k, j, i = record
            h, w = min(4, source.shape[1]-j), min(4, source.shape[2]-i)
            np.testing.assert_array_equal(images[0, :h, :w, 0], normalize_image(source[k, j:j+h, i:i+w], self.settings))
            np.testing.assert_array_equal(masks[0, :h, :w, 0], self.masks[variant][k, j:j+h, i:i+w])
        self.assertEqual(totals, {'variant-0': 105, 'variant-1': 105})
        self.assertEqual(foreground, {'variant-0': 3, 'variant-1': 0})

    def test_fixed_clip_is_shared_by_tiled_whole_slice_inference(self):
        class IdentityPredictor:
            def predict_on_batch(self, batch):
                return batch.copy()
        image = np.linspace(-.2, .8, 35, dtype=np.float32).reshape(5, 7)
        predicted = predict_axial_slice(IdentityPredictor(), image, self.settings, batch_size=3)
        np.testing.assert_array_equal(predicted, normalize_image(image, self.settings))
        np.testing.assert_array_equal(image, np.linspace(-.2, .8, 35, dtype=np.float32).reshape(5, 7))
        self.assertEqual(predicted.shape, image.shape)
        self.assertEqual(predicted.min(), 0)
        self.assertEqual(predicted.max(), 1)

    def test_deterministic_epoch_order_and_single_volume_cache(self):
        dataset = PatchDataset(self.path, self.settings)
        self.assertEqual(dataset.ordered_patches('train', 2), dataset.ordered_patches('train', 2))
        self.assertNotEqual(dataset.ordered_patches('train', 2), dataset.ordered_patches('train', 3))
        self.assertCountEqual(dataset.ordered_patches('train', 2), dataset.patches('train'))
        ids = [record[0] for record in dataset.ordered_patches('train', 2)]
        self.assertEqual(sum(a != b for a, b in zip(ids, ids[1:])), 1)
        with patch('fakect_segmentation.np.load', wraps=np.load) as load:
            list(dataset.batches('train', 2))
            self.assertEqual(load.call_count, 2)
        self.assertIsNotNone(dataset._cache)
        self.assertEqual(len(dataset._cache), 2)  # One image + its paired mask.

    def test_test_payload_is_not_opened_even_when_absent(self):
        (self.directory / self.manifest['samples'][-1]['path']).unlink()
        dataset = PatchDataset(self.path, self.settings)
        list(dataset.batches('train'))
        list(dataset.batches('validation'))
        self.assertNotIn('variant-3', dataset.loaded_variants)
        with self.assertRaisesRegex(ValueError, 'held untouched'):
            dataset.patches('test')
        with self.assertRaisesRegex(ValueError, 'Test payloads'):
            dataset.batch([(3, 0, 0, 0)])

    def test_duplicate_geometry_or_mask_cannot_cross_splits(self):
        for key in ('geometry_group', 'mask_sha256'):
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.manifest)
                manifest['samples'][2][key] = manifest['samples'][0][key]
                self.rewrite(manifest)
                with self.assertRaisesRegex(ValueError, 'multiple splits'):
                    inspect_manifest(self.path)

    def test_split_scope_and_paths_cannot_make_false_family_holdout_claim(self):
        for mutate, message in (
            (lambda m: m.update(split_mode='patient_holdout'), 'scenario_only'),
            (lambda m: m['samples'][2].update(anatomy_family='another-family'), 'shared anatomy'),
            (lambda m: m['samples'][0].update(path='../elsewhere.npz'), 'leaves'),
            (lambda m: m['samples'][0].update(path='/tmp/elsewhere.npz'), 'relative'),
            (lambda m: m['samples'][2].update(split='test'), 'nonempty train and validation'),
        ):
            with self.subTest(message=message):
                manifest = copy.deepcopy(self.manifest)
                mutate(manifest)
                self.rewrite(manifest)
                with self.assertRaisesRegex(ValueError, message):
                    inspect_manifest(self.path)

    def test_changed_file_and_nonbinary_mask_are_rejected(self):
        sample = self.manifest['samples'][0]
        path = self.directory / sample['path']
        with path.open('ab') as handle:
            handle.write(b'changed')
        dataset = PatchDataset(self.path, self.settings)
        with self.assertRaisesRegex(ValueError, 'file hash changed'):
            dataset.batch(dataset.patches('train')[:1])
        mask = self.masks['variant-0'].copy()
        mask.flat[0] = 2
        np.savez_compressed(path, image=self.images['variant-0'], mask=mask)
        sample['sha256'], sample['mask_sha256'] = digest(path), array_digest(mask)
        self.rewrite(self.manifest)
        dataset = PatchDataset(self.path, self.settings)
        with self.assertRaisesRegex(ValueError, 'uint8'):
            dataset.batch(dataset.patches('train')[:1])

    def test_model_settings_and_prediction_validation(self):
        for settings in ({'patch_size': [6, 8]}, {'slice_axis': 'i'}, {'clip_min': .5, 'clip_max': .5},
                         {'normalization': 'per_patient'}, {'seed': -1}, {'epochs': 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                model_settings({'model': settings})
        with self.assertRaisesRegex(ValueError, 'finite'):
            normalize_image(np.asarray([np.nan]), self.settings)
        class BrokenPredictor:
            def predict_on_batch(self, batch):
                return batch + 2
        with self.assertRaisesRegex(ValueError, 'invalid sigmoid'):
            predict_axial_slice(BrokenPredictor(), np.zeros((5, 7)), self.settings)

    def test_float32_clip_bounds_reject_overflow_underflow_and_rounded_equality(self):
        for lower, upper in ((1e308, 1.1e308), (-1e308, 1e308), (0, 1e-300),
                             (1., 1.00000001), (-3e38, 3e38)):
            with self.subTest(lower=lower, upper=upper):
                for action in (
                    lambda: validate_clip_bounds(lower, upper),
                    lambda: model_settings({'model': {'clip_min': lower, 'clip_max': upper}}),
                    lambda: normalize_image(np.asarray([0., .1, .2], np.float32),
                                             {'clip_min': lower, 'clip_max': upper}),
                ):
                    with self.assertRaisesRegex(ValueError, 'float32'):
                        action()

    def test_fixed_clip_saturates_large_inputs_without_nonfinite_intermediates(self):
        values = np.asarray([-1e308, 0., .25, .5, 1e308], np.float64)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            normalized = normalize_image(values, self.settings)
        np.testing.assert_array_equal(normalized, np.asarray([0, 0, .5, 1, 1], np.float32))
        self.assertFalse(emitted)
        # Small but representable intervals remain useful and finite.
        small = model_settings({'model': {'clip_min': 0, 'clip_max': 1e-30}})
        np.testing.assert_allclose(normalize_image(np.asarray([0, .5e-30, 1e-30]), small), [0, .5, 1])


class TensorFlowSmokeTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is absent; preprocessing tests still run')
    def test_tiny_cpu_fit_reload_and_tiled_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path, manifest, _, _ = make_manifest(directory / 'dataset', shape=(1, 16, 16))
            # A missing test payload proves the training path does not read it.
            (path.parent / manifest['samples'][-1]['path']).unlink()
            code = '''
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from fakect_segmentation import train_segmentation,load_segmentation,predict_axial_slice
config={'model':{'patch_size':[16,16],'epochs':1,'batch_size':1,'seed':8},'train':{'model_directory':sys.argv[3]}}
metadata=train_segmentation(config,sys.argv[2])
model,loaded=load_segmentation(sys.argv[3])
prediction=predict_axial_slice(model,np.full((19,23),.2,np.float32),loaded['settings'])
assert prediction.shape==(19,23) and np.isfinite(prediction).all()
assert not metadata['test_payloads_loaded'] and not metadata['test_evaluated']
assert metadata['epochs_completed']==1 and metadata['best_epoch']==1
assert metadata['history'][0]['validation']['valid_voxels']==256
assert metadata['loaded_variants']==['variant-0','variant-1','variant-2']
assert loaded['model_sha256']==metadata['model_sha256']
# Ensure padded pixels are ignored by BOTH components of the loss.
from fakect_segmentation import _loss_function
truth=np.zeros((1,4,4,2),np.float32);truth[:,0,0,1]=1
pred1=np.full((1,4,4,1),.1,np.float32);pred2=pred1.copy();pred2[:,1:,:,:]=.9
np.testing.assert_allclose(_loss_function(tf)(truth,pred1),_loss_function(tf)(truth,pred2))
print('CPU_SMOKE_PASSED',tf.__version__)
'''
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3',
                       TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1',
                       MPLCONFIGDIR=str(directory / 'matplotlib'))
            # TensorFlow import/graph initialization can be slow on the shared
            # cluster filesystem, independently of this three-patch fixture.
            try:
                result = subprocess.run([sys.executable, '-u', '-c', code, str(ROOT / 'src'), str(path), str(directory / 'model')],
                                        env=env, capture_output=True, text=True, timeout=360)
            except subprocess.TimeoutExpired as exc:
                self.fail(f'CPU smoke timed out during TensorFlow initialization/fit: {exc.stdout!r}\n{exc.stderr!r}')
            self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
            self.assertIn('CPU_SMOKE_PASSED', result.stdout)


if __name__ == '__main__':
    unittest.main()
