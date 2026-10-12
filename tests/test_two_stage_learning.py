"""Data geometry, leakage, change identities and actual two-stage CPU smoke."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_two_stage_learning import (VolumeLoader, binary_metrics, change_masks,
    change_probabilities, inspect_manifest, predict_volume, settings_for,
    tile_origins, training_origins, _batches, _summarize_records)


def fixture(directory):
    directory = Path(directory)
    rows = []
    for family, split in enumerate(('train', 'validation', 'test')):
        before = np.zeros((8, 8, 8), np.uint8)
        before[:, 2:6, 2:6] = 1
        source = directory/f'source-{family}.npz'
        np.savez_compressed(source, target_mask=before)
        for control in (False, True):
            after = before.copy()
            if not control:
                after[2:4, 3:5, 2] = 0
                after[4:6, 3:5, 6] = 1
            variant = directory/f'variant-{family}-{control}.npz'
            np.savez_compressed(variant, image=(after*.2).astype(np.float32), mask=after)
            rows.append(dict(sample_id=f'{family}:{control}', case_id=f'case-{family}', anatomy_family=f'family-{family}',
                             variant_id=str(control), split=split, source_npz=str(source), variant_npz=str(variant),
                             source_npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             variant_npz_sha256=hashlib.sha256(variant.read_bytes()).hexdigest(),
                             geometry={'spacing_ijk_mm': [1., 1., 1.]}, baseline=control))
    config = {'model': dict(patch_size_kji=[8, 8, 8], target_spacing_mm=[1., 1., 1.], base_filters=2, depth=1,
                            epochs=1, batch_size=1, patches_per_volume=2, seed=3)}
    return config, {'samples': rows}


class TwoStageLearningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config, self.manifest = fixture(self.temporary.name)

    # Purpose: Validate patch compatibility with U-Net downsampling depth.
    # Inputs: Tiny two-stage settings with depth 3, then a 12x8x8 patch.
    # Outputs: Divisible settings accepted; incompatible patch dimensions rejected.
    def test_settings_validate_depth_and_patch(self):
        self.config['model']['depth'] = 3
        self.assertEqual(settings_for(self.config, 2)['depth'], 3)
        self.config['model']['patch_size_kji'] = [12, 8, 8]
        with self.assertRaisesRegex(ValueError, 'divisible'):
            settings_for(self.config, 2)

    # Purpose: Prevent family and baseline-reference leakage across model splits.
    # Inputs: Three-family manifest edited to share family identity or a reference path across splits.
    # Outputs: Family-crossing and shared-reference ValueError.
    def test_rejects_family_and_reference_cross_split(self):
        self.manifest['samples'][2]['anatomy_family'] = 'family-0'
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            inspect_manifest(self.manifest)
        self.manifest['samples'][2]['anatomy_family'] = 'family-1'
        self.manifest['samples'][2]['source_npz'] = self.manifest['samples'][0]['source_npz']
        with self.assertRaisesRegex(ValueError, 'Shared reference'):
            inspect_manifest(self.manifest)

    # Purpose: Load checksum-pinned pairs while respecting allowed split access.
    # Inputs: 8-cube baseline/edit pairs and a training-only volume loader.
    # Outputs: Four added/removed voxels, exact reconstruction and rejection of a test-split load.
    def test_loader_supports_frozen_manifest_hashes_and_booleans(self):
        rows, _ = inspect_manifest(self.manifest)
        loader = VolumeLoader(rows, settings_for(self.config, 2), ['train'])
        volume = loader.load(rows[0])
        self.assertEqual(int(volume['added'].sum()), 4)
        self.assertEqual(int(volume['removed'].sum()), 4)
        self.assertTrue(np.array_equal((volume['after'].astype(bool) & ~volume['added']) | volume['removed'], volume['before']))
        with self.assertRaisesRegex(ValueError, 'split is unavailable'):
            loader.load(rows[4])

    # Purpose: Reject native payloads changed after manifest creation.
    # Inputs: Training variant NPZ replaced by corrupt bytes.
    # Outputs: Volume loading raises a hash-change error.
    def test_changed_payload_is_rejected(self):
        rows, _ = inspect_manifest(self.manifest)
        Path(rows[0]['variant_npz']).write_bytes(b'corrupt')
        loader = VolumeLoader(rows, settings_for(self.config, 1), ['train'])
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            loader.load(rows[0])

    # Purpose: Maintain reference reconstruction on a resampled physical grid.
    # Inputs: Native 8-cube pair resampled to 1.5/1/0.5-mm spacing.
    # Outputs: 16x8x6 masks retain the exact before/after change identity.
    def test_resampling_masks_preserves_boolean_identity(self):
        rows, _ = inspect_manifest(self.manifest)
        self.config['model']['target_spacing_mm'] = [1.5, 1., .5]
        settings = settings_for(self.config, 2)
        volume = VolumeLoader(rows, settings, ['train']).load(rows[0])
        self.assertEqual(volume['after'].shape, (16, 8, 6))
        self.assertTrue(np.array_equal((volume['after'].astype(bool) & ~volume['added']) | volume['removed'], volume['before']))

    # Purpose: Derive consistent binary and probabilistic change representations.
    # Inputs: Observed binary mask, reference probabilities and threshold 0.7.
    # Outputs: Added/removed masks reconstruct thresholded reference; probability identity also holds.
    def test_threshold_changes_reconstruct_at_nondefault_threshold(self):
        observed = np.array([0, 1, 0, 1], np.uint8)
        reference = np.array([.7, .65, .69, .1])
        added, removed = change_masks(observed, reference, .7)
        self.assertTrue(np.array_equal(added, [0, 1, 0, 1]))
        self.assertTrue(np.array_equal(removed, [1, 0, 0, 0]))
        self.assertTrue(np.array_equal((observed.astype(bool) & ~added) | removed, reference >= .7))
        add_prob, remove_prob = change_probabilities(observed, reference)
        self.assertTrue(np.allclose(observed-add_prob+remove_prob, reference))

    # Purpose: Include segmentation errors in changes derived from chained model input.
    # Inputs: Normal observed mask, deliberately imperfect predicted mask and the same reference.
    # Outputs: Observed input has no changes; predicted input creates two change voxels.
    def test_chained_changes_use_actual_input(self):
        actual_after = np.array([1, 1, 0], np.uint8)
        predicted_after = np.array([1, 0, 1], np.uint8)
        before = np.array([1, 1, 0], np.float32)
        a, r = change_masks(actual_after, before)
        self.assertEqual(int(a.sum()+r.sum()), 0)
        a, r = change_masks(predicted_after, before)
        self.assertEqual(int(a.sum()+r.sum()), 2)

    # Purpose: Verify tiled inference covers uneven volumes and padded patches.
    # Inputs: Identity predictor on random 3x7x5 and 13x19x11 arrays using 8-cube tiles.
    # Outputs: Full-volume predictions reproduce inputs within numerical tolerance.
    def test_overlap_prediction_covers_awkward_volume_and_padding(self):
        class Identity:
            def predict_on_batch(self, value):
                return value
        for shape in ((3, 7, 5), (13, 19, 11)):
            value = np.random.default_rng(1).random(shape, dtype=np.float32)
            prediction = predict_volume(Identity(), value, [8, 8, 8], 3)
            self.assertTrue(np.allclose(prediction, value, atol=1e-7))

    # Purpose: Separate exhaustive validation coverage from bounded random training patches.
    # Inputs: Two validation 8-cube volumes, 4-cube patches and a fixed sampling seed.
    # Outputs: All validation voxels counted once; two training origins stay within bounds.
    def test_validation_exhaustive_voxels_and_training_bounds(self):
        rows, _ = inspect_manifest(self.manifest)
        settings = settings_for(self.config, 2)
        settings['patch_size_kji'] = [4, 4, 4]
        loader = VolumeLoader(rows, settings, ['train', 'validation'])
        valid_voxels = sum(int(y[..., 1].sum()) for _, y in _batches(loader, rows, settings, 'validation'))
        self.assertEqual(valid_voxels, 2*8**3)
        volume = loader.load(rows[0])
        positions = training_origins(volume, settings, np.random.default_rng(3))
        self.assertTrue(all(all(0 <= n <= 4 for n in p) for p in positions))
        self.assertEqual(len(positions), 2)

    # Purpose: Handle empty targets and physical false-positive burden.
    # Inputs: Empty binary truth with empty or all-positive predictions and 2-mm3 voxels.
    # Outputs: Dice is one for empty agreement, zero for false positives, with 8 mm3 false-positive volume.
    def test_metrics_empty_and_false_positive(self):
        empty = np.zeros(4, bool)
        self.assertEqual(binary_metrics(empty, empty, 2.)['dice'], 1.)
        result = binary_metrics(empty, np.ones(4, bool), 2.)
        self.assertEqual(result['dice'], 0.)
        self.assertEqual(result['false_positive_mm3'], 8.)

    # Purpose: Avoid overweighting cases with more siblings in family summaries.
    # Inputs: Two zero-Dice siblings from case a and one perfect sibling from case b.
    # Outputs: Family-macro Dice is 0.5; sample-macro Dice is one third.
    def test_family_means_give_cases_equal_weight(self):
        records = [dict(case_id='a', anatomy_family='f', metrics={'aorta': {'dice': 0.}}),
                   dict(case_id='a', anatomy_family='f', metrics={'aorta': {'dice': 0.}}),
                   dict(case_id='b', anatomy_family='f', metrics={'aorta': {'dice': 1.}})]
        result = _summarize_records(records)['aorta']
        self.assertEqual(result['family_macro']['dice'], .5)
        self.assertAlmostEqual(result['sample_macro']['dice'], 1/3)


class TwoStageTensorFlowSmokeTests(unittest.TestCase):
    # Purpose: Exercise real CPU fitting, model reload and two-stage heldout evaluation.
    # Inputs: Tiny three-family dataset with test payloads hidden during both fits and restored for evaluation.
    # Outputs: Finite reloaded predictions, no fit leakage, exact change identities and tamper rejection.
    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
    def test_real_fit_reload_and_oracle_chained_test(self):
        code = r'''
import sys
import hashlib
from pathlib import Path
import numpy as np
root, directory = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(root/'src'), str(root/'tests')]
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from test_two_stage_learning import fixture
from fakect_two_stage_learning import fit_stage, load_stage, evaluate_stage, predict_volume, build_unet3d, build_training_step, settings_for
config, manifest = fixture(directory)
# Repeated batches and varying batch counts retain the same TensorSpec graph.
settings = settings_for(config, 2)
probe = build_unet3d(settings)
step = build_training_step(probe, settings)
for n in (1, 2, 1, 3, 1, 2, 1, 2, 1, 2):
    x = np.zeros((n,8,8,8,1), np.float32)
    y = np.zeros((n,8,8,8,3), np.float32)
    y[...,1] = 1
    probe.reset_metrics()
    assert np.isfinite(float(step(x,y)['loss']))
assert step.experimental_get_tracing_count() <= 2
# Make every test payload inaccessible during both fits.
hidden = []
for row in manifest['samples']:
    if row['split'] == 'test':
        for field in ('source_npz', 'variant_npz'):
            path = Path(row[field])
            if path.exists():
                other = path.with_suffix('.hidden')
                path.rename(other)
                hidden.append((path, other))
for stage in (1, 2):
    result = fit_stage(config, manifest, stage, directory/f'model{stage}')
    assert result['epochs_completed'] == 1
    assert result['train_function_traces'] <= 2
    assert result['history'][0]['train_seconds'] > 0
    assert result['history'][0]['validation_seconds'] > 0
    assert not result['test_payloads_loaded']
    assert len(result['loaded_samples']) == 4
    assert all(not sid.startswith('2:') for sid in result['loaded_samples'])
    model, meta = load_stage(directory/f'model{stage}')
    p = predict_volume(model, np.zeros((7, 11, 9), np.float32), [8, 8, 8])
    assert p.shape == (7, 11, 9) and np.isfinite(p).all()
for path, other in hidden:
    other.rename(path)
one = evaluate_stage(config, manifest, 1, directory/'model1', directory/'test1')
two = evaluate_stage(config, manifest, 2, directory/'model2', directory/'test2', directory/'model1')
for folder in ('test1', 'test2'):
    assert (directory/folder/'evaluation.json.sha256').read_text().strip() == hashlib.sha256((directory/folder/'evaluation.json').read_bytes()).hexdigest()
assert one['test_samples'] == two['test_samples'] == 2
assert 'chained_residual_added' in two['summary']
assert 'oracle_reference' in two['controls_summary']
assert two['records'][0]['metrics']['identity_added']['dice'] == 0
assert two['records'][1]['metrics']['identity_added']['dice'] == 1
for row in two['records']:
    with np.load(directory/'test2'/row['prediction_file']) as data:
        observed = data['after'].astype(bool)
        reference = data['oracle_reference_probability'] >= .5
        assert np.array_equal((observed & ~data['oracle_added_mask']) | data['oracle_removed_mask'], reference)
        s = data['stage1_aorta_probability'] >= .5
        r = data['chained_reference_probability'] >= .5
        assert np.array_equal((s & ~data['chained_added_mask']) | data['chained_removed_mask'], r)
metadata_path = directory/'model1'/'model-metadata.json'
original = metadata_path.read_bytes()
metadata_path.write_bytes(original+b' ')
try:
    load_stage(directory/'model1')
    raise AssertionError('Tampered model metadata accepted')
except ValueError as exc:
    assert 'metadata checksum' in str(exc)
metadata_path.write_bytes(original)
history_path = directory/'model1'/'history.json'
history_path.write_text('[]\n')
try:
    load_stage(directory/'model1')
    raise AssertionError('Tampered training history accepted')
except ValueError as exc:
    assert 'history differs' in str(exc)
print('TWO_STAGE_CPU_SMOKE_PASSED', tf.__version__, flush=True)
'''
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3',
                       TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1')
            completed = subprocess.run([sys.executable, '-u', '-c', code, str(ROOT), temporary],
                                       env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(completed.returncode, 0, completed.stdout+'\n'+completed.stderr)
            self.assertIn('TWO_STAGE_CPU_SMOKE_PASSED', completed.stdout)
            print(next(line for line in completed.stdout.splitlines() if 'TWO_STAGE_CPU_SMOKE_PASSED' in line))


if __name__ == '__main__':
    unittest.main()
