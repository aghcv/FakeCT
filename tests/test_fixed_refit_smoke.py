"""Actual CPU fit/reload for a frozen final refit without validation data."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FixedRefitTensorFlowSmokeTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is absent; fixed-refit CPU smoke skipped')
    def test_frozen_fixed_epoch_fit_ignores_outer_test_and_reloads(self):
        code = r'''
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'tests'))
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from test_model_experiment import ModelExperimentTests
from fakect_model_experiment import freeze_model_experiment, inspect_experiment_lock, plan_model_experiment
from fakect_model_evaluation import train_fixed_segmentation
from fakect_segmentation import load_segmentation, predict_axial_slice

fixture = ModelExperimentTests(methodName='runTest')
fixture.setUp()
try:
    # Real small NPZs, manifest files and registry receipts; only the registry
    # resolver/verifier are fixture mocks. Keep them active through lock reads.
    config = fixture.config
    config['training_protocol'] = 'fixed_epochs'
    config['split'].update(assignment={'family-0': 'train', 'family-1': 'train', 'family-2': 'test'},
                           allow_empty_validation=True)
    config['model'].update(patch_size=[16, 16], epochs=1, batch_size=2, seed=8)
    plan = plan_model_experiment(config)
    assert plan['ready'], plan
    frozen = freeze_model_experiment(config, fixture.raw, prepared_plan=plan)
    checked = inspect_experiment_lock(frozen['lock_path'])
    assert checked['split_sizes'] == {'train': 4, 'validation': 0, 'test': 2}
    expected_development = sorted(row['variant_id'] for row in checked['samples'] if row['split'] == 'train')
    test_variants = {row['variant_id'] for row in checked['samples'] if row['split'] == 'test'}
    for row in checked['samples']:
        if row['split'] == 'test':
            row['_path'].unlink()
    # Any attempt by the actual trainer to open an outer-test payload fails.
    model_directory = checked['lock']['model_directory']
    metadata = train_fixed_segmentation({'model': config['model'],
                                         'train': {'model_directory': model_directory}}, frozen['lock_path'])
    assert metadata['epochs_completed'] == 1
    assert metadata['split_sizes']['validation'] == 0
    assert metadata['loaded_variants'] == expected_development
    assert not test_variants.intersection(metadata['loaded_variants'])
    assert not metadata['test_payloads_loaded'] and not metadata['test_evaluated']
    assert not metadata['weights_include_test_selection']
    assert len(metadata['history']) == 1 and metadata['history'][0]['epoch'] == 1
    assert 'validation' not in metadata['history'][0]
    assert metadata['history'][0]['train_patches'] == 10
    assert np.isfinite(metadata['history'][0]['train_loss'])
    assert metadata['source_manifest_sha256'] == checked['sha256']
    model, loaded = load_segmentation(model_directory)
    prediction = predict_axial_slice(model, np.full((19, 23), .2, np.float32), loaded['settings'])
    assert prediction.shape == (19, 23) and np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    assert loaded['model_sha256'] == metadata['model_sha256']
    assert loaded['settings']['epochs'] == 1
    print('FIXED_REFIT_CPU_SMOKE_PASSED', tf.__version__, flush=True)
finally:
    fixture.doCleanups()
'''
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3',
                       TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1',
                       MPLCONFIGDIR=str(Path(temporary) / 'matplotlib'))
            try:
                completed = subprocess.run([sys.executable, '-u', '-c', code, str(ROOT)],
                                           env=env, capture_output=True, text=True, timeout=360)
            except subprocess.TimeoutExpired as exc:
                self.fail(f'Fixed-refit CPU smoke timed out: {exc.stdout!r}\n{exc.stderr!r}')
            self.assertEqual(completed.returncode, 0, completed.stdout + '\n' + completed.stderr)
            self.assertIn('FIXED_REFIT_CPU_SMOKE_PASSED', completed.stdout)
            print(next(line for line in completed.stdout.splitlines() if 'FIXED_REFIT_CPU_SMOKE_PASSED' in line))


if __name__ == '__main__':
    unittest.main()
