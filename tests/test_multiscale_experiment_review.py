"""Small staged-pilot integration, with synthetic volumes and real TF checkpoints.

No cohort payload is opened. The optional TF test runs only eight 16-cubed
synthetic scans, one epoch per model; detail padding is 32-cubed to exercise
validity and guarantee image-proposed coverage for bounded locator predictions.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_multiscale_experiment as exp
import fakect_multiscale_learning as learning


def fixture_rows():
    rows = []
    for split in ('train', 'validation'):
        for role, removed in [('original', 0), ('edited_negative', 4),
                              ('lower_positive', 12), ('higher_positive', 16)]:
            baseline = role == 'original'
            steps = [] if baseline else [
                dict(step_name='arch_outer_expand', operation='dilation', iteration=1,
                     status='applied', added_voxels=1, removed_voxels=0),
                dict(step_name='arch_inner_narrow', operation='erosion', iteration=1,
                     status='applied', added_voxels=0, removed_voxels=1),
                dict(step_name='coa_narrow', operation='erosion', iteration=1,
                     status='applied', added_voxels=0, removed_voxels=removed),
            ]
            rows.append(dict(sample_id=f'{split}:{role}', case_id=split,
                             anatomy_family=f'{split}_family', split=split,
                             baseline=baseline, pilot_role=role, step_outcomes=steps,
                             annotation={'coa_benchmark_positive': removed >= 8,
                                         exp.SCORE: removed / 40},
                             test_removed=removed))
    return rows


class MiniVolume:
    accessed = []
    returned_mask_value = None

    def __init__(self, row, verify_hashes=True):
        if row['split'] not in ('train', 'validation'):
            raise AssertionError('A test payload was accessed')
        self.accessed.append(row['sample_id'])
        self.spacing_ijk_mm = np.ones(3)
        self.mask_origin_ijk = np.zeros(3, dtype=int)
        self.mask_before = np.zeros((16, 16, 16), np.uint8)
        self.mask_before[6:10, 6:10, 6:10] = 1
        self.mask_after = self.mask_before.copy()
        if not row['baseline']:
            self.mask_after[5, 7, 7] = 1
            indexes = np.argwhere(self.mask_before)
            for index in indexes[:1 + row['test_removed']]:
                self.mask_after[tuple(index)] = 0
        self.image = np.full((16, 16, 16), .07, np.float32)
        self.image += np.indices(self.image.shape, dtype=np.float32)[2] * .002
        self.image += self.mask_after * .2

    def read_block(self, origin, shape, spacing):
        if tuple(spacing) != (1, 1, 1):
            raise AssertionError('Fixture supports unit sampling only')
        image = np.zeros(shape, np.float32)
        mask = np.zeros(shape, np.uint8)
        valid = np.zeros(shape, np.uint8)
        lo = np.maximum(origin, 0)
        hi = np.minimum(np.asarray(origin) + shape, self.image.shape)
        if np.all(hi > lo):
            src = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
            dst = tuple(slice(int(l - o), int(h - o)) for l, h, o in zip(lo, hi, origin))
            image[dst], mask[dst], valid[dst] = self.image[src], self.mask_after[src], 1
        if self.returned_mask_value is not None:
            mask.fill(self.returned_mask_value)
        return dict(image=image, mask=mask, valid=valid)

    def close(self):
        pass


def mini_overview(row, c):
    volume = MiniVolume(row)
    return dict(image=volume.image, valid_fraction=np.ones_like(volume.image),
                valid=np.ones_like(volume.image, np.uint8), spacing_ijk_mm=[1., 1., 1.],
                composition_key=row['sample_id'], cache_key=row['case_id'], stats={})


def fixture_config(directory):
    return {
        'experiment': {'output_directory': str(directory)},
        'input.overview': {'shape_kji': [16, 16, 16]},
        'input.detail': {'shape_kji': [32, 32, 32], 'spacing_ijk_mm': [1., 1., 1.], 'proposals': 2},
        'model': {'base_filters': 2, 'seed': 42, 'learning_rate': .001, 'clip_min': 0., 'clip_max': .5},
        'pilot': {'locator_epochs': 1, 'classifier_epochs': 1,
                  'minimum_changed_coverage': .9, 'require_gpu': False, 'gpu_memory_limit_mb': 1024},
    }


def fixture_lock(directory):
    lock = dict(samples=fixture_rows(), reserved_test_sample_ids=['test:never_open'],
                pilot_counts={'train': 4, 'validation': 4},
                annotation_acceptance={'use': 'synthetic_test_fixture'},
                sampling='four fixed synthetic cases per split',
                inference_contract='image only')
    exp.write_json(Path(directory) / 'experiment-lock.json', lock)
    (Path(directory) / 'input.ini').write_text('synthetic integration fixture\n')
    return lock


class InputIsolationTests(unittest.TestCase):
    def setUp(self):
        MiniVolume.accessed = []
        MiniVolume.returned_mask_value = None

    def tearDown(self):
        MiniVolume.returned_mask_value = None

    # Purpose: Verify classifier inputs ignore supervision returned by the image reader.
    # Inputs: MiniVolume detail blocks with mask changed to all ones.
    # Outputs: Overview/boxes/detail tensors stay identical and validity retains observed and padded voxels.
    def test_classifier_inputs_ignore_readers_supervision_mask(self):
        row = fixture_rows()[1]
        config = fixture_config('/tmp/not_written')
        proposal = dict(origins_kji=[[-8, -8, -8], [-4, -8, -8]],
                        proposal_boxes=[[.5, .5, .5, 2, 2, 2], [.75, .5, .5, 2, 2, 2]])
        with patch.object(learning, 'overview', side_effect=mini_overview), \
             patch.object(learning, 'ContextVolume', MiniVolume):
            inputs = learning.Inputs(config)
            before = inputs.classifier(row, proposal)
            MiniVolume.returned_mask_value = 1
            after = inputs.classifier(row, proposal)
        self.assertEqual(set(before), {'overview', 'proposal_boxes', 'detail_0', 'detail_1'})
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(before['detail_0'].shape, (1, 32, 32, 32, 2))
        self.assertTrue(np.any(before['detail_0'][..., 1] == 0))
        self.assertTrue(np.any(before['detail_0'][..., 1] == 1))

    # Purpose: Reject test preparation before image loading.
    # Inputs: Inspected fixture lock with one test-split row.
    # Outputs: Prepare raises ValueError and the overview reader is never called.
    def test_prepare_rejects_test_before_any_image_load(self):
        with tempfile.TemporaryDirectory() as directory:
            config, lock = fixture_config(directory), fixture_lock(directory)
            lock['samples'] = [{**lock['samples'][0], 'split': 'test'}]
            with patch.object(learning, 'overview') as reader, \
                 patch.object(exp, 'inspect_lock', return_value=lock):
                with self.assertRaisesRegex(ValueError, '[Tt]est|[Pp]artition|[Ss]plit'):
                    learning.prepare(config, lock)
                reader.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class StagedTensorFlowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_multiscale_network as network
        cls.network = network
        cls.tf = network._tensorflow()

    def setUp(self):
        self.tf.keras.backend.clear_session()
        MiniVolume.accessed = []

    def _patches(self, lock):
        stack = ExitStack()
        stack.enter_context(patch.object(learning, 'overview', side_effect=mini_overview))
        stack.enter_context(patch.object(learning, 'ContextVolume', MiniVolume))
        stack.enter_context(patch.object(learning, '_preview'))
        stack.enter_context(patch.object(exp, 'inspect_lock', return_value=lock))
        # GPU initialization policy has separate tests. Keep this fixture valid
        # when a prior TF test has already initialized the same process runtime.
        stack.enter_context(patch.object(learning, 'configure_tensorflow', return_value=(self.tf, self.network)))
        return stack

    # Purpose: Verify real multiscale preparation, fitting, and checkpoint reload isolation.
    # Inputs: Eight train/validation MiniVolume fixtures and small TensorFlow networks.
    # Outputs: Both checkpoints save, reload scores match, report exists, and reserved test payloads stay unread.
    def test_real_prepare_fit_save_reload_and_no_test_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            config, lock = fixture_config(directory), fixture_lock(directory)
            with self._patches(lock):
                preparation = learning.prepare(config, lock)
                result = learning.fit(config, lock)
                self.assertEqual(len(preparation['samples']), 8)
                self.assertTrue(result['classifier_trained'])
                self.assertFalse(result['test_evaluated'])
                self.assertIn(result['status'], ('learning_check_passed', 'learning_check_failed'))
                self.assertEqual(set(result['checkpoint_sha256']), {'locator.keras', 'classifier.keras'})
                self.assertEqual(len(result['predictions']), 8)
                self.assertTrue(exp.checked_json(out / 'proposal-audit.json')['passed'])
                loaded = self.network.load_model(out / 'model/classifier.keras')
                audits = {r['sample_id']: r for r in exp.checked_json(out / 'proposal-audit.json')['samples']}
                scores = learning._score_classifier(loaded, lock['samples'], learning.Inputs(config), audits)
                np.testing.assert_allclose([r['probability'] for r in scores],
                                           [r['probability'] for r in result['predictions']], atol=1e-6)
            self.assertTrue((out / 'report.html').exists())
            self.assertNotIn('test:never_open', MiniVolume.accessed)

    # Purpose: Stop classifier construction when locator coverage is insufficient.
    # Inputs: Changed coverage 0.95 and removed coverage 0.99 with build_classifier spy.
    # Outputs: Fit stops at coverage gate without training/saving a classifier or evaluating tests.
    def test_failed_coverage_stops_before_classifier_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            config, lock = fixture_config(directory), fixture_lock(directory)
            actual_coverage = exp.coverage
            def failed_coverage(volume, origins, c):
                result = actual_coverage(volume, origins, c)
                if result['changed_voxels']:
                    result.update(changed_coverage=.95, removed_coverage=.99)
                return result
            with self._patches(lock):
                learning.prepare(config, lock)
                with patch.object(exp, 'coverage', side_effect=failed_coverage), \
                     patch.object(self.network, 'build_classifier') as build:
                    result = learning.fit(config, lock)
                build.assert_not_called()
            self.assertEqual(result['status'], 'stopped_at_locator_coverage_gate')
            self.assertFalse(result['classifier_trained'])
            self.assertFalse(result['test_evaluated'])
            self.assertFalse((Path(directory) / 'model/classifier.keras').exists())


if __name__ == '__main__':
    unittest.main()
