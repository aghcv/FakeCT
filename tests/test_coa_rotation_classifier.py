"""Actual focused-network training on tiny fold-local rotation inputs.

The fixtures verify isolation and optimizer/checkpoint plumbing; they make no
claim that two epochs yield an accurate classifier. No cohort payload is read.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import fakect_coa_rotation_learning as learning
from fakect_coa_classifier_learning import weight_fingerprint
from fakect_tensorflow_steps import StableSteps
from test_multiscale_v3_integration import config


def fixture_rows():
    rows = []
    for case, split in [('train-a', 'train'), ('train-b', 'train'), ('validation-c', 'validation')]:
        for block in range(3):
            for positive in (False, True):
                rows.append({'sample_id': case + ':' + str(block) + ':' + str(int(positive)),
                             'case_id': case, 'anatomy_family': 'family-' + case, 'split': split,
                             'baseline': False,
                             'parameters': {'arch_outer_expand': {'distance_mm': float(block)},
                                            'arch_inner_narrow': {'distance_mm': .5},
                                            'coa_narrow': {'distance_mm': 3. if positive else 1.}},
                             'annotation': {'coa_benchmark_positive': positive,
                                            learning.exp.SCORE: .4 if positive else .1},
                             'synthetic_signal': .35 if positive else .12})
    return rows


class TinyFineInputs:
    accessed = []
    def __init__(self, runtime, limit_mb, allowed_rows):
        self.runtime = runtime
        self.allowed = {row['sample_id']: row['split'] for row in allowed_rows}
        self.cache, self.loads, self.hits = {}, 0, 0
    def classifier(self, row, proposal, *, cache=True):
        if self.allowed.get(row['sample_id']) != row['split']:
            raise AssertionError('Tiny reader opened a sample outside its stage')
        self.accessed.append(row['sample_id'])
        if row['sample_id'] not in self.cache:
            n = self.runtime['input.detail']['shape_kji']
            arrays = {}
            for index in range(self.runtime['input.detail']['proposals']):
                image = np.full((1, *n, 2), .03, np.float32)
                image[..., 1] = 1.
                image[:, 5:11, 6:10, 6:10, 0] = row['synthetic_signal'] + .01 * index
                if row['split'] == 'validation':
                    image[..., 0] += .015
                image.flags.writeable = False
                arrays['detail_' + str(index)] = image
            self.cache[row['sample_id']] = arrays
            self.loads += 1
        else:
            self.hits += 1
        return self.cache[row['sample_id']]
    def stats(self):
        resident = sum(a.nbytes for sample in self.cache.values() for a in sample.values())
        return {'resident_mib': resident / 1024**2, 'samples_resident': len(self.cache),
                'loads': self.loads, 'hits': self.hits}


class RotationClassifierIsolationTests(unittest.TestCase):
    def test_fitting_rejects_outer_test_before_any_image_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = config(directory)
            rows = fixture_rows()
            rows[-1]['split'] = 'final_test'
            forbidden = Mock(side_effect=AssertionError('Cannot open held-out images during fitting'))
            with patch.object(learning, 'StageFineInputs', forbidden):
                with self.assertRaisesRegex(ValueError, 'final-test'):
                    learning.fit_classifier(None, runtime, rows, {}, Path(directory) / 'classifier',
                                            {'training': {'cache_limit_mb': 1}}, epochs=2)
                forbidden.assert_not_called()

    def test_heldout_reader_requires_exact_stage_allowlist_before_payloads(self):
        runtime = config('/tmp/not-written')
        allowed = [{**fixture_rows()[0], 'split': 'final_test'}]
        reader = learning.StageFineInputs(runtime, 1, allowed)
        invalid = deepcopy(allowed[0])
        invalid['sample_id'] = 'reserved-infant:excluded'
        with self.assertRaisesRegex(ValueError, 'outside'):
            reader.classifier(invalid, {})
        invalid = {**allowed[0], 'split': 'train'}
        with self.assertRaisesRegex(ValueError, 'outside'):
            reader.classifier(invalid, {})

    def test_unbalanced_or_unmatched_companions_are_rejected(self):
        rows = fixture_rows()[:2]
        self.assertEqual(len(learning.matched_pairs(rows)), 1)
        for mutation in ('class', 'companion', 'case', 'split'):
            with self.subTest(mutation=mutation):
                changed = deepcopy(rows)
                if mutation == 'class':
                    changed[1]['annotation']['coa_benchmark_positive'] = False
                elif mutation == 'companion':
                    changed[1]['parameters']['arch_inner_narrow']['distance_mm'] += .1
                else:
                    changed[1][mutation if mutation == 'split' else 'case_id'] = 'different'
                with self.assertRaises(ValueError):
                    learning.matched_pairs(changed)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class RotationClassifierTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_coa_paired_steps as network
        cls.network = network
        cls.tf = network.focused._tensorflow()

    def setUp(self):
        self.tf.keras.backend.clear_session()
        TinyFineInputs.accessed = []

    def fit_fixture(self, directory, *, fixed_budget):
        runtime = config(directory)
        runtime['input.detail']['shape_kji'] = [16, 16, 16]
        runtime['model']['seed'] = 42
        rows = [row for row in fixture_rows() if not fixed_budget or row['split'] == 'train']
        proposals = {row['sample_id']: {'origins_kji': [[0, 0, 0], [0, 0, 0]]} for row in rows}
        c = {'training': {'cache_limit_mb': 1, 'classifier_learning_rate': .001, 'regression_weight': 1.}}
        reference = self.network.build_classifier([16, 16, 16], num_proposals=2, base_filters=2, seed=42)
        initial = weight_fingerprint(reference)
        del reference
        updates = []
        actual_make = self.network.make_steps
        def make(model, *args, **kwargs):
            actual = actual_make(model, *args, **kwargs)
            def train(images, targets):
                self.assertEqual(set(images), {'detail_0', 'detail_1'})
                self.assertTrue(all(tuple(image.shape) == (2, 16, 16, 16, 2) for image in images.values()))
                np.testing.assert_array_equal(np.asarray(targets['coa_logit']).ravel(), [0., 1.])
                np.testing.assert_allclose(np.asarray(targets['local_volume_loss']).ravel(), [.1, .4])
                before = int(np.asarray(actual.optimizer.iterations))
                value = actual.train(images, targets)
                self.assertEqual(int(np.asarray(actual.optimizer.iterations)), before + 1)
                updates.append(before + 1)
                return value
            train.experimental_get_tracing_count = actual.train.experimental_get_tracing_count
            return StableSteps(train, actual.predict, actual.optimizer)
        fake = ModuleType(self.network.__name__)
        fake.__dict__.update(vars(self.network))
        fake.make_steps = make
        output = Path(directory) / 'classifier'
        with patch.object(learning, 'StageFineInputs', TinyFineInputs), \
             patch.dict(sys.modules, {'fakect_coa_paired_steps': fake}):
            result = learning.fit_classifier(self.tf, runtime, rows, proposals, output, c,
                                             epochs=2, fixed_budget=fixed_budget)
        return result, output, rows, initial, updates

    def test_inner_classifier_fresh_balanced_updates_and_validation_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, rows, initial, updates = self.fit_fixture(directory, fixed_budget=False)
            self.assertEqual(result['epochs_completed'], 2)
            self.assertEqual(result['optimizer_updates'], 12)
            self.assertEqual(updates, list(range(1, 13)))
            self.assertEqual(result['fresh_initial_weights_sha256'], initial)
            self.assertEqual(result['training']['samples'], 12)
            self.assertEqual(result['validation']['samples'], 6)
            self.assertTrue(result['checkpoint_parity']['passed'])
            self.assertEqual(set(TinyFineInputs.accessed), {row['sample_id'] for row in rows})
            self.assertEqual(result['cache']['loads'], 18)
            history = learning.geometry.checked_json(output / 'history.json')
            chosen = min(history['epochs'], key=lambda epoch: epoch['validation']['bce'])
            self.assertEqual(result['selected_epoch'], chosen['epoch'])
            self.assertTrue(all(epoch['selection_bce'] == epoch['validation']['bce'] for epoch in history['epochs']))
            self.assertEqual(len(history['epochs'][0]['training_pair_order']), 6)
            expected_train = {row['sample_id'] for row in rows if row['split'] == 'train'}
            self.assertEqual({sample for pair in history['epochs'][0]['training_pair_order'] for sample in pair}, expected_train)

    def test_fixed_budget_classifier_saves_final_epoch_without_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, rows, initial, updates = self.fit_fixture(directory, fixed_budget=True)
            self.assertEqual(result['selected_epoch'], 2)
            self.assertEqual(result['optimizer_updates'], 12)
            self.assertEqual(updates, list(range(1, 13)))
            self.assertEqual(result['validation_sample_ids'], [])
            self.assertIsNone(result['validation'])
            self.assertEqual(result['fresh_initial_weights_sha256'], initial)
            self.assertTrue(result['checkpoint_parity']['passed'])
            self.assertEqual(set(TinyFineInputs.accessed), {row['sample_id'] for row in rows})
            history = learning.geometry.checked_json(output / 'history.json')
            self.assertTrue(all(epoch['selection_bce'] is None for epoch in history['epochs']))
            self.assertEqual(result['cache']['loads'], 12)


if __name__ == '__main__':
    unittest.main()
