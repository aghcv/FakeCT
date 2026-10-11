"""Scientific matching, split isolation and immutable comparison provenance."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import fakect_context_comparison_learning as comparison
import fakect_context_learning as learning


class Volume:
    def __init__(self, row):
        self.row, self.closed = row, False

    def model_shape(self, spacing):
        return (24, 24, 24)

    def foreground_points(self, spacing, max_points, seed):
        # Deliberately different sibling foreground; it must never be sampled.
        if not self.row['baseline']:
            raise AssertionError('Sampling used a sibling mask')
        return np.asarray([[12, 12, 12], [13, 12, 12]], np.int64)

    def read_block(self, origin, shape, spacing):
        mask = np.zeros(shape, np.uint8)
        mask[2:4, 2:4, 2:4] = 1
        return dict(image=mask.astype(np.float32)*.2, mask=mask, valid=np.ones(shape, np.uint8))

    def close(self):
        self.closed = True


class Model:
    def __init__(self):
        self.weights = [np.asarray([1, 2, 3], np.float32)]
        self.inputs = [SimpleNamespace(shape=(None, 8, 8, 8, 1))]
        self.outputs = [SimpleNamespace(shape=(None, 8, 8, 8, 1))]

    def save(self, path):
        Path(path).write_bytes(self.weights[0].tobytes())

    def count_params(self):
        return 3


class Step:
    def __call__(self, *arrays):
        return .3

    def experimental_get_tracing_count(self):
        return 1


class ComparisonTests(unittest.TestCase):
    def config(self):
        return dict(model=dict(patch_size_kji=[8, 8, 8], depth=1, base_filters=2,
            epochs=3), comparison=dict(updates_per_epoch=8, validation_every_epochs=2,
                                     volume_cache_size=2))

    def lock(self):
        rows = []
        for split, cases in [('train', ['a', 'b']), ('validation', ['c']), ('test', ['d'])]:
            for case in cases:
                for baseline in (True, False):
                    variant = 'baseline' if baseline else 'edited'
                    rows.append(dict(sample_id=case+':'+variant, case_id=case,
                        variant_id=variant, baseline=baseline, anatomy_family=case,
                        split=split, source_shape_kji=[24, 24, 24],
                        spacing_ijk_mm=[1, 1, 1], source_atn={'path':case+'.bin'}))
        return dict(samples=rows)

    def test_strict_settings_and_exact_balancing(self):
        for settings in (dict(updates_per_epoch=0), dict(volume_cache_size=3),
                         dict(unknown=1), dict(validation_every_epochs=True)):
            with self.assertRaises(ValueError):
                comparison.settings_for(dict(comparison=settings))
        lock = self.lock()
        for updates in (3, 6):
            with self.assertRaisesRegex(ValueError, 'divisible'):
                comparison.training_cases(lock, dict(updates_per_epoch=updates))
        lock['samples'][1]['source_shape_kji'] = [25, 24, 24]
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            comparison.training_cases(lock, dict(updates_per_epoch=8))

    def test_missing_baseline_and_family_leakage_rejected_before_io(self):
        lock = self.lock()
        lock['samples'][0]['baseline'] = False
        with self.assertRaisesRegex(ValueError, 'exactly one baseline'):
            comparison.training_cases(lock, dict(updates_per_epoch=8))
        lock = self.lock()
        lock['samples'][-1]['anatomy_family'] = 'a'
        with self.assertRaisesRegex(ValueError, 'crosses partitions'):
            comparison.training_cases(lock, dict(updates_per_epoch=8))

    def test_schedule_equal_origins_seeds_and_balanced_variants(self):
        config = self.config()
        settings = learning.settings_for(config)
        cases = comparison.training_cases(self.lock(), comparison.settings_for(config))
        def events(epoch):
            with comparison.VolumeCache() as cache:
                return list(comparison.epoch_events(cases, settings, epoch, cache))
        with patch.object(learning, '_volume', side_effect=Volume):
            first, again, second = events(0), events(0), events(1)
        self.assertEqual(first, again)
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 8)
        counts = comparison.Counter(event['sibling']['sample_id'] for event in first)
        self.assertEqual(counts, {'a:baseline':2, 'a:edited':2, 'b:baseline':2, 'b:edited':2})
        self.assertTrue(all(event['baseline']['baseline'] for event in first))
        for event in first:
            self.assertEqual(event['baseline']['case_id'], event['sibling']['case_id'])

    def test_lru_bound_hit_and_exception_cleanup(self):
        made = []
        def create(row):
            volume = Volume(row)
            made.append(volume)
            return volume
        loaded = set()
        rows = self.lock()['samples']
        with patch.object(learning, '_volume', side_effect=create):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                with comparison.VolumeCache(2, loaded) as cache:
                    first = cache.get(rows[0])
                    self.assertIs(cache.get(rows[0]), first)
                    cache.get(rows[1])
                    cache.get(rows[2])
                    self.assertTrue(first.closed)
                    self.assertEqual(len(cache.volumes), 2)
                    raise RuntimeError('stop')
        self.assertEqual(len(made), 3)
        self.assertTrue(all(volume.closed for volume in made))
        self.assertEqual(loaded, {row['sample_id'] for row in rows[:3]})

    def fake_runtime(self, stack, observed):
        model = Model()
        tf = SimpleNamespace(__version__='test', keras=SimpleNamespace(
            backend=SimpleNamespace(clear_session=lambda: None),
            utils=SimpleNamespace(set_random_seed=lambda seed: None),
            models=SimpleNamespace(load_model=lambda *args, **kwargs: model)),
            config=SimpleNamespace(experimental=SimpleNamespace(enable_op_determinism=lambda:None),
                                   list_logical_devices=lambda:[]))
        stack.enter_context(patch.object(learning, '_tensorflow', return_value=tf))
        stack.enter_context(patch.object(learning, 'build_model', side_effect=lambda settings:Model()))
        stack.enter_context(patch.object(learning, '_training_step', return_value=Step()))
        stack.enter_context(patch.object(comparison, 'package_version', return_value='test'))
        stack.enter_context(patch.object(comparison, '_component_hashes', return_value={'code':'123'}))
        def create(row):
            observed.append(row['sample_id'])
            return Volume(row)
        stack.enter_context(patch.object(learning, '_volume', side_effect=create))
        def score(model, rows, settings, loaded, phase):
            loaded.update(row['sample_id'] for row in rows)
            observed.extend(row['sample_id'] for row in rows)
            counts = dict(true_positive=4, false_positive=1, false_negative=1,
                          true_negative=4, valid_voxels=10)
            return [{**learning._identity(row), 'metrics':learning.metrics_from_counts(counts),
                     'loss':.2, 'tiles':4} for row in rows]
        stack.enter_context(patch.object(comparison, '_score', side_effect=score))

    def test_fit_matching_isolation_validation_selection_and_evaluation(self):
        observed = []
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.fake_runtime(stack, observed)
            config, lock = self.config(), self.lock()
            root = Path(tmp)
            baseline = comparison.fit_arm(config, lock, 'baseline_only', root/'baseline')
            self.assertNotIn('a:edited', observed)
            self.assertNotIn('b:edited', observed)
            self.assertNotIn('d:baseline', observed)
            self.assertNotIn('d:edited', observed)
            self.assertIn('c:edited', observed)  # Common validation siblings are allowed.
            self.assertEqual(baseline['best_epoch'], 2)  # Equal score keeps earliest.
            self.assertEqual(baseline['optimizer_updates'], 24)
            self.assertFalse((root/'baseline'/'INCOMPLETE').exists())
            siblings = comparison.fit_arm(config, lock, 'baseline_plus_siblings', root/'siblings')
            self.assertEqual(baseline['initial_weights_sha256'], siblings['initial_weights_sha256'])
            self.assertEqual(baseline['optimizer_updates'], siblings['optimizer_updates'])
            histories = [json.loads((root/arm/'history.json').read_text()) for arm in ('baseline','siblings')]
            self.assertEqual([row['physical_patch_events_sha256'] for row in histories[0]],
                             [row['physical_patch_events_sha256'] for row in histories[1]])
            self.assertEqual([row['validation_performed'] for row in histories[0]], [False,True,True])
            self.assertNotEqual(histories[0][0]['selected_patch_events_sha256'],
                                histories[1][0]['selected_patch_events_sha256'])
            result = comparison.evaluate_arm(config, lock, 'baseline_only', root/'baseline', root/'test')
            self.assertEqual(result['split'], 'test')
            self.assertEqual(result['loaded_samples'], ['d:baseline', 'd:edited'])
            self.assertTrue(result['test_payloads_loaded'])
            with self.assertRaisesRegex(ValueError, 'provenance'):
                comparison.load_arm(config, lock, 'baseline_plus_siblings', root/'baseline')
            with self.assertRaises(FileExistsError):
                comparison.fit_arm(config, lock, 'baseline_only', root/'baseline')
            (root/'baseline'/'validation-epoch-002.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'selection evidence'):
                comparison.load_arm(config, lock, 'baseline_only', root/'baseline')

    def test_failure_keeps_incomplete_and_closes_native_volumes(self):
        observed, volumes = [], []
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.fake_runtime(stack, observed)
            def create(row):
                volume = Volume(row)
                volumes.append(volume)
                return volume
            stack.enter_context(patch.object(learning, '_volume', side_effect=create))
            stack.enter_context(patch.object(learning, '_training_step', return_value=lambda *args:float('nan')))
            output = Path(tmp)/'failed'
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                comparison.fit_arm(self.config(), self.lock(), 'baseline_only', output)
            self.assertTrue((output/'INCOMPLETE').exists())
            self.assertFalse((output/'training.json').exists())
            self.assertTrue(all(volume.closed for volume in volumes))

    def test_initial_digest_tracks_values_and_shape_not_names(self):
        first, second = Model(), Model()
        self.assertEqual(comparison.weights_digest(first), comparison.weights_digest(second))
        second.weights[0][0] = 7
        self.assertNotEqual(comparison.weights_digest(first), comparison.weights_digest(second))


if __name__ == '__main__':
    unittest.main()
