"""Independent focused pilot: immutable reference, cache bounds, real training.

The metadata fixtures never open phantom data. A one-epoch TensorFlow fixture
trains 32 tiny images and evaluates eight validation images, with no test data.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]

import fakect_coa_pilot_experiment as exp
import fakect_coa_pilot_learning as learning
import fakect_coa_paired_experiment_v2 as parent
from test_coa_focused_experiment import fixture_contract, write_ini
from test_coa_paired_v2 import revision_config
from test_multiscale_experiment_review import MiniVolume, mini_overview


def pilot_config(root):
    root = Path(root)
    return {'experiment': {'schema_version': 'fakect.coa-focused-pilot/1',
                           'output_directory': str(root / 'pilot')},
            'reference': {'experiment_directory': str(root / 'passed-reference')},
            'training': {'epochs': 50, 'learning_rate': .001, 'seed': 42,
                         'cache_limit_mb': 5120, 'host_memory_limit_mb': 12288,
                         'gpu_memory_limit_mb': 12288, 'require_gpu': True}}


def passed_reference(root):
    root = Path(root)
    c = pilot_config(root)
    reference = Path(c['reference']['experiment_directory'])
    (reference / 'diagnostic/model').mkdir(parents=True)
    reference_config = revision_config(root)
    reference_config['experiment']['output_directory'] = str(reference)
    write_ini(reference / 'input.ini', reference_config)
    lock = fixture_contract(root)
    lock['parent_config']['input.detail']['shape_kji'] = [216, 192, 192]
    lock.update(config=reference_config,
                reference_initial_classifier_weights_sha256='3' * 64)
    selected = parent.select_diagnostic(reference_config, lock)
    lock['diagnostic_sample_ids'] = [row['sample_id'] for row in selected]
    parent.write_json(reference / 'experiment-lock.json', lock)
    source_digest = parent.sha256(reference / 'experiment-lock.json')
    checkpoint = reference / 'diagnostic/model/classifier.keras'
    checkpoint.write_bytes(b'passed synthetic focused diagnostic weights')
    records = []
    for row in selected:
        label = int(row['annotation']['coa_benchmark_positive'])
        records.append({'sample_id': row['sample_id'], 'case_id': row['case_id'],
                        'split': 'train', 'family': row['anatomy_family'],
                        'role': row['pilot_role'], 'label': label,
                        'score_label': row['annotation'][parent.SCORE],
                        'score_prediction': row['annotation'][parent.SCORE],
                        'probability': .95 if label else .05})
    result = {'stage': 'diagnostic', 'status': 'diagnostic_passed',
              'source_lock_sha256': source_digest, 'test_evaluated': False,
              'locator_unchanged': True, 'locator_retrained': False,
              'classifier_trained': True, 'selected_epoch': 194,
              'epochs_completed': 200, 'optimizer_steps': 400, 'batch_size': 2,
              'sample_presentations': 800, 'training_pairs_per_epoch': 2,
              'maximum_encoder_stride': 4, 'validation': None,
              'checkpoint_sha256': {'classifier.keras': parent.sha256(checkpoint),
                                    'locator.keras': lock['locator_checkpoint']['sha256']},
              'selected_checkpoint_prediction_parity': {'passed': True},
              'learning_gate': {'passed': True}, 'predictions': records}
    predictions = {'source_lock_sha256': source_digest, 'predictions': records,
                   'checkpoint_sha256': parent.sha256(checkpoint), 'selected_epoch': 194,
                   'test_evaluated': False}
    parent.write_json(reference / 'diagnostic/results.json', result)
    parent.write_json(reference / 'diagnostic/classifier-predictions.json', predictions)
    history = {'selected_epoch': 194, 'epochs': [
        {'epoch': epoch, 'checkpoint_selection_bce': .05 if epoch == 194 else .2,
         'predictions': deepcopy(records), 'validation': None}
        for epoch in range(1, 201)]}
    initialization = {'weights_sha256': '3' * 64, 'seed': 42,
                      'predictions': deepcopy(records), 'validation': None}
    audit = {'stage': 'diagnostic', 'source_lock_sha256': source_digest,
             'locator_retrained': False, 'locator_unchanged': True, 'test_evaluated': False,
             'batch_size': 2, 'maximum_encoder_stride': 4, 'optimizer_steps': 400,
             'training_pairs_per_epoch': 2, 'training_sample_count': 4,
             'validation_sample_count': 0, 'proposal_replay_passed': True,
             'initialization_seed': 42, 'shuffle_seed': 42,
             'initial_classifier_weights_sha256': '3' * 64,
             'reference_initial_classifier_weights_sha256': '3' * 64,
             'same_classifier_initialization_as_reference': True, 'selected_epoch': 194,
             'selected_classifier_checkpoint_sha256': parent.sha256(checkpoint),
             'selected_checkpoint_prediction_parity': {'passed': True}}
    detail = lock['parent_config']['input.detail']
    smoke = {'status': 'passed', 'checkpoint_parity': True, 'source_lock_sha256': source_digest,
             'batch_size': 2, 'detail_shape_kji': detail['shape_kji'], 'proposals': detail['proposals'],
             'traces': {'train': 1, 'predict': 1}, 'gpu_visible': True}
    parent.write_json(reference / 'diagnostic/classifier-history.json', history)
    parent.write_json(reference / 'diagnostic/initialization.json', initialization)
    parent.write_json(reference / 'diagnostic/continuation-audit.json', audit)
    parent.write_json(reference / 'runtime-smoke.json', smoke)
    return c, reference_config, lock, result


class PilotContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.c, self.reference_config, self.reference_lock, self.result = passed_reference(self.root)
        self.fake_parent = SimpleNamespace(**vars(parent))
        self.fake_parent.inspect_lock = Mock(return_value=deepcopy(self.reference_lock))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Earlier immutable modules are read through copied references; tests
        # do not mutate their globals or load any real cohort payload.
        self.stack.enter_context(patch.object(exp, 'parent', self.fake_parent))

    def test_clean_ini_roundtrip_and_bounded_cache(self):
        path = self.root / 'pilot.ini'
        write_ini(path, self.c)
        self.assertEqual(exp.load_config(path), self.c)
        for value in (0, 6145, 16384):
            with self.subTest(cache=value):
                changed = deepcopy(self.c)
                changed['training']['cache_limit_mb'] = value
                write_ini(path, changed)
                with self.assertRaises(ValueError):
                    exp.load_config(path)
        changed = deepcopy(self.c)
        changed['training']['unreviewed_option'] = True
        write_ini(path, changed)
        with self.assertRaises(ValueError):
            exp.load_config(path)

    def test_pilot_retains_frozen_32_8_panel_and_fresh_initialization(self):
        document = exp.plan(self.c)
        self.assertEqual(document['samples'], self.reference_lock['samples'])
        self.assertEqual(document['proposals'], self.reference_lock['proposals'])
        self.assertEqual(document['parent_config'], self.reference_lock['parent_config'])
        self.assertEqual(document['reserved_test_sample_ids'], self.reference_lock['reserved_test_sample_ids'])
        self.assertEqual(document['pilot_counts'], self.reference_lock['pilot_counts'])
        self.assertEqual(document['model_config'], self.reference_config['model'])
        self.assertEqual(document['reference_initial_classifier_weights_sha256'], '3' * 64)
        self.assertEqual(sum(row['split'] == 'train' for row in document['samples']), 32)
        self.assertEqual(sum(row['split'] == 'validation' for row in document['samples']), 8)
        reference = Path(self.c['reference']['experiment_directory'])
        for relative in ('input.ini', 'experiment-lock.json', 'diagnostic/results.json',
                         'diagnostic/classifier-predictions.json', 'diagnostic/model/classifier.keras'):
            path = reference / relative
            self.assertEqual(document['dependency_sha256'][str(path)], exp.sha256(path))

    def test_failed_or_forged_parent_gate_cannot_authorize_pilot(self):
        reference = Path(self.c['reference']['experiment_directory'])
        for mutation in ('status', 'gate', 'case_probability', 'epochs', 'checkpoint'):
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.result)
                predictions = parent.checked_json(reference / 'diagnostic/classifier-predictions.json')
                if mutation == 'status':
                    changed['status'] = 'diagnostic_failed'
                elif mutation == 'gate':
                    changed['learning_gate']['passed'] = False
                elif mutation == 'case_probability':
                    changed['predictions'][1]['probability'] = .51
                    predictions['predictions'] = changed['predictions']
                elif mutation == 'epochs':
                    changed['epochs_completed'] = 100
                else:
                    predictions['checkpoint_sha256'] = 'changed model'
                parent.write_json(reference / 'diagnostic/results.json', changed)
                parent.write_json(reference / 'diagnostic/classifier-predictions.json', predictions)
                with self.assertRaises(ValueError):
                    exp.plan(self.c)
                parent.write_json(reference / 'diagnostic/results.json', self.result)
                predictions.update(predictions=self.result['predictions'],
                                   checkpoint_sha256=self.result['checkpoint_sha256']['classifier.keras'])
                parent.write_json(reference / 'diagnostic/classifier-predictions.json', predictions)


    def test_insufficient_full_panel_cache_or_leaking_partition_is_rejected(self):
        changed_config = deepcopy(self.c)
        changed_config['training']['cache_limit_mb'] = 4000
        with self.assertRaisesRegex(ValueError, '[Cc]ache|RAM'):
            exp.plan(changed_config)
        for mutation in ('test', 'missing', 'split', 'roles'):
            with self.subTest(mutation=mutation):
                lock = deepcopy(self.reference_lock)
                if mutation == 'test':
                    lock['samples'][0]['split'] = 'test'
                elif mutation == 'missing':
                    lock['samples'].pop()
                elif mutation == 'split':
                    lock['samples'][0]['split'] = 'validation'
                else:
                    lock['samples'][0]['pilot_role'] = 'edited_negative'
                self.fake_parent.inspect_lock.return_value = lock
                with self.assertRaises(ValueError):
                    exp.plan(self.c)
        self.fake_parent.inspect_lock.return_value = deepcopy(self.reference_lock)

    def test_pilot_lock_pins_reference_geometry_panel_and_runtime_sources(self):
        source_names = ('src/inherited_runtime.py', 'src/pilot_runtime.py')
        for name in source_names:
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'IMMUTABLE_RUNTIME = True\n')
        self.stack.enter_context(patch.object(exp, 'ROOT', self.root))
        self.stack.enter_context(patch.object(exp, 'CODE_FILES', source_names))
        data = b'immutable synthetic pilot config\n'
        lock = exp.freeze(self.c, data)
        self.assertEqual(exp.inspect_lock(self.c, data), lock)
        out = Path(self.c['experiment']['output_directory'])
        reference = Path(self.c['reference']['experiment_directory'])
        paths = [self.root / 'src/inherited_runtime.py', out / 'source/src/pilot_runtime.py',
                 reference / 'diagnostic/model/classifier.keras',
                 reference / 'diagnostic/classifier-history.json']
        for path in paths:
            with self.subTest(path=str(path)):
                original = path.read_bytes()
                path.write_bytes(original + b'changed\n')
                try:
                    with self.assertRaisesRegex(ValueError, '[Cc]hanged'):
                        exp.inspect_lock(self.c, data)
                finally:
                    path.write_bytes(original)
                self.assertEqual(exp.inspect_lock(self.c, data), lock)
        for mutation in ('geometry', 'crop', 'sample'):
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.reference_lock)
                if mutation == 'geometry':
                    changed['parent_config']['input.detail']['spacing_ijk_mm'][0] += .01
                elif mutation == 'crop':
                    sample_id = changed['samples'][0]['sample_id']
                    changed['proposals'][sample_id]['origins_kji'][0][0] += 1
                else:
                    changed['samples'][0]['source_atn']['path'] = str(self.root / 'other.bin')
                self.fake_parent.inspect_lock.return_value = changed
                with self.assertRaises(ValueError):
                    exp.inspect_lock(self.c, data)
                self.fake_parent.inspect_lock.return_value = deepcopy(self.reference_lock)
                self.assertEqual(exp.inspect_lock(self.c, data), lock)

    def test_full_size_smoke_requires_full_cache_in_own_lock_and_gpu(self):
        lock = exp.plan(self.c)
        output = Path(self.c['experiment']['output_directory'])
        output.mkdir()
        (output / 'experiment-lock.json').write_bytes(b'new focused pilot lock')
        source_digest = exp.sha256(output / 'experiment-lock.json')
        detail = lock['parent_config']['input.detail']
        needed = lock['cache_plan']['input_cache_bytes']
        probe = {'status': 'passed', 'passed': True, 'source_lock_sha256': source_digest,
                 'sample_count': 40, 'resident_bytes': needed, 'resident_mib': needed / 1024**2,
                 'cache_limit_mib': 5120, 'host_guard_passed': True,
                 'host_peak_rss_mib': needed / 1024**2 + 1000, 'host_limit_mib': 12288,
                 'synthetic_data_only': True, 'physical_pages_touched': True,
                 'model_update_while_resident': True,
                 'resident_training_traces': {'train': 1, 'predict': 1}}
        smoke = {'status': 'passed', 'synthetic_data_only': True, 'checkpoint_parity': True,
                 'source_lock_sha256': source_digest, 'batch_size': 2,
                 'detail_shape_kji': detail['shape_kji'], 'proposals': detail['proposals'],
                 'traces': {'train': 1, 'predict': 1}, 'gpu_visible': True,
                 'samples': [{'loss': .7, 'gradient_global_norm': .2, 'full_cache_resident': True}],
                 'full_cache_memory_probe': probe}
        exp.write_json(output / 'runtime-smoke.json', smoke)
        self.assertEqual(exp.require_smoke(self.c, lock), smoke)
        for key, value in [('source_lock_sha256', 'different-lock'), ('gpu_visible', False),
                           ('batch_size', 1), ('samples', []),
                           ('samples', [{'loss': .7, 'gradient_global_norm': 0}]),
                           ('samples', [{'loss': .7, 'gradient_global_norm': .2}]),
                           ('samples', [{'loss': .7, 'gradient_global_norm': .2},
                                        {'loss': .7, 'gradient_global_norm': 0, 'full_cache_resident': True}])]:
            with self.subTest(key=key, value=value):
                exp.write_json(output / 'runtime-smoke.json', {**smoke, key: value})
                with self.assertRaises(ValueError):
                    exp.require_smoke(self.c, lock)
        for key, value in [('source_lock_sha256', 'parent-lock'), ('sample_count', 4),
                           ('resident_bytes', needed // 10), ('host_guard_passed', False),
                           ('host_peak_rss_mib', 13000), ('host_peak_rss_mib', 1),
                           ('cache_limit_mib', 2048), ('physical_pages_touched', False),
                           ('model_update_while_resident', False),
                           ('resident_training_traces', {'train': 2, 'predict': 1})]:
            with self.subTest(probe_field=key):
                invalid = deepcopy(smoke)
                invalid['full_cache_memory_probe'][key] = value
                exp.write_json(output / 'runtime-smoke.json', invalid)
                with self.assertRaises(ValueError):
                    exp.require_smoke(self.c, lock)


class PilotCacheTests(unittest.TestCase):
    def test_full_cache_probe_allocates_distinct_readonly_touched_image_buffers(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        runtime = {'experiment': {'output_directory': directory.name},
                   'input.detail': {'shape_kji': [8, 8, 8], 'proposals': 2},
                   'training': {'host_memory_limit_mb': 128}}
        checked = []
        class Guard:
            def check(self, stage, **kwargs):
                checked.append(stage)
                return {'rss_mib': 16., 'host_peak_rss_mib': 16.}
        retained, audit = learning.full_cache_memory_probe(runtime, Guard(), sample_count=40,
                                                          cache_limit_mb=1)
        self.assertEqual(len(retained), 40)
        self.assertTrue(all(set(sample) == {'detail_0', 'detail_1'} for sample in retained))
        arrays = [array for sample in retained for array in sample.values()]
        needed = 40 * 2 * 8**3 * 2 * np.dtype(np.float32).itemsize
        self.assertEqual(sum(array.nbytes for array in arrays), needed)
        self.assertEqual(audit['resident_bytes'], needed)
        self.assertEqual(audit['sample_count'], 40)
        self.assertTrue(audit['passed'])
        self.assertTrue(audit['host_guard_passed'])
        self.assertTrue(audit['physical_pages_touched'])
        self.assertTrue(checked)
        self.assertTrue(all(array.dtype == np.float32 and not array.flags.writeable for array in arrays))
        self.assertTrue(all(np.isfinite(array).all() for array in arrays))
        for index, array in enumerate(arrays):
            self.assertFalse(any(np.shares_memory(array, other) for other in arrays[index + 1:]))
        with self.assertRaises(ValueError):
            learning.full_cache_memory_probe(runtime, Guard(), sample_count=0, cache_limit_mb=1)
        with self.assertRaises(ValueError):
            learning.full_cache_memory_probe(runtime, Guard(), sample_count=40, cache_limit_mb=.01)

    def test_cache_reader_refuses_reserved_test_before_reading_any_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = fixture_contract(directory)
            row = deepcopy(contract['samples'][0])
            row['split'] = 'test'
            forbidden = Mock(side_effect=AssertionError('Reserved test data must remain unopened'))
            with patch.object(learning, 'ContextVolume', forbidden):
                reader = learning.FineInputs(contract['parent_config'], 1)
                with self.assertRaises(ValueError):
                    reader.classifier(row, contract['proposals'][row['sample_id']])
                forbidden.assert_not_called()


    def test_preload_rejects_test_duplicates_and_insufficient_cache_before_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = fixture_contract(directory)
            for mutation in ('test', 'duplicate', 'insufficient_cache'):
                with self.subTest(mutation=mutation):
                    rows = deepcopy(contract['samples'])
                    if mutation == 'test':
                        rows[-1]['split'] = 'test'
                    elif mutation == 'duplicate':
                        rows.append(deepcopy(rows[0]))
                    reader = learning.FineInputs(contract['parent_config'],
                                                 1 if mutation == 'insufficient_cache' else 128)
                    forbidden = Mock(side_effect=AssertionError('Invalid panel must be rejected before image I/O'))
                    guard = SimpleNamespace(check=Mock(return_value={'rss_mib': 16.}))
                    with patch.object(learning, 'ContextVolume', forbidden):
                        with self.assertRaises(ValueError):
                            learning.preload_inputs(reader, rows, contract['proposals'], guard)
                        forbidden.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class PilotTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_coa_paired_steps as network
        import fakect_multiscale_network_v3 as locator_network
        cls.network, cls.locator_network = network, locator_network
        cls.tf = locator_network._tensorflow()

    def setUp(self):
        self.tf.keras.backend.clear_session()
        MiniVolume.accessed = []
        MiniVolume.returned_mask_value = None

    def test_full_32_8_pilot_trains_fresh_then_selects_validation_checkpoint(self):
        from test_multiscale_v3_integration import config as parent_config
        import fakect_multiscale_experiment_v3 as geometry
        from fakect_coa_classifier_learning import weight_fingerprint
        from fakect_tensorflow_steps import StableSteps

        class FixtureOverviewInputs:
            def __init__(self, c):
                self.c, self.cache = c, {}
            def coarse(self, row):
                overview = mini_overview({**row, 'test_removed': 4}, self.c)
                self.cache[row['sample_id']] = overview
                image = geometry.normalized_input(overview['image'], overview['valid_fraction'], self.c)[None]
                return overview, image

        class PilotVolume(MiniVolume):
            def __init__(self, row, verify_hashes=True):
                super().__init__(row, verify_hashes=verify_hashes)
                if row['split'] == 'validation':
                    # A distinct validation image distribution makes using
                    # training BCE for checkpoint selection detectable.
                    self.image += .06

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c, lock = pilot_config(root), fixture_contract(root)
            c['training'].update(epochs=1, require_gpu=False)
            lock['parent_config'] = parent_config(root / 'parent')
            lock['model_config'] = {'base_filters': 2, 'regression_weight': 1.}
            for row in lock['samples']:
                row['test_removed'] = {'original': 0, 'edited_negative': 4,
                                       'lower_positive': 12, 'higher_positive': 16}[row['pilot_role']]
            shape = lock['parent_config']['input.detail']['shape_kji']
            reference_model = self.network.build_classifier(shape, num_proposals=2,
                                                             base_filters=2, seed=42)
            initial_digest = weight_fingerprint(reference_model)
            lock['reference_initial_classifier_weights_sha256'] = initial_digest
            del reference_model
            output = Path(c['experiment']['output_directory'])
            output.mkdir()
            (output / 'input.ini').write_bytes(b'fresh synthetic focused pilot config')
            locator = self.locator_network.build_locator((16, 16, 16), base_filters=2, seed=42)
            checkpoint = Path(lock['locator_checkpoint']['path'])
            locator.save(checkpoint)
            locator_digest = exp.sha256(checkpoint)
            lock['locator_checkpoint']['sha256'] = locator_digest
            reader = FixtureOverviewInputs(lock['parent_config'])
            for row in lock['samples']:
                overview, image = reader.coarse(row)
                predicted = self.locator_network.predict_boxes(locator, image)[0]
                origins, boxes = geometry.proposal_origins(predicted, overview, lock['parent_config'])
                lock['proposals'][row['sample_id']] = {'predicted_box': predicted.tolist(),
                    'origins_kji': origins, 'proposal_boxes': boxes.tolist(), 'coverage_passed': True}
            cache_bytes = len(lock['samples']) * 2 * np.prod(shape) * 2 * np.dtype(np.float32).itemsize
            lock['cache_plan'] = {'samples': 40, 'input_cache_bytes': int(cache_bytes),
                                  'input_cache_mib': float(cache_bytes / 1024**2),
                                  'cache_limit_mib': 5120}
            exp.write_json(output / 'experiment-lock.json', lock)
            MiniVolume.accessed = []
            updates = []
            real_make = self.network.make_steps

            def observe_steps(model, *args, **kwargs):
                actual = real_make(model, *args, **kwargs)
                def observe_train(values, targets):
                    np.testing.assert_array_equal(np.asarray(targets['coa_logit']).ravel(), [0., 1.])
                    self.assertEqual(set(values), {'detail_0', 'detail_1'})
                    self.assertTrue(all(tuple(value.shape) == (2, 32, 32, 32, 2)
                                        for value in values.values()))
                    before = int(np.asarray(actual.optimizer.iterations))
                    result = actual.train(values, targets)
                    self.assertEqual(int(np.asarray(actual.optimizer.iterations)), before + 1)
                    updates.append(before + 1)
                    return result
                observe_train.experimental_get_tracing_count = actual.train.experimental_get_tracing_count
                return StableSteps(observe_train, actual.predict, actual.optimizer)

            fake_network = ModuleType(self.network.__name__)
            fake_network.__dict__.update(vars(self.network))
            fake_network.make_steps = observe_steps
            fake_locator = SimpleNamespace(**vars(self.locator_network))
            fake_locator.build_locator = Mock(side_effect=AssertionError('Pilot must reuse the frozen locator'))
            fake_legacy = SimpleNamespace(**vars(learning.legacy))
            fake_legacy.Inputs = FixtureOverviewInputs
            fake_legacy.configure_tensorflow = lambda runtime: (self.tf, fake_locator)
            with ExitStack() as stack:
                stack.enter_context(patch.object(exp, 'inspect_lock', return_value=lock))
                stack.enter_context(patch.object(exp, 'require_smoke', return_value={'status': 'passed'}))
                stack.enter_context(patch.object(exp, 'require_diagnostic',
                                                  return_value={'scope': 'passed synthetic parent',
                                                                'dependency_sha256': {}}))
                stack.enter_context(patch.object(learning, 'ContextVolume', PilotVolume))
                stack.enter_context(patch.object(learning, 'legacy', fake_legacy))
                stack.enter_context(patch.dict(sys.modules, {'fakect_coa_paired_steps': fake_network}))
                stack.enter_context(patch.object(learning, '_render'))
                result = learning.fit(c, lock, diagnostic=False)
            fake_locator.build_locator.assert_not_called()
            self.assertEqual(updates, list(range(1, 17)))
            self.assertEqual(result['optimizer_steps'], 16)
            self.assertEqual(result['epochs_completed'], 1)
            self.assertEqual(result['training']['samples'], 32)
            self.assertEqual(result['validation']['samples'], 8)
            self.assertFalse(result['test_evaluated'])
            self.assertTrue(result['locator_unchanged'])
            self.assertEqual(exp.sha256(checkpoint), locator_digest)
            expected_ids = {row['sample_id'] for row in lock['samples']}
            self.assertEqual(set(MiniVolume.accessed), expected_ids)
            self.assertTrue(result['selected_checkpoint_prediction_parity']['passed'])
            self.assertEqual(result['training_traces']['classifier'], {'train': 1, 'predict': 1})
            folder = output / 'pilot'
            self.assertEqual(exp.checked_json(folder / 'initialization.json')['weights_sha256'], initial_digest)
            audit = exp.checked_json(folder / 'continuation-audit.json')
            self.assertTrue(audit['same_classifier_initialization_as_reference'])
            self.assertEqual(audit['validation_sample_count'], 8)
            self.assertTrue(exp.checked_json(folder / 'proposal-replay.json')['passed'])
            history = exp.checked_json(folder / 'classifier-history.json')
            self.assertEqual(result['selected_epoch'], min(history['epochs'],
                key=lambda row: row['validation']['bce'])['epoch'])
            self.assertTrue(all(row['checkpoint_selection_bce'] == row['validation']['bce']
                                for row in history['epochs']))
            self.assertNotAlmostEqual(history['epochs'][0]['training']['bce'],
                                      history['epochs'][0]['validation']['bce'], places=7)
            self.assertEqual(result['image_cache']['samples_resident'], 40)
            self.assertEqual(result['image_cache']['loads'], 40)


if __name__ == '__main__':
    unittest.main()
