"""Single-factor control integrity and one-epoch real TensorFlow fitting.

Only synthetic metadata/images are used; reserved phantom payloads are never read.
"""
from contextlib import ExitStack
from copy import deepcopy
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import fakect_coa_pilot_control_experiment as exp
import fakect_coa_pilot_control_learning as learning
import fakect_coa_pilot_experiment as parent
from test_coa_focused_experiment import fixture_contract, write_ini
from test_coa_pilot import pilot_config
from test_multiscale_experiment_review import MiniVolume, mini_overview


def control_config(root):
    c = pilot_config(root)
    c['experiment'].update(schema_version=exp.SCHEMA, output_directory=str(Path(root)/'control'))
    c['reference']['experiment_directory'] = str(Path(root)/'reference')
    c['training']['learning_rate'] = .0003
    return c


def completed_reference(root):
    c = control_config(root)
    reference = Path(c['reference']['experiment_directory'])
    (reference/'pilot/model').mkdir(parents=True)
    rc = pilot_config(root)
    rc['experiment']['output_directory'] = str(reference)
    write_ini(reference/'input.ini', rc)
    lock = fixture_contract(root)
    lock['parent_config']['input.detail']['shape_kji'] = [216, 192, 192]
    lock.update(model_config={'base_filters': 4, 'regression_weight': 1.},
                reference_initial_classifier_weights_sha256='3'*64,
                diagnostic_dependency={'scope': 'passed synthetic diagnostic'},
                cache_plan={'samples': 40, 'input_cache_bytes': 5096079360,
                            'input_cache_mib': 4860., 'cache_limit_mib': 5120})
    exp.write_json(reference/'experiment-lock.json', lock)
    digest = exp.sha256(reference/'experiment-lock.json')
    checkpoint = reference/'pilot/model/classifier.keras'
    checkpoint.write_bytes(b'completed synthetic pilot checkpoint')
    records = [{'sample_id': row['sample_id'], 'case_id': row['case_id'],
                'family': row['anatomy_family'], 'split': row['split'], 'role': row['pilot_role'],
                'label': int(row['annotation']['coa_benchmark_positive']),
                'score_label': row['annotation'][exp.SCORE], 'score_prediction': .1,
                'probability': .7 if row['annotation']['coa_benchmark_positive'] else .3}
               for row in lock['samples']]
    pairs = [[row['sample_id'] for row in pair] for pair in learning.pair_rows(
        [row for row in lock['samples'] if row['split'] == 'train'])]
    history = {'selected_epoch': 37, 'epochs': [
        {'epoch': epoch, 'checkpoint_selection_bce': .6 if epoch == 37 else .8,
         'validation': {'bce': .6 if epoch == 37 else .8}, 'predictions': deepcopy(records),
         'training_pair_order': deepcopy(pairs), 'training_sample_order': [value for pair in pairs for value in pair],
         'batch_size': 2, 'optimizer_updates_this_epoch': 16}
        for epoch in range(1, 51)]}
    common = {'source_lock_sha256': digest, 'test_evaluated': False, 'selected_epoch': 37}
    result = {**common, 'stage': 'pilot', 'status': 'learning_check_failed',
        'epochs_completed': 50, 'classifier_trained': True, 'batch_size': 2,
        'optimizer_steps': 800, 'training_pairs_per_epoch': 16, 'sample_presentations': 1600,
        'training': {'samples': 32}, 'validation': {'samples': 8, 'accuracy': .5},
        'locator_retrained': False, 'locator_unchanged': True, 'full_cache_loaded': True,
        'selected_checkpoint_prediction_parity': {'passed': True}, 'predictions': records,
        'checkpoint_sha256': {'classifier.keras': exp.sha256(checkpoint),
                              'locator.keras': lock['locator_checkpoint']['sha256']}}
    initial = {'weights_sha256': '3'*64, 'seed': 42, 'predictions': deepcopy(records)}
    audit = {**common, 'locator_retrained': False, 'locator_unchanged': True,
        'selected_checkpoint_prediction_parity': {'passed': True}, 'full_cache_loaded': True,
        'proposal_replay_passed': True, 'initial_classifier_weights_sha256': '3'*64,
        'reference_initial_classifier_weights_sha256': '3'*64,
        'same_classifier_initialization_as_reference': True, 'initialization_seed': 42, 'shuffle_seed': 42,
        'selected_classifier_checkpoint_sha256': exp.sha256(checkpoint)}
    predictions = {**common, 'checkpoint_sha256': exp.sha256(checkpoint), 'predictions': deepcopy(records)}
    cache = {**common, 'all_development_samples_resident': True, 'preloaded_before_first_gradient': True,
             'sample_count': 40, 'resident_bytes': 5096079360}
    documents = {'results': result, 'history': history, 'initialization': initial,
                 'audit': audit, 'predictions': predictions, 'replay': {'passed': True},
                 'cache': cache, 'smoke': {'source_lock_sha256': digest}}
    for key, name in exp.REFERENCE_FILES.items():
        exp.write_json(reference/name, documents[key])
    return c, rc, lock, documents


class ControlContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.c, self.rc, self.reference_lock, self.documents = completed_reference(self.root)
        self.fake_parent = SimpleNamespace(**vars(parent))
        self.fake_parent.inspect_lock = Mock(return_value=deepcopy(self.reference_lock))
        self.fake_parent.require_diagnostic = Mock(return_value={'scope': 'passed synthetic diagnostic'})
        self.fake_parent.require_smoke = Mock(return_value={'status': 'passed'})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(exp, 'parent', self.fake_parent))

    # Purpose: Enforce the declared single-factor pilot control configuration.
    # Inputs: Valid control INI and altered epoch, learning-rate, or cache-limit settings.
    # Outputs: Valid INI round-trips; every undeclared rate or budget change raises ValueError.
    def test_clean_ini_allows_only_declared_rate_and_budget(self):
        path = self.root/'control.ini'
        write_ini(path, self.c)
        self.assertEqual(exp.load_config(path), self.c)
        for key, value in (('epochs', 51), ('epochs', 200), ('learning_rate', .001),
                           ('learning_rate', .0001), ('learning_rate', float('nan')),
                           ('cache_limit_mb', 7000)):
            with self.subTest(key=key, value=value):
                changed = deepcopy(self.c)
                changed['training'][key] = value
                write_ini(path, changed)
                with self.assertRaises(ValueError):
                    exp.load_config(path)

    # Purpose: Allow completed reference evidence despite failed learning validation.
    # Inputs: Completed 50-epoch reference with learning_check_failed and selected epoch 37.
    # Outputs: Plan keeps samples, crops, model, cache, and initialization with rate changed to 0.0003.
    def test_completed_reference_retained_even_when_validation_or_learning_gate_fails(self):
        plan = exp.plan(self.c)
        self.assertEqual(plan['reference_summary']['status'], 'learning_check_failed')
        self.assertEqual(plan['reference_summary']['selected_epoch'], 37)
        self.assertEqual(plan['samples'], self.reference_lock['samples'])
        self.assertEqual(plan['proposals'], self.reference_lock['proposals'])
        self.assertEqual(plan['parent_config'], self.reference_lock['parent_config'])
        self.assertEqual(plan['model_config'], self.reference_lock['model_config'])
        self.assertEqual(plan['reference_initial_classifier_weights_sha256'], '3'*64)
        self.assertEqual(plan['cache_plan']['input_cache_bytes'], 5096079360)
        self.assertEqual(plan['training_contract']['changed_factor'],
                         {'learning_rate': {'reference': .001, 'control': .0003}})

    # Purpose: Reject changes to training factors beyond the control rate.
    # Inputs: Altered epochs, seed, cache/host/GPU limits, or GPU requirement.
    # Outputs: Planning raises ValueError for each changed factor.
    def test_every_other_training_factor_is_rejected(self):
        for key, value in (('epochs', 49), ('seed', 43), ('cache_limit_mb', 5000),
                           ('host_memory_limit_mb', 11000), ('gpu_memory_limit_mb', 11000),
                           ('require_gpu', False)):
            with self.subTest(key=key):
                changed = deepcopy(self.c)
                changed['training'][key] = value
                with self.assertRaises(ValueError):
                    exp.plan(changed)

    # Purpose: Verify reference completion, checkpoint, crop, cache, and initialization evidence.
    # Inputs: Incomplete budget, test access, locator changes, bad hashes, replay, or residency flags.
    # Outputs: Planning rejects every invalid reference evidence mutation.
    def test_parent_completion_reload_crop_cache_and_initialization_failures_rejected(self):
        mutations = [('results', 'epochs_completed', 49), ('results', 'optimizer_steps', 799),
                     ('results', 'test_evaluated', True), ('results', 'locator_unchanged', False),
                     ('audit', 'same_classifier_initialization_as_reference', False),
                     ('audit', 'selected_epoch', 50), ('initialization', 'weights_sha256', '4'*64),
                     ('predictions', 'checkpoint_sha256', '4'*64), ('replay', 'passed', False),
                     ('cache', 'resident_bytes', 1), ('cache', 'preloaded_before_first_gradient', False)]
        reference = Path(self.c['reference']['experiment_directory'])
        for key, field, value in mutations:
            with self.subTest(key=key, field=field):
                changed = deepcopy(self.documents[key])
                changed[field] = value
                path = reference/exp.REFERENCE_FILES[key]
                exp.write_json(path, changed)
                try:
                    with self.assertRaises(ValueError):
                        exp.plan(self.c)
                finally:
                    exp.write_json(path, self.documents[key])

    # Purpose: Require full balanced history and validation-based checkpoint selection.
    # Inputs: Missing epoch, wrong selection BCE, reserved sample order, or incorrect prediction labels.
    # Outputs: Planning raises ValueError for each altered history.
    def test_complete_history_must_preserve_validation_selection_and_balanced_updates(self):
        reference = Path(self.c['reference']['experiment_directory'])
        path = reference/exp.REFERENCE_FILES['history']
        for change in ('missing_epoch', 'wrong_selection_bce', 'wrong_order', 'wrong_predictions'):
            with self.subTest(change=change):
                history = deepcopy(self.documents['history'])
                if change == 'missing_epoch':
                    history['epochs'].pop()
                elif change == 'wrong_selection_bce':
                    history['epochs'][0]['checkpoint_selection_bce'] = .1
                elif change == 'wrong_order':
                    history['epochs'][0]['training_sample_order'][0] = '260612:reserved'
                else:
                    history['epochs'][0]['predictions'][0]['label'] = 1
                exp.write_json(path, history)
                try:
                    with self.assertRaises(ValueError):
                        exp.plan(self.c)
                finally:
                    exp.write_json(path, self.documents['history'])

    # Purpose: Verify control locks pin source and completed reference evidence.
    # Inputs: Frozen control followed by source/snapshot/history/checkpoint or parent contract mutations.
    # Outputs: Initial inspection matches; dependency, panel, crop, model, and geometry changes are rejected.
    def test_lock_pins_new_sources_and_completed_reference_evidence(self):
        source_names = ('src/inherited_runtime.py', 'src/control_runtime.py')
        for name in source_names:
            path = self.root/name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'IMMUTABLE_RUNTIME = True\n')
        self.stack.enter_context(patch.object(exp, 'ROOT', self.root))
        self.stack.enter_context(patch.object(exp, 'CODE_FILES', source_names))
        content = b'immutable synthetic control input\n'
        lock = exp.freeze(self.c, content)
        self.assertEqual(exp.inspect_lock(self.c, content), lock)
        reference = Path(self.c['reference']['experiment_directory'])
        for path in (self.root/source_names[0], Path(self.c['experiment']['output_directory'])/'source'/source_names[1],
                     reference/'pilot/classifier-history.json', reference/'pilot/model/classifier.keras'):
            with self.subTest(path=str(path)):
                original = path.read_bytes()
                path.write_bytes(original+b'changed\n')
                try:
                    with self.assertRaisesRegex(ValueError, '[Cc]hanged'):
                        exp.inspect_lock(self.c, content)
                finally:
                    path.write_bytes(original)
        for field in ('samples', 'proposals', 'model_config', 'parent_config'):
            with self.subTest(field=field):
                modified = deepcopy(self.reference_lock)
                if field == 'samples':
                    modified[field][0]['source_atn']['path'] = '/different/native.bin'
                elif field == 'proposals':
                    modified[field][modified['samples'][0]['sample_id']]['origins_kji'][0][0] += 1
                elif field == 'model_config':
                    modified[field]['regression_weight'] = 0.
                else:
                    modified[field]['input.detail']['spacing_ijk_mm'][0] += .01
                self.fake_parent.inspect_lock.return_value = modified
                try:
                    with self.assertRaises(ValueError):
                        exp.inspect_lock(self.c, content)
                finally:
                    self.fake_parent.inspect_lock.return_value = deepcopy(self.reference_lock)

    # Purpose: Verify the control numerical learner matches the reference implementation.
    # Inputs: Parsed pilot/control source with experiment imports normalized and report functions removed.
    # Outputs: Remaining executable ASTs are identical.
    def test_numerical_learner_is_identical_except_report_and_experiment_binding(self):
        original = (ROOT/'src/fakect_coa_pilot_learning.py').read_text()
        control = (ROOT/'src/fakect_coa_pilot_control_learning.py').read_text()
        control = control.replace('import fakect_coa_pilot_control_experiment as exp',
                                  'import fakect_coa_pilot_experiment as exp')
        trees = [ast.parse(text) for text in (original, control)]
        for tree in trees:
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.FunctionDef) and node.name == 'write_report')]
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))

    # Purpose: Verify the control report states its single changed training factor.
    # Inputs: Generated control report HTML.
    # Outputs: Report describes Adam rate 0.001 to 0.0003 and the matching 50-epoch budget.
    def test_control_report_explains_the_single_factor(self):
        path = Path(learning.write_report(self.root/'report-fixture'))
        document = path.read_text()
        self.assertIn('Only Adam learning rate changes from 0.001 to 0.0003', document)
        self.assertIn('the 50-epoch budget match the completed', document)

    # Purpose: Verify Slurm runs smoke and fit through the existing control CLI.
    # Inputs: Control GPU submission script.
    # Outputs: Exactly two crun commands target the real CLI with smoke then fit stages.
    def test_slurm_uses_existing_control_cli_in_separate_smoke_then_fit_processes(self):
        script = (ROOT/'scripts/research/slurm/fakect_coa_pilot_control_gpu.sh').read_text()
        commands = [line for line in script.splitlines() if line.startswith('crun ')]
        self.assertEqual(len(commands), 2)
        for command, stage in zip(commands, ('smoke', 'fit')):
            arguments = command.split()
            target = arguments[arguments.index('python')+1]
            self.assertEqual(target, 'scripts/research/train_coa_pilot_control.py')
            self.assertTrue((ROOT/target).is_file())
            self.assertIn('--stage '+stage, command)

    # Purpose: Reject diagnostic mode before loading control payloads.
    # Inputs: Control fit invoked with diagnostic=True and an empty lock.
    # Outputs: Fit raises ValueError.
    def test_diagnostic_mode_is_rejected_without_payload_reads(self):
        with self.assertRaises(ValueError):
            learning.fit(self.c, {}, diagnostic=True)

    # Purpose: Require production smoke with a resident cache, GPU, and new lock.
    # Inputs: Valid 40-sample smoke plus altered lock, batch, gradient, cache, memory, and trace evidence.
    # Outputs: Valid evidence loads; each incomplete or inconsistent smoke raises ValueError.
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


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class ControlTensorFlowTests(unittest.TestCase):
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

    # Purpose: Verify the lower-rate pilot reuses its locator and validation checkpoint.
    # Inputs: Fresh TensorFlow control with 32 train/8 validation samples and one epoch at 0.0003.
    # Outputs: Sixteen updates preserve locator hashes, load 40 inputs, keep one trace, and select validation BCE.
    def test_fresh_32_8_control_uses_lower_rate_and_validation_checkpoint(self):
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
            c, lock = control_config(root), fixture_contract(root)
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
                self.assertEqual(kwargs["learning_rate"], .0003)
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
