"""Controlled 200-epoch revision: reference integrity and tiny real training.

Metadata fixtures do not open phantom images. The TensorFlow check executes
one synthetic epoch (two balanced updates), never a 200-epoch clinical run.
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

import fakect_coa_paired_experiment as parent
import fakect_coa_paired_experiment_v2 as exp
import fakect_coa_paired_learning_v2 as learning
from test_coa_focused_experiment import fixture_config, fixture_contract, write_ini
from test_multiscale_experiment_review import MiniVolume, mini_overview


def revision_config(root):
    root = Path(root)
    c = fixture_config(root)
    c['experiment'].update(schema_version='fakect.coa-paired/2',
                           output_directory=str(root / 'paired-v2'))
    c['locator'].update(focused_experiment_directory=str(root / 'focused-reference'),
                        paired_experiment_directory=str(root / 'paired-reference'))
    c['diagnostic']['epochs'] = 200
    return c


def reference_fixture(root):
    """A completed, failed 100-epoch reference, independent of cohort files."""
    c, contract = revision_config(root), fixture_contract(root)
    reference_config = deepcopy(c)
    reference_config['experiment'].update(schema_version='fakect.coa-paired/1',
                                          output_directory=c['locator']['paired_experiment_directory'])
    reference_config['locator'].pop('paired_experiment_directory')
    reference_config['diagnostic']['epochs'] = 100
    lock_digest, checkpoint_digest, initial_digest = '1' * 64, '2' * 64, '3' * 64
    selected = exp.select_diagnostic(c, contract)
    reference_lock = deepcopy(contract)
    reference_lock.update(config=reference_config,
                          diagnostic_sample_ids=[row['sample_id'] for row in selected],
                          reference_initial_classifier_weights_sha256=initial_digest)
    records = []
    for row in selected:
        label = int(row['annotation']['coa_benchmark_positive'])
        records.append({'sample_id': row['sample_id'], 'case_id': row['case_id'],
                        'split': 'train', 'family': row['anatomy_family'],
                        'role': row['pilot_role'], 'label': label,
                        'score_label': row['annotation'][exp.SCORE],
                        'score_prediction': .3, 'probability': .55 if label else .45})
    result = {'stage': 'diagnostic', 'status': 'diagnostic_failed',
              'source_lock_sha256': lock_digest, 'classifier_trained': True,
              'locator_retrained': False, 'locator_unchanged': True,
              'test_evaluated': False, 'validation': None,
              'epochs_completed': 100, 'batch_size': 2, 'optimizer_steps': 200,
              'training_pairs_per_epoch': 2, 'sample_presentations': 400,
              'maximum_encoder_stride': 4, 'selected_epoch': 98,
              'checkpoint_sha256': {'classifier.keras': checkpoint_digest,
                                    'locator.keras': contract['locator_checkpoint']['sha256']},
              'selected_checkpoint_prediction_parity': {'passed': True},
              'learning_gate': {'passed': False}, 'predictions': records}
    audit = {'stage': 'diagnostic', 'source_lock_sha256': lock_digest,
             'locator_retrained': False, 'locator_unchanged': True,
             'test_evaluated': False, 'batch_size': 2, 'maximum_encoder_stride': 4,
             'optimizer_steps': 200, 'training_pairs_per_epoch': 2,
             'training_sample_count': 4, 'validation_sample_count': 0,
             'proposal_replay_passed': True,
             'initialization_seed': 42, 'shuffle_seed': 42,
             'initial_classifier_weights_sha256': initial_digest,
             'reference_initial_classifier_weights_sha256': initial_digest,
             'same_classifier_initialization_as_reference': True,
             'selected_epoch': 98, 'selected_classifier_checkpoint_sha256': checkpoint_digest,
             'selected_checkpoint_prediction_parity': {'passed': True}}
    history = {'selected_epoch': 98, 'epochs': [
        {'epoch': epoch, 'checkpoint_selection_bce': .6 if epoch == 98 else .7,
         'predictions': deepcopy(records), 'validation': None,
         'training_sample_order': [row['sample_id'] for row in selected]}
        for epoch in range(1, 101)]}
    predictions = {'source_lock_sha256': lock_digest, 'selected_epoch': 98,
                   'checkpoint_sha256': checkpoint_digest, 'test_evaluated': False,
                   'predictions': deepcopy(records)}
    detail = contract['parent_config']['input.detail']
    smoke = {'status': 'passed', 'checkpoint_parity': True,
             'source_lock_sha256': lock_digest, 'batch_size': 2,
             'detail_shape_kji': detail['shape_kji'], 'proposals': detail['proposals'],
             'traces': {'train': 1, 'predict': 1}, 'gpu_visible': True}
    initialization = {'weights_sha256': initial_digest, 'seed': 42,
                      'predictions': deepcopy(records), 'validation': None}
    documents = {'results': result, 'history': history, 'initialization': initialization,
                 'audit': audit, 'predictions': predictions, 'smoke': smoke}
    return c, contract, reference_config, reference_lock, documents, lock_digest, checkpoint_digest


class RevisionReferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.c, self.contract, self.reference_config, self.reference_lock,
         self.documents, self.lock_digest, self.checkpoint_digest) = reference_fixture(self.root)

    def validate(self, **changes):
        values = {'c': self.c, 'contract': self.contract,
                  'reference_config': self.reference_config, 'reference_lock': self.reference_lock,
                  'documents': self.documents, 'lock_digest': self.lock_digest,
                  'checkpoint_digest': self.checkpoint_digest}
        values.update(changes)
        return exp.validate_reference(values['c'], values['contract'], values['reference_config'],
                                      values['reference_lock'], values['documents'],
                                      lock_digest=values['lock_digest'],
                                      checkpoint_digest=values['checkpoint_digest'])

    def test_failed_reference_can_authorize_only_controlled_longer_diagnostic(self):
        before = deepcopy((self.c, self.contract, self.reference_config,
                           self.reference_lock, self.documents))
        self.validate()
        self.assertEqual((self.c, self.contract, self.reference_config,
                          self.reference_lock, self.documents), before)
        self.assertFalse(self.documents['results']['learning_gate']['passed'])
        output = Path(self.c['experiment']['output_directory'])
        output.mkdir()
        (output / 'experiment-lock.json').write_bytes(b'new revision lock')
        folder = output / 'diagnostic'
        folder.mkdir()
        # A completed failed reference is evidence for the longer diagnostic,
        # never permission to start a larger pilot under the new revision.
        exp.write_json(folder / 'results.json', self.documents['results'])
        exp.write_json(folder / 'classifier-predictions.json', self.documents['predictions'])
        with self.assertRaises(ValueError):
            exp.require_diagnostic(self.c, self.reference_lock)

    def test_only_diagnostic_epoch_budget_may_change(self):
        for section, key, value in [
                ('diagnostic', 'epochs', 199), ('diagnostic', 'case_ids', ['260613']),
                ('diagnostic', 'learning_rate', .002), ('diagnostic', 'maximum_bce', .5),
                ('diagnostic', 'minimum_probability_gap', .1), ('model', 'base_filters', 8),
                ('model', 'regression_weight', .5), ('training', 'seed', 43),
                ('training', 'learning_rate', .002)]:
            with self.subTest(section=section, key=key):
                changed = deepcopy(self.c)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    self.validate(c=changed)

    def test_reference_panel_crop_geometry_and_labels_cannot_change(self):
        for mutation in ('sample', 'crop', 'geometry', 'label', 'test'):
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.contract)
                if mutation == 'sample':
                    changed['samples'].pop()
                elif mutation == 'crop':
                    sample_id = changed['samples'][1]['sample_id']
                    changed['proposals'][sample_id]['origins_kji'][0][0] += 1
                elif mutation == 'geometry':
                    changed['parent_config']['input.detail']['spacing_ijk_mm'][0] = 1.
                elif mutation == 'label':
                    changed['samples'][1]['annotation']['coa_benchmark_positive'] = True
                else:
                    changed['reserved_test_sample_ids'].append('260613:edited_negative')
                with self.assertRaises(ValueError):
                    self.validate(contract=changed)

    def test_reference_completion_checkpoint_and_initialization_must_agree(self):
        changes = [('results', 'epochs_completed', 99), ('results', 'optimizer_steps', 199),
                   ('results', 'test_evaluated', True), ('results', 'locator_unchanged', False),
                   ('results', 'source_lock_sha256', 'changed'),
                   ('predictions', 'checkpoint_sha256', 'changed'),
                   ('predictions', 'selected_epoch', 99), ('audit', 'shuffle_seed', 43),
                   ('audit', 'same_classifier_initialization_as_reference', False),
                   ('initialization', 'weights_sha256', 'changed'), ('smoke', 'batch_size', 1)]
        for document, key, value in changes:
            with self.subTest(document=document, key=key):
                changed = deepcopy(self.documents)
                changed[document][key] = value
                with self.assertRaises(ValueError):
                    self.validate(documents=changed)
        with self.assertRaises(ValueError):
            self.validate(checkpoint_digest='changed checkpoint')

    def test_reference_history_must_be_complete_and_selected_by_original_rule(self):
        for mutation in ('missing_epoch', 'duplicate_epoch', 'better_checkpoint', 'wrong_records'):
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.documents)
                epochs = changed['history']['epochs']
                if mutation == 'missing_epoch':
                    epochs.pop()
                elif mutation == 'duplicate_epoch':
                    epochs[-1]['epoch'] = 99
                elif mutation == 'better_checkpoint':
                    epochs[49]['checkpoint_selection_bce'] = .1
                else:
                    epochs[97]['predictions'][0]['sample_id'] = 'different sibling'
                with self.assertRaises(ValueError):
                    self.validate(documents=changed)


class RevisionResourceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.c = revision_config(self.root)
        self.lock = fixture_contract(self.root)

    def test_new_revision_allows_200_without_relaxing_frozen_r1(self):
        path = self.root / 'revision.ini'
        write_ini(path, self.c)
        self.assertEqual(exp.load_config(path), self.c)
        old = deepcopy(self.c)
        old['experiment']['schema_version'] = 'fakect.coa-paired/1'
        old['locator'].pop('paired_experiment_directory')
        write_ini(path, old)
        with self.assertRaisesRegex(ValueError, 'epochs'):
            parent.load_config(path)
        old['diagnostic']['epochs'] = 100
        write_ini(path, old)
        self.assertEqual(parent.load_config(path), old)

    def test_new_revision_smoke_requires_own_lock_gpu_and_balanced_batch(self):
        output = Path(self.c['experiment']['output_directory'])
        output.mkdir()
        (output / 'experiment-lock.json').write_bytes(b'new synthetic revision lock')
        detail = self.lock['parent_config']['input.detail']
        smoke = {'status': 'passed', 'checkpoint_parity': True,
                 'source_lock_sha256': exp.sha256(output / 'experiment-lock.json'),
                 'detail_shape_kji': detail['shape_kji'], 'proposals': detail['proposals'],
                 'traces': {'train': 1, 'predict': 1}, 'gpu_visible': True,
                 'batch_size': 2}
        exp.write_json(output / 'runtime-smoke.json', smoke)
        self.assertEqual(exp.require_smoke(self.c, self.lock), smoke)
        for key, value in [('source_lock_sha256', 'r1-lock'), ('gpu_visible', False),
                           ('batch_size', 1), ('checkpoint_parity', False)]:
            with self.subTest(key=key):
                exp.write_json(output / 'runtime-smoke.json', {**smoke, key: value})
                with self.assertRaises(ValueError):
                    exp.require_smoke(self.c, self.lock)


class RevisionBindingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.c, self.contract, self.reference_config, self.reference_lock,
         self.documents, _, _) = reference_fixture(self.root)
        self.reference = Path(self.c['locator']['paired_experiment_directory'])
        (self.reference / 'diagnostic/model').mkdir(parents=True)
        write_ini(self.reference / 'input.ini', self.reference_config)
        exp.write_json(self.reference / 'experiment-lock.json', self.reference_lock)
        lock_digest = exp.sha256(self.reference / 'experiment-lock.json')
        checkpoint = self.reference / 'diagnostic/model/classifier.keras'
        checkpoint.write_bytes(b'immutable synthetic paired reference weights')
        checkpoint_digest = exp.sha256(checkpoint)
        for key in ('results', 'audit', 'predictions', 'smoke'):
            self.documents[key]['source_lock_sha256'] = lock_digest
        self.documents['results']['checkpoint_sha256']['classifier.keras'] = checkpoint_digest
        self.documents['audit']['selected_classifier_checkpoint_sha256'] = checkpoint_digest
        self.documents['predictions']['checkpoint_sha256'] = checkpoint_digest
        names = {'results': 'diagnostic/results.json', 'history': 'diagnostic/classifier-history.json',
                 'initialization': 'diagnostic/initialization.json', 'audit': 'diagnostic/continuation-audit.json',
                 'predictions': 'diagnostic/classifier-predictions.json', 'smoke': 'runtime-smoke.json'}
        for key, name in names.items():
            exp.write_json(self.reference / name, self.documents[key])
        fake_classifier = SimpleNamespace(**vars(exp.classifier))
        fake_classifier.source_contract = lambda c: deepcopy(self.contract)
        fake_parent = SimpleNamespace(**vars(exp.paired_parent))
        fake_parent.inspect_lock = Mock(return_value=deepcopy(self.reference_lock))
        self.fake_parent = fake_parent
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(exp, 'classifier', fake_classifier))
        self.stack.enter_context(patch.object(exp, 'paired_parent', fake_parent))

    def test_reference_artifacts_and_inherited_sources_are_pinned_and_immutable(self):
        self.assertTrue(set(parent.CODE_FILES).issubset(exp.CODE_FILES))
        self.assertIn('src/fakect_coa_paired_steps.py', exp.CODE_FILES)
        contract = exp.source_contract(self.c)
        self.assertEqual(contract['paired_reference']['learning_status'], 'diagnostic_failed')
        for relative in ('input.ini', 'experiment-lock.json', 'diagnostic/results.json',
                         'diagnostic/classifier-history.json', 'diagnostic/model/classifier.keras'):
            path = self.reference / relative
            self.assertEqual(contract['dependency_sha256'][str(path)], exp.sha256(path))
        source_names = ('src/frozen_r1.py', 'src/new_v2.py')
        for name in source_names:
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('IMMUTABLE_RUNTIME = True\n')
        self.stack.enter_context(patch.object(exp, 'ROOT', self.root))
        self.stack.enter_context(patch.object(exp, 'CODE_FILES', source_names))
        input_bytes = b'controlled synthetic 200-epoch revision\n'
        lock = exp.freeze(self.c, input_bytes)
        self.assertEqual(lock['training_contract']['controlled_changes'],
                         {'diagnostic_epochs': {'reference': 100, 'current': 200}})
        self.assertEqual(exp.inspect_lock(self.c, input_bytes), lock)
        output = Path(self.c['experiment']['output_directory'])
        paths = [self.root / 'src/frozen_r1.py', self.root / 'src/new_v2.py',
                 output / 'source/src/frozen_r1.py', output / 'input.ini',
                 self.reference / 'input.ini', self.reference / 'diagnostic/classifier-history.json',
                 self.reference / 'diagnostic/model/classifier.keras']
        for path in paths:
            with self.subTest(path=str(path)):
                original = path.read_bytes()
                path.write_bytes(original + b'changed\n')
                try:
                    with self.assertRaisesRegex(ValueError, '[Cc]hanged'):
                        exp.inspect_lock(self.c, input_bytes)
                finally:
                    path.write_bytes(original)
                self.assertEqual(exp.inspect_lock(self.c, input_bytes), lock)
        with self.assertRaises(FileExistsError):
            exp.freeze(self.c, input_bytes)

    def test_reference_failure_or_incomplete_markers_stop_before_reference_load(self):
        self.fake_parent.inspect_lock.reset_mock()
        for relative in ('INCOMPLETE', 'failure.json', 'diagnostic/INCOMPLETE',
                         'diagnostic/failure.json'):
            with self.subTest(marker=relative):
                marker = self.reference / relative
                marker.touch()
                try:
                    with self.assertRaisesRegex(ValueError, '[Ff]ailed|incomplete'):
                        exp.source_contract(self.c)
                    self.fake_parent.inspect_lock.assert_not_called()
                finally:
                    marker.unlink()


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class RevisionTensorFlowTests(unittest.TestCase):
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

    def test_tiny_v2_training_reuses_initialization_and_keeps_locator_frozen(self):
        from test_multiscale_v3_integration import config as parent_config
        import fakect_multiscale_experiment_v3 as geometry
        from fakect_coa_classifier_learning import weight_fingerprint
        from fakect_tensorflow_steps import StableSteps

        def stable_overview(row, c):
            return mini_overview({**row, 'test_removed': 4}, c)

        class FixtureOverviewInputs:
            def __init__(self, c):
                self.c, self.cache = c, {}
            def coarse(self, row):
                overview = stable_overview(row, self.c)
                self.cache[row['sample_id']] = overview
                image = geometry.normalized_input(overview['image'], overview['valid_fraction'], self.c)[None]
                return overview, image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c, lock = revision_config(root), fixture_contract(root)
            # Exercise the real learner cheaply, with the production epoch cap
            # independently checked above. No cohort or test image is loaded.
            c['training']['require_gpu'] = False
            c['model']['base_filters'] = 2
            c['diagnostic']['epochs'] = 1
            lock['parent_config'] = parent_config(root / 'parent')
            for row in lock['samples']:
                row['test_removed'] = {'original': 0, 'edited_negative': 4,
                                       'lower_positive': 12, 'higher_positive': 16}[row['pilot_role']]
            selected = exp.select_diagnostic(c, lock)
            lock['diagnostic_sample_ids'] = [row['sample_id'] for row in selected]
            shape = lock['parent_config']['input.detail']['shape_kji']
            reference = self.network.build_classifier(shape, num_proposals=2,
                                                       base_filters=2, seed=42)
            initial_digest = weight_fingerprint(reference)
            lock['reference_initial_classifier_weights_sha256'] = initial_digest
            del reference
            output = Path(c['experiment']['output_directory'])
            output.mkdir()
            (output / 'input.ini').write_bytes(b'synthetic v2 diagnostic config')
            locator = self.locator_network.build_locator((16, 16, 16), base_filters=2, seed=42)
            checkpoint = Path(lock['locator_checkpoint']['path'])
            locator.save(checkpoint)
            locator_hash = exp.sha256(checkpoint)
            lock['locator_checkpoint']['sha256'] = locator_hash
            reader = FixtureOverviewInputs(lock['parent_config'])
            for row in selected:
                overview, image = reader.coarse(row)
                predicted = self.locator_network.predict_boxes(locator, image)[0]
                origins, boxes = geometry.proposal_origins(predicted, overview, lock['parent_config'])
                lock['proposals'][row['sample_id']] = {
                    'predicted_box': predicted.tolist(), 'origins_kji': origins,
                    'proposal_boxes': boxes.tolist(), 'coverage_passed': True}
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
                    after = int(np.asarray(actual.optimizer.iterations))
                    self.assertEqual(after, before + 1)
                    updates.append(after)
                    return result
                observe_train.experimental_get_tracing_count = actual.train.experimental_get_tracing_count
                return StableSteps(observe_train, actual.predict, actual.optimizer)

            # Patch only new-module references and temporary import aliases;
            # earlier frozen module globals remain unchanged.
            fake_network = ModuleType(self.network.__name__)
            fake_network.__dict__.update(vars(self.network))
            fake_network.make_steps = observe_steps
            fake_locator = SimpleNamespace(**vars(self.locator_network))
            fake_locator.build_locator = Mock(side_effect=AssertionError('Revision must reuse frozen locator'))
            fake_legacy = SimpleNamespace(**vars(learning.legacy))
            fake_legacy.Inputs = FixtureOverviewInputs
            fake_legacy.configure_tensorflow = lambda runtime: (self.tf, fake_locator)
            with ExitStack() as stack:
                stack.enter_context(patch.object(exp, 'inspect_lock', return_value=lock))
                stack.enter_context(patch.object(exp, 'require_smoke',
                                                  return_value={'status': 'passed', 'batch_size': 2}))
                stack.enter_context(patch.object(learning, 'ContextVolume', MiniVolume))
                stack.enter_context(patch.object(learning, 'legacy', fake_legacy))
                stack.enter_context(patch.dict(sys.modules, {'fakect_coa_paired_steps': fake_network}))
                stack.enter_context(patch.object(learning, '_render'))
                result = learning.fit(c, lock, diagnostic=True)
                fake_locator.build_locator.assert_not_called()
            self.assertEqual(updates, [1, 2])
            self.assertEqual(result['epochs_completed'], 1)
            self.assertEqual(result['optimizer_steps'], 2)
            self.assertFalse(result['test_evaluated'])
            self.assertIsNone(result['validation'])
            self.assertTrue(result['locator_unchanged'])
            self.assertEqual(exp.sha256(checkpoint), locator_hash)
            self.assertEqual(set(MiniVolume.accessed), set(lock['diagnostic_sample_ids']))
            self.assertTrue(result['selected_checkpoint_prediction_parity']['passed'])
            self.assertEqual(result['training_traces']['classifier'], {'train': 1, 'predict': 1})
            folder = output / 'diagnostic'
            self.assertEqual(exp.checked_json(folder / 'initialization.json')['weights_sha256'], initial_digest)
            audit = exp.checked_json(folder / 'continuation-audit.json')
            self.assertTrue(audit['same_classifier_initialization_as_reference'])
            self.assertEqual(audit['reference_initial_classifier_weights_sha256'], initial_digest)
            self.assertTrue(exp.checked_json(folder / 'proposal-replay.json')['passed'])
            self.assertEqual(exp.checked_json(folder / 'classifier-history.json')['selected_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
