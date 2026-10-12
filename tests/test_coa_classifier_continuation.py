"""Classifier-only continuation integrity and tiny real TensorFlow integration.

All images are 16-cubed synthetic fixtures. Reserved cohort payloads are never
opened. Original locator training is deliberately forbidden in the fit test.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import fakect_multiscale_experiment_v3 as parent
import fakect_multiscale_learning_v3 as parent_learning
import fakect_coa_classifier_experiment as exp
import fakect_coa_classifier_learning as learning
from test_multiscale_experiment_review import MiniVolume, fixture_rows
from test_multiscale_v2_integration import ShiftedValidationVolume, overview
from test_multiscale_v3_integration import config as parent_config


class SyntheticSources:
    """A complete metadata-only parent and recovered review, with real hashes."""
    def __init__(self, directory):
        self.root = Path(directory)
        self.parent = self.root / 'parent'
        self.review = self.root / 'review'
        self.output = self.root / 'classifier'
        self.parent.mkdir()
        self.review.mkdir()
        (self.parent / 'model').mkdir()
        self.parent_config = parent_config(self.parent)
        self.parent_config['experiment']['schema_version'] = parent.SCHEMA
        self.parent_config['split'] = {'case_counts': [1, 1, 1], 'seed': 42}
        self.rows = fixture_rows()
        for row in self.rows:
            row.update(original_reference_mask_sha256=row['case_id'] + '-reference',
                       spacing_ijk_mm=[1., 1., 1.], source_shape_kji=[16, 16, 16],
                       source_atn={'path': str(self.root / (row['case_id'] + '.bin'))})
            row['annotation'].update(sample_id=row['sample_id'], case_id=row['case_id'],
                                     anatomy_family=row['anatomy_family'], errors=[])
        self.parent_bytes = b'synthetic parent input; parsed by fixture only\n'
        (self.parent / 'input.ini').write_bytes(self.parent_bytes)
        self.parent_lock = {
            'schema_version': parent.SCHEMA, 'ready': True,
            'config': self.parent_config, 'samples': self.rows,
            'reserved_test_sample_ids': ['test:never_open'],
            'design': {'families': {'train_family': ['train'],
                                    'validation_family': ['validation'], 'test_family': ['test']},
                       'folds': [{'family_assignment': {
                'train_family': 'train', 'validation_family': 'validation',
                'test_family': 'test'}}]},
            'pilot_counts': {split: {'samples': 4, 'positive': 2}
                             for split in ('train', 'validation')},
            'annotation_acceptance': {'use': 'accepted_engineering_proxy',
                                     'threshold': .2, 'clinical_review': False},
            'sampling': 'Synthetic paired panel', 'inference_contract': 'Image only',
            'input_sha256': hashlib.sha256(self.parent_bytes).hexdigest(),
            'code_sha256': {},
        }
        self.write_parent_lock()
        self.checkpoint = self.parent / 'model/locator.keras'
        self.checkpoint.write_bytes(b'synthetic checkpoint for metadata checks')
        self.prepared = {'complete': True,
                         'lock_sha256': parent.sha256(self.parent / 'experiment-lock.json'),
                         'samples': []}
        self.proposals = []
        for row in self.rows:
            box = [.5, .5, .5, .25, .25, .25]
            ov = {'image': SimpleNamespace(shape=(16, 16, 16)),
                  'spacing_ijk_mm': [1., 1., 1.]}
            origins, boxes = parent.proposal_origins(box, ov, self.parent_config)
            self.prepared['samples'].append({
                'sample_id': row['sample_id'], 'split': row['split'],
                'target_box_kji': box, 'overview_spacing_ijk_mm': [1., 1., 1.]})
            self.proposals.append({
                'sample_id': row['sample_id'], 'split': row['split'],
                'predicted_box': box, 'origins_kji': origins,
                'proposal_boxes': boxes.tolist(), 'target_coverage': 1.,
                'before_coverage': 1., 'changed_coverage': None if row['baseline'] else 1.,
                'added_coverage': None if row['baseline'] else 1.,
                'removed_coverage': None if row['baseline'] else 1.,
                'changed_voxels': 0 if row['baseline'] else row['test_removed'] + 2,
                'coverage_passed': True, 'center_error_mm': 0.})
        parent.write_json(self.parent / 'preparation.json', self.prepared)
        parent.write_json(self.parent / 'locator-history.json', {'epochs': [
            {'epoch': 1, 'validation_loss': 1.}, {'epoch': 2, 'validation_loss': .5}]})
        parent.write_json(self.parent / 'runtime-smoke.json', {
            'status': 'passed', 'checkpoint_parity': True,
            'classifier_shape_kji': [32, 32, 32], 'classifier_traces': {'train': 1, 'predict': 1}})
        self.audit = {'passed': True, 'failed_sample_ids': [], 'samples': self.proposals}
        self.robustness = {'passed': True, 'response_gate_passed': True,
                           'native_coverage_gate_passed': True,
                           'image_dependence_gate_passed': True, 'test_evaluated': False,
                           'probes': [], 'identities': []}
        for row in self.rows:
            if not row['baseline']:
                continue
            identity = {'sample_id': row['sample_id'], 'split': row['split'],
                        'real_error_mm': .5, 'blank_error_mm': 5.}
            self.robustness['identities'].append(identity)
            for axis, name in enumerate(('k', 'j', 'i')):
                for sign in ('-', '+'):
                    self.robustness['probes'].append({**identity, 'probe': name + sign,
                        'axis': axis, 'eligible': True, 'coverage_passed': True,
                        'relative_response_error': .1,
                        'native_coverage': {'target_coverage': 1., 'before_coverage': 1.,
                                            'changed_coverage': None, 'removed_coverage': None}})
        self.write_review()
        self.c = {
            'experiment': {'schema_version': 'fakect.coa-classifier/1',
                           'output_directory': str(self.output)},
            'locator': {'experiment_directory': str(self.parent),
                        'review_directory': str(self.review)},
            'training': {'epochs': 2, 'learning_rate': .001, 'seed': 42,
                         'host_memory_limit_mb': 12288, 'gpu_memory_limit_mb': 1024,
                         'require_gpu': False},
        }

    def write_parent_lock(self):
        parent.write_json(self.parent / 'experiment-lock.json', self.parent_lock)
        parent.write_json(self.review / 'experiment-lock.json', self.parent_lock)

    def write_review(self):
        digest = parent.sha256(self.checkpoint)
        self.provenance = {
            'schema_version': 'fakect.multiscale-review/1',
            'parent_experiment': str(self.parent),
            'parent_lock_sha256': parent.sha256(self.parent / 'experiment-lock.json'),
            'checkpoint': str(self.checkpoint), 'checkpoint_sha256': digest,
            'selected_epoch': 2, 'evaluation_only': True,
            'test_evaluated': False, 'source_sha256': {},
        }
        self.results = {
            'status': 'locator_ready_for_classifier', 'evaluation_only': True,
            'classifier_trained': False, 'test_evaluated': False,
            'coverage_passed': True, 'passing_samples': len(self.rows),
            'total_samples': len(self.rows), 'failed_samples': [],
            'robustness_passed': True, 'selected_epoch': 2,
            'checkpoint_sha256': digest,
        }
        for name, value in [('review-provenance.json', self.provenance),
                            ('results.json', self.results),
                            ('proposal-audit.json', self.audit),
                            ('locator-robustness.json', self.robustness)]:
            parent.write_json(self.review / name, value)

    def parsed_parent(self):
        stack = ExitStack()
        stack.enter_context(patch.object(parent, 'load_config', return_value=self.parent_config))
        stack.enter_context(patch.object(parent, 'inspect_lock', return_value=self.parent_lock))
        return stack

    def documents(self):
        return {name: parent.checked_json(self.review / (name + '.json')) for name in (
            'review-provenance', 'results', 'proposal-audit', 'locator-robustness', 'experiment-lock')}


class ContinuationMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = SyntheticSources(self.directory.name)

    def validate(self, *, lock=None, prepared=None, documents=None):
        fixture = self.fixture
        return exp.validate_review(fixture.parent_config,
            fixture.parent_lock if lock is None else lock,
            fixture.prepared if prepared is None else prepared,
            fixture.documents() if documents is None else documents)

    # Purpose: Verify reviewed proposals join exactly without changing image-derived crops.
    # Inputs: Synthetic parent samples, preparation, audit, and robustness documents.
    # Outputs: Every sample joins with its original proposal origins.
    def test_exact_review_join_keeps_image_only_proposals(self):
        proposals = self.validate()
        self.assertEqual(set(proposals), {r['sample_id'] for r in self.fixture.rows})
        for row in self.fixture.proposals:
            self.assertEqual(proposals[row['sample_id']]['origins_kji'], row['origins_kji'])

    # Purpose: Exclude reserved test partitions from classifier continuation.
    # Inputs: Sample changed to test split, reserved ID, or test family.
    # Outputs: Sample validation raises ValueError for each leakage route.
    def test_reserved_test_split_id_and_family_cannot_enter_continuation(self):
        changes = [dict(split='test'), dict(sample_id='test:never_open'),
                   dict(anatomy_family='test_family')]
        for change in changes:
            with self.subTest(change=change):
                lock = deepcopy(self.fixture.parent_lock)
                lock['samples'][0].update(change)
                with self.assertRaises(ValueError):
                    exp.validate_samples(lock)

    # Purpose: Prevent partition overlap in family, case, reference, and source identity.
    # Inputs: Validation sample reusing a training family, case, mask hash, or source.
    # Outputs: Sample validation rejects every cross-partition identity collision.
    def test_family_case_reference_and_source_do_not_cross_partitions(self):
        for key in ('anatomy_family', 'case_id', 'original_reference_mask_sha256', 'source_atn'):
            with self.subTest(key=key):
                lock = deepcopy(self.fixture.parent_lock)
                lock['samples'][4][key] = lock['samples'][0][key]
                with self.assertRaises(ValueError):
                    exp.validate_samples(lock)

    # Purpose: Require exact proposal membership and split assignments.
    # Inputs: Missing, duplicate, extra, or wrong-split audit records.
    # Outputs: Review validation raises ValueError.
    def test_missing_duplicate_extra_or_wrong_split_proposal_rejected(self):
        mutations = (
            lambda rows: rows.pop(),
            lambda rows: rows.append(deepcopy(rows[0])),
            lambda rows: rows.append({**rows[0], 'sample_id': 'unselected:scenario'}),
            lambda rows: rows[0].update(split='validation'),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                docs = self.fixture.documents()
                mutate(docs['proposal-audit']['samples'])
                with self.assertRaises(ValueError):
                    self.validate(documents=docs)

    # Purpose: Require proposal coverage of original anatomy and removed tissue.
    # Inputs: Target/before coverage 0.8, removed coverage 0.99, or failed coverage flag.
    # Outputs: Review validation rejects each insufficient proposal.
    def test_review_cannot_approve_missing_erosion_or_original_anatomy(self):
        mutations = (
            lambda rows: rows[0].update(target_coverage=.8),
            lambda rows: rows[0].update(before_coverage=.8),
            lambda rows: rows[1].update(removed_coverage=.99),
            lambda rows: rows[1].update(coverage_passed=False),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                docs = self.fixture.documents()
                # Keep global approval true to detect trust in summary flags.
                mutate(docs['proposal-audit']['samples'])
                with self.assertRaises(ValueError):
                    self.validate(documents=docs)

    # Purpose: Require each robustness gate independently.
    # Inputs: Passed summary with one robustness gate flag set false.
    # Outputs: Review validation raises ValueError for every failed gate.
    def test_each_robustness_gate_is_required_even_if_summary_passed(self):
        for key in ('passed', 'response_gate_passed', 'native_coverage_gate_passed',
                    'image_dependence_gate_passed'):
            with self.subTest(key=key):
                docs = self.fixture.documents()
                docs['locator-robustness'][key] = False
                with self.assertRaises(ValueError):
                    self.validate(documents=docs)

    # Purpose: Verify robustness flags cannot conceal invalid numeric evidence.
    # Inputs: Response error 2, blank error 0, or ineligible validation probes.
    # Outputs: Review validation rejects each altered probe family.
    def test_robustness_flags_cannot_override_bad_numeric_probes(self):
        for field, value in [('relative_response_error', 2.), ('blank_error_mm', 0.),
                             ('eligible', False)]:
            with self.subTest(field=field):
                docs = self.fixture.documents()
                for row in docs['locator-robustness']['probes']:
                    if row['split'] == 'validation':
                        row[field] = value
                if field == 'blank_error_mm':
                    for row in docs['locator-robustness']['identities']:
                        row[field] = value
                with self.assertRaises(ValueError):
                    self.validate(documents=docs)

    # Purpose: Verify preparation and copied review lock match parent evidence.
    # Inputs: Missing preparation row or changed pilot role in the copied lock.
    # Outputs: Review validation rejects both mismatches.
    def test_preparation_and_copied_parent_lock_must_match(self):
        prepared = deepcopy(self.fixture.prepared)
        prepared['samples'].pop()
        with self.assertRaises(ValueError):
            self.validate(prepared=prepared)
        docs = self.fixture.documents()
        docs['experiment-lock']['samples'][0]['pilot_role'] = 'changed-after-review'
        with self.assertRaises(ValueError):
            self.validate(documents=docs)

    # Purpose: Prevent reviewed crop geometry from moving after prediction.
    # Inputs: First origin or proposal-box coordinate increased by one.
    # Outputs: Review validation raises ValueError.
    def test_proposal_geometry_cannot_be_moved_after_image_prediction(self):
        for field in ('origins_kji', 'proposal_boxes'):
            with self.subTest(field=field):
                docs = self.fixture.documents()
                docs['proposal-audit']['samples'][0][field][0][0] += 1
                with self.assertRaises(ValueError):
                    self.validate(documents=docs)

    # Purpose: Verify continuation runtime retains inherited image geometry.
    # Inputs: Parent configuration and new two-epoch classifier settings.
    # Outputs: New output/epochs apply, overview/detail geometry remains equal, and parent config is intact.
    def test_runtime_reuses_geometry_and_applies_new_classifier_settings(self):
        lock = {'parent_config': deepcopy(self.fixture.parent_config)}
        runtime = exp.runtime_config(self.fixture.c, lock)
        self.assertEqual(runtime['experiment']['output_directory'], str(self.fixture.output))
        self.assertEqual(runtime['pilot']['classifier_epochs'], 2)
        self.assertEqual(runtime['input.detail'], self.fixture.parent_config['input.detail'])
        self.assertEqual(runtime['input.overview'], self.fixture.parent_config['input.overview'])
        self.assertEqual(self.fixture.parent_config['experiment']['output_directory'], str(self.fixture.parent))

    # Purpose: Reject test samples before TensorFlow or image access.
    # Inputs: Verified lock containing a test-split sample and configure/input spies.
    # Outputs: Fit raises ValueError without configuring TensorFlow or opening inputs.
    def test_fit_rejects_test_before_tensorflow_or_image_reader(self):
        fixture = self.fixture
        fixture.output.mkdir()
        (fixture.output / 'input.ini').write_text('synthetic input\n')
        lock = {'parent_config': fixture.parent_config,
                'samples': [{**fixture.rows[0], 'split': 'test'}]}
        with patch.object(exp, 'inspect_lock', return_value=lock), \
             patch.object(parent_learning, 'configure_tensorflow') as configure, \
             patch.object(parent_learning, 'Inputs') as inputs:
            with self.assertRaisesRegex(ValueError, '[Tt]est'):
                learning.fit(fixture.c, lock)
            configure.assert_not_called()
            inputs.assert_not_called()

    # Purpose: Verify fit uses the inspected lock rather than caller-supplied samples.
    # Inputs: Caller training panel replaced by an inspected lock containing a test row.
    # Outputs: Fit rejects the verified test row before TensorFlow or image access.
    def test_fit_uses_verified_lock_instead_of_unverified_caller_panel(self):
        fixture = self.fixture
        fixture.output.mkdir()
        (fixture.output / 'input.ini').write_text('synthetic input\n')
        caller = {'parent_config': fixture.parent_config, 'samples': fixture.rows}
        verified = {**caller, 'samples': [{**fixture.rows[0], 'split': 'test'}]}
        with patch.object(exp, 'inspect_lock', return_value=verified), \
             patch.object(parent_learning, 'configure_tensorflow') as configure, \
             patch.object(parent_learning, 'Inputs') as inputs:
            with self.assertRaisesRegex(ValueError, '[Tt]est'):
                learning.fit(fixture.c, caller)
            configure.assert_not_called()
            inputs.assert_not_called()

    # Purpose: Require exact replay of native voxel origins and finite boxes.
    # Inputs: Proposal origin shifted by one, then a proposal box containing NaN.
    # Outputs: Replay rejects altered origins and nonfinite geometry.
    def test_proposal_replay_demands_exact_voxel_origin(self):
        proposal = self.fixture.proposals[0]
        origins = deepcopy(proposal['origins_kji'])
        origins[0][0] += 1
        with self.assertRaisesRegex(ValueError, 'origins'):
            learning.compare_proposal(proposal, proposal['predicted_box'],
                                      origins, proposal['proposal_boxes'])
        boxes = np.array(proposal['proposal_boxes'])
        boxes[0, 0] = np.nan
        with self.assertRaises(ValueError):
            learning.compare_proposal(proposal, proposal['predicted_box'],
                                      proposal['origins_kji'], boxes)

    # Purpose: Verify continuation provenance pins the reviewed locator and dependencies.
    # Inputs: Synthetic parent/review artifacts and selected locator epoch 2.
    # Outputs: Contract records epoch 2, checkpoint digest, and required artifact hashes.
    def test_source_contract_pins_parent_review_and_exact_checkpoint(self):
        fixture = self.fixture
        with fixture.parsed_parent():
            contract = exp.source_contract(fixture.c)
        self.assertEqual(contract['locator_checkpoint']['selected_epoch'], 2)
        self.assertEqual(contract['locator_checkpoint']['sha256'], exp.sha256(fixture.checkpoint))
        expected = {str(fixture.parent / 'experiment-lock.json'), str(fixture.checkpoint),
                    str(fixture.review / 'results.json'), str(fixture.review / 'proposal-audit.json')}
        self.assertTrue(expected <= set(contract['dependency_sha256']))

    # Purpose: Reject substituted parent, checkpoint, or selected-epoch provenance.
    # Inputs: Changed parent path/hash, checkpoint path/hash, epoch 1, or checkpoint bytes.
    # Outputs: Source contract validation raises ValueError for each substitution.
    def test_source_contract_rejects_checkpoint_parent_and_epoch_substitution(self):
        changes = [('parent_experiment', str(self.fixture.root / 'other')),
                   ('parent_lock_sha256', '0' * 64),
                   ('checkpoint', str(self.fixture.root / 'other.keras')),
                   ('checkpoint_sha256', '0' * 64), ('selected_epoch', 1)]
        for field, value in changes:
            with self.subTest(field=field):
                altered = {**self.fixture.provenance, field: value}
                exp.write_json(self.fixture.review / 'review-provenance.json', altered)
                with self.fixture.parsed_parent(), self.assertRaises(ValueError):
                    exp.source_contract(self.fixture.c)
        exp.write_json(self.fixture.review / 'review-provenance.json', self.fixture.provenance)
        self.fixture.checkpoint.write_bytes(b'changed checkpoint')
        with self.fixture.parsed_parent(), self.assertRaises(ValueError):
            exp.source_contract(self.fixture.c)

    # Purpose: Reject rewritten reviews even when their sidecars are refreshed.
    # Inputs: Frozen continuation followed by changed review results and a new checksum.
    # Outputs: Initial inspection matches; later dependency inspection raises ValueError.
    def test_frozen_dependency_rejects_rewritten_review_with_new_sidecar(self):
        fixture = self.fixture
        data = b'synthetic continuation input\n'
        with fixture.parsed_parent():
            lock = exp.freeze(fixture.c, data)
            self.assertEqual(exp.inspect_lock(fixture.c, data), lock)
            altered = {**fixture.results, 'note': 'An externally revised review'}
            # A valid new sidecar must not let an existing lock trust new data.
            exp.write_json(fixture.review / 'results.json', altered)
            with self.assertRaisesRegex(ValueError, '[Dd]ependency|[Cc]hanged'):
                exp.inspect_lock(fixture.c, data)

    # Purpose: Enforce strict continuation configuration and independent output.
    # Inputs: Valid INI, then unknown fields, NaN learning rate, or output beneath parent.
    # Outputs: Valid config round-trips and each malformed variant raises ValueError.
    def test_strict_ini_rejects_unknown_fields_nan_and_parent_output(self):
        fixture = self.fixture
        config_path = fixture.root / 'classifier.ini'
        def write(values):
            config_path.write_text('\n\n'.join('[' + section + ']\n' + '\n'.join(
                key + ' = ' + str(value) for key, value in fields.items())
                for section, fields in values.items()) + '\n')
        write(fixture.c)
        self.assertEqual(exp.load_config(config_path), fixture.c)
        for mutation in ('unknown', 'nan', 'parent_output'):
            with self.subTest(mutation=mutation):
                bad = deepcopy(fixture.c)
                if mutation == 'unknown':
                    bad['training']['unrecognized'] = 7
                elif mutation == 'nan':
                    bad['training']['learning_rate'] = 'nan'
                else:
                    bad['experiment']['output_directory'] = str(fixture.parent / 'new-experiment')
                write(bad)
                with self.assertRaises(ValueError):
                    exp.load_config(config_path)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class ContinuationTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_multiscale_network_v3 as network
        cls.network = network
        cls.tf = network._tensorflow()

    def setUp(self):
        self.tf.keras.backend.clear_session()
        MiniVolume.accessed = []
        ShiftedValidationVolume.returned_mask_value = None

    # Purpose: Verify classifier continuation trains and reloads a frozen locator encoder.
    # Inputs: Saved small TensorFlow locator, synthetic panels, verified proposals, and two training epochs.
    # Outputs: Eight updates use one trace; reload matches, encoder hashes stay fixed, and test IDs stay unread.
    def test_saved_locator_classifier_fit_single_trace_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticSources(directory)
            fixture.output.mkdir()
            (fixture.output / 'input.ini').write_text('synthetic classifier continuation\n')
            locator = self.network.build_locator((16, 16, 16), base_filters=2, seed=42)
            locator.save(fixture.checkpoint)
            original_locator_hash = exp.sha256(fixture.checkpoint)
            encoder_before = learning.weight_fingerprint(locator.get_layer('overview_encoder'))
            proposals = {}
            with patch.object(parent_learning, 'overview', side_effect=overview), \
                 patch.object(parent_learning, 'ContextVolume', ShiftedValidationVolume):
                inputs = parent_learning.Inputs(fixture.parent_config)
                for row in fixture.rows:
                    ov, image = inputs.coarse(row)
                    predicted = self.network.predict_boxes(locator, image)[0]
                    origins, boxes = parent.proposal_origins(predicted, ov, fixture.parent_config)
                    proposals[row['sample_id']] = dict(
                        predicted_box=predicted.tolist(), origins_kji=origins,
                        proposal_boxes=boxes.tolist(), coverage_passed=True)
            lock = dict(parent_config=fixture.parent_config, samples=fixture.rows,
                        proposals=proposals, reserved_test_sample_ids=['test:never_open'],
                        locator_checkpoint=dict(path=str(fixture.checkpoint),
                            sha256=original_locator_hash, selected_epoch=2))
            with ExitStack() as stack:
                stack.enter_context(patch.object(exp, 'inspect_lock', return_value=lock))
                stack.enter_context(patch.object(parent_learning, 'overview', side_effect=overview))
                stack.enter_context(patch.object(parent_learning, 'ContextVolume', ShiftedValidationVolume))
                stack.enter_context(patch.object(parent_learning, 'configure_tensorflow',
                                                return_value=(self.tf, self.network)))
                stack.enter_context(patch.object(learning, '_render'))
                forbidden = stack.enter_context(patch.object(self.network, 'build_locator',
                    side_effect=AssertionError('Continuation must not construct/train a locator')))
                forbidden_fit = stack.enter_context(patch.object(parent_learning, 'fit',
                    side_effect=AssertionError('Continuation must not retrain the locator')))
                result = learning.fit(fixture.c, lock)
                forbidden.assert_not_called()
                forbidden_fit.assert_not_called()
                self.assertTrue(result['classifier_trained'])
                self.assertFalse(result['locator_retrained'])
                self.assertFalse(result['test_evaluated'])
                self.assertTrue(result['overview_encoder_unchanged'])
                self.assertEqual(result['optimizer_steps'], 8)
                self.assertEqual(result['training_traces']['classifier'], {'train': 1, 'predict': 1})
                self.assertEqual(result['training_traces']['selected_classifier'], {'train': 0, 'predict': 1})
                loaded = self.network.load_model(fixture.output / 'model/classifier.keras')
                encoder = loaded.get_layer('overview_encoder')
                self.assertFalse(encoder.trainable)
                self.assertEqual(encoder.trainable_variables, [])
                self.assertEqual(learning.weight_fingerprint(encoder), encoder_before)
                actual = parent_learning._score_classifier(loaded, fixture.rows,
                    parent_learning.Inputs(exp.runtime_config(fixture.c, lock)), proposals)
                np.testing.assert_allclose([r['probability'] for r in actual],
                    [r['probability'] for r in result['predictions']], atol=1e-6)
                np.testing.assert_allclose([r['score_prediction'] for r in actual],
                    [r['score_prediction'] for r in result['predictions']], atol=1e-6)
            self.assertEqual(exp.sha256(fixture.checkpoint), original_locator_hash)
            audit = exp.checked_json(fixture.output / 'continuation-audit.json')
            self.assertEqual(audit['locator_weights_before_sha256'], audit['locator_weights_after_sha256'])
            self.assertEqual(audit['overview_encoder_before_sha256'], audit['overview_encoder_after_sha256'])
            history = exp.checked_json(fixture.output / 'classifier-history.json')
            best = min(history['epochs'], key=lambda row: row['checkpoint_selection_bce'])
            self.assertEqual(result['selected_epoch'], best['epoch'])
            for epoch in history['epochs']:
                self.assertEqual(set(epoch['training_sample_order']),
                    {row['sample_id'] for row in fixture.rows if row['split'] == 'train'})
            self.assertNotIn('test:never_open', MiniVolume.accessed)
            self.assertTrue(exp.checked_json(fixture.output / 'proposal-replay.json')['passed'])


if __name__ == '__main__':
    unittest.main()
