"""Focused-classifier contract tests using metadata and tiny text fixtures.

No phantom images or reserved test payloads are opened. These tests protect the
matched sibling diagnostic and frozen continuation boundaries, rather than
testing arbitrary configuration implementation details.
"""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import fakect_coa_focused_experiment as exp
import fakect_coa_focused_learning as learning
from fakect_multiscale_experiment import SCORE
from test_multiscale_experiment_review import MiniVolume


def fixture_contract(root):
    """Same 32/8 balanced panel shape, without real cohort filenames/images."""
    root = Path(root)
    cases = [('260613', 'young', 'train'), ('260618', 'young', 'train'),
             ('260616', 'older', 'train'), ('260621', 'older', 'train'),
             ('260614', 'middle', 'train'), ('260619', 'middle', 'train'),
             ('260615', 'adolescent', 'train'), ('260620', 'adolescent', 'train'),
             ('260602', 'adult', 'validation'), ('260611', 'adult', 'validation')]
    rows, proposals = [], {}
    for case, family, split in cases:
        for role, score in [('original', 0.), ('edited_negative', .1),
                            ('lower_positive', .25), ('higher_positive', .55)]:
            sample_id = case + ':' + role
            baseline = role == 'original'
            companions = {} if baseline else {'arch_outer_expand': {'distance_mm': 2.}}
            parameters = {} if baseline else {**deepcopy(companions),
                                              'coa_narrow': {'distance_mm': score * 10}}
            row = {'sample_id': sample_id, 'case_id': case, 'anatomy_family': family,
                   'split': split, 'pilot_role': role, 'baseline': baseline,
                   'parameters': parameters, 'pilot_companion_parameters': companions,
                   'source_atn': {'path': str(root / (case + '.bin'))},
                   'original_reference_mask_sha256': case + '-reference',
                   'annotation': {'sample_id': sample_id, 'case_id': case,
                                  'anatomy_family': family, 'coa_benchmark_positive': score >= .2,
                                  SCORE: score, 'errors': []}}
            rows.append(row)
            proposals[sample_id] = {'sample_id': sample_id, 'split': split,
                                   'origins_kji': [[8, 10, 12], [20, 10, 12]],
                                   'predicted_box': [.5, .5, .5, .4, .2, .2],
                                   'proposal_boxes': [[.4, .5, .5, .4, .2, .2],
                                                      [.6, .5, .5, .4, .2, .2]],
                                   'coverage_passed': True}
    dependency = root / 'locator.keras'
    dependency.write_bytes(b'immutable synthetic locator checkpoint')
    families = {}
    for case, family, _ in cases:
        families.setdefault(family, []).append(case)
    families['infant'] = ['260612', '260617']
    return {'parent_config': {
                'experiment': {'output_directory': str(root / 'parent')},
                'input.overview': {'shape_kji': [16, 16, 16]},
                'input.detail': {'shape_kji': [32, 32, 32],
                                 'spacing_ijk_mm': [.6, .6, .6], 'proposals': 2},
                'model': {'base_filters': 2, 'seed': 7, 'learning_rate': .0001,
                          'clip_min': 0., 'clip_max': .5},
                'pilot': {'classifier_epochs': 1, 'require_gpu': False,
                          'gpu_memory_limit_mb': 1024},
                'training': {'host_memory_limit_mb': 2048, 'maximum_traces': 1}},
            'samples': rows, 'proposals': proposals,
            'reserved_test_sample_ids': ['260612:never_open', '260617:never_open'],
            'pilot_counts': {'train': {'samples': 32, 'positive': 16},
                             'validation': {'samples': 8, 'positive': 4}},
            'design': {'families': families, 'folds': [{'family_assignment': {
                'young': 'train', 'older': 'train', 'middle': 'train',
                'adolescent': 'train', 'adult': 'validation', 'infant': 'test'}}]},
            'annotation_acceptance': {'threshold': .2, 'clinical_review': False,
                                     'use': 'accepted_engineering_proxy'},
            'locator_checkpoint': {'path': str(dependency), 'selected_epoch': 99,
                                   'sha256': hashlib.sha256(dependency.read_bytes()).hexdigest()},
            'dependency_sha256': {str(dependency): hashlib.sha256(dependency.read_bytes()).hexdigest()}}


def fixture_config(root):
    root = Path(root)
    return {'experiment': {'schema_version': 'fakect.coa-focused/1',
                           'output_directory': str(root / 'focused')},
            'locator': {'experiment_directory': str(root / 'parent'),
                        'review_directory': str(root / 'review')},
            'diagnostic': {'case_ids': ['260613', '260616'], 'epochs': 40,
                           'learning_rate': .001, 'cache_limit_mb': 2048,
                           'minimum_probability_gap': .3, 'maximum_bce': .3},
            'training': {'epochs': 10, 'learning_rate': .001, 'seed': 42,
                         'host_memory_limit_mb': 12288, 'gpu_memory_limit_mb': 12288,
                         'require_gpu': True},
            'model': {'base_filters': 4, 'regression_weight': 1.}}


def write_ini(path, config):
    sections = []
    for section, fields in config.items():
        lines = ['[' + section + ']']
        for key, value in fields.items():
            if isinstance(value, list):
                value = ','.join(map(str, value))
            if isinstance(value, bool):
                value = str(value).lower()
            lines.append(key + ' = ' + str(value))
        sections.append('\n'.join(lines))
    Path(path).write_text('\n\n'.join(sections) + '\n')


class FocusedContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = fixture_config(self.root)
        self.contract = fixture_contract(self.root)

    def test_diagnostic_uses_matched_training_siblings_and_keeps_parent_panel(self):
        before = deepcopy(self.contract)
        selected = exp.select_diagnostic(self.config, self.contract)
        self.assertEqual(len(selected), 4)
        self.assertEqual({r['case_id'] for r in selected}, {'260613', '260616'})
        self.assertEqual({r['pilot_role'] for r in selected},
                         {'edited_negative', 'higher_positive'})
        self.assertTrue(all(r['split'] == 'train' and not r['baseline'] for r in selected))
        self.assertEqual(self.contract, before)
        self.assertEqual(len(self.contract['samples']), 40)

    def test_validation_reserved_test_unknown_and_repeated_cases_rejected(self):
        for cases in (['260602'], ['260612'], ['999999'], ['260613', '260613']):
            with self.subTest(cases=cases):
                config = deepcopy(self.config)
                config['diagnostic']['case_ids'] = cases
                with self.assertRaises(ValueError):
                    exp.select_diagnostic(config, self.contract)

    def test_missing_duplicate_or_relabelled_diagnostic_role_rejected(self):
        for mutation in ('missing', 'duplicate', 'label'):
            with self.subTest(mutation=mutation):
                contract = deepcopy(self.contract)
                negative = next(r for r in contract['samples']
                                if r['sample_id'] == '260613:edited_negative')
                if mutation == 'missing':
                    contract['samples'].remove(negative)
                elif mutation == 'duplicate':
                    contract['samples'].append(deepcopy(negative))
                else:
                    negative['annotation']['coa_benchmark_positive'] = True
                with self.assertRaises(ValueError):
                    exp.select_diagnostic(self.config, contract)

    def test_score_and_class_must_agree_with_frozen_engineering_threshold(self):
        changes = [('edited_negative', .3, False), ('higher_positive', .1, True),
                   ('higher_positive', float('nan'), True)]
        for role, score, positive in changes:
            with self.subTest(role=role, score=score):
                contract = deepcopy(self.contract)
                row = next(r for r in contract['samples'] if r['sample_id'] == '260613:' + role)
                row['annotation'].update({SCORE: score, 'coa_benchmark_positive': positive})
                with self.assertRaises(ValueError):
                    exp.select_diagnostic(self.config, contract)

    def test_companion_edits_and_reviewed_sampling_must_match_within_pair(self):
        for mutation in ('declared_companions', 'actual_companions', 'crop_origin'):
            with self.subTest(mutation=mutation):
                contract = deepcopy(self.contract)
                row = next(r for r in contract['samples']
                           if r['sample_id'] == '260613:higher_positive')
                if mutation == 'declared_companions':
                    row['pilot_companion_parameters']['arch_outer_expand']['distance_mm'] = 4.
                elif mutation == 'actual_companions':
                    row['parameters']['arch_outer_expand']['distance_mm'] = 4.
                else:
                    contract['proposals'][row['sample_id']]['origins_kji'][0][0] += 1
                with self.assertRaises(ValueError):
                    exp.select_diagnostic(self.config, contract)

    def test_ini_roundtrip_keeps_user_facing_values(self):
        path = self.root / 'focused.ini'
        write_ini(path, self.config)
        self.assertEqual(exp.load_config(path), self.config)

    def test_ini_rejects_unknown_fields_sections_and_defaults(self):
        path = self.root / 'focused.ini'
        for mutation in ('field', 'section', 'defaults'):
            with self.subTest(mutation=mutation):
                config = deepcopy(self.config)
                if mutation == 'field':
                    config['diagnostic']['new_unchecked_parameter'] = 1
                elif mutation == 'section':
                    config['new_section'] = {'x': 1}
                write_ini(path, config)
                if mutation == 'defaults':
                    path.write_text('[DEFAULT]\nunchecked = 1\n\n' + path.read_text())
                with self.assertRaises(ValueError):
                    exp.load_config(path)

    def test_ini_rejects_nonfinite_or_invalid_resource_and_learning_values(self):
        path = self.root / 'focused.ini'
        changes = [('diagnostic', 'epochs', 0), ('diagnostic', 'learning_rate', 'nan'),
                   ('diagnostic', 'cache_limit_mb', 0),
                   ('diagnostic', 'minimum_probability_gap', 1.1),
                   ('diagnostic', 'maximum_bce', 0),
                   ('training', 'epochs', 0), ('training', 'learning_rate', 'inf'),
                   ('training', 'host_memory_limit_mb', 0),
                   ('training', 'gpu_memory_limit_mb', 0),
                   ('model', 'base_filters', 0), ('model', 'regression_weight', -1)]
        for section, key, value in changes:
            with self.subTest(section=section, key=key, value=value):
                config = deepcopy(self.config)
                config[section][key] = value
                write_ini(path, config)
                with self.assertRaises(ValueError):
                    exp.load_config(path)

    def test_output_cannot_be_parent_or_nested_under_review(self):
        path = self.root / 'focused.ini'
        for out in (self.root / 'parent', self.root / 'parent/new',
                    self.root / 'review', self.root / 'review/new'):
            with self.subTest(out=out):
                config = deepcopy(self.config)
                config['experiment']['output_directory'] = str(out)
                write_ini(path, config)
                with self.assertRaises(ValueError):
                    exp.load_config(path)

    def test_plan_retains_full_panel_and_cache_budget_is_enforced(self):
        with patch.object(exp.classifier, 'source_contract', return_value=self.contract):
            document = exp.plan(self.config)
        self.assertEqual(document['samples'], self.contract['samples'])
        self.assertEqual(document['pilot_counts'], self.contract['pilot_counts'])
        self.assertEqual(set(document['diagnostic_sample_ids']), {
            '260613:edited_negative', '260613:higher_positive',
            '260616:edited_negative', '260616:higher_positive'})
        self.assertEqual(document['diagnostic_counts']['samples'], 4)
        self.assertEqual(document['diagnostic_counts']['positive'], 2)
        self.assertFalse(document['training_contract']['reserved_test_evaluated'])
        self.assertEqual(document['training_contract']['feature_spacing_ijk_mm'], [2.4, 2.4, 2.4])
        oversized = deepcopy(self.contract)
        oversized['parent_config']['input.detail']['shape_kji'] = [1024, 1024, 1024]
        with patch.object(exp.classifier, 'source_contract', return_value=oversized):
            with self.assertRaisesRegex(ValueError, '[Cc]ache|RAM'):
                exp.plan(self.config)

    def test_runtime_keeps_physical_geometry_and_applies_classifier_settings(self):
        before = deepcopy(self.contract)
        runtime = exp.runtime_config(self.config, self.contract)
        self.assertEqual(runtime['input.detail'], self.contract['parent_config']['input.detail'])
        self.assertEqual(runtime['input.overview'], self.contract['parent_config']['input.overview'])
        self.assertEqual(runtime['model']['base_filters'], 4)
        self.assertEqual(runtime['model']['seed'], 42)
        self.assertEqual(runtime['pilot']['classifier_epochs'], 10)
        self.assertEqual(runtime['experiment']['output_directory'], self.config['experiment']['output_directory'])
        self.assertEqual(self.contract, before)

    def test_frozen_sources_snapshots_input_and_dependencies_cannot_change(self):
        runtime = self.root / 'src/runtime.py'
        runtime.parent.mkdir()
        runtime.write_text('FROZEN_RUNTIME = 1\n')
        data = b'synthetic focused config\n'
        with patch.object(exp, 'ROOT', self.root), \
             patch.object(exp, 'CODE_FILES', ('src/runtime.py',)), \
             patch.object(exp.classifier, 'source_contract', return_value=self.contract):
            lock = exp.freeze(self.config, data)
            self.assertEqual(exp.inspect_lock(self.config, data), lock)
            with self.assertRaises(FileExistsError):
                exp.freeze(self.config, data)
            output = Path(self.config['experiment']['output_directory'])
            paths = [runtime, output / 'source/src/runtime.py', output / 'input.ini',
                     self.root / 'locator.keras']
            for path in paths:
                with self.subTest(path=path):
                    original = path.read_bytes()
                    path.write_bytes(original + b'changed\n')
                    try:
                        with self.assertRaisesRegex(ValueError, '[Cc]hanged'):
                            exp.inspect_lock(self.config, data)
                    finally:
                        path.write_bytes(original)
                    self.assertEqual(exp.inspect_lock(self.config, data), lock)
            with self.assertRaises(ValueError):
                exp.inspect_lock(self.config, data + b'changed caller input\n')

    def _stage_fixture(self):
        output = Path(self.config['experiment']['output_directory'])
        output.mkdir()
        (output / 'experiment-lock.json').write_text('immutable focused experiment\n')
        folder = output / 'diagnostic'
        (folder / 'model').mkdir(parents=True)
        checkpoint = folder / 'model/classifier.keras'
        checkpoint.write_bytes(b'synthetic diagnostic weights')
        selected = exp.select_diagnostic(self.config, self.contract)
        rows = []
        for row in selected:
            label = int(row['annotation']['coa_benchmark_positive'])
            rows.append({'sample_id': row['sample_id'], 'case_id': row['case_id'],
                         'split': 'train', 'family': row['anatomy_family'],
                         'role': row['pilot_role'], 'label': label,
                         'score_label': row['annotation'][SCORE],
                         'score_prediction': row['annotation'][SCORE],
                         'probability': .95 if label else .05})
        result = {'stage': 'diagnostic', 'status': 'diagnostic_passed',
                  'source_lock_sha256': exp.sha256(output / 'experiment-lock.json'),
                  'test_evaluated': False, 'locator_unchanged': True,
                  'selected_epoch': 3,
                  'checkpoint_sha256': {'classifier.keras': exp.sha256(checkpoint)},
                  'learning_gate': {'passed': True},
                  'selected_checkpoint_prediction_parity': {'passed': True},
                  'predictions': rows}
        predictions = {'predictions': rows, 'checkpoint_sha256': exp.sha256(checkpoint),
                       'selected_epoch': 3, 'test_evaluated': False}
        return output, result, predictions

    def _write_diagnostic(self, output, result, predictions):
        exp.write_json(output / 'diagnostic/results.json', result)
        exp.write_json(output / 'diagnostic/classifier-predictions.json', predictions)

    def test_diagnostic_gate_binds_four_matched_labels_and_checkpoint(self):
        output, result, predictions = self._stage_fixture()
        self._write_diagnostic(output, result, predictions)
        dependencies = exp.require_diagnostic(self.config, self.contract)
        self.assertEqual(len(dependencies['dependency_sha256']), 3)
        self.assertEqual(dependencies['source_lock_sha256'],
                         exp.sha256(output / 'experiment-lock.json'))
        (output / 'diagnostic/model/classifier.keras').write_bytes(b'other model')
        with self.assertRaisesRegex(ValueError, '[Cc]heckpoint'):
            exp.require_diagnostic(self.config, self.contract)

    def test_diagnostic_summary_cannot_override_failed_case_or_label_join(self):
        output, valid_result, valid_predictions = self._stage_fixture()
        changes = [('probability', .51), ('label', 0), ('score_label', .25),
                   ('split', 'validation'), ('case_id', '260602'),
                   ('family', 'adult'), ('role', 'original')]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                result = deepcopy(valid_result)
                # Change a positive record, retaining the global approved flag.
                row = next(r for r in result['predictions'] if r['label'] == 1)
                row[key] = value
                predictions = deepcopy(valid_predictions)
                predictions['predictions'] = result['predictions']
                self._write_diagnostic(output, result, predictions)
                with self.assertRaises(ValueError):
                    exp.require_diagnostic(self.config, self.contract)

    def test_diagnostic_gate_rejects_wrong_source_missing_records_and_approval_flags(self):
        output, valid_result, valid_predictions = self._stage_fixture()
        for mutation in ('source', 'missing', 'duplicate', 'test', 'locator', 'parity', 'gate'):
            with self.subTest(mutation=mutation):
                result = deepcopy(valid_result)
                if mutation == 'source':
                    result['source_lock_sha256'] = '0' * 64
                elif mutation == 'missing':
                    result['predictions'].pop()
                elif mutation == 'duplicate':
                    result['predictions'][-1] = deepcopy(result['predictions'][0])
                elif mutation == 'test':
                    result['test_evaluated'] = True
                elif mutation == 'locator':
                    result['locator_unchanged'] = False
                elif mutation == 'parity':
                    result['selected_checkpoint_prediction_parity']['passed'] = False
                else:
                    result['learning_gate']['passed'] = False
                predictions = deepcopy(valid_predictions)
                predictions['predictions'] = result['predictions']
                self._write_diagnostic(output, result, predictions)
                with self.assertRaises(ValueError):
                    exp.require_diagnostic(self.config, self.contract)

    def test_full_size_smoke_is_architecture_and_lock_specific(self):
        output, _, _ = self._stage_fixture()
        detail = self.contract['parent_config']['input.detail']
        valid = {'status': 'passed', 'checkpoint_parity': True,
                 'source_lock_sha256': exp.sha256(output / 'experiment-lock.json'),
                 'detail_shape_kji': detail['shape_kji'], 'proposals': detail['proposals'],
                 'traces': {'train': 1, 'predict': 1}, 'gpu_visible': True}
        exp.write_json(output / 'runtime-smoke.json', valid)
        self.assertEqual(exp.require_smoke(self.config, self.contract), valid)
        changes = [('source_lock_sha256', '0' * 64), ('status', 'failed'),
                   ('checkpoint_parity', False), ('detail_shape_kji', [16, 16, 16]),
                   ('proposals', 1), ('traces', {'train': 2, 'predict': 1}),
                   ('gpu_visible', False)]
        for key, value in changes:
            with self.subTest(key=key):
                exp.write_json(output / 'runtime-smoke.json', {**valid, key: value})
                with self.assertRaises(ValueError):
                    exp.require_smoke(self.config, self.contract)

    def test_failed_or_incomplete_diagnostic_cannot_authorize_full_fit(self):
        output, result, predictions = self._stage_fixture()
        self._write_diagnostic(output, result, predictions)
        for name in ('failure.json', 'INCOMPLETE'):
            with self.subTest(marker=name):
                marker = output / 'diagnostic' / name
                marker.touch()
                try:
                    with self.assertRaises(ValueError):
                        exp.require_diagnostic(self.config, self.contract)
                finally:
                    marker.unlink()

    def test_gate_requires_each_case_to_learn_instead_of_pooled_average(self):
        _, result, _ = self._stage_fixture()
        rows = result['predictions']
        self.assertTrue(learning.diagnostic_gate(self.config, rows)['passed'])
        bad_pair = deepcopy(rows)
        for row in bad_pair:
            if row['case_id'] == '260616':
                row['probability'] = .65 if row['label'] else .25
        # Both decisions and the .4 gap pass. Only this case's BCE exceeds .3;
        # pooled BCE is still below .3 because the other case is well learned.
        self.assertFalse(learning.diagnostic_gate(self.config, bad_pair)['passed'])
        for mutation in ('wrong_split', 'missing', 'duplicate_class'):
            with self.subTest(mutation=mutation):
                invalid = deepcopy(rows)
                if mutation == 'wrong_split':
                    invalid[0]['split'] = 'validation'
                elif mutation == 'missing':
                    invalid.pop()
                else:
                    invalid[1]['label'] = invalid[0]['label']
                with self.assertRaises(ValueError):
                    learning.diagnostic_gate(self.config, invalid)

    def test_cache_is_bounded_readonly_image_only_and_excludes_test_payloads(self):
        runtime = deepcopy(self.contract['parent_config'])
        runtime['input.detail']['spacing_ijk_mm'] = [1., 1., 1.]
        proposal = {'origins_kji': [[-8, -8, -8], [-4, -8, -8]]}
        rows = deepcopy(self.contract['samples'])
        for row in rows:
            row['test_removed'] = 4
        MiniVolume.accessed = []
        self.addCleanup(setattr, MiniVolume, 'returned_mask_value', None)
        with patch.object(learning, 'ContextVolume', MiniVolume):
            inputs = learning.FineInputs(runtime, .5)
            first = inputs.classifier(rows[1], proposal)
            self.assertEqual(set(first), {'detail_0', 'detail_1'})
            self.assertTrue(all(value.dtype == np.float32 and not value.flags.writeable
                                for value in first.values()))
            self.assertIs(inputs.classifier(rows[1], proposal), first)
            inputs.classifier(rows[2], proposal)
            self.assertEqual(inputs.stats()['samples_resident'], 1)
            self.assertEqual(inputs.stats()['loads'], 2)
            self.assertEqual(inputs.stats()['hits'], 1)
            self.assertLessEqual(inputs.stats()['peak_mib'], inputs.stats()['limit_mib'])
            # Targets returned by the reader never enter predictor channels.
            MiniVolume.returned_mask_value = 1
            altered = inputs.classifier(rows[1], proposal, cache=False)
            for key in first:
                np.testing.assert_array_equal(first[key], altered[key])
            before = list(MiniVolume.accessed)
            with self.assertRaisesRegex(ValueError, '[Tt]est'):
                inputs.classifier({**rows[1], 'split': 'test'}, proposal)
            self.assertEqual(MiniVolume.accessed, before)
            with self.assertRaises(ValueError):
                inputs.classifier(rows[1], {'origins_kji': [[-.5, 0., 0.], [0., 0., 0.]]})
            too_small = learning.FineInputs(runtime, .1)
            with self.assertRaisesRegex(ValueError, '[Bb]udget'):
                too_small.classifier(rows[1], proposal)

    def test_signal_difference_audit_rejects_invalid_arrays(self):
        before = np.zeros((2, 2, 2), np.float32)
        after = before.copy()
        after[0, 0, 0] = 2
        summary = learning.difference_summary(before, after)
        self.assertEqual(summary['shape'], [2, 2, 2])
        self.assertEqual(summary['different_fraction'], .125)
        self.assertEqual(summary['maximum_absolute_difference'], 2.)
        self.assertAlmostEqual(summary['rms_difference'], np.sqrt(.5))
        with self.assertRaises(ValueError):
            learning.difference_summary(before, np.zeros((2, 2)))
        after[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            learning.difference_summary(before, after)

    def test_fit_binds_verified_lock_and_rejects_test_before_reading(self):
        output = Path(self.config['experiment']['output_directory'])
        output.mkdir()
        (output / 'input.ini').write_text('synthetic config\n')
        caller = deepcopy(self.contract)
        caller['diagnostic_sample_ids'] = ['260613:edited_negative']
        exp.write_json(output / 'experiment-lock.json', caller)
        verified = deepcopy(caller)
        verified['samples'][1]['split'] = 'test'
        with patch.object(exp, 'inspect_lock', return_value=verified), \
             patch.object(exp, 'require_smoke'), \
             patch.object(learning.legacy, 'configure_tensorflow') as configure, \
             patch.object(learning, 'ContextVolume') as reader:
            with self.assertRaises(ValueError):
                learning.fit(self.config, caller, diagnostic=True)
            configure.assert_not_called()
            reader.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class FocusedTensorFlowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_coa_focused_network as network
        import fakect_multiscale_network_v3 as locator_network
        cls.network = network
        cls.locator_network = locator_network
        cls.tf = network._tensorflow()

    def setUp(self):
        self.tf.keras.backend.clear_session()
        MiniVolume.accessed = []
        MiniVolume.returned_mask_value = None

    def test_diagnostic_reads_only_four_training_siblings_and_reloads_exactly(self):
        from test_multiscale_v3_integration import config as parent_config
        from test_multiscale_experiment_review import mini_overview
        import fakect_multiscale_experiment_v3 as geometry
        def stable_overview(row, supplied_config):
            # This synthetic coarse view is insensitive to the focal lesion,
            # while native fine images retain their differing removed voxels.
            # It makes the diagnostic pair use identical image-only origins.
            return mini_overview({**row, 'test_removed': 4}, supplied_config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture_config(root)
            config['training']['require_gpu'] = False
            config['model']['base_filters'] = 2
            config['diagnostic']['epochs'] = 2
            contract = fixture_contract(root)
            contract['parent_config'] = parent_config(root / 'parent')
            for row in contract['samples']:
                row['test_removed'] = {'original': 0, 'edited_negative': 4,
                                       'lower_positive': 12, 'higher_positive': 16}[row['pilot_role']]
            output = Path(config['experiment']['output_directory'])
            output.mkdir()
            (output / 'input.ini').write_text('synthetic diagnostic config\n')
            locator = self.locator_network.build_locator((16, 16, 16), base_filters=2, seed=42)
            checkpoint = Path(contract['locator_checkpoint']['path'])
            locator.save(checkpoint)
            locator_hash = exp.sha256(checkpoint)
            contract['locator_checkpoint']['sha256'] = locator_hash
            contract['diagnostic_sample_ids'] = [r['sample_id'] for r in contract['samples']
                if r['case_id'] in config['diagnostic']['case_ids']
                and r['pilot_role'] in ('edited_negative', 'higher_positive')]
            selected = [r for r in contract['samples'] if r['sample_id'] in contract['diagnostic_sample_ids']]
            # Build the reviewed record from the actual saved locator and the
            # same synthetic image reader. Replay itself is not mocked.
            with patch.object(learning.legacy, 'overview', side_effect=stable_overview):
                inputs = learning.legacy.Inputs(contract['parent_config'])
                for row in selected:
                    ov, image = inputs.coarse(row)
                    predicted = self.locator_network.predict_boxes(locator, image)[0]
                    origins, boxes = geometry.proposal_origins(predicted, ov, contract['parent_config'])
                    contract['proposals'][row['sample_id']] = {
                        'predicted_box': predicted.tolist(), 'origins_kji': origins,
                        'proposal_boxes': boxes.tolist(), 'coverage_passed': True}
            self.assertEqual(len(exp.select_diagnostic(config, contract)), 4)
            exp.write_json(output / 'experiment-lock.json', contract)
            MiniVolume.accessed = []
            with ExitStack() as stack:
                stack.enter_context(patch.object(exp, 'inspect_lock', return_value=contract))
                stack.enter_context(patch.object(exp, 'require_smoke', return_value={'status': 'passed'}))
                stack.enter_context(patch.object(learning.legacy, 'overview', side_effect=stable_overview))
                stack.enter_context(patch.object(learning, 'ContextVolume', MiniVolume))
                stack.enter_context(patch.object(learning.legacy, 'configure_tensorflow',
                                                return_value=(self.tf, self.locator_network)))
                stack.enter_context(patch.object(learning, '_render'))
                forbidden = stack.enter_context(patch.object(self.locator_network, 'build_locator',
                    side_effect=AssertionError('The focused classifier cannot retrain its locator')))
                result = learning.fit(config, contract, diagnostic=True)
                forbidden.assert_not_called()
            self.assertEqual(set(MiniVolume.accessed), set(contract['diagnostic_sample_ids']))
            self.assertFalse(any('260612' in item or '260617' in item for item in MiniVolume.accessed))
            self.assertEqual(result['optimizer_steps'], 8)
            self.assertEqual(result['epochs_completed'], 2)
            self.assertTrue(result['locator_unchanged'])
            self.assertFalse(result['test_evaluated'])
            self.assertIsNone(result['validation'])
            self.assertEqual(result['training_traces']['classifier'], {'train': 1, 'predict': 1})
            self.assertEqual(result['training_traces']['selected_classifier'], {'train': 0, 'predict': 1})
            self.assertTrue(result['selected_checkpoint_prediction_parity']['passed'])
            self.assertEqual(exp.sha256(checkpoint), locator_hash)
            folder = output / 'diagnostic'
            replay = exp.checked_json(folder / 'proposal-replay.json')
            self.assertTrue(replay['passed'])
            self.assertEqual({row['sample_id'] for row in replay['samples']}, set(contract['diagnostic_sample_ids']))
            history = exp.checked_json(folder / 'classifier-history.json')
            self.assertEqual(result['selected_epoch'],
                             min(history['epochs'], key=lambda row: row['checkpoint_selection_bce'])['epoch'])
            for epoch in history['epochs']:
                self.assertEqual(set(epoch['training_sample_order']), set(contract['diagnostic_sample_ids']))
            self.assertEqual(len(exp.checked_json(folder / 'sibling-sensitivity-selected.json')['pairs']), 2)
            with self.assertRaises(FileExistsError):
                with patch.object(exp, 'inspect_lock', return_value=contract), \
                     patch.object(exp, 'require_smoke'):
                    learning.fit(config, contract, diagnostic=True)


if __name__ == '__main__':
    unittest.main()
