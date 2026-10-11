"""Classification loss ablation preserves frozen panels and parent provenance.

Synthetic completed parent artifacts exercise real checksum, history and role
validation. TensorFlow and external source-inventory checks are replaced only
at their boundaries; no cohort image payload exists in these fixtures.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import fakect_coa_classification_experiment as exp
import fakect_multiscale_experiment as base
from test_coa_budget_experiment import ParentFixture, INITIAL, predictions


class ClassificationFixture(ParentFixture):
    def complete_comparison(self):
        """Materialize a separate real frozen budget plan and five final runs."""
        self.comparison = self.output
        self.comparison_config = deepcopy(self.c)
        self.comparison_content = self.content
        lock = exp.budget.freeze(self.c, self.content)
        digest = base.sha256(self.comparison / 'experiment-lock.json')
        smoke = base.checked_json(self.parent / 'runtime-smoke.json')
        smoke['source_lock_sha256'] = digest
        smoke['full_cache_memory_probe']['source_lock_sha256'] = digest
        base.write_json(self.comparison / 'runtime-smoke.json', smoke)
        for stage in lock['stages']:
            folder = self.comparison / 'stages' / stage['id']
            ancestor = self.parent / 'stages' / stage['id']
            (folder / 'classifier').mkdir(parents=True)
            (folder / 'final-test').mkdir()
            for name in ('stage-input.json', 'proposal-audit.json', 'final-test/proposal-audit.json',
                         'classifier/initialization.json'):
                shutil.copyfile(ancestor / name, folder / name)
                shutil.copyfile(Path(str(ancestor / name) + '.sha256'),
                                Path(str(folder / name) + '.sha256'))
            checkpoint = folder / 'classifier/classifier.keras'
            checkpoint.write_bytes(b'completed final 200 weights ' + stage['id'].encode())
            receipt = {'checkpoint': str(checkpoint), 'checkpoint_sha256': base.sha256(checkpoint),
                       'selected_epoch': 200, 'epochs_completed': 200, 'optimizer_updates': 4800,
                       'training_sample_ids': stage['train_sample_ids'],
                       'validation_sample_ids': [], 'validation': None,
                       'fresh_initial_weights_sha256': INITIAL}
            base.write_json(folder / 'classifier/receipt.json', receipt)
            order = [stage['train_sample_ids'][i:i + 2] for i in range(0, 48, 2)]
            history = {'selected_epoch': 200, 'checkpoint_selection': 'final fixed-budget epoch',
                       'epochs': [{'epoch': epoch, 'selection_bce': None, 'validation': None,
                                   'optimizer_updates': epoch * 24, 'training_pair_order': order,
                                   'training': {'bce': .1, 'accuracy': 1., 'score_mae': .1}}
                                  for epoch in range(1, 201)]}
            base.write_json(folder / 'classifier/history.json', history)
            rows = [{**r, 'split': 'final_test'} for r in lock['samples']
                    if r['case_id'] in stage['final_test_case_ids']]
            final = predictions(rows)
            base.write_json(folder / 'final-test/predictions.json', {'predictions': final})
            base.write_json(folder / 'results.json', {
                'status': 'completed', 'stage_id': stage['id'], 'protocol': 'family_holdout',
                'source_lock_sha256': digest, 'parent_lock_sha256': lock['parent_lock_sha256'],
                'classifier_selected_epoch': 200, 'classifier': receipt,
                'final_predictions': final, 'reserved_test_evaluated': False,
                'final_test_evaluated': True})
        self.output = self.root / 'classification'
        self.c = {'experiment': {'schema_version': exp.SCHEMA, 'output_directory': str(self.output)},
                  'reference': {'experiment_directory': str(self.parent),
                                'comparison_directory': str(self.comparison)},
                  'training': {'classifier_epochs': 200, 'checkpoint_interval': 50,
                               'regression_weight': 0.}}
        self.content = ('[experiment]\nschema_version = ' + exp.SCHEMA + '\noutput_directory = ' +
                        str(self.output) + '\n[reference]\nexperiment_directory = ' + str(self.parent) +
                        '\ncomparison_directory = ' + str(self.comparison) +
                        '\n[training]\nclassifier_epochs = 200\ncheckpoint_interval = 50\n'
                        'regression_weight = 0\n').encode()
        return lock

    def comparison_folder(self):
        return self.comparison / 'stages' / self.document['stages'][0]['id']

    def update_comparison_receipt(self, change):
        folder = self.comparison_folder()
        receipt = base.checked_json(folder / 'classifier/receipt.json')
        change(receipt)
        base.write_json(folder / 'classifier/receipt.json', receipt)
        result = base.checked_json(folder / 'results.json')
        result['classifier'] = receipt
        base.write_json(folder / 'results.json', result)


class ClassificationExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.f = ClassificationFixture(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(exp.rotation, 'inspect_lock', return_value=self.f.document))
        for module in (exp, exp.budget):
            self.stack.enter_context(patch.object(module, 'ROOT', self.f.code))
            self.stack.enter_context(patch.object(module, 'CODE_FILES', self.f.code_files))
        self.f.complete_comparison()

    def test_plan_preserves_five_folds_panels_crops_and_initialization(self):
        lock = exp.plan(self.f.c)
        self.assertEqual(lock['schema_version'], exp.SCHEMA)
        self.assertEqual((lock['classifier_epochs'], lock['regression_weight'], lock['severity_trained']),
                         (200, 0., False))
        self.assertEqual(lock['unchanged_training_settings'], self.f.parent_c['training'])
        expected = deepcopy(self.f.parent_c['training'])
        expected.update(classifier_epochs=200, regression_weight=0.)
        self.assertEqual(lock['intended_training_settings'], expected)
        comparison = base.checked_json(self.f.comparison / 'experiment-lock.json')
        for field in ('stages', 'samples', 'runtime_geometry', 'fold_manifest', 'reserved_case_ids'):
            self.assertEqual(lock[field], comparison[field])
        self.assertEqual(lock['reserved_case_ids'], ['260612', '260617'])
        self.assertEqual(len(lock['stages']), 5)
        for stage in lock['stages']:
            self.assertEqual((len(stage['train_sample_ids']), len(stage['final_test_sample_ids'])), (48, 12))
            self.assertEqual(stage['expected_initial_weights_sha256'], INITIAL)
            self.assertFalse(set(stage['train_case_ids']) & set(stage['final_test_case_ids']))
            self.assertFalse(set(stage['train_case_ids'] + stage['final_test_case_ids']) & {'260612', '260617'})
        for path in (self.f.comparison / 'experiment-lock.json',
                     self.f.comparison_folder() / 'classifier/history.json',
                     self.f.stage_folder() / 'final-test/proposal-audit.json'):
            self.assertEqual(lock['dependency_sha256'][str(path)], base.sha256(path))

    def test_config_enforces_zero_loss_weight_and_fixed_budget(self):
        path = self.f.root / 'classification.ini'
        path.write_bytes(self.f.content)
        self.assertEqual(exp.load_config(path), self.f.c)
        mutations = ((b'regression_weight = 0', b'regression_weight = 1'),
                     (b'regression_weight = 0', b'regression_weight = nan'),
                     (b'classifier_epochs = 200', b'classifier_epochs = 100'),
                     (b'checkpoint_interval = 50', b'checkpoint_interval = 30'))
        for old, new in mutations:
            with self.subTest(new=new):
                path.write_bytes(self.f.content.replace(old, new))
                with self.assertRaises(ValueError):
                    exp.load_config(path)
        path.write_bytes(self.f.content + b'learning_rate = .03\n')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            exp.load_config(path)

    def test_config_protects_parent_and_comparison_directories(self):
        path = self.f.root / 'classification.ini'
        for output in (self.f.parent, self.f.parent / 'nested', self.f.comparison,
                       self.f.comparison / 'nested', self.f.root):
            with self.subTest(output=str(output)):
                path.write_bytes(self.f.content.replace(str(self.f.output).encode(), str(output).encode()))
                with self.assertRaisesRegex(ValueError, 'separate'):
                    exp.load_config(path)

    def test_comparison_must_be_the_completed_supervised_200_epoch_control(self):
        path = self.f.comparison / 'experiment-lock.json'
        original = base.checked_json(path)
        for change in (lambda v: v.update(schema_version='wrong/1'),
                       lambda v: v.update(classifier_epochs=100),
                       lambda v: v.update(parent_directory=str(self.f.root / 'other')),
                       lambda v: v['unchanged_training_settings'].update(regression_weight=0.)):
            with self.subTest(change=change):
                altered = deepcopy(original)
                change(altered)
                base.write_json(path, altered)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    exp.plan(self.f.c)
                base.write_json(path, original)

    def test_missing_or_failed_comparison_fold_blocks_plan(self):
        folder = self.f.comparison_folder()
        for name in ('INCOMPLETE', 'failure.json'):
            with self.subTest(marker=name):
                marker = folder / name
                marker.touch()
                with self.assertRaisesRegex(ValueError, 'finished cleanly'):
                    exp.plan(self.f.c)
                marker.unlink()

    def test_comparison_training_selection_and_weight_identity_are_checked(self):
        folder = self.f.comparison_folder()
        old_receipt = base.checked_json(folder / 'classifier/receipt.json')
        old_result = base.checked_json(folder / 'results.json')
        mutations = ({'selected_epoch': 100}, {'epochs_completed': 100}, {'optimizer_updates': 2400},
                     {'validation_sample_ids': ['260612:baseline']}, {'validation': {}},
                     {'fresh_initial_weights_sha256': 'b' * 64},
                     {'training_sample_ids': old_receipt['training_sample_ids'][:-1]})
        for fields in mutations:
            with self.subTest(fields=fields):
                self.f.update_comparison_receipt(lambda v: v.update(fields))
                with self.assertRaisesRegex(ValueError, 'completion, initialization'):
                    exp.plan(self.f.c)
                base.write_json(folder / 'classifier/receipt.json', old_receipt)
                base.write_json(folder / 'results.json', old_result)

    def test_incomplete_or_heldout_selection_history_blocks_plan(self):
        path = self.f.comparison_folder() / 'classifier/history.json'
        original = base.checked_json(path)
        for mutation in ('missing_epoch', 'selection', 'heldout_pair', 'updates'):
            with self.subTest(mutation=mutation):
                value = deepcopy(original)
                if mutation == 'missing_epoch':
                    value['epochs'].pop()
                elif mutation == 'selection':
                    value['epochs'][0]['selection_bce'] = .1
                elif mutation == 'heldout_pair':
                    value['epochs'][0]['training_pair_order'][0][0] = '260612:baseline'
                else:
                    value['epochs'][0]['optimizer_updates'] = 25
                base.write_json(path, value)
                with self.assertRaises(ValueError):
                    exp.plan(self.f.c)
                base.write_json(path, original)

    def test_comparison_cannot_recenter_training_or_heldout_crops(self):
        for name in ('proposal-audit.json', 'final-test/proposal-audit.json'):
            path = self.f.comparison_folder() / name
            original = base.checked_json(path)
            self.f.rewrite(path, lambda v: v['samples'][0].update(origins_kji=[[1, 0, 0], [0, 0, 0]]))
            with self.assertRaisesRegex(ValueError, 'crop coordinates'):
                exp.plan(self.f.c)
            base.write_json(path, original)

    def test_duplicate_comparison_predictions_block_plan(self):
        path = self.f.comparison_folder() / 'results.json'
        self.f.rewrite(path, lambda v: v['final_predictions'].append(deepcopy(v['final_predictions'][0])))
        with self.assertRaisesRegex(ValueError, 'completion, initialization'):
            exp.plan(self.f.c)

    def test_freeze_is_separate_and_pins_live_and_snapshot_sources(self):
        parent_before = base.sha256(self.f.parent / 'experiment-lock.json')
        comparison_before = base.sha256(self.f.comparison / 'experiment-lock.json')
        lock = exp.freeze(self.f.c, self.f.content)
        self.assertEqual(exp.inspect_lock(self.f.c, self.f.content), lock)
        self.assertEqual(set(lock['code_sha256']), set(self.f.code_files))
        self.assertEqual(base.sha256(self.f.parent / 'experiment-lock.json'), parent_before)
        self.assertEqual(base.sha256(self.f.comparison / 'experiment-lock.json'), comparison_before)
        with self.assertRaises(FileExistsError):
            exp.freeze(self.f.c, self.f.content)
        for path in (self.f.code / self.f.code_files[0], self.f.output / 'source' / self.f.code_files[0]):
            with self.subTest(path=str(path)):
                original = path.read_bytes()
                path.write_bytes(original + b'# changed\n')
                with self.assertRaisesRegex(ValueError, 'implementation changed'):
                    exp.inspect_lock(self.f.c, self.f.content)
                path.write_bytes(original)

    def test_parent_and_comparison_artifacts_cannot_change_after_freeze(self):
        exp.freeze(self.f.c, self.f.content)
        paths = (self.f.parent / 'experiment-lock.json',
                 self.f.stage_folder() / 'locator/locator.keras',
                 self.f.comparison_folder() / 'classifier/history.json',
                 self.f.comparison_folder() / 'classifier/classifier.keras',
                 self.f.comparison_folder() / 'final-test/predictions.json')
        for path in paths:
            with self.subTest(path=str(path)):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, 'Pinned parent artifact'):
                    exp.inspect_lock(self.f.c, self.f.content)
                path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'input changed'):
            exp.inspect_lock(self.f.c, self.f.content + b'\n')

    def test_runtime_changes_only_output_epoch_budget_and_loss_weight(self):
        lock = exp.plan(self.f.c)
        parent_before = deepcopy(self.f.parent_c)
        run_c, runtime = exp.runtime_settings(self.f.c, lock, lock['stages'][0], self.f.output / 'stage')
        expected = deepcopy(parent_before)
        expected['experiment']['output_directory'] = str(self.f.output)
        expected['training'].update(classifier_epochs=200, regression_weight=0.)
        self.assertEqual(run_c, expected)
        self.assertEqual(self.f.parent_c, parent_before)
        self.assertEqual(runtime['model'], self.f.runtime['model'])
        self.assertEqual(runtime['input.detail'], self.f.runtime['input.detail'])

    def test_own_smoke_must_bind_new_lock_and_explicit_zero_weight(self):
        lock = exp.freeze(self.f.c, self.f.content)
        smoke = base.checked_json(self.f.parent / 'runtime-smoke.json')
        base.write_json(self.f.output / 'runtime-smoke.json', smoke)
        with self.assertRaises(ValueError):
            exp.require_smoke(self.f.c, lock)
        digest = base.sha256(self.f.output / 'experiment-lock.json')
        smoke['source_lock_sha256'] = digest
        smoke['full_cache_memory_probe']['source_lock_sha256'] = digest
        base.write_json(self.f.output / 'runtime-smoke.json', smoke)
        with self.assertRaisesRegex(ValueError, 'classification-only loss'):
            exp.require_smoke(self.f.c, lock)
        smoke.update(regression_weight=0., severity_trained=False)
        base.write_json(self.f.output / 'runtime-smoke.json', smoke)
        self.assertEqual(exp.require_smoke(self.f.c, lock), smoke)
        smoke['regression_weight'] = 1.
        base.write_json(self.f.output / 'runtime-smoke.json', smoke)
        with self.assertRaisesRegex(ValueError, 'classification-only loss'):
            exp.require_smoke(self.f.c, lock)

    def test_fit_finishes_before_heldout_reads_and_outputs_hide_untrained_severity(self):
        import fakect_coa_classification_learning as classification
        import fakect_coa_rotation_learning as learning
        import fakect_coa_paired_steps as network
        # Freeze a new synthetic comparison after establishing failed coverage;
        # finished comparison dependencies must never be modified in place.
        shutil.rmtree(self.f.comparison)
        self.f.rewrite(self.f.stage_folder() / 'final-test/proposal-audit.json',
                       lambda v: v['samples'][0].update(coverage_passed=False))
        self.f.output = self.f.comparison
        self.f.c = deepcopy(self.f.comparison_config)
        self.f.content = self.f.comparison_content
        self.f.complete_comparison()
        lock = exp.freeze(self.f.c, self.f.content)
        stage, events = lock['stages'][0], []

        def fit(tf, runtime, rows, proposals, output, run_c, **kwargs):
            events.append('fit')
            self.assertEqual(run_c['training']['regression_weight'], 0.)
            self.assertEqual({r['split'] for r in rows}, {'train'})
            self.assertEqual(set(proposals), set(stage['train_sample_ids']))
            self.assertEqual({r['sample_id'] for r in rows}, set(stage['train_sample_ids']))
            self.assertEqual(kwargs, {'epochs': 200, 'checkpoint_interval': 50, 'fixed_budget': True,
                                     'expected_initial_weights_sha256': INITIAL})
            output.mkdir()
            checkpoint = output / 'classifier.keras'
            checkpoint.write_bytes(b'new classification weights')
            base.write_json(output / 'initialization.json', {'predictions': predictions(rows)})
            return {'checkpoint': str(checkpoint), 'checkpoint_sha256': base.sha256(checkpoint),
                    'selected_epoch': 200, 'epochs_completed': 200, 'optimizer_updates': 4800,
                    'validation_sample_ids': [], 'validation': None,
                    'regression_weight': 0., 'severity_trained': False,
                    'fresh_initial_weights_sha256': INITIAL}

        def reader(runtime, limit, rows):
            self.assertEqual(events, ['fit'])
            events.append('heldout-reader')
            self.assertEqual({r['split'] for r in rows}, {'final_test'})
            self.assertEqual({r['sample_id'] for r in rows}, set(stage['final_test_sample_ids']))
            return Mock()

        def score(tf, steps, rows, inputs, proposals, guard):
            self.assertEqual(events, ['fit', 'heldout-reader'])
            events.append('score')
            return predictions(rows)

        with patch.object(classification, 'fit_classifier', side_effect=fit), \
             patch.object(learning.legacy, 'configure_tensorflow', return_value=(object(), object())), \
             patch.object(learning.legacy, 'RuntimeGuard', return_value=Mock()), \
             patch.object(learning, 'StageFineInputs', side_effect=reader), \
             patch.object(learning, '_score', side_effect=score), \
             patch.object(network, 'load_model', return_value=Mock()), \
             patch.object(network, 'make_steps', return_value=Mock()) as steps:
            result = exp.run_stage(self.f.c, lock, stage)
        self.assertEqual(events, ['fit', 'heldout-reader', 'score'])
        self.assertEqual(steps.call_args.kwargs['regression_weight'], 0.)
        self.assertEqual(len(result['final_predictions']), 12)
        self.assertTrue(all(r['score_prediction'] is None for r in result['final_predictions']))
        self.assertEqual(sum(r['pipeline_failure'] for r in result['final_predictions']), 1)
        self.assertFalse(result['reserved_test_evaluated'])
        self.assertFalse(result['severity_trained'])
        self.assertFalse(result['locator_retrained'])
        self.assertFalse((self.f.output / 'stages' / stage['id'] / 'INCOMPLETE').exists())

    def test_fit_contract_failure_blocks_every_heldout_read(self):
        import fakect_coa_classification_learning as classification
        import fakect_coa_rotation_learning as learning
        lock = exp.freeze(self.f.c, self.f.content)
        stage = lock['stages'][0]
        valid = {'selected_epoch': 200, 'epochs_completed': 200, 'optimizer_updates': 4800,
                 'validation_sample_ids': [], 'validation': None, 'regression_weight': 0.,
                 'severity_trained': False, 'fresh_initial_weights_sha256': INITIAL}
        for changed in ({'selected_epoch': 199}, {'epochs_completed': 199}, {'optimizer_updates': 4799},
                        {'regression_weight': 1.}, {'severity_trained': True},
                        {'fresh_initial_weights_sha256': 'b' * 64},
                        {'validation_sample_ids': ['260612:baseline']}):
            with self.subTest(changed=changed), \
                 patch.object(classification, 'fit_classifier', return_value={**valid, **changed}), \
                 patch.object(learning.legacy, 'configure_tensorflow', return_value=(object(), object())), \
                 patch.object(learning, 'StageFineInputs') as reader, \
                 patch.object(learning, '_score') as score:
                with self.assertRaisesRegex(ValueError, 'final-epoch policy'):
                    exp.run_stage(self.f.c, lock, stage)
                reader.assert_not_called()
                score.assert_not_called()
                shutil.rmtree(self.f.output / 'stages' / stage['id'])


class ClassificationInventoryTests(unittest.TestCase):
    def test_runtime_extends_every_frozen_budget_dependency(self):
        self.assertTrue(set(exp.budget.CODE_FILES).issubset(exp.CODE_FILES))
        self.assertTrue({'src/fakect_coa_classification_experiment.py',
                         'src/fakect_coa_classification_learning.py',
                         'src/fakect_coa_classification_report.py',
                         'scripts/train_coa_classification.py',
                         'scripts/slurm/fakect_coa_classification_gpu.sh',
                         'scripts/slurm/fakect_coa_classification_report.sh'}.issubset(exp.CODE_FILES))
        self.assertEqual(len(exp.CODE_FILES), len(set(exp.CODE_FILES)))


if __name__ == '__main__':
    unittest.main()
