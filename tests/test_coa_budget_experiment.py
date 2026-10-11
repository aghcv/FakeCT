"""Budget controls inherit immutable primary folds without opening cohort images.

The parent fixtures contain five completed folds, real artifact checksums and
checkpoint bytes. TensorFlow and the parent's external source inventory are
replaced at their boundaries; no fixture image payload exists to be read.
"""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_coa_budget_experiment as exp
import fakect_multiscale_experiment as base


SCORE = 'coa_local_target_volume_loss_fraction'
INITIAL = 'a' * 64
FAMILIES = [('adult', ['260602', '260611']), ('1yr', ['260613', '260618']),
            ('5yr', ['260614', '260619']), ('10yr', ['260615', '260620']),
            ('15yr', ['260616', '260621'])]


def predictions(rows):
    return [{'sample_id': r['sample_id'], 'case_id': r['case_id'],
             'family': r['anatomy_family'], 'split': r.get('split', 'train'),
             'label': int(r['annotation']['coa_benchmark_positive']),
             'logit': 0., 'probability': .5, 'score_label': r['annotation'][SCORE],
             'score_prediction': .25} for r in rows]


class ParentFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.parent = self.root / 'parent'
        self.parent.mkdir()
        self.output = self.root / 'control'
        self.code = self.root / 'code'
        self.code_files = ('src/fixture.py', 'scripts/fixture.py')
        for name in self.code_files:
            p = self.code / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# immutable source fixture\n')
        self.c = {'experiment': {'schema_version': exp.SCHEMA, 'output_directory': str(self.output)},
                  'reference': {'experiment_directory': str(self.parent)},
                  'training': {'classifier_epochs': 200, 'checkpoint_interval': 50}}
        self.content = ('[experiment]\nschema_version = ' + exp.SCHEMA + '\noutput_directory = ' +
                        str(self.output) + '\n[reference]\nexperiment_directory = ' + str(self.parent) +
                        '\n[training]\nclassifier_epochs = 200\ncheckpoint_interval = 50\n').encode()
        self.runtime = {'experiment': {'output_directory': str(self.parent)},
                        'model': {'seed': 42, 'base_filters': 4, 'learning_rate': .001},
                        'training': {'host_memory_limit_mb': 16384},
                        'pilot': {'require_gpu': True, 'gpu_memory_limit_mb': 12288},
                        'input.overview': {'shape_kji': [2, 2, 2]},
                        'input.detail': {'shape_kji': [2, 2, 2], 'proposals': 2}}
        self.parent_c = {'experiment': {'output_directory': str(self.parent)},
                         'training': {'locator_epochs': 120, 'classifier_epochs': 50,
                                      'locator_learning_rate': .0003, 'classifier_learning_rate': .001,
                                      'regression_weight': 1., 'cache_limit_mb': 1,
                                      'host_memory_limit_mb': 16384, 'gpu_memory_limit_mb': 12288,
                                      'require_gpu': True}}
        rows, originals, inventory, stages = [], [], [], []
        for family, cases in FAMILIES:
            for case in cases:
                inventory.append({'case_id': case, 'anatomy_family': family})
                originals.append({'sample_id': case + ':baseline', 'case_id': case,
                                  'anatomy_family': family, 'baseline': True})
                for block in range(3):
                    for positive in (False, True):
                        rows.append({'sample_id': f'{case}:{block}:{int(positive)}', 'case_id': case,
                                     'anatomy_family': family, 'baseline': False,
                                     'parameters': {'arch_outer_expand': {'distance_mm': float(block)},
                                                    'coa_narrow': {'distance_mm': 3. if positive else 1.}},
                                     'annotation': {'coa_benchmark_positive': positive,
                                                    SCORE: .35 if positive else .1},
                                     'payload_must_not_exist': str(self.root / 'no-images' / case)})
        all_cases = [case for _, cases in FAMILIES for case in cases]
        for family, heldout in FAMILIES:
            train = [case for case in all_cases if case not in heldout]
            stages.append({'id': family + '-refit', 'outer_fold_id': family, 'protocol': 'family_holdout',
                           'stage': 'outer_refit', 'seed': 42, 'execution_eligible': True,
                           'train_case_ids': train, 'validation_case_ids': [],
                           'final_test_case_ids': heldout,
                           'train_sample_ids': [r['sample_id'] for r in rows if r['case_id'] in train],
                           'final_test_sample_ids': [r['sample_id'] for r in rows if r['case_id'] in heldout],
                           'parent_inner_stage_ids': []})
        self.document = {'config': self.parent_c, 'schema_version': 'parent-fixture/1',
                         'stages': stages, 'samples': rows, 'original_samples': originals,
                         'inventory': inventory, 'reserved_case_ids': ['260612', '260617'],
                         'runtime_geometry': self.runtime, 'prediction_threshold': .5}
        (self.parent / 'input.ini').write_bytes(b'parent input\n')
        base.write_json(self.parent / 'experiment-lock.json', self.document)
        lock_sha = base.sha256(self.parent / 'experiment-lock.json')
        # This uses the real inherited smoke validator, but needs no TF process.
        per_sample = 2 * 2 * 2 * 2 * 2 * 4
        count = 1024**2 // per_sample
        base.write_json(self.parent / 'runtime-smoke.json', {
            'status': 'passed', 'checkpoint_parity': True, 'source_lock_sha256': lock_sha,
            'overview_shape': [2, 2, 2], 'detail_shape': [2, 2, 2],
            'traces': {'train': 1, 'predict': 1}, 'locator_gradient_norm': 1.,
            'classifier_gradient_norm': 1., 'gpu_visible': True,
            'host_peak_rss_mib': 100., 'gpu_peak_mib': 10.,
            'full_cache_memory_probe': {'source_lock_sha256': lock_sha,
                'model_update_while_resident': True, 'physical_pages_touched': True,
                'sample_count': count, 'resident_bytes': count * per_sample,
                'passed': True, 'host_guard_passed': True}})
        for stage in stages:
            folder = self.parent / 'stages' / stage['id']
            for name in ('locator', 'classifier', 'final-test'):
                (folder / name).mkdir(parents=True, exist_ok=True)
            train_rows = [{**r, 'split': 'train'} for r in rows if r['case_id'] in stage['train_case_ids']]
            test_rows = [{**r, 'split': 'final_test'} for r in rows if r['case_id'] in stage['final_test_case_ids']]
            ids = [r['sample_id'] for r in originals if r['case_id'] in stage['train_case_ids']]
            locator_file, classifier_file = folder / 'locator/locator.keras', folder / 'classifier/classifier.keras'
            locator_file.write_bytes(b'fold locator ' + stage['id'].encode())
            classifier_file.write_bytes(b'parent classifier ' + stage['id'].encode())
            locator = {'checkpoint': str(locator_file), 'checkpoint_sha256': base.sha256(locator_file),
                       'training_sample_ids': ids, 'validation_sample_ids': [],
                       'prior_training_sample_ids': ids, 'outer_evaluation_used': False,
                       'reserved_test_evaluated': False}
            classifier = {'checkpoint': str(classifier_file), 'checkpoint_sha256': base.sha256(classifier_file),
                          'training_sample_ids': stage['train_sample_ids'], 'validation_sample_ids': [],
                          'fresh_initial_weights_sha256': INITIAL}
            base.write_json(folder / 'locator/locator-result.json', locator)
            base.write_json(folder / 'classifier/receipt.json', classifier)
            base.write_json(folder / 'classifier/initialization.json', {'weights_sha256': INITIAL,
                                                                       'predictions': predictions(train_rows)})
            for role, panel, audit_name in [('train', train_rows, 'proposal-audit.json'),
                                            ('final_test', test_rows, 'final-test/proposal-audit.json')]:
                sample_records = [{'sample_id': r['sample_id'], 'split': role,
                                   'coverage_passed': True, 'predicted_box': [.5] * 6,
                                   'origins_kji': [[0, 0, 0], [0, 0, 0]],
                                   'proposal_boxes': [[.5] * 6, [.5] * 6]} for r in panel]
                base.write_json(folder / audit_name, {'passed': True, 'samples': sample_records,
                                                       'selection_allowed': False})
            final = predictions(test_rows)
            base.write_json(folder / 'final-test/predictions.json', {'predictions': final})
            base.write_json(folder / 'stage-input.json', {'stage': stage, 'source_lock_sha256': lock_sha})
            base.write_json(folder / 'results.json', {'status': 'completed', 'stage_id': stage['id'],
                'source_lock_sha256': lock_sha, 'reserved_test_evaluated': False,
                'final_test_evaluated': True, 'locator': locator, 'classifier': classifier,
                'final_predictions': final})

    def stage_folder(self):
        return self.parent / 'stages' / self.document['stages'][0]['id']

    def rewrite(self, path, update):
        value = base.checked_json(path)
        update(value)
        base.write_json(path, value)

    def rewrite_parent_document(self, document):
        """Keep completion provenance coherent while exercising plan semantics."""
        base.write_json(self.parent / 'experiment-lock.json', document)
        digest = base.sha256(self.parent / 'experiment-lock.json')
        smoke = base.checked_json(self.parent / 'runtime-smoke.json')
        smoke['source_lock_sha256'] = digest
        smoke['full_cache_memory_probe']['source_lock_sha256'] = digest
        base.write_json(self.parent / 'runtime-smoke.json', smoke)
        for stage in self.document['stages']:
            for name in ('results.json', 'stage-input.json'):
                path = self.parent / 'stages' / stage['id'] / name
                self.rewrite(path, lambda value: value.update(source_lock_sha256=digest))


class BudgetExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.f = ParentFixture(self.temporary.name)
        self.parent_inspection = patch.object(exp.rotation, 'inspect_lock', return_value=self.f.document)
        self.parent_inspection.start()
        self.addCleanup(self.parent_inspection.stop)
        self.root_patch = patch.object(exp, 'ROOT', self.f.code)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.source_patch = patch.object(exp, 'CODE_FILES', self.f.code_files)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def test_plan_retains_exact_five_primary_partitions_and_training_settings(self):
        lock = exp.plan(self.f.c)
        self.assertEqual(len(lock['stages']), 5)
        self.assertEqual({s['id'] for s in lock['stages']}, {s['id'] for s in self.f.document['stages']})
        self.assertEqual(lock['classifier_epochs'], 200)
        self.assertEqual(lock['unchanged_training_settings'], self.f.parent_c['training'])
        self.assertEqual(lock['runtime_geometry'], self.f.runtime)
        self.assertEqual(len(lock['samples']), 60)
        self.assertEqual(len(lock['fold_manifest']['folds']), 5)
        for stage, original in zip(lock['stages'], self.f.document['stages']):
            for field in ('train_case_ids', 'final_test_case_ids', 'train_sample_ids', 'final_test_sample_ids'):
                self.assertEqual(stage[field], original[field])
            self.assertEqual((len(stage['train_sample_ids']), len(stage['final_test_sample_ids'])), (48, 12))
            self.assertFalse(set(stage['train_case_ids']) & set(stage['final_test_case_ids']))
            self.assertFalse(set(stage['train_case_ids'] + stage['final_test_case_ids']) & {'260612', '260617'})
            self.assertEqual(stage['expected_initial_weights_sha256'], INITIAL)
            self.assertEqual(stage['parent_inner_stage_ids'], [])

    def test_requires_five_unique_seed42_primary_folds(self):
        for mutation in ('count', 'fold', 'seed', 'eligible', 'validation'):
            with self.subTest(mutation=mutation):
                document = deepcopy(self.f.document)
                if mutation == 'count':
                    document['stages'].pop()
                elif mutation == 'fold':
                    document['stages'][1]['outer_fold_id'] = document['stages'][0]['outer_fold_id']
                elif mutation == 'seed':
                    document['stages'][0]['seed'] = 7
                elif mutation == 'eligible':
                    document['stages'][0]['execution_eligible'] = False
                else:
                    document['stages'][0]['validation_case_ids'] = ['260612']
                self.f.rewrite_parent_document(document)
                message = 'five reviewed' if mutation in ('count', 'fold', 'seed') else 'eligible'
                with self.assertRaisesRegex(ValueError, message):
                    exp.plan(self.f.c)
        self.f.rewrite_parent_document(self.f.document)

    def test_locator_prior_cannot_include_heldout_or_reserved_originals(self):
        folder = self.f.stage_folder()
        receipt_path, result_path = folder / 'locator/locator-result.json', folder / 'results.json'
        original_receipt, original_result = base.checked_json(receipt_path), base.checked_json(result_path)
        for field, bad in [('prior_training_sample_ids', '260602:baseline'),
                           ('training_sample_ids', '260612:baseline')]:
            with self.subTest(field=field):
                changed = deepcopy(original_receipt)
                changed[field].append(bad)
                result = deepcopy(original_result)
                result['locator'] = changed
                base.write_json(receipt_path, changed)
                base.write_json(result_path, result)
                with self.assertRaisesRegex(ValueError, 'isolation'):
                    exp.plan(self.f.c)

    def test_classifier_receipt_cannot_add_heldout_training_sample(self):
        folder = self.f.stage_folder()
        receipt_path, result_path = folder / 'classifier/receipt.json', folder / 'results.json'
        receipt, result = base.checked_json(receipt_path), base.checked_json(result_path)
        receipt['training_sample_ids'].append(self.f.document['stages'][0]['final_test_sample_ids'][0])
        result['classifier'] = receipt
        base.write_json(receipt_path, receipt)
        base.write_json(result_path, result)
        with self.assertRaisesRegex(ValueError, 'isolation'):
            exp.plan(self.f.c)

    def test_modified_receipt_and_checkpoint_are_rejected(self):
        folder = self.f.stage_folder()
        receipt = folder / 'classifier/receipt.json'
        self.f.rewrite(receipt, lambda row: row.update(fresh_initial_weights_sha256='b' * 64))
        with self.assertRaisesRegex(ValueError, 'receipts'):
            exp.plan(self.f.c)
        base.write_json(receipt, base.checked_json(folder / 'results.json')['classifier'])
        checkpoint = Path(base.checked_json(receipt)['checkpoint'])
        checkpoint.write_bytes(b'changed learned weights')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            exp.plan(self.f.c)

    def test_reviewed_crop_panel_roles_and_training_coverage_are_enforced(self):
        path = self.f.stage_folder() / 'proposal-audit.json'
        original = base.checked_json(path)
        for mutation in ('duplicate', 'missing', 'role', 'coverage', 'selection'):
            with self.subTest(mutation=mutation):
                audit = deepcopy(original)
                if mutation == 'duplicate':
                    audit['samples'].append(deepcopy(audit['samples'][0]))
                elif mutation == 'missing':
                    audit['samples'].pop()
                elif mutation == 'role':
                    audit['samples'][0]['split'] = 'final_test'
                elif mutation == 'coverage':
                    audit['samples'][0]['coverage_passed'] = False
                else:
                    audit['selection_allowed'] = True
                base.write_json(path, audit)
                with self.assertRaises(ValueError):
                    exp.plan(self.f.c)

    def test_parent_lock_checksum_and_completion_markers_are_required(self):
        path = self.f.parent / 'experiment-lock.json'
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            exp.plan(self.f.c)
        path.write_bytes(original)
        marker = self.f.stage_folder() / 'INCOMPLETE'
        marker.touch()
        with self.assertRaisesRegex(ValueError, 'finish cleanly'):
            exp.plan(self.f.c)

    def test_output_and_all_parent_artifact_pins_are_immutable(self):
        lock = exp.freeze(self.f.c, self.f.content)
        self.assertEqual(set(lock['code_sha256']), set(self.f.code_files))
        self.assertEqual(exp.inspect_lock(self.f.c, self.f.content), lock)
        with self.assertRaises(FileExistsError):
            exp.freeze(self.f.c, self.f.content)
        dependency = self.f.stage_folder() / 'final-test/proposal-audit.json'
        self.f.rewrite(dependency, lambda audit: audit['samples'][0].update(origins_kji=[[1, 0, 0], [0, 0, 0]]))
        with self.assertRaisesRegex(ValueError, 'Pinned parent artifact'):
            exp.inspect_lock(self.f.c, self.f.content)

    def test_parent_checkpoint_and_parent_lock_changes_after_freeze_are_rejected(self):
        exp.freeze(self.f.c, self.f.content)
        paths = [self.f.parent / 'experiment-lock.json', self.f.stage_folder() / 'locator/locator.keras']
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, 'Pinned parent artifact'):
                    exp.inspect_lock(self.f.c, self.f.content)
                path.write_bytes(original)

    def test_live_and_snapshot_sources_and_input_bytes_are_pinned(self):
        exp.freeze(self.f.c, self.f.content)
        source = self.f.code / self.f.code_files[0]
        original = source.read_bytes()
        source.write_bytes(original + b'# changed\n')
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            exp.inspect_lock(self.f.c, self.f.content)
        source.write_bytes(original)
        snapshot = self.f.output / 'source' / self.f.code_files[0]
        snapshot.write_bytes(original + b'# snapshot changed\n')
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            exp.inspect_lock(self.f.c, self.f.content)
        snapshot.write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'input changed'):
            exp.inspect_lock(self.f.c, self.f.content + b'\n')

    def test_config_requires_separate_output_and_exact_budget_fields(self):
        path = self.f.root / 'control.ini'
        path.write_bytes(self.f.content)
        self.assertEqual(exp.load_config(path), self.f.c)
        for output in (self.f.parent, self.f.parent / 'nested', self.f.root):
            path.write_bytes(self.f.content.replace(str(self.f.output).encode(), str(output).encode()))
            with self.assertRaisesRegex(ValueError, 'separate'):
                exp.load_config(path)
        path.write_bytes(self.f.content.replace(b'checkpoint_interval = 50', b'checkpoint_interval = 30'))
        with self.assertRaisesRegex(ValueError, 'divide'):
            exp.load_config(path)
        path.write_bytes(self.f.content + b'learning_rate = .03\n')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            exp.load_config(path)

    def test_runtime_and_own_smoke_inherit_limits_but_use_new_lock(self):
        lock = exp.freeze(self.f.c, self.f.content)
        run_c, runtime = exp.runtime_settings(self.f.c, lock, lock['stages'][0], self.f.output / 'stage')
        expected = deepcopy(self.f.parent_c)
        expected['experiment']['output_directory'] = str(self.f.output)
        expected['training']['classifier_epochs'] = 200
        self.assertEqual(run_c, expected)
        self.assertEqual(runtime['model'], self.f.runtime['model'])
        self.assertEqual(runtime['input.detail'], self.f.runtime['input.detail'])
        # Copying the parent's successful smoke is insufficient: its lock hash
        # differs. Both its top-level and physical-cache proof must be rebound.
        smoke = base.checked_json(self.f.parent / 'runtime-smoke.json')
        base.write_json(self.f.output / 'runtime-smoke.json', smoke)
        with self.assertRaisesRegex(ValueError, 'Own production-size smoke'):
            exp.require_smoke(self.f.c, lock)
        digest = base.sha256(self.f.output / 'experiment-lock.json')
        smoke['source_lock_sha256'] = digest
        smoke['full_cache_memory_probe']['source_lock_sha256'] = digest
        base.write_json(self.f.output / 'runtime-smoke.json', smoke)
        exp.require_smoke(self.f.c, lock)
        smoke['full_cache_memory_probe']['physical_pages_touched'] = False
        base.write_json(self.f.output / 'runtime-smoke.json', smoke)
        with self.assertRaises(ValueError):
            exp.require_smoke(self.f.c, lock)

    def test_final_inputs_are_opened_only_after_train_only_fit_and_failures_remain(self):
        import fakect_coa_budget_learning as budget_learning
        import fakect_coa_rotation_learning as learning
        import fakect_coa_paired_steps as network
        audit_path = self.f.stage_folder() / 'final-test/proposal-audit.json'
        # QA failure must be retained, rather than suppressing an outer sample.
        # Establish the failed coverage record before freezing the control.
        self.f.rewrite(audit_path, lambda audit: audit['samples'][0].update(coverage_passed=False))
        lock = exp.freeze(self.f.c, self.f.content)
        stage = lock['stages'][0]
        events = []
        def fit(tf, runtime, rows, proposals, output, run_c, **kwargs):
            events.append('fit')
            self.assertEqual({r['sample_id'] for r in rows}, set(stage['train_sample_ids']))
            self.assertEqual({r['split'] for r in rows}, {'train'})
            self.assertEqual(set(proposals), set(stage['train_sample_ids']))
            self.assertFalse({r['case_id'] for r in rows} & set(stage['final_test_case_ids'] + ['260612', '260617']))
            self.assertEqual(kwargs, {'epochs': 200, 'checkpoint_interval': 50, 'fixed_budget': True,
                                     'expected_initial_weights_sha256': INITIAL})
            output = Path(output)
            output.mkdir()
            checkpoint = output / 'classifier.keras'
            checkpoint.write_bytes(b'new final weights')
            base.write_json(output / 'initialization.json', {'predictions': predictions(rows)})
            return {'checkpoint': str(checkpoint), 'checkpoint_sha256': base.sha256(checkpoint),
                    'selected_epoch': 200, 'epochs_completed': 200, 'optimizer_updates': 4800,
                    'validation_sample_ids': [], 'validation': None, 'fresh_initial_weights_sha256': INITIAL}
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
        with patch.object(budget_learning, 'fit_classifier', side_effect=fit), \
             patch.object(learning.legacy, 'configure_tensorflow', return_value=(object(), object())), \
             patch.object(learning.legacy, 'RuntimeGuard', return_value=Mock()), \
             patch.object(learning, 'StageFineInputs', side_effect=reader), \
             patch.object(learning, '_score', side_effect=score), \
             patch.object(network, 'load_model', return_value=Mock()), \
             patch.object(network, 'make_steps', return_value=Mock()):
            result = exp.run_stage(self.f.c, lock, stage)
        self.assertEqual(events, ['fit', 'heldout-reader', 'score'])
        self.assertEqual(len(result['final_predictions']), 12)
        self.assertEqual(sum(r['pipeline_failure'] for r in result['final_predictions']), 1)
        self.assertFalse(result['reserved_test_evaluated'])
        self.assertFalse(result['locator_retrained'])
        self.assertFalse((self.f.output / 'stages' / stage['id'] / 'INCOMPLETE').exists())


class BudgetRuntimeInventoryTests(unittest.TestCase):
    def test_new_runtime_extends_every_frozen_rotation_source(self):
        self.assertTrue(set(exp.rotation.CODE_FILES).issubset(exp.CODE_FILES))
        self.assertTrue({'src/fakect_coa_budget_experiment.py', 'src/fakect_coa_budget_learning.py',
                         'src/fakect_coa_budget_report.py', 'scripts/train_coa_budget.py',
                         'scripts/slurm/fakect_coa_budget_gpu.sh'}.issubset(exp.CODE_FILES))
        self.assertEqual(len(exp.CODE_FILES), len(set(exp.CODE_FILES)))


if __name__ == '__main__':
    unittest.main()
