"""Exercise fold orchestration with real archives/locks and a tiny fake backend."""
import copy
import json
from pathlib import Path
import statistics
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
import test_model_experiment as model_fixture
from fakect_model_evaluation import (
    fit_evaluation, freeze_evaluation, plan_evaluation, summarize_evaluation,
    test_evaluation as evaluate_folds,
)
from fakect_model_experiment import inspect_experiment_lock
from fakect_segmentation import MODEL_SCHEMA, PatchDataset


class ForegroundPredictor:
    """Predict all foreground so scores vary with each real test mask's size."""
    def predict_on_batch(self, batch):
        return np.ones_like(batch)


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        # Compose the existing archive/registry fixture without inheriting its
        # test methods or collecting its TestCase class a second time.
        self.fixture = model_fixture.ModelExperimentTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = copy.deepcopy(self.fixture.config)
        self.config['model']['epochs'] = 3
        self.config['evaluation'] = {'strategy': 'holdout', 'case_counts': [1, 1, 1],
                                     'seeds': [7], 'threshold': .5}
        self.raw = self.fixture.raw
        self.fits = []
        self.prediction_models = []
        self.normal = patch('fakect_segmentation.train_segmentation',
                            side_effect=lambda c, p: self.fake_fit(c, p, fixed=False)).start()
        self.fixed = patch('fakect_model_evaluation.train_fixed_segmentation',
                           side_effect=lambda c, p: self.fake_fit(c, p, fixed=True)).start()
        self.loader = patch('fakect_evaluation_metrics.load_segmentation',
                            side_effect=self.load_predictor).start()

    def fake_fit(self, config, manifest_path, *, fixed):
        dataset = PatchDataset(manifest_path, config['model'])
        checked = dataset.manifest
        loaded_paths = []
        numpy_load = np.load
        def observe_load(path, *args, **kwargs):
            loaded_paths.append(Path(path))
            return numpy_load(path, *args, **kwargs)
        # Real PatchDataset verifies bytes/arrays, normalizes, tiles and pads.
        # The fake backend substitutes only expensive TensorFlow optimization.
        with patch('fakect_segmentation.np.load', side_effect=observe_load):
            for split in ('train', 'validation'):
                for image, mask, valid in dataset.batches(split):
                    self.assertEqual(image.shape, mask.shape)
                    self.assertGreater(int(valid.sum()), 0)
        held_out_paths = {s['_path'] for s in checked['samples'] if s['split'] == 'test'}
        self.assertFalse(set(loaded_paths) & held_out_paths)
        self.assertEqual({s['variant_id'] for s in checked['samples'] if s['split'] != 'test'},
                         dataset.loaded_variants)
        directory = Path(config['train']['model_directory'])
        directory.mkdir(parents=True, exist_ok=False)
        settings = config['model']
        model_path = directory/'model.keras'
        model_path.write_bytes(f"fake:{manifest_path}:{settings['seed']}:{settings['epochs']}".encode())
        fold_name = directory.parent.name
        best = int(fold_name.rsplit('-', 1)[1]) if fold_name.startswith('inner-') else 2
        best = min(best, settings['epochs'])
        metadata = {'schema_version': MODEL_SCHEMA, 'architecture': settings['architecture'],
                    'settings': settings, 'source_manifest': str(checked['path']),
                    'source_manifest_sha256': checked['sha256'], 'model_file': 'model.keras',
                    'model_sha256': model_fixture.digest(model_path),
                    'epochs_completed': settings['epochs'], 'best_epoch': best,
                    'test_payloads_loaded': False, 'loaded_variants': sorted(dataset.loaded_variants),
                    'history': [{'epoch': epoch, 'validation': {'loss': float(abs(epoch-best)+1)}}
                                for epoch in range(1, settings['epochs']+1)]}
        (directory/'model-metadata.json').write_text(json.dumps(metadata))
        self.fits.append({'fixed': fixed, 'path': checked['path'], 'settings': copy.deepcopy(settings),
                          'samples': checked['samples'], 'loaded_paths': loaded_paths,
                          'split_sizes': checked['split_sizes']})
        return metadata

    def load_predictor(self, directory):
        directory = Path(directory)
        self.prediction_models.append(directory)
        return ForegroundPredictor(), json.loads((directory/'model-metadata.json').read_text())

    def freeze_and_fit(self):
        frozen = freeze_evaluation(self.config, self.raw)
        fitted = fit_evaluation(self.config, self.raw)
        self.assertTrue(fitted['model_fitted'])
        self.assertFalse(fitted['test_evaluated'])
        return frozen

    def enable_loo(self, *, nested=False, seeds=(7,)):
        if nested:
            self.fixture.make_dataset('case-3', 'family-3', 3)
            self.config['experiment']['datasets'].append('case-3')
        self.config['evaluation'] = {
            'strategy': 'nested_leave_one_out' if nested else 'leave_one_out',
            'seeds': list(seeds), 'threshold': .5}
        if not nested:
            self.config['evaluation']['validation_families'] = 1

    def results(self):
        return [json.loads(path.read_text()) for path in
                sorted(Path(self.config['experiment']['output_directory']).glob('folds/*/seed-*/evaluation.json'))]

    # Purpose: Verify holdout evaluation writes reports and resumes without redoing work.
    # Inputs: Frozen synthetic holdout with mocked fitting and foreground prediction.
    # Outputs: One fit and prediction produce correct macro Dice; repeat runs resume and source hashes stay intact.
    def test_holdout_writes_report_and_resumes_completed_fit_and_test(self):
        original = {sample['_path']: model_fixture.digest(sample['_path'])
                    for manifest in self.fixture.manifests.values() for sample in manifest['samples']}
        frozen = self.freeze_and_fit()
        self.assertEqual(frozen['fit_count'], 1)
        self.assertEqual(len(self.fits), 1)
        fit_evaluation(self.config, self.raw)
        self.assertEqual(len(self.fits), 1)
        result = evaluate_folds(self.config, self.raw)
        self.assertTrue(Path(result['report']).is_file())
        self.assertTrue(Path(result['summary']).is_file())
        rows = self.results()[0]['samples']
        expected = statistics.mean(2*row['true_positive'] /
                                   (row['valid_voxels']+row['true_positive']) for row in rows)
        self.assertAlmostEqual(result['macro_average']['dice'], expected)
        self.assertEqual(len(self.prediction_models), 1)
        evaluate_folds(self.config, self.raw)
        self.assertEqual(len(self.prediction_models), 1)
        self.assertEqual(original, {path: model_fixture.digest(path) for path in original})

    # Purpose: Verify repeated leave-one-family-out folds retain membership and family weights.
    # Inputs: Three-family leave-one-out design with seeds 7 and 11.
    # Outputs: Six fits/results use stable family panels and equal-family macro averaging.
    def test_leave_one_family_out_repeats_keep_test_membership_and_family_weight(self):
        self.enable_loo(seeds=(7, 11))
        frozen = self.freeze_and_fit()
        self.assertEqual(frozen['fit_count'], 6)
        self.assertEqual(len(self.fits), 6)
        summary_path = evaluate_folds(self.config, self.raw)['summary']
        results = self.results()
        self.assertEqual(len(results), 6)
        membership = {}
        by_family = {}
        for result in results:
            cases = {(row['variant_id'], row['case_id'], row['anatomy_family'])
                     for row in result['samples']}
            self.assertEqual(membership.setdefault(result['fold_id'], cases), cases)
            self.assertEqual(len({family for _, _, family in cases}), 1)
            for row in result['samples']:
                by_family.setdefault(row['anatomy_family'], []).append(row['dice'])
        self.assertEqual(len(by_family), 3)
        summary = json.loads(Path(summary_path).read_text())
        self.assertAlmostEqual(summary['macro_average']['dice'],
                               statistics.mean(statistics.mean(rows) for rows in by_family.values()))

    # Purpose: Verify nested evaluation refits at the inner selected epoch.
    # Inputs: Four-family nested leave-one-out plan and mocked inner best epochs.
    # Outputs: Twelve inner fits plus four fresh refits use median epoch 2; tests evaluate only outer models.
    def test_nested_folds_refit_all_development_families_at_inner_median_epoch(self):
        self.enable_loo(nested=True)
        frozen = self.freeze_and_fit()
        self.assertEqual(frozen['fit_count'], 16)
        self.assertEqual(self.normal.call_count, 12)
        self.assertEqual(self.fixed.call_count, 4)
        outer_tests = {fold['id']: set(fold['test_families']) for fold in frozen['design']['folds']}
        for fit in self.fits:
            self.assertEqual(fit['split_sizes']['test'], 2)
            self.assertEqual({row['anatomy_family'] for row in fit['samples'] if row['split'] == 'test'},
                             outer_tests[fit['path'].parents[2].name])
            if fit['fixed']:
                self.assertEqual(fit['settings']['epochs'], 2)
                self.assertEqual(fit['split_sizes'], {'train': 6, 'validation': 0, 'test': 2})
            else:
                self.assertEqual(fit['split_sizes'], {'train': 4, 'validation': 2, 'test': 2})
        evaluate_folds(self.config, self.raw)
        self.assertEqual(len(self.results()), 4)
        self.assertEqual(len(self.prediction_models), 4)  # Never evaluate the inner models.
        fit_evaluation(self.config, self.raw)
        self.assertEqual(len(self.fits), 16)

    # Purpose: Reject inner selected epochs exceeding their training budgets.
    # Inputs: Completed nested fit with best_epoch greater than settings.epochs.
    # Outputs: Evaluation raises ValueError before loading any predictor or releasing results.
    def test_nested_epoch_metadata_outside_budget_is_rejected_before_testing(self):
        self.enable_loo(nested=True)
        self.freeze_and_fit()
        path = next(Path(self.config['experiment']['output_directory']).glob(
            'folds/*/seed-*/inner-*/model/model-metadata.json'))
        metadata = json.loads(path.read_text())
        metadata['best_epoch'] = metadata['settings']['epochs']+1
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'best epoch'):
            evaluate_folds(self.config, self.raw)
        self.assertEqual(self.prediction_models, [])
        self.assertEqual(self.results(), [])

    # Purpose: Verify explicit evaluation design supersedes legacy split fractions.
    # Inputs: Holdout config with legacy validation 0.05/test 0.9 fractions.
    # Outputs: Planning is ready with one fit and creates no output directory.
    def test_evaluation_design_does_not_inherit_legacy_fraction_failure(self):
        self.config['split'].update(validation_fraction=.05, test_fraction=.9)
        planned = plan_evaluation(self.config)
        self.assertTrue(planned['ready'], planned['errors'])
        self.assertEqual(planned['fit_count'], 1)
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())

    # Purpose: Verify changed parent INI bytes stop fitting.
    # Inputs: Frozen holdout with changed caller input or changed saved input.ini.
    # Outputs: Fit raises ValueError before any model training.
    def test_parent_ini_changes_fail_before_fitting_or_evaluating(self):
        freeze_evaluation(self.config, self.raw)
        with self.assertRaisesRegex(ValueError, 'INI differs'):
            fit_evaluation(self.config, self.raw+b'changed')
        output = Path(self.config['experiment']['output_directory'])
        (output/'input.ini').write_bytes(self.raw+b'changed')
        with self.assertRaisesRegex(ValueError, 'INI changed'):
            fit_evaluation(self.config, self.raw)
        self.assertEqual(self.fits, [])

    # Purpose: Reject cached evaluation after model or metadata provenance changes.
    # Inputs: Completed evaluation with altered metadata, then weights and refreshed model hash.
    # Outputs: Summary/evaluation raises ValueError without invoking another predictor.
    def test_cached_results_reject_changed_model_or_metadata_provenance(self):
        self.freeze_and_fit()
        evaluate_folds(self.config, self.raw)
        model_dir = self.prediction_models[0]
        metadata_path = model_dir/'model-metadata.json'
        before = metadata_path.read_bytes()
        metadata = json.loads(before)
        metadata['extra_review_note'] = 'changed metadata after evaluation'
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            summarize_evaluation(self.config, self.raw)
        metadata_path.write_bytes(before)
        model_path = model_dir/'model.keras'
        model_path.write_bytes(model_path.read_bytes()+b'different trained weights')
        metadata = json.loads(before)
        metadata['model_sha256'] = model_fixture.digest(model_path)
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            evaluate_folds(self.config, self.raw)
        self.assertEqual(len(self.prediction_models), 1)

    # Purpose: Require all outer models before releasing any held-out result.
    # Inputs: Completed leave-one-out fitting with the last outer model removed.
    # Outputs: Evaluation raises FileNotFoundError with no predictors loaded or results released.
    def test_missing_outer_model_prevents_any_test_result_release(self):
        self.enable_loo()
        self.freeze_and_fit()
        last = sorted(Path(self.config['experiment']['output_directory']).glob(
            'folds/*/seed-*/outer-model/model/model.keras'))[-1]
        last.unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate_folds(self.config, self.raw)
        self.assertEqual(self.prediction_models, [])
        self.assertEqual(self.results(), [])

    # Purpose: Reject resumed fold locks tied to foreign parent snapshots.
    # Inputs: Valid resealed fold lock containing another input.ini snapshot.
    # Outputs: Fit raises ValueError for differing fold lock and does not retrain.
    def test_resumed_fold_rejects_foreign_parent_snapshot(self):
        self.freeze_and_fit()
        path = self.fits[0]['path']
        changed = json.loads(path.read_text())
        (path.parent/'input.ini').write_bytes(b'other parent input')
        changed['input_config_sha256'] = model_fixture.digest(path.parent/'input.ini')
        self.fixture.reseal_lock(path, changed)
        # The fold is internally intact but belongs to a different input file.
        inspect_experiment_lock(path)
        with self.assertRaisesRegex(ValueError, 'fold lock differs'):
            fit_evaluation(self.config, self.raw)
        self.assertEqual(len(self.fits), 1)

    # Purpose: Require nested inner locks to share the frozen parent input.
    # Inputs: Inner fold resealed against different parent input bytes.
    # Outputs: Evaluation rejects the changed assignment before prediction.
    def test_nested_inner_lock_must_reference_the_same_parent_input(self):
        self.enable_loo(nested=True)
        self.freeze_and_fit()
        path = next(fit['path'] for fit in self.fits if not fit['fixed'])
        changed = json.loads(path.read_text())
        (path.parent/'input.ini').write_bytes(b'other nested parent input')
        changed['input_config_sha256'] = model_fixture.digest(path.parent/'input.ini')
        self.fixture.reseal_lock(path, changed)
        with self.assertRaisesRegex(ValueError, 'Inner model assignment changed'):
            evaluate_folds(self.config, self.raw)
        self.assertEqual(self.prediction_models, [])


if __name__ == '__main__':
    unittest.main()
