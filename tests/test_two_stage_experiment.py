"""Exercise two-model INI, shared family partitions and frozen run bindings."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_two_stage_experiment import load_config, plan, prepare, freeze, inspect_lock, fit, _evaluate_one, sha256
from fakect_two_stage_learning import inspect_manifest, settings_for


class TwoStageExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ini = self.root/'pilot.ini'
        self.raw = (ROOT/'archive/research/configs/models/coa-two-stage.ini').read_bytes()
        self.ini.write_bytes(self.raw)
        self.config = load_config(self.ini, repo_root=self.root)
        project = Path(self.config['data']['project'])
        project.parent.mkdir(parents=True)
        project.write_text('{"pilot":"project-a"}')
        self.directory = Path(self.config['data']['directory'])
        self.directory.mkdir(parents=True)
        (self.directory/'dataset-manifest.json').write_text('{"test":"manifest"}')
        samples = []
        for family in range(6):
            for sex in range(2):
                case = f'case-{family}-{sex}'
                for variant in range(2):
                    samples.append({'sample_id': f'{case}:v{variant}', 'case_id': case,
                        'anatomy_family': f'family-{family}', 'variant_id': f'v{variant}',
                        'source_npz': f'/not-loaded/{case}/source.npz',
                        'variant_npz': f'/not-loaded/{case}/v{variant}.npz',
                        'before_mask_sha256': hashlib.sha256(str(family).encode()).hexdigest(),
                        'changes_npz': f'/not-loaded/{case}/change{variant}.npz',
                        'geometry': {'crop_shape_kji': [12, 16, 20], 'spacing_ijk_mm': [1., 1., 1.]}})
        self.manifest = {'samples': samples, 'family_status': 'provisional',
                         'project_sha256': hashlib.sha256(project.read_bytes()).hexdigest()}
        self.reader = patch('fakect_two_stage_data.inspect_dataset',
                            side_effect=lambda *a, **kw: deepcopy(self.manifest)).start()
        self.addCleanup(patch.stopall)

    def write_ini(self, content):
        self.ini.write_text(content)
        return load_config(self.ini, repo_root=self.root)

    def test_comment_friendly_ini_and_separate_models(self):
        self.assertEqual(self.config['split']['case_counts'], [8, 2, 2])
        self.assertEqual(self.config['stage1']['change_weight'], 1.)
        self.assertEqual(self.config['stage2']['change_weight'], 5.)
        self.assertEqual(self.config['stage1']['target_spacing_mm'], [.6]*3)

    def test_unknown_ini_option_and_bad_model_parameters_fail(self):
        for find, replacement in [('batch_size = 1 ', 'batch_size = 17 '),
                                  ('base_filters = 4 ', 'base_filters = 1 '),
                                  ('0.6,0.6,0.6 ', 'nan,0.6,0.6 '),
                                  ('64,64,64 ', '256,64,64 '),
                                  ('architecture = unet3d', 'architecture_typo = unet3d')]:
            with self.subTest(find=find), self.assertRaises(ValueError):
                self.write_ini(self.raw.decode().replace(find, replacement))

    def test_stage_grids_cannot_disagree(self):
        content = self.raw.decode().replace('[model.reconstruction]',
                    '[model.reconstruction]\ntarget_spacing_mm = 1,1,1')
        with self.assertRaisesRegex(ValueError, 'same physical grid'):
            self.write_ini(content)

    def test_family_assignment_is_shared_deterministic_and_complete(self):
        first, second = plan(self.config), plan(self.config)
        self.assertEqual(first['design'], second['design'])
        self.assertEqual(first['split_counts'], {'train': 16, 'validation': 4, 'test': 4})
        self.assertEqual(first['design']['counts']['cases'], 12)
        by_family = {}
        for row in first['samples']:
            by_family.setdefault(row['anatomy_family'], set()).add(row['split'])
        self.assertTrue(all(len(parts) == 1 for parts in by_family.values()))
        self.assertEqual(first['family_status'], 'provisional')
        self.assertEqual(first['scope'], 'engineering')
        self.assertEqual(first['model_grids']['case-0-0']['model_shape_kji'], [20, 27, 34])

    def test_identical_baseline_cannot_cross_partitions(self):
        assigned = plan(self.config)['samples']
        train = next(row for row in assigned if row['split'] == 'train')
        test = next(row for row in assigned if row['split'] == 'test')
        for row in self.manifest['samples']:
            if row['anatomy_family'] == test['anatomy_family']:
                row['before_mask_sha256'] = train['before_mask_sha256']
        with self.assertRaisesRegex(ValueError, 'Identical baseline'):
            plan(self.config)

    def test_oversized_resampled_grid_rejected_before_training(self):
        config = deepcopy(self.config)
        config['model']['target_spacing_mm'] = [.001]*3
        config['stage1']['target_spacing_mm'] = [.001]*3
        with self.assertRaisesRegex(ValueError, 'voxel crop limit'):
            plan(config)

    def test_missing_prepared_data_reports_not_ready(self):
        (self.directory/'dataset-manifest.json').unlink()
        result = plan(self.config)
        self.assertFalse(result['ready'])
        self.assertEqual(result['next_stage'], 'prepare')

    def test_existing_data_is_bound_to_project_and_export_option(self):
        self.assertEqual(prepare(self.config)['family_status'], 'provisional')
        config = deepcopy(self.config)
        config['data']['write_change_masks'] = False
        with self.assertRaisesRegex(ValueError, 'write_change_masks'):
            prepare(config)
        Path(self.config['data']['project']).write_text('{"different":true}')
        with self.assertRaisesRegex(ValueError, 'Requested project'):
            plan(self.config)

    def test_lock_preserves_data_source_and_code_bindings(self):
        result = freeze(self.config, self.raw)
        inspected = inspect_lock(self.config, self.raw)
        self.assertEqual(result, inspected)
        self.assertIn('fakect_segmentation.py', result['code_sha256'])
        self.assertFalse(any('model1' in str(p) for p in Path(self.config['experiment']['output_directory']).iterdir()))
        with self.assertRaises(FileExistsError):
            freeze(self.config, self.raw)

    def test_changed_config_or_implementation_requires_new_run(self):
        freeze(self.config, self.raw)
        config = deepcopy(self.config)
        config['stage1']['epochs'] += 1
        with self.assertRaisesRegex(ValueError, 'INI differs'):
            inspect_lock(config, self.raw)
        with patch('fakect_two_stage_experiment._code_hashes', return_value={'changed': 'code'}):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                inspect_lock(self.config, self.raw)

    def test_changed_manifest_or_lock_cannot_be_silently_reused(self):
        freeze(self.config, self.raw)
        path = self.directory/'dataset-manifest.json'
        path.write_text('tampered data manifest')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            inspect_lock(self.config, self.raw)
        lock_path = Path(self.config['experiment']['output_directory'])/'experiment-lock.json'
        lock_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'lock checksum'):
            inspect_lock(self.config, self.raw)

    def write_signed_json(self, path, document):
        path.write_text(json.dumps(document, allow_nan=False))
        path.with_name(path.name+'.sha256').write_text(sha256(path)+'\n')

    def completed_model(self, lock, stage=1):
        directory = Path(self.config['experiment']['output_directory'])/f'model{stage}'
        directory.mkdir()
        (directory/'model.keras').write_bytes(f'completed model {stage}'.encode())
        settings = settings_for(self.config, stage)
        history = [{'epoch': n+1, 'validation_loss': 1/(n+1)} for n in range(settings['epochs'])]
        (directory/'history.json').write_text(json.dumps(history))
        _, fingerprint = inspect_manifest(lock)
        metadata = dict(schema_version='fakect.two-stage-model/1', stage=stage, settings=settings,
                        manifest_fingerprint=fingerprint, model_sha256=sha256(directory/'model.keras'),
                        source_sha256=sha256(ROOT/'src'/'fakect_two_stage_learning.py'),
                        history_sha256=sha256(directory/'history.json'), epochs_completed=settings['epochs'],
                        best_epoch=settings['epochs'], best_validation_loss=history[-1]['validation_loss'],
                        history=history, test_payloads_loaded=False,
                        loaded_samples=[row['sample_id'] for row in lock['samples'] if row['split'] != 'test'])
        self.write_signed_json(directory/'model-metadata.json', metadata)
        return directory, metadata

    def completed_evaluation(self, lock, stage=1):
        root = Path(self.config['experiment']['output_directory'])
        for number in range(1, stage+1):
            if not (root/f'model{number}').exists():
                self.completed_model(lock, number)
        output = root/f'test-model{stage}'
        output.mkdir()
        rows = []
        for index, source in enumerate(row for row in lock['samples'] if row['split'] == 'test'):
            prediction = output/f'{index:04d}-prediction.npz'
            prediction.write_bytes(b'pinned prediction fixture')
            rows.append({**{key: source[key] for key in ('sample_id', 'case_id', 'anatomy_family')},
                         'prediction_file': prediction.name, 'prediction_sha256': sha256(prediction),
                         'metrics': {'aorta': {'dice': .8}}})
        _, fingerprint = inspect_manifest(lock)
        result = dict(schema_version='fakect.two-stage-evaluation/1', stage=stage,
                      manifest_fingerprint=fingerprint, model_sha256=sha256(root/f'model{stage}'/'model.keras'),
                      stage1_model_sha256=sha256(root/'model1'/'model.keras') if stage == 2 else None,
                      threshold=settings_for(self.config, stage)['threshold'],
                      target_spacing_ijk_mm=settings_for(self.config, stage)['target_spacing_mm'],
                      test_samples=len(rows), records=rows, summary={'aorta': {'family_macro': {'dice': .8}}})
        self.write_signed_json(output/'evaluation.json', result)
        return output, result

    def test_completed_fit_is_reused_without_tensorflow_or_payload_reads(self):
        lock = freeze(self.config, self.raw)
        directory, metadata = self.completed_model(lock)
        original = {path: sha256(path) for path in directory.iterdir()}
        with patch('fakect_two_stage_learning.fit_stage') as train:
            result = fit(self.config, self.raw, 1)
        train.assert_not_called()
        self.assertTrue(result['reused'])
        self.assertEqual(result['model_sha256'], metadata['model_sha256'])
        self.assertEqual(original, {path: sha256(path) for path in directory.iterdir()})

    def test_completed_fit_rejects_changed_weights_metadata_or_history(self):
        lock = freeze(self.config, self.raw)
        directory, metadata = self.completed_model(lock)
        for name in ('model.keras', 'model-metadata.json', 'history.json'):
            path = directory/name
            original = path.read_bytes()
            with self.subTest(name=name):
                path.write_bytes(original+b' ')
                with self.assertRaises((ValueError, OSError)):
                    fit(self.config, self.raw, 1)
                path.write_bytes(original)

    def test_partial_fit_is_preserved_and_not_restarted(self):
        freeze(self.config, self.raw)
        directory = Path(self.config['experiment']['output_directory'])/'model1'
        directory.mkdir()
        partial = directory/'history.json'; partial.write_text('[{"epoch":1}]')
        with self.assertRaisesRegex(FileExistsError, 'not empty'):
            fit(self.config, self.raw, 1)
        self.assertEqual(partial.read_text(), '[{"epoch":1}]')

    def test_completed_evaluation_reuses_exact_predictions_and_scores(self):
        lock = freeze(self.config, self.raw)
        output, result = self.completed_evaluation(lock, 2)
        original = {path: sha256(path) for path in output.iterdir()}
        with patch('fakect_two_stage_learning.evaluate_stage') as evaluate:
            reused = _evaluate_one(self.config, lock, 2, output.parent)
        evaluate.assert_not_called()
        self.assertEqual(result, reused)
        self.assertEqual(original, {path: sha256(path) for path in output.iterdir()})

    def test_completed_evaluation_rejects_changed_prediction_or_numeric_scores(self):
        lock = freeze(self.config, self.raw)
        output, result = self.completed_evaluation(lock)
        prediction = output/result['records'][0]['prediction_file']
        original = prediction.read_bytes()
        prediction.write_bytes(original+b'changed')
        with self.assertRaisesRegex(ValueError, 'prediction checksum'):
            _evaluate_one(self.config, lock, 1, output.parent)
        prediction.write_bytes(original)
        result['summary']['aorta']['family_macro']['dice'] = .999
        (output/'evaluation.json').write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            _evaluate_one(self.config, lock, 1, output.parent)

    def test_completed_evaluation_rejects_wrong_grid_counts_and_partition(self):
        lock = freeze(self.config, self.raw)
        output, result = self.completed_evaluation(lock)
        for key, value in [('target_spacing_ijk_mm', [1., 1., 1.]), ('test_samples', 999),
                           ('records', result['records'][:-1])]:
            changed = deepcopy(result); changed[key] = value
            self.write_signed_json(output/'evaluation.json', changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                _evaluate_one(self.config, lock, 1, output.parent)

    def test_partial_evaluation_does_not_overwrite_completed_predictions(self):
        lock = freeze(self.config, self.raw)
        root, metadata = self.completed_model(lock)
        output = root.parent/'test-model1'; output.mkdir()
        prediction = output/'0000-prediction.npz'; prediction.write_bytes(b'partial run')
        with patch('fakect_two_stage_learning.load_stage', return_value=(object(), metadata)):
            with self.assertRaisesRegex(FileExistsError, 'not empty'):
                _evaluate_one(self.config, lock, 1, root.parent)
        self.assertEqual(prediction.read_bytes(), b'partial run')


if __name__ == '__main__':
    unittest.main()
