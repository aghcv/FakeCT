"""Mask-only experiments preserve splits/provenance and never require Model 1."""
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
from fakect_mask_model import load_config, plan, freeze, inspect_lock, fit, evaluate, sha256
from fakect_two_stage_learning import inspect_manifest, settings_for


class MaskModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = (ROOT/'archive/research/configs/models/coa-mask-model.ini').read_bytes()
        self.ini = self.root/'model.ini'
        self.ini.write_bytes(self.raw)
        self.config = load_config(self.ini, repo_root=self.root)
        self.path = Path(self.config['data']['manifest'])
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{}')
        samples = []
        for family in range(6):
            for member in range(2):
                case = f'case-{family}-{member}'
                for variant in range(2):
                    samples.append({'sample_id': f'{case}:v{variant}', 'case_id': case,
                        'anatomy_family': f'family-{family}', 'variant_id': f'v{variant}',
                        'source_npz': f'/not-loaded/{case}/source.npz',
                        'variant_npz': f'/not-loaded/{case}/v{variant}.npz',
                        'before_mask_sha256': hashlib.sha256(str(family).encode()).hexdigest(),
                        'geometry': {'crop_shape_kji': [12, 16, 20],
                                     'spacing_ijk_mm': [1., 1., 1.]}})
        self.manifest = {'samples': samples, 'family_status': 'provisional'}
        self.reader = patch('fakect_two_stage_data.inspect_dataset',
            side_effect=lambda *args, **kwargs: deepcopy(self.manifest)).start()
        self.addCleanup(patch.stopall)

    def signed(self, path, value):
        path.write_text(json.dumps(value))
        path.with_name(path.name+'.sha256').write_text(sha256(path)+'\n')

    def completed_model(self, lock):
        directory = Path(self.config['experiment']['output_directory'])/'model2'
        directory.mkdir()
        (directory/'model.keras').write_bytes(b'model fixture')
        (directory/'history.json').write_text('[{"epoch":5}]')
        _, fingerprint = inspect_manifest(lock)
        settings = settings_for(self.config, 2)
        metadata = {'stage': 2, 'settings': settings, 'manifest_fingerprint': fingerprint,
            'model_sha256': sha256(directory/'model.keras'),
            'history_sha256': sha256(directory/'history.json'),
            'source_sha256': sha256(ROOT/'src/fakect_two_stage_learning.py'),
            'test_payloads_loaded': False, 'epochs_completed': settings['epochs']}
        self.signed(directory/'model-metadata.json', metadata)
        return directory, metadata

    def completed_evaluation(self, lock, metadata):
        directory = Path(self.config['experiment']['output_directory'])/'test-model2'
        directory.mkdir()
        records = []
        for number, row in enumerate(r for r in lock['samples'] if r['split'] == 'test'):
            path = directory/f'{number:04d}-prediction.npz'
            path.write_bytes(b'prediction fixture')
            records.append({'sample_id': row['sample_id'], 'prediction_file': path.name,
                            'prediction_sha256': sha256(path)})
        result = {'stage': 2, 'manifest_fingerprint': metadata['manifest_fingerprint'],
            'model_sha256': metadata['model_sha256'], 'stage1_model_sha256': None,
            'threshold': .5, 'target_spacing_ijk_mm': [.6]*3,
            'test_samples': len(records), 'records': records}
        self.signed(directory/'evaluation.json', result)
        return directory, result

    def test_ini_and_plan_have_no_learned_segmentation_stage(self):
        self.assertNotIn('stage1', self.config)
        result = plan(self.config)
        self.assertTrue(result['ready'])
        self.assertEqual(result['target_scope'], 'selected_lineage')
        self.assertEqual(result['model_settings']['stage'], 2)
        self.assertEqual(result['split_counts'], {'train': 16, 'validation': 4, 'test': 4})
        self.assertEqual(result['design'], plan(self.config)['design'])
        self.assertEqual(result['model_grids']['case-0-0']['model_shape_kji'], [20, 27, 34])
        assignments = {}
        for row in result['samples']:
            assignments.setdefault(row['anatomy_family'], set()).add(row['split'])
        self.assertTrue(all(len(parts) == 1 for parts in assignments.values()))
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())

    def test_invalid_scope_unknown_keys_and_model_settings_rejected(self):
        for find, replacement in [('selected_lineage ', 'whole_thoracic '),
                                  ('base_filters = 4 ', 'base_filters = 1 '),
                                  ('batch_size = 1 ', 'batch_size = 17 '),
                                  ('0.6,0.6,0.6 ', 'nan,0.6,0.6 '),
                                  ('manifest = ', 'manifest_typo = ')]:
            with self.subTest(find=find), self.assertRaises(ValueError):
                self.ini.write_text(self.raw.decode().replace(find, replacement))
                load_config(self.ini, repo_root=self.root)

    def test_missing_dataset_is_not_ready(self):
        self.path.unlink()
        self.assertFalse(plan(self.config)['ready'])

    def test_shared_reference_leakage_is_rejected(self):
        samples = plan(self.config)['samples']
        train = next(r for r in samples if r['split'] == 'train')
        test = next(r for r in samples if r['split'] == 'test')
        for row in self.manifest['samples']:
            if row['anatomy_family'] == test['anatomy_family']:
                row['before_mask_sha256'] = train['before_mask_sha256']
        with self.assertRaisesRegex(ValueError, 'reference anatomy crosses splits'):
            plan(self.config)

    def test_oversized_grid_is_rejected(self):
        config = deepcopy(self.config)
        config['model']['target_spacing_mm'] = [.001]*3
        with self.assertRaisesRegex(ValueError, 'voxel crop limit'):
            plan(config)

    def test_freeze_locks_config_data_and_source_snapshots(self):
        lock = freeze(self.config, self.raw)
        self.assertEqual(lock, inspect_lock(self.config, self.raw))
        directory = Path(self.config['experiment']['output_directory'])
        self.assertTrue((directory/'source/fakect_mask_model.py').is_file())
        with self.assertRaises(FileExistsError):
            freeze(self.config, self.raw)
        (directory/'source/fakect_mask_model.py').write_text('changed snapshot')
        with self.assertRaisesRegex(ValueError, 'snapshot checksum'):
            inspect_lock(self.config, self.raw)

    def test_changed_ini_dataset_or_code_fails_reuse(self):
        freeze(self.config, self.raw)
        with self.assertRaisesRegex(ValueError, 'INI differs'):
            inspect_lock(self.config, self.raw+b'\n')
        with patch('fakect_mask_model._code_hashes', return_value={'changed': 'code'}):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                inspect_lock(self.config, self.raw)
        self.path.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            inspect_lock(self.config, self.raw)

    def test_fit_delegates_only_to_stage_two(self):
        lock = freeze(self.config, self.raw)
        with patch('fakect_two_stage_learning.fit_stage', return_value={'trained': True}) as backend:
            self.assertEqual(fit(self.config, self.raw), {'trained': True})
        self.assertEqual(backend.call_args.args[:3], (self.config, lock, 2))
        self.assertEqual(backend.call_args.args[3].name, 'model2')

    def test_completed_fit_reuses_checked_weights_without_tensorflow(self):
        lock = freeze(self.config, self.raw)
        directory, _ = self.completed_model(lock)
        with patch('fakect_two_stage_learning.fit_stage', side_effect=AssertionError('refit')):
            self.assertTrue(fit(self.config, self.raw)['reused'])
        (directory/'model.keras').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            fit(self.config, self.raw)

    def test_partial_fit_does_not_restart(self):
        freeze(self.config, self.raw)
        directory = Path(self.config['experiment']['output_directory'])/'model2'
        directory.mkdir()
        (directory/'history.json').write_text('[]')
        with self.assertRaisesRegex(FileExistsError, 'Partial model'):
            fit(self.config, self.raw)

    def test_test_stage_explicitly_bypasses_model_one(self):
        lock = freeze(self.config, self.raw)
        self.completed_model(lock)
        with patch('fakect_two_stage_learning.evaluate_stage', return_value={'stage': 2}) as backend:
            result = evaluate(self.config, self.raw)
        self.assertEqual(backend.call_args.args[2], 2)
        self.assertIsNone(backend.call_args.kwargs['stage1_model_path'])
        self.assertEqual(result['schema_version'], 'fakect.mask-model-results/1')
        self.assertNotIn('stage1', result)

    def test_completed_evaluation_reuses_exact_prediction_checksums(self):
        lock = freeze(self.config, self.raw)
        _, metadata = self.completed_model(lock)
        directory, original = self.completed_evaluation(lock, metadata)
        with patch('fakect_two_stage_learning.evaluate_stage', side_effect=AssertionError('retest')):
            self.assertEqual(evaluate(self.config, self.raw)['stage2'], original)
        (directory/original['records'][0]['prediction_file']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'prediction checksum'):
            evaluate(self.config, self.raw)

    def test_chained_evaluation_cannot_masquerade_as_mask_only(self):
        lock = freeze(self.config, self.raw)
        _, metadata = self.completed_model(lock)
        directory, original = self.completed_evaluation(lock, metadata)
        original['stage1_model_sha256'] = 'unexpected-model1'
        self.signed(directory/'evaluation.json', original)
        with self.assertRaisesRegex(ValueError, 'mask model/partition'):
            evaluate(self.config, self.raw)

    def test_missing_predictions_or_partial_evaluation_are_rejected(self):
        lock = freeze(self.config, self.raw)
        _, metadata = self.completed_model(lock)
        directory, original = self.completed_evaluation(lock, metadata)
        del original['records'][0]['prediction_file']
        self.signed(directory/'evaluation.json', original)
        with self.assertRaisesRegex(ValueError, 'missing required predictions'):
            evaluate(self.config, self.raw)
        (directory/'evaluation.json').unlink()
        with self.assertRaisesRegex(FileExistsError, 'Partial evaluation'):
            evaluate(self.config, self.raw)


if __name__ == '__main__':
    unittest.main()
