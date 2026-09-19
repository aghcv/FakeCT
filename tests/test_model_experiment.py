"""Family splits and immutable model-only experiments over prepared artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from fakect_model_experiment import (fit_model_experiment, freeze_model_experiment,
                                     inspect_experiment_lock, load_model_experiment_config,
                                     plan_model_experiment)
from fakect_segmentation import PatchDataset, model_settings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


class ModelExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entries = {}
        self.manifests = {}
        for index in range(3):
            self.make_dataset(f'case-{index}', f'family-{index}', index)
        self.config = {
            'experiment': {'schema_version': 'fakect.model-experiment/1',
                           'registry': str(self.root/'registry'), 'datasets': list(self.entries),
                           'output_directory': str(self.root/'experiment')},
            'split': {'mode': 'anatomy_family', 'validation_fraction': .2,
                      'test_fraction': .2, 'seed': 42},
            'model': model_settings({'model': {'patch_size': [4, 4], 'batch_size': 2}})}
        self.raw = b'fixture immutable INI\n'
        # The registry verifier itself has separate payload/inventory tests.
        # Here receipts are real files so experiment pinning is not mocked.
        self.addCleanup(patch.stopall)
        self.registry = patch('fakect_cohort_registry.load_registry_entry', side_effect=self.load_entry).start()
        self.verifier = patch('fakect_cohort_registry.verify_prepared_manifest', side_effect=self.verify_manifest).start()

    def make_dataset(self, name, family, index, shape=None):
        directory = self.root/name
        directory.mkdir()
        shape = shape or (2+index, 5+index, 7)
        samples = []
        for variant in range(2):
            mask = np.zeros(shape, np.uint8)
            mask.ravel()[index:index+variant+2] = 1
            image = (.1 + .1*index + .1*mask).astype(np.float32)
            path = directory/f'variant-{variant}.npz'
            np.savez_compressed(path, image=image, mask=mask)
            samples.append({'variant_id': f'variant-{variant}', 'path': path.name,
                            '_path': str(path), 'split': 'unassigned', 'shape_kji': list(shape),
                            'sha256': digest(path), 'image_sha256': array_digest(image),
                            'mask_sha256': array_digest(mask), 'geometry_group': f'{family}:{variant}'})
        manifest = {'schema_version': 'fakect.training-dataset/1', 'complete': True,
                    'split_mode': 'unassigned', 'anatomy_family': family,
                    'case_id': f'anatomy-{index}', 'frame': 1, 'samples': samples,
                    'geometry': {'array_order': 'kji', 'spacing_ijk_mm': [1., 1., 1.],
                                 'crop_shape_kji': list(shape), 'crop_origin_ijk': [0, 0, 0],
                                 'orientation': 'Native fixture index grid'},
                    'image_units': 'cm^-1', 'image_method': 'attenuation_copy_proxy',
                    'target_scope': 'selected_lineage', 'target_semantics': 'Reviewed ROI ancestry',
                    'source_files': {'act': {'path': f'/unavailable/raw-{index}.bin'}},
                    'source_crop_sha256': {'act': hashlib.sha256(str(index).encode()).hexdigest()}}
        path = directory/'dataset-manifest.json'
        path.write_text(json.dumps(manifest))
        receipt = directory/'registry-entry.json'
        receipt.write_text(json.dumps({'family': family, 'name': name}))
        entry = {'name': name, 'anatomy_family': family, 'family_verified': True,
                 'manifest_path': str(path), 'manifest_sha256': digest(path),
                 'dataset_fingerprint': hashlib.sha256(name.encode()).hexdigest(),
                 'case_id': manifest['case_id'], 'frame': 1, 'registry_entry_path': str(receipt),
                 'manifest': manifest}
        self.entries[name] = entry
        self.manifests[str(path)] = manifest
        return entry

    def load_entry(self, registry, name, verify=True):
        if name not in self.entries:
            raise FileNotFoundError(name)
        return copy.deepcopy(self.entries[name])

    def verify_manifest(self, path, verify_payloads=True):
        return copy.deepcopy(self.manifests[str(path)])

    def freeze(self):
        return Path(freeze_model_experiment(self.config, self.raw)['lock_path'])

    def ini(self):
        return '''[experiment]
schema_version = fakect.model-experiment/1
registry = registry
datasets = case-0, case-1, case-2
output_directory = experiment

[split]
mode = anatomy_family
validation_fraction = .2
test_fraction = .2
seed = 42

[model]
architecture = unet2d
patch_size = 4,4
slice_axis = k
normalization = fixed_clip
clip_min = 0
clip_max = .5
epochs = 1
batch_size = 2
learning_rate = .001
seed = 1729
'''

    def test_ini_contains_only_model_inputs_and_repo_relative_paths(self):
        path = self.root/'model.ini'
        path.write_text(self.ini())
        self.assertEqual(load_model_experiment_config(path, repo_root=self.root), self.config)
        for changed in (self.ini()+'\n[roi]\ncenter_ijk=1,2,3\n',
                        self.ini().replace('case-0, case-1, case-2', 'case-0, case-0'),
                        self.ini().replace('validation_fraction = .2', 'validation_fraction = nan')):
            path.write_text(changed)
            with self.assertRaises(ValueError):
                load_model_experiment_config(path, repo_root=self.root)

    def test_validation_reports_missing_and_unverified_together_without_writes(self):
        self.config['experiment']['datasets'].append('pending-phantom')
        self.entries['case-0']['family_verified'] = False
        report = plan_model_experiment(self.config)
        self.assertFalse(report['ready'])
        self.assertEqual(report['missing_datasets'], ['pending-phantom'])
        self.assertEqual(report['unverified_datasets'], ['case-0'])
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())

    def test_three_verified_families_are_required(self):
        self.config['experiment']['datasets'] = ['case-0', 'case-1']
        report = plan_model_experiment(self.config)
        self.assertFalse(report['ready'])
        self.assertTrue(any('three verified' in e for e in report['errors']))
        with self.assertRaisesRegex(ValueError, 'not ready'):
            self.freeze()
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())

    def test_family_partition_deterministic_with_repeated_family_and_namespaced_ids(self):
        extra = self.make_dataset('case-0-later', 'family-0', 4)
        self.config['experiment']['datasets'].append(extra['name'])
        report = plan_model_experiment(self.config)
        self.assertTrue(report['ready'], report['errors'])
        self.config['experiment']['datasets'].reverse()
        reordered = plan_model_experiment(self.config)
        self.assertEqual(report['family_assignment'], reordered['family_assignment'])
        self.assertEqual(len({s['variant_id'] for s in report['samples']}), 8)
        self.assertEqual(len({s['split'] for s in report['samples'] if s['anatomy_family'] == 'family-0'}), 1)
        self.assertEqual(set(s['split'] for s in report['samples']), {'train', 'validation', 'test'})
        self.assertTrue(all(s['split'] == 'unassigned' for m in self.manifests.values() for s in m['samples']))

    def test_same_case_or_source_cannot_be_declared_independent(self):
        for key in ('case_id', 'source_files', 'source_crop_sha256'):
            with self.subTest(key=key):
                old = copy.deepcopy(self.entries['case-1']['manifest'][key])
                self.entries['case-1']['manifest'][key] = copy.deepcopy(self.entries['case-0']['manifest'][key])
                report = plan_model_experiment(self.config)
                self.assertFalse(report['ready'])
                self.assertTrue(any('Shared source' in e for e in report['errors']))
                self.entries['case-1']['manifest'][key] = old

    def test_duplicate_sample_or_geometry_across_families_is_rejected(self):
        one, two = self.entries['case-0']['manifest']['samples'][0], self.entries['case-1']['manifest']['samples'][0]
        two['sha256'] = one['sha256']
        self.assertTrue(any('Duplicate sample' in e for e in plan_model_experiment(self.config)['errors']))
        two['sha256'] = 'd'*64
        two['shape_kji'], two['mask_sha256'] = one['shape_kji'], one['mask_sha256']
        self.assertTrue(any('Duplicate sample' in e for e in plan_model_experiment(self.config)['errors']))

    def test_incompatible_units_spacing_orientation_and_target_scope_rejected(self):
        manifest = self.entries['case-1']['manifest']
        for owner, key, value in ((manifest, 'image_units', 'HU'),
                                  (manifest['geometry'], 'spacing_ijk_mm', [2., 1., 1.]),
                                  (manifest['geometry'], 'orientation', 'Different orientation'),
                                  (manifest, 'target_scope', 'whole_anatomy')):
            with self.subTest(key=key):
                original = owner[key]
                owner[key] = value
                self.assertFalse(plan_model_experiment(self.config)['ready'])
                owner[key] = original

    def test_lock_is_immutable_and_does_not_require_raw_sources(self):
        path = self.freeze()
        self.assertEqual(inspect_experiment_lock(path)['split_sizes'], {'train': 2, 'validation': 2, 'test': 2})
        with self.assertRaises(FileExistsError):
            self.freeze()
        with path.open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            inspect_experiment_lock(path)

    def test_registry_and_manifest_mutation_are_detected(self):
        path = self.freeze()
        receipt = Path(self.entries['case-0']['registry_entry_path'])
        before = receipt.read_bytes()
        receipt.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError, 'receipt or prepared manifest changed'):
            inspect_experiment_lock(path)
        receipt.write_bytes(before)
        manifest = Path(self.entries['case-0']['manifest_path'])
        manifest.write_bytes(manifest.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'receipt or prepared manifest changed'):
            inspect_experiment_lock(path)

    def test_variable_crops_are_tiled_completely_and_test_payloads_never_open(self):
        path = self.freeze()
        locked = inspect_experiment_lock(path)
        test_paths = {s['_path'] for s in locked['samples'] if s['split'] == 'test'}
        for test_path in test_paths:
            test_path.unlink()  # Fit/index construction must not open the held-out arrays.
        with patch('fakect_segmentation.np.load', wraps=np.load) as load:
            dataset = PatchDataset(path, self.config['model'])
            counts = {'train': 0, 'validation': 0}
            for split in counts:
                for _, _, valid in dataset.batches(split):
                    counts[split] += int(valid.sum())
            self.assertTrue(all(Path(call.args[0]) not in test_paths for call in load.call_args_list))
        for split, count in counts.items():
            self.assertEqual(count, sum(np.prod(s['shape_kji']) for s in locked['samples'] if s['split'] == split))
        self.assertTrue(all(not call.kwargs['verify'] for call in self.registry.call_args_list[-3:]))
        self.assertTrue(all(not call.kwargs['verify_payloads'] for call in self.verifier.call_args_list))

    def test_fit_uses_frozen_model_and_checks_training_bytes_before_runner(self):
        path = self.freeze()
        with patch('fakect_segmentation.train_segmentation', return_value={'fitted': True}) as fit:
            result = fit_model_experiment(self.config, self.raw)
            self.assertTrue(result['fitted'])
            self.assertEqual(fit.call_args.args[1], path)
            self.assertEqual(fit.call_args.args[0]['model'], self.config['model'])
        changed = copy.deepcopy(self.config)
        changed['model']['epochs'] += 1
        with self.assertRaisesRegex(ValueError, 'differs from its frozen lock'):
            fit_model_experiment(changed, self.raw)
        samples = inspect_experiment_lock(path)['samples']
        train = next(s for s in samples if s['split'] == 'train')
        with train['_path'].open('ab') as stream:
            stream.write(b'mutated')
        with patch('fakect_segmentation.train_segmentation') as fit:
            with self.assertRaisesRegex(ValueError, 'sample changed'):
                fit_model_experiment(self.config, self.raw)
            fit.assert_not_called()

    def test_cli_validation_never_fits_or_writes_and_reports_readiness(self):
        import model_experiment
        config_path = self.root/'model.ini'
        config_path.write_text(self.ini())
        with patch('model_experiment.load_model_experiment_config', return_value=self.config), \
                patch('model_experiment.fit_model_experiment') as fit:
            result = model_experiment.run(config_path, stage='fit', validate_only=True)
        self.assertTrue(result['ready'])
        self.assertNotIn('samples', result)
        fit.assert_not_called()
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())


if __name__ == '__main__':
    unittest.main()
