"""Bounded paired-data planning/export checks on a tiny signed-label phantom."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_morphology import apply_morphology
from fakect_roi import prepare_crop, resolve_preview
from fakect_training_data import (plan_variants, prepare_training_dataset,
                                   dataset_fingerprint, validate_dataset, validate_study_plan)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        directory = self.root / '001'
        directory.mkdir()
        shape = (31, 31, 31)
        k, j, i = np.indices(shape)
        self.labels = np.full(shape, 20, dtype='<f4')
        self.labels[(i - 15)**2 + (j - 15)**2 <= 4] = -7
        self.labels[:, 2:4, 2:4] = 8  # A second artery is not the training target.
        self.scalar = np.where(self.labels == -7, .026, .019).astype('<f4')
        self.scalar += (k / 1e6).astype('<f4')
        self.labels.tofile(directory / '001_act_1.bin')
        self.scalar.tofile(directory / '001_atn_1.bin')
        par, log = directory / '001.par', directory / '001.log'
        par.write_text('test geometry parameters')
        log.write_text('test scalar and source metadata')
        catalog = {'policy_version': 'test', 'sources': {},
                   'categories': [{'id': code, 'name': name} for code, name in
                                  [(0, 'background'), (1, 'soft_tissue'), (5, 'artery'), (255, 'unknown')]],
                   'records': [{'original_id': value, 'original_name': name,
                                'classification': {'tissue_id': code, 'tissue_name': tissue}}
                               for value, name, code, tissue in
                               [(0, 'background', 0, 'background'), (20, 'soft', 1, 'soft_tissue'),
                                (-7, 'aorta', 5, 'artery'), (8, 'other_artery', 5, 'artery')]]}
        catalog_path = self.root / 'catalog.json'
        catalog_path.write_text(json.dumps(catalog))
        audit = {'cases': [{'case_id': '001', 'directory': str(directory),
                           'shape_kji': shape, 'spacing_ijk_mm': [1., 1., 1.],
                           'par_path': str(par), 'log_path': str(log)}],
                 'source_metadata_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in (par, log)}}
        audit_path = self.root / 'audit.json'
        audit_path.write_text(json.dumps(audit))
        self.config = {
            'study': {'schema_version': 'fakect.study/1', 'name': 'test'},
            'input': {'root': self.root, 'case_id': '001', 'frame': 1,
                      'catalog': catalog_path, 'audit': audit_path},
            'selection': {'tissue': 'artery', 'source_ids': (-7,)},
            'roi': {'shape': 'sphere', 'center_ijk': (15, 15, 15), 'radius_mm': 5.,
                    'crop_half_width_mm': 15., 'coordinate_reviewed': False},
            'preview': {'slice_ijk': None, 'volume_stride': 2, 'context_tissues': (),
                        'overlay_opacity': .2, 'volume_opacity': .25, 'context_opacity': .08},
            'output': {'directory': self.root / 'preview'},
            'edit': {'operation': 'erosion', 'distance_mm': 1., 'profile': 'uniform',
                     'profile_axis': 'k', 'shape_k': 10., 'shape_window': (0., 1.)},
            'reassignment': {'allowed_tissues': ('soft_tissue',), 'max_distance_mm': 3.,
                             'unresolved': 'preserve'},
            'train': {'stage': 'prepare', 'dataset_directory': self.root / 'dataset',
                      'model_directory': self.root / 'model', 'target_source_ids': (-7,),
                      'anatomy_family': 'test_original_family', 'operations': ('erosion', 'dilation'),
                      'distances_mm': (.5, 1., 2.), 'shape_ks': (10.,), 'include_baseline': True,
                      'max_variants': 10, 'image_method': 'attenuation_copy_proxy',
                      'split_mode': 'scenario_only', 'validation_fraction': .2,
                      'test_fraction': .2, 'split_seed': 42},
            'model': {'architecture': 'unet2d', 'patch_size': (16, 16), 'slice_axis': 'k',
                      'normalization': 'fixed_clip', 'clip_min': 0., 'clip_max': .3,
                      'epochs': 1, 'batch_size': 2, 'learning_rate': .001, 'seed': 42}}
        self.resolved = resolve_preview(self.config)

    def test_cartesian_plan_is_stable_bounded_and_whole_variant_split(self):
        planned = plan_variants(self.config)
        self.assertEqual(len(planned), 7)
        self.assertEqual(planned[0]['variant_id'], 'baseline')
        self.assertEqual(planned[0]['split'], 'train')
        self.assertEqual(planned[0]['operation'], 'none')
        self.assertEqual({variant['split'] for variant in planned}, {'train', 'validation', 'test'})
        reordered = deepcopy(self.config)
        reordered['train']['operations'] = tuple(reversed(reordered['train']['operations']))
        reordered['train']['distances_mm'] = tuple(reversed(reordered['train']['distances_mm']))
        self.assertEqual(planned, plan_variants(reordered))
        self.assertFalse(self.config['train']['dataset_directory'].exists())
        invalid = [('max_variants', 6), ('operations', ('erosion', 'erosion')),
                   ('distances_mm', (1., 1.)), ('distances_mm', (float('nan'),)),
                   ('shape_ks', (0,)), ('validation_fraction', .9), ('split_mode', 'patient'),
                   ('image_method', 'recovered_ct'), ('split_seed', -1)]
        for field, value in invalid:
            config = deepcopy(self.config)
            config['train'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan_variants(config)

    def test_prepared_pairs_preserve_signed_source_and_full_target_outside_roi(self):
        source_bytes = {name: path.read_bytes() for name, path in self.resolved['source_files'].items()}
        result = prepare_training_dataset(self.config, self.resolved)
        manifest = validate_dataset(self.config, self.resolved)
        self.assertEqual(result['sample_count'], 7)
        self.assertEqual(manifest['split_mode'], 'scenario_only')
        self.assertIn('not HU', manifest['image_warning'])
        self.assertFalse((self.config['train']['dataset_directory'] / 'INCOMPLETE').exists())
        baseline = next(sample for sample in manifest['samples'] if sample['variant_id'] == 'baseline')
        directory = self.config['train']['dataset_directory']
        with np.load(directory / baseline['path'], allow_pickle=False) as data:
            self.assertEqual(data['image'].dtype, np.dtype('float32'))
            self.assertEqual(data['mask'].dtype, np.dtype('uint8'))
            np.testing.assert_array_equal(data['mask'], self.labels == -7)
            np.testing.assert_array_equal(data['edited_labels'], self.labels)
            np.testing.assert_array_equal(data['image'], self.scalar / .1)
            self.assertTrue(data['mask'][0, 15, 15])  # Far outside the spherical ROI.
            self.assertFalse(data['mask'][0, 2, 2])   # Same tissue, different original artery.
            self.assertFalse(data['changed_mask'].any())
        with np.load(directory / 'source.npz', allow_pickle=False) as data:
            np.testing.assert_array_equal(data['original_labels'], self.labels)
            np.testing.assert_array_equal(data['original_attenuation_per_pixel'], self.scalar)
            self.assertEqual(data['original_labels'].dtype, np.dtype('<f4'))
        for name, path in self.resolved['source_files'].items():
            self.assertEqual(path.read_bytes(), source_bytes[name])
        with self.assertRaises(FileExistsError):
            prepare_training_dataset(self.config, self.resolved)

    def test_every_variant_starts_from_original_and_proxy_agrees_with_edit(self):
        prepare_training_dataset(self.config, self.resolved)
        manifest = validate_dataset(self.config, self.resolved)
        arrays, _ = prepare_crop(self.resolved, self.config)
        original = {name: array.copy() for name, array in arrays.items()}
        for sample in manifest['samples']:
            config = deepcopy(self.config)
            for key in ('operation', 'distance_mm', 'shape_k', 'shape_window', 'profile', 'profile_axis'):
                config['edit'][key] = sample[key]
            expected = apply_morphology(arrays, self.resolved, config)
            with np.load(self.config['train']['dataset_directory'] / sample['path'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['edited_labels'], expected['edited_labels'])
                np.testing.assert_array_equal(data['mask'], expected['edited_labels'] == -7)
                np.testing.assert_array_equal(data['image'], expected['attenuation_proxy_per_pixel'] / .1)
                np.testing.assert_array_equal(data['changed_mask'], expected['changed_mask'])
        for name, array in arrays.items():
            np.testing.assert_array_equal(array, original[name])

    def test_noop_variants_remain_recorded_but_duplicate_geometry_never_crosses_splits(self):
        config = deepcopy(self.config)
        config['train']['distances_mm'] = (.1, .2, .3)
        prepare_training_dataset(config, self.resolved)
        manifest = validate_dataset(config, self.resolved)
        self.assertEqual(manifest['sample_count'], 7)
        self.assertEqual(manifest['zero_change_count'], 7)
        self.assertEqual(manifest['split_counts'], {'train': 7, 'validation': 0, 'test': 0})
        self.assertEqual(manifest['geometry_group_counts']['train'], 1)
        self.assertTrue(any(sample['planned_split'] != sample['split'] for sample in manifest['samples']))
        self.assertTrue(any('fitting must not proceed' in warning for warning in manifest['warnings']))
        group_splits = {}
        for sample in manifest['samples']:
            self.assertEqual(group_splits.setdefault(sample['geometry_group'], sample['split']), sample['split'])

    def test_fingerprint_binds_data_but_allows_model_and_stage_comparisons(self):
        initial = dataset_fingerprint(self.config, self.resolved)
        config = deepcopy(self.config)
        config['train']['stage'] = 'fit'
        config['train']['model_directory'] = self.root / 'another-model'
        config['train']['dataset_directory'] = self.root / 'another-destination'
        config['output']['directory'] = self.root / 'another-preview'
        config['model']['epochs'] = 100
        config['model']['clip_max'] = 100
        config['preview']['overlay_opacity'] = .99
        self.assertEqual(dataset_fingerprint(config, self.resolved), initial)
        config['train']['distances_mm'] = (1.,)
        self.assertNotEqual(dataset_fingerprint(config, self.resolved), initial)
        config = deepcopy(self.config)
        config['train']['target_source_ids'] = (8,)
        self.assertNotEqual(dataset_fingerprint(config, self.resolved), initial)
        self.config['input']['catalog'].write_text('{}')
        with self.assertRaisesRegex(ValueError, 'catalog changed'):
            dataset_fingerprint(self.config, self.resolved)

    def test_existing_artifacts_and_source_changes_are_detected_before_reuse(self):
        prepare_training_dataset(self.config, self.resolved)
        changed = deepcopy(self.config)
        changed['roi']['radius_mm'] = 4.
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_dataset(changed, self.resolved)
        manifest = validate_dataset(self.config, self.resolved)
        archive = self.config['train']['dataset_directory'] / manifest['samples'][0]['path']
        with archive.open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            validate_dataset(self.config, self.resolved)

    def test_changed_source_stat_and_audit_metadata_invalidate_frozen_data(self):
        prepare_training_dataset(self.config, self.resolved)
        before = dataset_fingerprint(self.config, self.resolved)
        source = self.resolved['source_files']['act']
        with source.open('r+b') as handle:
            handle.write(np.asarray([21.], dtype='<f4').tobytes())
        self.assertNotEqual(dataset_fingerprint(self.config, self.resolved), before)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_dataset(self.config, self.resolved)
        before = dataset_fingerprint(self.config, self.resolved)
        Path(self.resolved['case']['par_path']).write_text('changed geometry parameters')
        self.assertNotEqual(dataset_fingerprint(self.config, self.resolved), before)

    def test_manifest_tampering_and_duplicate_geometry_splits_are_rejected(self):
        prepare_training_dataset(self.config, self.resolved)
        directory = self.config['train']['dataset_directory']
        path = directory / 'dataset-manifest.json'
        content = path.read_bytes()
        path.write_bytes(content + b' ')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_dataset(self.config, self.resolved)
        manifest = json.loads(content)
        original = manifest['samples'][0]
        duplicate = next(sample for sample in manifest['samples'][1:]
                         if sample['mask_sha256'] == original['mask_sha256'])
        duplicate['split'] = 'test'
        path.write_text(json.dumps(manifest))
        (directory / 'dataset-manifest.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'geometry crosses'):
            validate_dataset(self.config, self.resolved)

    def test_explicit_anatomical_target_and_edit_scope_are_required(self):
        for targets in ((), (-7, -7), (1234,), (20,), (0,), (True,)):
            config = deepcopy(self.config)
            config['train']['target_source_ids'] = targets
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                prepare_training_dataset(config, self.resolved)
        resolved = deepcopy(self.resolved)
        resolved['source_ids'] = (-7, 8)
        with self.assertRaisesRegex(ValueError, 'subset'):
            prepare_training_dataset(self.config, resolved)
        self.assertFalse(self.config['train']['dataset_directory'].exists())

    def test_metadata_plan_checks_all_edit_strengths_and_targets_without_crop_reads(self):
        with patch('fakect_training_data.prepare_crop', side_effect=AssertionError('must not read voxels')):
            self.assertEqual(validate_study_plan(self.config, self.resolved), plan_variants(self.config))
            bad_target = deepcopy(self.config)
            bad_target['train']['target_source_ids'] = (8,)
            with self.assertRaisesRegex(ValueError, 'subset'):
                validate_study_plan(bad_target, self.resolved)
            bad_halo = deepcopy(self.config)
            bad_halo['train']['distances_mm'] = (1., 20.)
            # The original edit distance remains valid; the larger planned
            # variant must still be checked before advertising the study plan.
            with self.assertRaisesRegex(ValueError, 'halo|context'):
                validate_study_plan(bad_halo, self.resolved)
        self.assertFalse(self.config['train']['dataset_directory'].exists())

    def test_exact_input_snapshot_is_optional_and_bound_to_artifact_inventory(self):
        input_bytes = b'\xef\xbb\xbf# Exact original comments\r\n[study]\r\nname = tiny-study\r\n'
        prepare_training_dataset(self.config, self.resolved, input_bytes=input_bytes)
        manifest = validate_dataset(self.config, self.resolved)
        snapshot = self.config['train']['dataset_directory'] / 'input.ini'
        self.assertEqual(snapshot.read_bytes(), input_bytes)
        self.assertEqual(manifest['input_snapshot'], 'input.ini')
        checksum = hashlib.sha256(input_bytes).hexdigest()
        self.assertEqual(manifest['input_config_sha256'], checksum)
        self.assertEqual(manifest['artifacts_sha256']['input.ini'], checksum)
        snapshot.write_bytes(input_bytes + b'# changed\n')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            validate_dataset(self.config, self.resolved)
        with self.assertRaisesRegex(ValueError, 'input_bytes'):
            prepare_training_dataset(self.config, self.resolved, input_bytes='text is not exact bytes')

    def test_failed_variant_leaves_incomplete_dataset_without_publishable_manifest(self):
        with patch('fakect_training_data.apply_morphology', side_effect=ValueError('synthetic failure')):
            with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                prepare_training_dataset(self.config, self.resolved)
        output = self.config['train']['dataset_directory']
        self.assertTrue((output / 'INCOMPLETE').exists())
        self.assertFalse((output / 'dataset-manifest.json').exists())
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
            validate_dataset(self.config, self.resolved)


if __name__ == '__main__':
    unittest.main()
