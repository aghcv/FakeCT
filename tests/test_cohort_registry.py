"""Frozen cohort storage validates recorded data independently of generator state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import fakect_cohort_registry as registry


def fixture_dataset(directory, *, lineage=True, unassigned=False):
    directory.mkdir()
    (directory/'variants').mkdir()
    geometry = {'array_order': 'kji', 'crop_shape_kji': [3, 4, 5], 'spacing_ijk_mm': [1., 1., 1.]}
    labels = np.zeros((3, 4, 5), dtype=np.int32)
    labels[1, 1:3, 1:4] = 42
    target = labels == 42
    attenuation = np.full(labels.shape, .02, dtype=np.float32)
    image = np.asarray(attenuation/.1, dtype=np.float32)
    roi = np.ones(labels.shape, dtype=bool)
    np.savez_compressed(directory/'source.npz', original_labels=labels,
                        original_attenuation_per_pixel=attenuation, target_mask=target.astype(np.uint8),
                        roi_mask=roi, geometry_json=np.array(json.dumps(geometry)))
    (directory/'input.ini').write_text('[study]\nschema_version = historical-generator/1\n')
    (directory/'data-config.json').write_text(json.dumps({'source': '/unavailable/source.bin', 'code_sha256': {'old.py': 'a'*64}}))
    sample = {'variant_id': 'baseline', 'case_id': '260602', 'frame': 1,
              'anatomy_family': 'unverified', 'geometry': geometry,
              'split': 'unassigned' if unassigned else 'train',
              'geometry_group': 'unverified:mask', 'dataset_fingerprint': 'b'*64,
              'image_method': 'attenuation_copy_proxy', 'image_units': 'cm^-1',
              'image_sha256': registry._array_hash(image), 'mask_sha256': registry._array_hash(target.astype(np.uint8)),
              'foreground_voxels': int(target.sum()),
              'mask_semantics': registry.LINEAGE_SEMANTICS if lineage else registry.ID_SEMANTICS}
    arrays = dict(image=image, mask=target.astype(np.uint8), edited_labels=labels,
                  changed_mask=np.zeros(labels.shape, dtype=bool))
    if lineage:
        sample['target_scope'] = 'selected_lineage'
        arrays.update(target_origin_index=np.arange(labels.size).reshape(labels.shape),
                      edit_region_mask=roi, scalar_changed_mask=np.zeros(labels.shape, dtype=bool))
    np.savez_compressed(directory/'variants/baseline.npz', **arrays,
                        metadata_json=np.array(json.dumps(sample)))
    sample.update(path='variants/baseline.npz', sha256=registry._hash(directory/'variants/baseline.npz'))
    inventory = {str(path.relative_to(directory)): registry._hash(path) for path in directory.rglob('*') if path.is_file()}
    manifest = {'schema_version': registry.DATASET_SCHEMA, 'complete': True,
                'case_id': '260602', 'frame': 1, 'anatomy_family': 'unverified',
                'dataset_fingerprint': 'b'*64, 'geometry': geometry,
                'image_method': 'attenuation_copy_proxy', 'image_units': 'cm^-1',
                'input_snapshot': 'input.ini', 'input_config_sha256': inventory['input.ini'],
                'samples': [sample], 'sample_count': 1, 'artifacts_sha256': inventory,
                'split_mode': 'unassigned' if unassigned else 'scenario_only',
                'split_counts': {'unassigned': 1} if unassigned else {'train': 1, 'validation': 0, 'test': 0},
                'source_files': {'act': {'path': '/unavailable/source.bin'}},
                'provenance': {'code_sha256': {'no-longer-present.py': 'a'*64}}}
    if lineage:
        manifest.update(target_scope='selected_lineage', target_semantics=registry.LINEAGE_SEMANTICS)
    else:
        manifest['target_source_ids'] = [42]
    save_manifest(directory, manifest)
    return directory/'dataset-manifest.json'


def save_manifest(directory, manifest):
    path = directory/'dataset-manifest.json'
    path.write_text(json.dumps(manifest))
    (directory/'dataset-manifest.sha256').write_text(registry._hash(path)+'\n')


class CohortRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = fixture_dataset(self.root/'data')
        self.registry = self.root/'registry'

    def mutate_manifest(self, change):
        manifest = json.loads(self.dataset.read_text())
        change(manifest)
        save_manifest(self.dataset.parent, manifest)

    def rewrite_payload(self, change):
        manifest = json.loads(self.dataset.read_text())
        path = self.dataset.parent/manifest['samples'][0]['path']
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        change(arrays)
        np.savez_compressed(path, **arrays)
        checksum = registry._hash(path)
        manifest['samples'][0]['sha256'] = checksum
        manifest['artifacts_sha256'][manifest['samples'][0]['path']] = checksum
        save_manifest(self.dataset.parent, manifest)

    def test_frozen_dataset_survives_source_and_code_unavailability(self):
        result = registry.verify_prepared_manifest(self.dataset)
        self.assertEqual(result['samples'][0]['shape_kji'], [3, 4, 5])
        self.assertEqual(Path(result['samples'][0]['_path']).parent.name, 'variants')

    def test_missing_registry_revision_is_distinct_from_corrupt_published_data(self):
        with self.assertRaises(FileNotFoundError):
            registry.load_registry_entry(self.registry, 'not-yet-prepared')
        self.assertFalse(self.registry.exists())
        registered = registry.register_dataset(self.registry, self.dataset, 'case-v1', 'family-a')
        Path(registered['manifest_path']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unsafe artifact'):
            registry.load_registry_entry(self.registry, 'case-v1')

    def test_native_xcat_integer_valued_float_source_labels_are_preserved(self):
        path = self.dataset.parent/'source.npz'
        with np.load(path) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays['original_labels'] = arrays['original_labels'].astype(np.float32)
        np.savez_compressed(path, **arrays)
        self.mutate_manifest(lambda m: m['artifacts_sha256'].update({'source.npz': registry._hash(path)}))
        self.assertEqual(registry.verify_prepared_manifest(self.dataset)['sample_count'], 1)
        arrays['original_labels'][0, 0, 0] = .5
        np.savez_compressed(path, **arrays)
        self.mutate_manifest(lambda m: m['artifacts_sha256'].update({'source.npz': registry._hash(path)}))
        with self.assertRaisesRegex(ValueError, 'Source labels'):
            registry.verify_prepared_manifest(self.dataset)

    def test_legacy_original_id_and_unassigned_datasets_supported(self):
        for index, options in enumerate(({'lineage': False}, {'unassigned': True})):
            path = fixture_dataset(self.root/f'other{index}', **options)
            self.assertEqual(registry.verify_prepared_manifest(path)['sample_count'], 1)

    def test_registration_copies_once_preserves_snapshots_and_is_idempotent(self):
        first = registry.register_dataset(self.registry, self.dataset, 'coa-260602-v1', 'family-a')
        saved = Path(first['manifest_path'])
        self.assertFalse(first['already_registered'])
        self.assertFalse(first['family_verified'])
        self.assertEqual((saved.parent/'input.ini').read_bytes(), (self.dataset.parent/'input.ini').read_bytes())
        second = registry.register_dataset(self.registry, self.dataset, 'coa-260602-v1', 'family-a')
        self.assertTrue(second['already_registered'])
        self.assertEqual(first['registered_at_utc'], second['registered_at_utc'])
        alias = registry.register_dataset(self.registry, self.dataset, 'coa-260602-v1-reviewed', 'family-a', True)
        self.assertEqual(saved, Path(alias['manifest_path']))
        self.assertEqual(len(list((self.registry/'objects').iterdir())), 1)
        shutil.rmtree(self.dataset.parent)
        self.assertEqual(registry.load_registry_entry(self.registry, 'coa-260602-v1')['sample_count'], 1)
        self.assertEqual(len(registry.list_registry_entries(self.registry)), 2)

    def test_registration_never_replaces_existing_policy_or_bytes(self):
        first = registry.register_dataset(self.registry, self.dataset, 'case-v1', 'family-a')
        receipt = Path(first['registry_entry_path']).read_bytes()
        with self.assertRaisesRegex(ValueError, 'different contents or family policy'):
            registry.register_dataset(self.registry, self.dataset, 'case-v1', 'family-b')
        with self.assertRaisesRegex(ValueError, 'different contents or family policy'):
            registry.register_dataset(self.registry, self.dataset, 'case-v1', 'family-a', True)
        self.mutate_manifest(lambda m: m.update(extra_note='new-revision'))
        with self.assertRaisesRegex(ValueError, 'different contents or family policy'):
            registry.register_dataset(self.registry, self.dataset, 'case-v1', 'family-a')
        self.assertEqual(Path(first['registry_entry_path']).read_bytes(), receipt)

    def test_copy_failure_cleans_staging_and_publishes_nothing(self):
        original = registry.shutil.copyfile
        count = [0]
        def failing_copy(src, dst):
            count[0] += 1
            if count[0] == 3:
                raise OSError('simulated disk failure')
            return original(src, dst)
        with patch.object(registry.shutil, 'copyfile', side_effect=failing_copy):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                registry.register_dataset(self.registry, self.dataset, 'case-v1', 'family-a')
        self.assertFalse(list((self.registry/'objects').iterdir()))
        self.assertFalse(list((self.registry/'entries').iterdir()))
        self.assertFalse(list((self.registry/'.staging').iterdir()))
        self.assertFalse((self.registry/'.register.lock').exists())

    def test_manifest_checksum_incomplete_and_payload_tampering_rejected(self):
        original = self.dataset.read_bytes()
        self.dataset.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError, 'manifest checksum'):
            registry.verify_prepared_manifest(self.dataset)
        self.dataset.write_bytes(original)
        marker = self.dataset.parent/'INCOMPLETE'
        marker.touch()
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
            registry.verify_prepared_manifest(self.dataset)
        marker.unlink()
        payload = self.dataset.parent/'variants/baseline.npz'
        payload.write_bytes(payload.read_bytes()+b'tampered')
        with self.assertRaisesRegex(ValueError, 'artifact checksum'):
            registry.verify_prepared_manifest(self.dataset)

    def test_duplicate_ids_unsafe_paths_and_symlinks_rejected(self):
        original = self.dataset.read_bytes()
        self.mutate_manifest(lambda m: (m['samples'].append(deepcopy(m['samples'][0])), m.update(sample_count=2)))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            registry.verify_prepared_manifest(self.dataset, verify_payloads=False)
        self.dataset.write_bytes(original)
        (self.dataset.parent/'dataset-manifest.sha256').write_text(registry._hash(self.dataset))
        self.mutate_manifest(lambda m: m['artifacts_sha256'].update({'../outside': 'a'*64}))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            registry.verify_prepared_manifest(self.dataset)
        self.dataset.write_bytes(original)
        (self.dataset.parent/'dataset-manifest.sha256').write_text(registry._hash(self.dataset))
        payload = self.dataset.parent/'variants/baseline.npz'
        outside = self.root/'outside.npz'
        payload.rename(outside); payload.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            registry.verify_prepared_manifest(self.dataset)

    def test_units_mask_semantics_and_geometry_metadata_rejected(self):
        original = json.loads(self.dataset.read_text())
        for key, value in (('image_units', 'HU'), ('target_semantics', 'ROI clipped'), ('split_mode', 'anything')):
            changed = deepcopy(original); changed[key] = value
            save_manifest(self.dataset.parent, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                registry.verify_prepared_manifest(self.dataset, verify_payloads=False)
        save_manifest(self.dataset.parent, original)
        self.mutate_manifest(lambda m: m['samples'][0].update(image_units='HU'))
        with self.assertRaisesRegex(ValueError, 'metadata disagrees'):
            registry.verify_prepared_manifest(self.dataset)

    def test_full_verification_enforces_array_contracts_even_if_archive_rehashed(self):
        self.rewrite_payload(lambda arrays: arrays['mask'].__setitem__((0, 0, 0), 2))
        with self.assertRaisesRegex(ValueError, 'binary'):
            registry.verify_prepared_manifest(self.dataset)

    def test_metadata_mode_never_opens_npz_arrays_or_hashes_npz_bytes(self):
        original_hash = registry._hash
        def protected_hash(path):
            if Path(path).suffix == '.npz':
                raise AssertionError('NPZ payload must not be read')
            return original_hash(path)
        with patch.object(registry.np, 'load', side_effect=AssertionError('No array read')), \
                patch.object(registry, '_hash', side_effect=protected_hash):
            result = registry.verify_prepared_manifest(self.dataset, verify_payloads=False)
        self.assertEqual(result['sample_count'], 1)

    def test_receipt_tamper_and_object_tamper_detected(self):
        result = registry.register_dataset(self.registry, self.dataset, 'case-v1', 'family-a')
        path = Path(result['registry_entry_path'])
        original = path.read_bytes()
        entry = json.loads(original); entry['family_verified'] = True
        path.write_text(json.dumps(entry))
        with self.assertRaisesRegex(ValueError, 'entry checksum'):
            registry.load_registry_entry(self.registry, 'case-v1')
        path.write_bytes(original)
        snapshot = Path(result['manifest_path']).parent/'input.ini'
        snapshot.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'artifact checksum'):
            registry.load_registry_entry(self.registry, 'case-v1')


if __name__ == '__main__':
    unittest.main()
