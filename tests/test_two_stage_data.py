"""Native pair annotation keeps before/after reconstruction identities exact."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
import fakect_two_stage_data as data
import fakect_cohort_registry as registry
from test_cohort_registry import fixture_dataset, save_manifest


class TwoStageDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepared = fixture_dataset(self.root/'prepared', unassigned=True)
        manifest = json.loads(self.prepared.read_text())
        baseline = manifest['samples'][0]
        with np.load(self.prepared.parent/baseline['path']) as archive:
            arrays = {key: archive[key] for key in archive.files if key != 'metadata_json'}
        arrays['mask'][1, 1, 1] = 0
        arrays['mask'][1, 1, 4] = 1
        arrays['edited_labels'][1, 1, 1] = 0
        arrays['edited_labels'][1, 1, 4] = 42
        arrays['changed_mask'][1, 1, [1, 4]] = True
        arrays['target_origin_index'][1, 1, 1] = -1
        arrays['target_origin_index'][1, 1, 4] = np.ravel_multi_index((1, 1, 2), arrays['mask'].shape)
        sample = {key: value for key, value in baseline.items() if key not in ('path', 'sha256')}
        sample.update(variant_id='edited', mask_sha256=registry._array_hash(arrays['mask']),
                      geometry_group='unverified:edited', operation='recipe')
        np.savez_compressed(self.prepared.parent/'variants/edited.npz', **arrays,
                            metadata_json=np.array(json.dumps(sample)))
        sample.update(path='variants/edited.npz', sha256=registry._hash(self.prepared.parent/'variants/edited.npz'))
        manifest['samples'].append(sample)
        manifest['sample_count'] = 2
        manifest['split_counts'] = {'unassigned': 2}
        manifest['artifacts_sha256'][sample['path']] = sample['sha256']
        save_manifest(self.prepared.parent, manifest)
        self.project = self.root/'project.json'
        self.project.write_text(json.dumps({'schema_version': 'fakect.cohort-project/1',
            'cases': [{'case_id': '260602', 'provisional_family': 'unverified',
                       'dataset_manifest': str(self.prepared)}]}))
        self.output = self.root/'two-stage'

    def prepare(self, **kwargs):
        return data.prepare_dataset(self.project, self.output, **kwargs)

    def rewrite_output(self, document):
        path = self.output/'dataset-manifest.json'
        path.write_text(json.dumps(document))
        (self.output/'dataset-manifest.sha256').write_text(registry._hash(path)+'\n')

    def test_exact_masks_and_reconstruction_identity(self):
        result = self.prepare()
        self.assertEqual(result['sample_count'], 2)
        self.assertEqual(result['baseline_count'], 1)
        self.assertFalse(result['family_verified'])
        baseline, edit = result['samples']
        self.assertEqual((baseline['added_voxels'], baseline['removed_voxels']), (0, 0))
        self.assertEqual((edit['added_voxels'], edit['removed_voxels']), (1, 1))
        with np.load(edit['source_npz']) as archive:
            before = archive['target_mask'].astype(bool)
        with np.load(edit['variant_npz']) as archive:
            after = archive['mask'].astype(bool)
        with np.load(edit['changes_npz']) as archive:
            added, removed = archive['added_mask'].astype(bool), archive['removed_mask'].astype(bool)
        self.assertTrue(np.array_equal((after & ~added) | removed, before))
        self.assertTrue(np.array_equal((before & ~removed) | added, after))
        self.assertFalse(np.any(removed & after))
        self.assertTrue(data.inspect_dataset(result['manifest_path'])['payloads_verified'])

    def test_prepared_dataset_bytes_are_not_modified(self):
        inventory = {p: registry._hash(p) for p in self.prepared.parent.rglob('*') if p.is_file()}
        self.prepare()
        self.assertEqual(inventory, {p: registry._hash(p) for p in inventory})

    def test_optional_change_sidecars(self):
        result = self.prepare(write_changes=False)
        self.assertFalse((self.output/'changes').exists())
        self.assertTrue(all('changes_npz' not in sample for sample in result['samples']))
        self.assertEqual(data.inspect_dataset(result['manifest_path'])['sample_count'], 2)

    def test_metadata_only_never_opens_array_payloads(self):
        result = self.prepare()
        with patch.object(np, 'load', side_effect=AssertionError('NPZ touched')):
            inspected = data.inspect_dataset(result['manifest_path'], verify_payloads=False)
        self.assertFalse(inspected['payloads_verified'])

    def test_existing_output_never_overwritten(self):
        result = self.prepare()
        original = Path(result['manifest_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(Path(result['manifest_path']).read_bytes(), original)

    def test_missing_dataset_does_not_reserve_output(self):
        self.prepared.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_incomplete_output_rejected(self):
        result = self.prepare()
        (self.output/'INCOMPLETE').write_text('interrupted')
        with self.assertRaisesRegex(ValueError, 'complete'):
            data.inspect_dataset(result['manifest_path'])

    def test_changed_native_payload_rejected(self):
        result = self.prepare()
        with Path(result['samples'][0]['variant_npz']).open('ab') as handle:
            handle.write(b'tampering')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            data.inspect_dataset(result['manifest_path'])

    def test_sidecar_hash_and_boolean_identity_both_checked(self):
        result = self.prepare()
        sample = result['samples'][1]
        path = Path(sample['changes_npz'])
        with np.load(path) as archive:
            added, removed = archive['added_mask'], archive['removed_mask']
        removed[:] = 0
        np.savez_compressed(path, added_mask=added, removed_mask=removed)
        with self.assertRaisesRegex(ValueError, 'artifact checksum'):
            data.inspect_dataset(result['manifest_path'])
        sample['changes_npz_sha256'] = registry._hash(path)
        result['artifacts_sha256'][str(path.relative_to(self.output))] = registry._hash(path)
        self.rewrite_output(result)
        with self.assertRaisesRegex(ValueError, 'Saved net change masks'):
            data.inspect_dataset(result['manifest_path'])

    def test_families_cannot_be_silently_marked_verified(self):
        result = self.prepare()
        result['family_verified'] = True
        self.rewrite_output(result)
        with self.assertRaisesRegex(ValueError, 'contract'):
            data.inspect_dataset(result['manifest_path'], verify_payloads=False)

    def test_sample_paths_and_family_bound_to_prepared_metadata(self):
        result = self.prepare()
        for field, value in (('anatomy_family', 'other'), ('variant_npz', '/tmp/other.npz')):
            modified = deepcopy(result)
            modified['samples'][0][field] = value
            self.rewrite_output(modified)
            with self.assertRaisesRegex(ValueError, 'pinned prepared sample'):
                data.inspect_dataset(result['manifest_path'], verify_payloads=False)

    def test_binary_mask_shape_contract(self):
        with self.assertRaisesRegex(ValueError, 'binary'):
            data._changes(np.zeros((2, 2, 2), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
