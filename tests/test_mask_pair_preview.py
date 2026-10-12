import sys
from pathlib import Path
import unittest
import json
import tempfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fakect_mask_pair_preview import transfer_pair, write_preview, sha256


class TransferTests(unittest.TestCase):
    def example(self):
        before=np.zeros((3,3,3),bool);before[1,1,1]=True
        after=before.copy();after[1,1,2]=True
        origins=np.full(before.shape,-1,np.int32)
        origins[after]=13
        return before,after,origins

    # Purpose: Verify mask transfer preserves broad reference context at native offsets.
    # Inputs: 3-cubed edit at offset 11/21/31 within 5-cubed broad mask at 10/20/30.
    # Outputs: New context retained, one translated addition, and exact before/after reconstruction.
    def test_preserves_new_context_and_physical_offsets(self):
        b,a,o=self.example();base=np.zeros((5,5,5),bool)
        base[2,2,2]=True;base[0,0,0]=True
        result=transfer_pair(base,[10,20,30],b,a,[11,21,31],o)
        self.assertTrue(result['after'][0,0,0])
        self.assertTrue(result['after'][2,2,3])
        self.assertEqual(result['counts']['added'],1)
        self.assertTrue(np.array_equal(result['before'],(result['after'] & ~result['added']) | result['removed']))

    # Purpose: Verify transfer expands the crop to include tracked growth.
    # Inputs: Single-voxel broad mask and neighboring tracked addition.
    # Outputs: Output shape 1x1x2 with one addition.
    def test_growth_can_leave_automatic_crop(self):
        b,a,o=self.example();base=np.ones((1,1,1),bool)
        result=transfer_pair(base,[2,2,2],b,a,[1,1,1],o)
        self.assertEqual(result['after'].shape,(1,1,2))
        self.assertEqual(result['counts']['added'],1)

    # Purpose: Verify additions are recomputed against the broader reference.
    # Inputs: Broad reference already containing the old edit addition.
    # Outputs: Zero new additions and one previously added voxel already in reference.
    def test_rederive_additions_for_broader_target(self):
        b,a,o=self.example();base=a.copy()
        result=transfer_pair(base,[0,0,0],b,a,[0,0,0],o)
        self.assertEqual(result['counts']['added'],0)
        self.assertEqual(result['counts']['old_additions_already_in_broad_reference'],1)

    # Purpose: Verify transferred growth requires included native ancestors.
    # Inputs: Broad mask excluding the tracked dilation ancestor.
    # Outputs: ValueError identifying excluded dilation ancestors.
    def test_rejects_dilation_from_excluded_anatomy(self):
        b,a,o=self.example();base=np.zeros_like(b);base[0,0,0]=True
        with self.assertRaisesRegex(ValueError,'dilation ancestors'):
            transfer_pair(base,[0,0,0],b,a,[0,0,0],o)

    # Purpose: Verify transferred removal cannot erase anatomy outside the new target.
    # Inputs: Removed old target and empty broad reference.
    # Outputs: ValueError identifying out-of-scope removed voxels.
    def test_rejects_removal_outside_new_scope(self):
        b,a,o=self.example();a[:]=False;o[:]=-1;base=np.zeros_like(b)
        with self.assertRaisesRegex(ValueError,'removed voxels'):
            transfer_pair(base,[0,0,0],b,a,[0,0,0],o)

    # Purpose: Verify pure removal transfers without additions.
    # Inputs: One-voxel before mask, empty after mask, and matching broad reference.
    # Outputs: One removed voxel and zero additions.
    def test_removed_voxels_and_empty_additions(self):
        b,a,o=self.example();a[:]=False;o[:]=-1
        r=transfer_pair(b,[0,0,0],b,a,[0,0,0],o)
        self.assertEqual(r['counts']['removed'],1)
        self.assertEqual(r['counts']['added'],0)

    # Purpose: Verify transfer enforces ancestry validity and allocation budget.
    # Inputs: Addition assigned wrong root then valid pair with max_voxels 1.
    # Outputs: ValueError for invalid lineage and voxel budget.
    def test_bad_lineage_and_budget_are_rejected(self):
        b,a,o=self.example();o[1,1,2]=0
        with self.assertRaisesRegex(ValueError,'lineage'):
            transfer_pair(b,[0,0,0],b,a,[0,0,0],o)
        b,a,o=self.example()
        with self.assertRaisesRegex(ValueError,'budget'):
            transfer_pair(b,[0,0,0],b,a,[0,0,0],o,max_voxels=1)


class FileExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.anatomy = self.root/'mask.npz'
        self.selector_path = self.root/'selection.json'
        self.prepared_path = self.root/'prepared.json'
        self.manifest_path = self.root/'dataset-manifest.json'
        self.source = self.root/'source.npz'
        self.variant = self.root/'variant.npz'
        self.output = self.root/'preview'
        before = np.zeros((3,3,3), bool)
        before[1,1,1] = True
        after = before.copy()
        after[1,1,2] = True
        origins = np.full(before.shape, -1, np.int32)
        origins[after] = 13
        base = before.copy()
        base[0,0,0] = True
        self.mask_data = {'mask': base, 'crop_origin_ijk': np.array([4,5,6]),
                          'spacing_ijk_mm': np.array([.8,.9,1.]), 'case_id': np.array('fixture'),
                          'frame': np.array(1), 'source_shape_kji': np.array([10,20,30])}
        np.savez_compressed(self.anatomy, **self.mask_data)
        np.savez_compressed(self.source, target_mask=before)
        np.savez_compressed(self.variant, mask=after, target_origin_index=origins)
        act = {'path': str(self.root/'act.bin'), 'bytes': 24000, 'mtime_ns': 123456}
        self.selector = {'schema_version': 'fakect.phantom-anatomy/1', 'complete': True,
                         'case_id': 'fixture', 'frame': 1, 'status': 'complete_provisional_review',
                         'provisional': True, 'coordinate_reviewed': False, 'mask_npz': 'mask.npz',
                         'act_source': act.copy(),
                         'geometry': {'shape_kji': [10,20,30], 'crop_shape_kji': [3,3,3],
                                      'crop_low_ijk': [4,5,6], 'crop_high_ijk_exclusive': [7,8,9],
                                      'spacing_ijk_mm': [.8,.9,1.]}}
        self.prepared = {'case_id': 'fixture', 'frame': 1, 'source_files': {'act': act.copy()}}
        self.row = {'case_id': 'fixture', 'variant_id': 'edited', 'sample_id': 'fixture:edited',
                    'baseline': False, 'source_npz': str(self.source), 'variant_npz': str(self.variant),
                    'prepared_manifest': str(self.prepared_path),
                    'geometry': {'spacing_ijk_mm': [.8,.9,1.], 'source_shape_kji': [10,20,30],
                                 'crop_shape_kji': [3,3,3], 'crop_origin_ijk': [4,5,6],
                                 'crop_high_ijk_exclusive': [7,8,9]}}
        self.save_selector()
        self.save_manifest()

    def save_selector(self):
        self.selector['mask_npz_sha256'] = sha256(self.anatomy)
        self.selector_path.write_text(json.dumps(self.selector))
        self.selector_path.with_suffix('.json.sha256').write_text(sha256(self.selector_path))

    def save_manifest(self):
        self.prepared_path.write_text(json.dumps(self.prepared))
        self.row.update(prepared_manifest_sha256=sha256(self.prepared_path),
                        source_npz_sha256=sha256(self.source), variant_npz_sha256=sha256(self.variant))
        self.manifest_path.write_text(json.dumps({'schema_version': 'fakect.two-stage-dataset/1',
                                                  'samples': [self.row]}))
        self.manifest_path.with_suffix('.sha256').write_text(sha256(self.manifest_path))

    def export(self):
        return write_preview(self.anatomy, case_id='fixture', dataset_manifest=self.manifest_path,
                             output_directory=self.output)

    # Purpose: Verify file export binds automatic-selection provenance and transferred masks.
    # Inputs: Checksummed provisional anatomy selector and baseline/edited NPZ fixture.
    # Outputs: Review-only result, exact provenance, preserved broad mask, one addition, and overwrite refusal.
    def test_complete_file_export_has_selector_provenance_and_masks(self):
        result = self.export()
        self.assertFalse(result['training_ready'])
        self.assertEqual(result['source_selection_manifest_sha256'], sha256(self.selector_path))
        self.assertTrue(result['source_selection_provisional'])
        self.assertFalse(result['source_selection_coordinate_reviewed'])
        self.assertEqual(result['source_selection_status'], 'complete_provisional_review')
        self.assertEqual(result['source_act']['mtime_ns'], 123456)
        self.assertEqual(len(result['exporter_code_sha256']), 64)
        self.assertFalse((self.output/'INCOMPLETE').exists())
        row = result['samples'][0]
        self.assertEqual(row['status'], 'ready_for_visual_review')
        with np.load(self.output/row['path']) as data:
            self.assertTrue(data['before'][0,0,0])
            self.assertEqual(data['added'].sum(), 1)
            np.testing.assert_array_equal(data['before'], (data['after'] & ~data['added']) | data['removed'])
        with self.assertRaises(FileExistsError):
            self.export()

    # Purpose: Verify anatomy payload changes invalidate selector checksums.
    # Inputs: Modified anatomy mask saved without updating selection checksum.
    # Outputs: ValueError for mask checksum and no output directory.
    def test_tampered_mask_is_not_accepted_under_old_selection_checksum(self):
        self.mask_data['mask'][0,0,1] = True
        np.savez_compressed(self.anatomy, **self.mask_data)
        with self.assertRaisesRegex(ValueError, 'mask checksum'):
            self.export()
        self.assertFalse(self.output.exists())

    # Purpose: Verify selection metadata changes require a matching digest.
    # Inputs: Selection JSON with appended whitespace and old checksum.
    # Outputs: ValueError for selection manifest checksum.
    def test_changed_selection_manifest_requires_matching_checksum(self):
        self.selector_path.write_text(self.selector_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'selection manifest checksum'):
            self.export()

    # Purpose: Verify anatomy mask origins match selector geometry.
    # Inputs: Mask origin shifted one voxel with fresh mask checksum.
    # Outputs: ValueError naming crop_low_ijk disagreement.
    def test_shifted_mask_origin_disagrees_with_selector_geometry(self):
        self.mask_data['crop_origin_ijk'][0] += 1
        np.savez_compressed(self.anatomy, **self.mask_data)
        self.save_selector()
        with self.assertRaisesRegex(ValueError, 'crop_low_ijk'):
            self.export()

    # Purpose: Verify transferred masks belong to the declared frame.
    # Inputs: Anatomy mask changed to frame 2 and selector checksum refreshed.
    # Outputs: ValueError naming frame disagreement.
    def test_frame_mismatch_is_rejected(self):
        self.mask_data['frame'] = np.array(2)
        np.savez_compressed(self.anatomy, **self.mask_data)
        self.save_selector()
        with self.assertRaisesRegex(ValueError, 'frame disagrees'):
            self.export()

    # Purpose: Verify transfer compares native ACT source revisions.
    # Inputs: Prepared ACT mtime changed while case/grid remain identical.
    # Outputs: ValueError for mismatched ACT source revisions.
    def test_old_act_revision_is_rejected_even_when_case_and_grid_match(self):
        self.prepared['source_files']['act']['mtime_ns'] += 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'ACT source revisions'):
            self.export()

    # Purpose: Verify file transfer validates native payload shapes.
    # Inputs: 4x3x3 source target against declared 3-cubed crop.
    # Outputs: ValueError for payload shape and INCOMPLETE export retained.
    def test_payload_shape_must_match_declared_native_crop(self):
        np.savez_compressed(self.source, target_mask=np.zeros((4,3,3), bool))
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'payload shape'):
            self.export()
        self.assertTrue((self.output/'INCOMPLETE').exists())

    # Purpose: Verify automatic anatomy crop stays inside native source bounds.
    # Inputs: Anatomy origin/bounds moved to i=29 through 32 in width-30 source.
    # Outputs: ValueError for crop extending beyond its source.
    def test_automatic_crop_cannot_leave_source_bounds(self):
        self.mask_data['crop_origin_ijk'][0] = 29
        self.selector['geometry']['crop_low_ijk'][0] = 29
        self.selector['geometry']['crop_high_ijk_exclusive'][0] = 32
        np.savez_compressed(self.anatomy, **self.mask_data)
        self.save_selector()
        with self.assertRaisesRegex(ValueError, 'leaves its declared source'):
            self.export()

    # Purpose: Verify file export requires completed checksummed selection evidence.
    # Inputs: Missing selection digest then automatic selection INCOMPLETE marker.
    # Outputs: ValueError for missing checksum and unfinished selection.
    def test_missing_selection_and_unfinished_selection_fail(self):
        self.selector_path.with_suffix('.json.sha256').unlink()
        with self.assertRaisesRegex(ValueError, 'checksummed selection'):
            self.export()
        self.save_selector()
        (self.root/'INCOMPLETE').write_text('pending')
        with self.assertRaisesRegex(ValueError, 'complete automatic'):
            self.export()

    # Purpose: Verify selector identity and edit crop bounds agree.
    # Inputs: Wrong selector case then inconsistent edit high bound.
    # Outputs: ValueError for wrong case/frame and invalid edit crop bounds.
    def test_wrong_case_and_inconsistent_edit_bounds_fail(self):
        self.selector['case_id'] = 'other'
        self.save_selector()
        with self.assertRaisesRegex(ValueError, 'wrong case/frame'):
            self.export()
        self.selector['case_id'] = 'fixture'
        self.save_selector()
        self.row['geometry']['crop_high_ijk_exclusive'][0] += 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Edit crop bounds'):
            self.export()


if __name__=='__main__':unittest.main()
