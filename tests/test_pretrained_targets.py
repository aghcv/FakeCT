"""Connected branch annotations, common-box boundaries and sibling ancestry."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from fakect_pretrained_targets import BranchInclusiveTemplate
import test_pretrained_adapter as adapter_tests
from test_pretrained_adapter import sha


class PretrainedTargetTests(unittest.TestCase):
    def setUp(self):
        fixture = adapter_tests.PretrainedAdapterTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups); self.f = fixture
        labels = fixture.native_labels.copy()
        labels[3, 2, 4:9] = 1547  # Connected continuation past old reach/edit crop.
        labels[4, 2, 4] = 2912   # Connected coronary candidate.
        labels[3, 3, 3] = 3      # Pulmonary cannot connect a systemic island.
        labels[3, 4, 3] = 1547
        labels[2, 5, 6] = 2922   # Disconnected same-aorta-ID island.
        labels[3, 5, 6] = 1547   # Branch connected only to that excluded island.
        self.native_labels = labels
        self.original = labels[1:5, 1:6, 1:7].copy()
        np.savez_compressed(fixture.source_npz, original_labels=self.original,
                            geometry_json=json.dumps(fixture.source_geometry))
        np.savez_compressed(fixture.variant_npz, edited_labels=self.original,
                            target_origin_index=np.full(self.original.shape, -1, np.int32))
        self.act = fixture.root / 'act.bin'; labels.astype('<f4').tofile(self.act)
        stat = self.act.stat()
        source_act = dict(path=str(self.act), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        paths = {2922: ('dias_aorta', '01_Cardiovascular/Heart/Aorta'),
                 1547: ('arteries', '01_Cardiovascular/Systemic_Arteries'),
                 2912: ('dias_rca1', '01_Cardiovascular/Coronary_Circulation/Coronary_Arteries'),
                 3: ('pulmonary', '01_Cardiovascular/Pulmonary_Circulation')}
        catalog = {'records': [dict(original_id=i, original_name=name,
                    classification=dict(tissue_name='artery', status='classified', hierarchy_path=path))
                    for i, (name, path) in paths.items()]}
        self.catalog = fixture.root / 'catalog.json'; self.catalog.write_text(json.dumps(catalog))
        with np.load(fixture.core_path) as data:
            arrays = {k:data[k] for k in data.files}
        branch = np.zeros_like(arrays['core_mask']); branch[2, 1, 2] = 1
        arrays.update(branch_mask=branch, mask=arrays['core_mask'] | branch)
        np.savez_compressed(fixture.core_path, **arrays)
        self.selection = dict(fixture.selection, mask_npz_sha256=sha(fixture.core_path),
            catalog_path=str(self.catalog), catalog_sha256=sha(self.catalog), act_source=source_act,
            settings=dict(connectivity=6), selected_source_ids=[2922, 1547])
        fixture.selection_path.write_text(json.dumps(self.selection))
        self.baseline = dict(fixture.row, baseline=True, sample_id='fixture:baseline',
            source_npz_sha256=sha(fixture.source_npz), variant_npz_sha256=sha(fixture.variant_npz),
            source_act=source_act, selection_manifest_sha256=sha(fixture.selection_path))
        self.grid = dict(case_id='fixture', native_origin_ijk=[0,0,0],
                         native_shape_kji=list(fixture.shape), native_spacing_ijk_mm=fixture.spacing.tolist())

    def template(self, **kwargs):
        return BranchInclusiveTemplate(self.baseline, self.selection, self.f.evidence, self.grid, **kwargs)

    def variant(self, labels, origins=None, name='edited'):
        path = self.f.root / (name + '.npz')
        if origins is None: origins = np.full(labels.shape, -1, np.int32)
        np.savez_compressed(path, edited_labels=labels, target_origin_index=origins)
        return dict(self.baseline, baseline=False, sample_id='fixture:' + name,
                    variant_npz=str(path), variant_npz_sha256=sha(path))

    # Purpose: Verify common target box retains connected branch continuations.
    # Inputs: Native aorta/core plus connected systemic/coronary branches.
    # Outputs: Target has nine voxels, six branches/five continuation cells, no changes, and exact source IDs.
    def test_common_box_preserves_connections_without_radial_branch_cut(self):
        template = self.template()
        masks = template.baseline_provider.read_block([0,0,0], self.f.shape)
        self.assertTrue(masks['target'][3, 2, 4:9].all())
        self.assertTrue(masks['target'][4, 2, 4])
        self.assertEqual(int(masks['target'].sum()), 9)
        self.assertEqual(template.counts['core_voxels'], 3)
        self.assertEqual(template.counts['branch_voxels'], 6)
        self.assertEqual(template.counts['branch_continuation_voxels'], 5)
        self.assertEqual(int(masks['changed'].sum()), 0)
        self.assertEqual(template.metadata['selected_source_ids'], [1547,2912,2922])

    # Purpose: Exclude pulmonary-mediated paths and disconnected same-ID islands.
    # Inputs: Native fixture with pulmonary connector and isolated aorta/systemic labels.
    # Outputs: All disconnected or pulmonary-connected points stay outside the target.
    def test_pulmonary_cannot_carry_path_and_same_id_islands_stay_excluded(self):
        masks = self.template().baseline_provider.read_block([0,0,0], self.f.shape)
        for point in ((3,3,3), (3,4,3), (2,5,6), (3,5,6)):
            self.assertEqual(masks['target'][point], 0)

    # Purpose: Verify sibling growth and release follow selected branch ancestry.
    # Inputs: One branch voxel released and one new branch voxel with valid origin index.
    # Outputs: Target gains/loses one branch cell, retains continuation, and preserves the three-cell aorta core.
    def test_sibling_branch_growth_and_release_use_selected_component_ancestry(self):
        template = self.template(); labels = self.original.copy()
        labels[2,1,3] = 100  # Release native branch(i4,j2,k3).
        labels[2,2,4] = 1547 # Grow native(i5,j3,k3), outside frozen proximal crop.
        origins = np.full(labels.shape, -1, np.int32)
        origins[2,2,4] = np.ravel_multi_index((2,1,4), labels.shape)
        row = self.variant(labels, origins); provider = template.provider(row)
        masks = provider.read_block([0,0,0], self.f.shape)
        self.assertEqual(masks['target'][3,2,4], 0)
        self.assertEqual(masks['target'][3,3,5], 1)
        self.assertTrue(masks['target'][3,2,7:9].all()) # Outside edit crop unchanged.
        extent = template.extent(row)
        self.assertEqual(extent['added_target_voxels'], 1)
        self.assertEqual(extent['removed_target_voxels'], 1)
        self.assertEqual(extent['added_aorta_core_voxels'], 0)
        self.assertEqual(extent['removed_aorta_core_voxels'], 0)
        self.assertEqual(extent['current_target_voxels'], 9)
        self.assertEqual(int(provider.core_provider.read_block([0,0,0], self.f.shape)['target'].sum()), 3)

    # Purpose: Reject promotion of growth originating from disconnected generic-ID islands.
    # Inputs: New systemic-label voxel with ancestry in an excluded native island.
    # Outputs: New voxel is not selected and changed target count stays zero.
    def test_growth_from_disconnected_generic_id_island_not_promoted(self):
        template = self.template(); labels = self.original.copy()
        labels[2,3,3] = 1547
        origins = np.full(labels.shape, -1, np.int32)
        origins[2,3,3] = np.ravel_multi_index((2,3,2), labels.shape) # Disconnected branch.
        provider = template.provider(self.variant(labels, origins))
        self.assertEqual(provider.read_block([0,0,0], self.f.shape)['target'][3,4,4], 0)
        self.assertEqual(template.extent(self.variant(labels, origins, 'again'))['changed_voxels'], 0)

    # Purpose: Reject missing or non-target ancestry for branch growth.
    # Inputs: New branch voxel with absent ancestry or origin index zero.
    # Outputs: Provider construction raises ValueError for invalid ancestry.
    def test_invalid_growth_ancestry_rejected(self):
        labels = self.original.copy(); labels[2,2,4] = 1547
        with self.assertRaisesRegex(ValueError, 'ancestry'):
            self.template().provider(self.variant(labels))
        origins = np.full(labels.shape, -1, np.int32); origins[2,2,4] = 0 # Soft tissue ancestor.
        with self.assertRaisesRegex(ValueError, 'ancestry'):
            self.template().provider(self.variant(labels, origins, 'soft'))

    # Purpose: Verify domain caps apply consistently without radial ROI clipping.
    # Inputs: Common native box shortened to [7,8,8].
    # Outputs: Domain excludes capped slabs/columns while preserving in-box branch continuation.
    def test_domain_and_box_caps_apply_to_all_masks_without_roi_clipping(self):
        self.grid['native_shape_kji'] = [7,8,8]
        provider = self.template().baseline_provider
        masks = provider.read_block([0,0,0], self.f.shape)
        self.assertFalse(masks['domain'][0].any())
        self.assertFalse(masks['domain'][5:].any())
        self.assertFalse(masks['domain'][:,:,8].any())
        self.assertTrue(masks['target'][3,2,7])
        self.assertEqual(masks['target'][3,2,8], 0)
        self.assertEqual(int(masks['domain'].sum()), 4*8*8)

    # Purpose: Require sibling case geometry and selection identity to match.
    # Inputs: Different case ID, crop origin, or selection manifest hash.
    # Outputs: Provider raises ValueError for case geometry mismatch.
    def test_sibling_source_case_geometry_mismatch_rejected(self):
        template = self.template()
        for key, value in (('case_id', 'different'), ('edit_crop_origin_ijk', [2,1,1]),
                           ('selection_manifest_sha256', 'changed')):
            row = deepcopy(self.baseline); row[key] = value
            with self.assertRaisesRegex(ValueError, 'case geometry'):
                template.provider(row)

    # Purpose: Verify explicit proximal branch mode preserves the old union.
    # Inputs: Branch-inclusive template constructed with branch_extent proximal.
    # Outputs: Target has four voxels, no continuation voxels, and excludes distal branch cells.
    def test_proximal_alternative_preserves_old_union_explicitly(self):
        template = self.template(branch_extent='proximal')
        self.assertEqual(template.counts['target_voxels'], 4)
        self.assertEqual(template.counts['branch_continuation_voxels'], 0)
        self.assertEqual(template.baseline_provider.read_block([0,0,0], self.f.shape)['target'][3,2,5], 0)

    # Purpose: Verify annotation cache reload avoids rereading native ACT.
    # Inputs: Saved annotation cache and a native read mock that must not run.
    # Outputs: Reload matches targets/counts/extents; requesting proximal semantics rejects dependencies.
    def test_cache_reload_is_identical_without_native_act_reread(self):
        cache = self.f.root / 'annotation'; original = self.template(annotation_directory=cache)
        self.assertTrue(original.annotation_manifest_path.is_file())
        with patch('fakect_pretrained_targets.read_crop', side_effect=AssertionError('ACT reread')):
            restored = self.template(annotation_directory=cache)
        np.testing.assert_array_equal(original.baseline_target, restored.baseline_target)
        self.assertEqual(original.counts, restored.counts)
        self.assertEqual(original.extent(self.baseline), restored.extent(self.baseline))
        with self.assertRaisesRegex(ValueError, 'dependencies'):
            self.template(annotation_directory=cache, branch_extent='proximal')

    # Purpose: Reject incomplete or altered annotation caches.
    # Inputs: INCOMPLETE marker, then bytes appended to annotation.npz.
    # Outputs: Cache loading raises ValueError for incompleteness or changed artifacts.
    def test_tampered_or_incomplete_cache_rejected(self):
        cache = self.f.root / 'annotation'; self.template(annotation_directory=cache)
        (cache / 'INCOMPLETE').touch()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.template(annotation_directory=cache)
        (cache / 'INCOMPLETE').unlink()
        with (cache / 'annotation.npz').open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.template(annotation_directory=cache)

    # Purpose: Require original ACT and frozen source labels to agree.
    # Inputs: Source NPZ label changed with its digest refreshed.
    # Outputs: Template construction raises ValueError for ACT label disagreement.
    def test_original_act_and_frozen_edit_labels_must_agree(self):
        original = self.original.copy(); original[2,1,3] = 100
        np.savez_compressed(self.f.source_npz, original_labels=original,
                            geometry_json=json.dumps(self.f.source_geometry))
        self.baseline['source_npz_sha256'] = sha(self.f.source_npz)
        with self.assertRaisesRegex(ValueError, 'ACT.*labels disagree'):
            self.template()

    # Purpose: Verify bounded ACT reads preserve connectivity across slab boundaries.
    # Inputs: Native fixture with MAX_CROP_VOXELS reduced to 144.
    # Outputs: Four slab reads still recover all nine target voxels.
    def test_native_reads_are_bounded_and_connectivity_crosses_read_slabs(self):
        with patch('fakect_pretrained_targets.MAX_CROP_VOXELS', 144):
            template = self.template()
        self.assertEqual(template.counts['target_voxels'], 9)
        self.assertEqual(len(template.native_read['slab_reads']), 4)

    # Purpose: Reject cached annotations prepared for a different common box.
    # Inputs: Annotation cache followed by native shape changed to [7,8,8].
    # Outputs: Cache loading raises ValueError for dependency mismatch.
    def test_annotation_cache_rejects_changed_common_box(self):
        cache = self.f.root / 'annotation'; self.template(annotation_directory=cache)
        self.grid['native_shape_kji'] = [7,8,8]
        with self.assertRaisesRegex(ValueError, 'dependencies'):
            self.template(annotation_directory=cache)

    # Purpose: Verify native ACT stat changes invalidate annotation caches.
    # Inputs: Saved annotation cache followed by appended ACT bytes.
    # Outputs: Cache loading raises ValueError for source size/stat change.
    def test_native_act_stat_change_invalidates_cached_annotation(self):
        cache = self.f.root / 'annotation'; self.template(annotation_directory=cache)
        with self.act.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'ACT source size/stat'):
            self.template(annotation_directory=cache)

    def replace_native_labels(self, labels):
        """Update matching raw ACT / frozen source contracts for a fixture."""
        labels = np.asarray(labels, np.float32)
        labels.astype('<f4').tofile(self.act)
        self.original = labels[1:5,1:6,1:7].copy()
        np.savez_compressed(self.f.source_npz, original_labels=self.original,
                            geometry_json=json.dumps(self.f.source_geometry))
        np.savez_compressed(self.f.variant_npz, edited_labels=self.original,
                            target_origin_index=np.full(self.original.shape, -1, np.int32))
        stat = self.act.stat(); record = dict(path=str(self.act),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        self.selection['act_source'] = record
        self.f.selection_path.write_text(json.dumps(self.selection))
        self.baseline.update(source_act=record, source_npz_sha256=sha(self.f.source_npz),
            variant_npz_sha256=sha(self.f.variant_npz), selection_manifest_sha256=sha(self.f.selection_path))

    # Purpose: Preserve negative nonarterial IDs without absolute-value mapping.
    # Inputs: Native label -1547 without a corresponding arterial catalog entry.
    # Outputs: Negative cell is excluded, target stays nine voxels, and negative activity count is one.
    def test_signed_nonarterial_ids_are_preserved_without_absolute_value(self):
        labels = self.native_labels.astype(np.float32)
        labels[3,3,3] = -1547.0  # Adjacent to core: no abs() conversion.
        self.replace_native_labels(labels)
        template = self.template()
        masks = template.baseline_provider.read_block([0,0,0], self.f.shape)
        self.assertEqual(masks['target'][3,3,3], 0)
        self.assertEqual(template.counts['target_voxels'], 9)
        self.assertEqual(template.native_read['activity_code_counts']['negative_activity_voxels'], 1)

    # Purpose: Select signed arterial IDs only through exact catalog identity.
    # Inputs: Catalog explicitly adding arterial ID -1547 and one matching native cell.
    # Outputs: The cell is selected and metadata includes signed ID -1547.
    def test_signed_arterial_catalog_id_is_selected_by_its_exact_identity(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['records'].append(dict(original_id=-1547,original_name='signed_systemic',
            classification=dict(tissue_name='artery',status='classified',hierarchy_path='01_Cardiovascular/Systemic_Arteries')))
        self.catalog.write_text(json.dumps(catalog)); self.selection['catalog_sha256'] = sha(self.catalog)
        labels = self.native_labels.astype(np.float32); labels[3,3,3] = -1547.0
        self.replace_native_labels(labels)
        template = self.template()
        self.assertEqual(template.baseline_provider.read_block([0,0,0],self.f.shape)['target'][3,3,3], 1)
        self.assertIn(-1547,template.metadata['selected_source_ids'])

    # Purpose: Reject fractional and overflowing activity codes before integer casts.
    # Inputs: Native values 1547.125, 2^31, and float32 maximum.
    # Outputs: Template raises ValueError for nonintegral or out-of-int32 codes.
    def test_fractional_and_overflow_codes_fail_before_any_id_cast(self):
        for value, message in ((1547.125, 'integral signed'), (2.**31, 'signed int32'), (np.finfo(np.float32).max, 'signed int32')):
            labels = self.native_labels.astype(np.float32); labels[0,0,0] = value
            self.replace_native_labels(labels)
            with self.assertRaisesRegex(ValueError,message): self.template()


if __name__ == '__main__': unittest.main()
