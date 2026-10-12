"""Calibration, physical orientation, streaming, and native core transfer."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from fakect_pretrained_adapter import (ThoracicCoreProvider, attenuation_to_hu,
    crop_ras_affine, export_mask_nifti, export_thorax_nifti, plan_thorax_crop)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoded(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        content = stream.read()
    assert struct.unpack_from('<i', content, 0)[0] == 348
    dim = struct.unpack_from('<8h', content, 40)
    datatype = struct.unpack_from('<h', content, 70)[0]
    offset = int(struct.unpack_from('<f', content, 108)[0])
    dtype = {16:'<f4', 2:'u1'}[datatype]
    shape_kji = tuple(dim[1:4][::-1])
    image = np.frombuffer(content[offset:], dtype=dtype).reshape(shape_kji)
    affine = np.eye(4)
    affine[:3] = np.asarray(struct.unpack_from('<12f', content, 280)).reshape(3, 4)
    return image, affine, content


class PretrainedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.shape = (7, 8, 9)
        self.spacing = np.asarray([1., 1.2, 2.])
        self.water = .1614
        k,j,i = np.indices(self.shape)
        self.native = (self.water + .002*k + .0004*j + .0001*i).astype(np.float32)
        self.native_labels = np.full(self.shape, 100, dtype=np.int32)
        self.native_labels[2:5, 2, 3] = 2922
        self.native_labels[3, 2, 4] = 1186  # Proximal branch is not core truth.
        source_atn = self.root/'source.bin'
        (self.native * np.float32(self.spacing[0]/10)).astype('<f4').tofile(source_atn)
        self.edit_origin = [1, 1, 1]
        self.edit_shape = (4, 5, 6)
        section = np.s_[1:5, 1:6, 1:7]
        original = self.native_labels[section].copy()
        self.labels = original.copy()
        self.labels[1, 1, 2] = 100  # Remove core at native(3,2,2).
        self.labels[2, 1, 0] = 2922 # Grow beyond old atlas tight X crop to i=1.
        self.origins = np.full(self.edit_shape, -1, dtype=np.int32)
        self.origins[2, 1, 0] = np.ravel_multi_index((2,1,2), self.edit_shape)
        self.overlay = self.native[section].copy()
        self.overlay[1, 1, 2] -= .02
        self.overlay[2, 1, 0] += .03
        self.source_npz = self.root/'source.npz'
        self.source_geometry = {'crop_origin_ijk':self.edit_origin,
            'crop_shape_kji':list(self.edit_shape),'spacing_ijk_mm':self.spacing.tolist()}
        np.savez_compressed(self.source_npz, original_labels=original,
                            geometry_json=json.dumps(self.source_geometry))
        self.variant_npz = self.root/'variant.npz'
        self.write_variant()
        broad_before = np.isin(original,[2922,1186]).astype(np.uint8)
        broad_after = np.isin(self.labels,[2922,1186]).astype(np.uint8)
        self.pair_npz = self.root/'pair.npz'
        self.pair_data = dict(before=broad_before, after=broad_after,
            added=(broad_after & ~broad_before).astype(np.uint8),
            removed=(broad_before & ~broad_after).astype(np.uint8),
            crop_origin_ijk=np.asarray(self.edit_origin), spacing_ijk_mm=self.spacing)
        np.savez_compressed(self.pair_npz, **self.pair_data)
        core_origin = [2,1,1]
        core_shape = (4,4,4)
        core = np.zeros(core_shape,dtype=np.uint8)
        core[1:4,1,1] = 1
        self.core_path = self.root/'core.npz'
        np.savez_compressed(self.core_path,core_mask=core,crop_origin_ijk=core_origin,
            source_shape_kji=self.shape,spacing_ijk_mm=self.spacing)
        self.selection = {'z_slab_mm':[3.,9.], 'anchor_source_ids':[2922],
            'mask_npz':'core.npz','mask_npz_sha256':sha(self.core_path)}
        self.selection_path = self.root/'selection.json'
        self.selection_path.write_text(json.dumps(self.selection))
        stat = source_atn.stat()
        self.row = {'sample_id':'fixture:sibling','case_id':'fixture','anatomy_family':'family',
            'baseline':False,'source_atn':{'path':str(source_atn),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},
            'source_shape_kji':list(self.shape),'spacing_ijk_mm':self.spacing.tolist(),
            'source_npz':str(self.source_npz),'source_npz_sha256':sha(self.source_npz),
            'variant_npz':str(self.variant_npz),'variant_npz_sha256':sha(self.variant_npz),
            'edit_crop_origin_ijk':self.edit_origin,'edit_crop_shape_kji':list(self.edit_shape),
            'mask_npz':str(self.pair_npz),'mask_npz_sha256':sha(self.pair_npz),
            'selection_manifest':str(self.selection_path),'selection_manifest_sha256':sha(self.selection_path),
            'source_atn_units':'per_pixel','image_units':'cm^-1'}
        affine = np.diag([-1.,-1.2,2.,1.]);affine[:3,3] = [4.,4.2,1.]
        self.evidence = {'case_id':'fixture','calibration':{'mu_water_cm_inverse':self.water,
            'source_atn_to_cm_inverse_scale':10.,'photon_energy_keV':120},
            'orientation':{'affine_ras_ijk':affine.tolist(),'spacing_ijk_mm':self.spacing.tolist(),
                           'status':'supported_by_generator_notes_and_named_surface_landmarks',
                           'shape_kji':list(self.shape),
                           'grid_boundary_origin_raw_mm':[-4.5,-4.8,0.]}}

    def write_variant(self):
        np.savez_compressed(self.variant_npz,image=self.overlay,edited_labels=self.labels,
                            target_origin_index=self.origins)
        if hasattr(self,'row'):
            self.row['variant_npz_sha256'] = sha(self.variant_npz)

    # Purpose: Verify attenuation-to-HU calibration and input preservation.
    # Inputs: Attenuation values zero, water, half-water and double-water; zero water calibration.
    # Outputs: HU -1000, 0, -500, 1000 with unchanged inputs; invalid calibration rejected.
    def test_known_hu_values_and_input_is_not_modified(self):
        mu = np.asarray([0.,self.water,.5*self.water,2*self.water],dtype=np.float32)
        old = mu.copy()
        np.testing.assert_allclose(attenuation_to_hu(mu,self.water),[-1000,0,-500,1000],atol=1e-4)
        np.testing.assert_array_equal(mu,old)
        with self.assertRaisesRegex(ValueError,'positive'):
            attenuation_to_hu(mu,0)

    # Purpose: Verify thorax cropping keeps full XY, native spacing and RAS offsets.
    # Inputs: Anisotropic 7x8x9 fixture with slab selection and context 2 or 0 mm.
    # Outputs: Expected crop bounds, native spacing and correctly translated half-voxel RAS affine.
    def test_full_xy_crop_native_spacing_and_half_voxel_ras_offset(self):
        plan = plan_thorax_crop(self.row,self.selection,self.evidence,context_mm=2)
        self.assertEqual(plan['shape_kji'],[6,8,9])
        self.assertEqual(plan['origin_ijk'],[0,0,0])
        no_context = plan_thorax_crop(self.row,self.selection,self.evidence,context_mm=0)
        self.assertEqual(no_context['origin_ijk'],[0,0,1])
        affine = np.asarray(no_context['affine_ras_ijk'])
        np.testing.assert_allclose(affine @ [2,3,0,1],[2.,.6,3.,1.],atol=1e-6)
        np.testing.assert_array_equal(no_context['spacing_ijk_mm'],self.spacing)
        direct = crop_ras_affine(self.evidence,[2,3,1])
        np.testing.assert_allclose(direct[:3,3],[2.,.6,3.],atol=1e-6)

    # Purpose: Verify core transfer excludes branches and allows lineage growth beyond old bounds.
    # Inputs: Core/branch labels, one removed core voxel and one lineage-owned addition outside atlas XY crop.
    # Outputs: Three target/baseline voxels, two changes, branch excluded and correct longitudinal domain.
    def test_core_transfer_excludes_branches_and_allows_growth_outside_old_xy_crop(self):
        provider = ThoracicCoreProvider(self.row,self.selection,self.evidence)
        masks = provider.read_block([0,0,0],self.shape)
        self.assertEqual(masks['target'].sum(),3)
        self.assertEqual(masks['baseline'].sum(),3)
        self.assertEqual(masks['changed'].sum(),2)
        self.assertEqual(masks['target'][3,2,4],0)  # Branch excluded.
        self.assertEqual(masks['target'][3,2,1],1)  # Added beyond atlas old X minimum.
        self.assertEqual(masks['target'][2,2,3],0)  # Eroded voxel.
        self.assertFalse(masks['domain'][0].any())
        self.assertTrue(masks['domain'][1:5].all())
        self.assertFalse(masks['domain'][5:].any())

    # Purpose: Reject new core voxels without verified original-core ancestry.
    # Inputs: Added voxel origin redirected to original soft tissue.
    # Outputs: ValueError identifies invalid ancestry.
    def test_new_core_requires_verified_original_core_ancestry(self):
        self.origins[2,1,0] = 0  # Original soft tissue.
        self.write_variant()
        with self.assertRaisesRegex(ValueError,'ancestry'):
            ThoracicCoreProvider(self.row,self.selection,self.evidence)

    # Purpose: Reject frozen anchor labels that disagree with the atlas core.
    # Inputs: Selected original core voxel relabeled as soft tissue with updated checksum.
    # Outputs: ValueError reports anchor/core mismatch.
    def test_frozen_original_anchor_labels_must_match_atlas_core(self):
        with np.load(self.source_npz) as source:
            original = source['original_labels'].copy()
        original[1,1,2] = 100  # Frozen selected core must still be an aorta label.
        np.savez_compressed(self.source_npz,original_labels=original,
                            geometry_json=json.dumps(self.source_geometry))
        self.row['source_npz_sha256'] = sha(self.source_npz)
        with self.assertRaisesRegex(ValueError,'do not match'):
            ThoracicCoreProvider(self.row,self.selection,self.evidence)

    # Purpose: Verify disconnected original anchor islands remain excluded.
    # Inputs: Same-ID island added outside atlas core with self-referencing ancestry.
    # Outputs: Island absent from target/changed masks and outside-core anchor count recorded.
    def test_disconnected_original_anchor_islands_remain_excluded(self):
        with np.load(self.source_npz) as source:
            original = source['original_labels'].copy()
        original[0,0,0] = 2922  # Existing island deliberately absent from atlas core.
        self.labels[0,0,0] = 2922
        self.origins[0,0,0] = 0
        np.savez_compressed(self.source_npz,original_labels=original,
                            geometry_json=json.dumps(self.source_geometry))
        self.row['source_npz_sha256'] = sha(self.source_npz)
        self.write_variant()
        provider = ThoracicCoreProvider(self.row,self.selection,self.evidence)
        result = provider.read_block(self.edit_origin,self.edit_shape)
        self.assertEqual(result['target'][0,0,0],0)
        self.assertEqual(result['changed'][0,0,0],0)
        self.assertGreater(provider.metadata['original_anchor_voxels_outside_selected_core_in_edit_crop'],0)

    # Purpose: Verify unchanged baselines produce identical target and reference masks.
    # Inputs: Baseline row restored to original labels/image with zero net changes.
    # Outputs: Target equals baseline and changed voxel count zero.
    def test_unchanged_baseline_has_no_target_changes(self):
        self.row['baseline'] = True
        with np.load(self.source_npz) as data:
            self.labels = data['original_labels'].copy()
        self.origins.fill(-1)
        self.overlay = self.native[1:5,1:6,1:7].copy()
        self.write_variant()
        self.pair_data['after'] = self.pair_data['before'].copy()
        self.pair_data['added'].fill(0);self.pair_data['removed'].fill(0)
        np.savez_compressed(self.pair_npz,**self.pair_data)
        self.row['mask_npz_sha256'] = sha(self.pair_npz)
        provider = ThoracicCoreProvider(self.row,self.selection,self.evidence)
        masks = provider.read_block([0,0,0],self.shape)
        np.testing.assert_array_equal(masks['target'],masks['baseline'])
        self.assertEqual(masks['changed'].sum(),0)

    # Purpose: Verify streamed NIfTI preserves HU units, orientation and native overlays.
    # Inputs: Two-slice streaming export of image plus target/baseline/domain/changed masks.
    # Outputs: Correct HU/affines/header units, bounded reads, exact mask counts and no partial files.
    def test_streamed_nifti_storage_units_orientation_and_native_edit_overlay(self):
        image_path = self.root/'pilot.nii.gz'
        mask_paths = {k:self.root/(k+'.nii.gz') for k in ['target','baseline','domain','changed']}
        result = export_thorax_nifti(self.row,self.selection,self.evidence,image_path,
            context_mm=2,slab_slices=2,mask_output_paths=mask_paths)
        image,affine,header = decoded(image_path)
        raw = np.fromfile(self.row['source_atn']['path'],'<f4').reshape(self.shape)
        expected_mu = raw * np.float32(10./self.spacing[0])
        expected_mu[1:5,1:6,1:7] = self.overlay
        expected_hu = 1000.*(expected_mu[:6]/np.float32(self.water)-1.)
        np.testing.assert_allclose(image,expected_hu,rtol=0,atol=1e-4)
        np.testing.assert_allclose(affine,self.evidence['orientation']['affine_ras_ijk'],atol=1e-6)
        self.assertEqual(header[344:348],b'n+1\0')
        self.assertEqual(header[123],2)
        self.assertEqual(struct.unpack_from('<hh',header,252),(1,1))
        np.testing.assert_allclose(struct.unpack_from('<3f',header,256),[0,0,1],atol=1e-6)
        self.assertEqual(result['statistics']['peak_read_voxels'],2*8*9)
        self.assertEqual(result['statistics']['native_voxels_read'],6*8*9)
        self.assertEqual(result['statistics']['mask_voxel_counts'],
                         {'target':3,'baseline':3,'domain':4*8*9,'changed':2})
        for key,path in mask_paths.items():
            mask,mask_affine,_ = decoded(path)
            self.assertEqual(mask.dtype,np.uint8)
            self.assertEqual(mask.shape,image.shape)
            np.testing.assert_array_equal(mask_affine,affine)
            self.assertEqual(int(mask.sum()),result['statistics']['mask_voxel_counts'][key])
        self.assertFalse(list(self.root.glob('*.partial')))

    # Purpose: Verify small mask NIfTI exports preserve dimensions and crop offsets.
    # Inputs: 2x3x4 binary mask at crop origin 2,3,1.
    # Outputs: Exact mask, expected RAS landmark, correct shape and overwrite rejection.
    def test_small_mask_export_keeps_ijk_header_and_crop_offset(self):
        mask = np.zeros((2,3,4),np.uint8);mask[1,2,3] = 1
        path = self.root/'small.nii'
        result = export_mask_nifti(mask,path,origin_ijk=[2,3,1],evidence=self.evidence)
        actual,affine,_ = decoded(path)
        np.testing.assert_array_equal(actual,mask)
        np.testing.assert_allclose(affine @ [3,2,1,1],[-1.,-1.8,5.,1.],atol=1e-6)
        self.assertEqual(result['shape_kji'],[2,3,4])
        with self.assertRaisesRegex(ValueError,'overwrite'):
            export_mask_nifti(mask,path,origin_ijk=[2,3,1],evidence=self.evidence)

    # Purpose: Reject unsupported units, calibration evidence, case identity or crop budget.
    # Inputs: Wrong attenuation units, different case, max_voxels one and bad conversion scale.
    # Outputs: ValueError identifies each incompatible export prerequisite.
    def test_units_evidence_case_and_budget_fail_closed(self):
        bad = copy.deepcopy(self.row);bad['source_atn_units'] = 'cm^-1'
        with self.assertRaisesRegex(ValueError,'per-pixel'):
            plan_thorax_crop(bad,self.selection,self.evidence)
        evidence = copy.deepcopy(self.evidence);evidence['case_id'] = 'other'
        with self.assertRaisesRegex(ValueError,'different phantom'):
            plan_thorax_crop(self.row,self.selection,evidence)
        with self.assertRaisesRegex(ValueError,'budget'):
            plan_thorax_crop(self.row,self.selection,self.evidence,max_voxels=1)
        evidence = copy.deepcopy(self.evidence);evidence['calibration']['source_atn_to_cm_inverse_scale'] = 1.
        with self.assertRaisesRegex(ValueError,'conversion'):
            plan_thorax_crop(self.row,self.selection,evidence)


if __name__ == '__main__':
    unittest.main()
