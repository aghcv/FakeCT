"""Canonical-grid supervision, sampling, and native quantitative reference."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from fakect_pretrained_adapter import ThoracicCoreProvider
from fakect_pretrained_data import (NativeCoreTemplate, PretrainedCaseVolume,
    audit_resampling_ceiling, choose_nested_rosters, export_native_reference,
    fixed_case_grid, inference_tiles, model_to_native_mask, native_core_extent,
    normalize_hu, CT_MEAN, CT_STD)
import test_pretrained_adapter as adapter_tests
from test_pretrained_adapter import decoded, sha


class PretrainedDataTests(unittest.TestCase):
    def setUp(self):
        self.fixture=adapter_tests.PretrainedAdapterTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture
        self.selection=deepcopy(self.f.selection)
        self.selection['geometry']={'crop_low_ijk':[2,1,1],
                                   'crop_high_ijk_exclusive':[6,5,5]}

    def isotropic(self):
        f=self.f;f.spacing=np.ones(3);f.row['spacing_ijk_mm']=[1.]*3
        f.source_geometry['spacing_ijk_mm']=[1.]*3
        with np.load(f.source_npz) as source:labels=source['original_labels'].copy()
        np.savez_compressed(f.source_npz,original_labels=labels,
                            geometry_json=json.dumps(f.source_geometry))
        f.row['source_npz_sha256']=sha(f.source_npz)
        f.pair_data['spacing_ijk_mm']=f.spacing
        np.savez_compressed(f.pair_npz,**f.pair_data);f.row['mask_npz_sha256']=sha(f.pair_npz)
        with np.load(f.core_path) as source:
            data={k:source[k] for k in source.files}
        data['spacing_ijk_mm']=f.spacing
        np.savez_compressed(f.core_path,**data)
        f.selection['mask_npz_sha256']=sha(f.core_path)
        f.selection['z_slab_mm']=[1.5,4.5]
        f.selection_path.write_text(json.dumps(f.selection))
        f.row['selection_manifest_sha256']=sha(f.selection_path)
        self.selection=deepcopy(f.selection)
        self.selection['geometry']={'crop_low_ijk':[2,1,1],'crop_high_ijk_exclusive':[6,5,5]}
        affine=np.diag([-1.,-1.,1.,1.]);affine[:3,3]=[4.,3.5,.5]
        f.evidence['orientation'].update(affine_ras_ijk=affine.tolist(),
            spacing_ijk_mm=[1.]*3,grid_boundary_origin_raw_mm=[-4.5,-4.,0.])

    def baseline(self):
        f=self.f;row=deepcopy(f.row);row.update(baseline=True,sample_id='fixture:baseline')
        with np.load(f.source_npz) as source:labels=source['original_labels']
        variant=f.root/'baseline-variant.npz'
        np.savez_compressed(variant,image=f.native[1:5,1:6,1:7],edited_labels=labels,
                            target_origin_index=np.full(f.edit_shape,-1,np.int32))
        pair=f.root/'baseline-pair.npz'
        data=dict(f.pair_data);data['after']=data['before'].copy()
        data['added']=np.zeros_like(data['before']);data['removed']=np.zeros_like(data['before'])
        np.savez_compressed(pair,**data)
        row.update(variant_npz=str(variant),variant_npz_sha256=sha(variant),
                   mask_npz=str(pair),mask_npz_sha256=sha(pair))
        return row

    def grid(self,spacing=1.):
        return fixed_case_grid(self.f.row,self.selection,self.f.evidence,
                               spacing_mm=spacing,padding_xy_mm=1.,context_z_mm=1.)

    def volume(self,grid=None,**kwargs):
        volume=PretrainedCaseVolume(self.f.row,self.grid() if grid is None else grid,
                                   self.selection,self.f.evidence,**kwargs)
        self.addCleanup(volume.close);return volume

    # Purpose: Verify fixed pretrained grids depend on source geometry and use canonical RAS.
    # Inputs: Isotropic edited/baseline rows with the same fixed selection.
    # Outputs: Identical 6-cube grid, expected origin, R/A/S axes and matching physical center.
    def test_fixed_grid_is_target_independent_and_canonical_ras(self):
        self.isotropic();grid=self.grid()
        baseline=self.baseline()
        again=fixed_case_grid(baseline,self.selection,self.f.evidence,
                              spacing_mm=1.,padding_xy_mm=1.,context_z_mm=1.)
        self.assertEqual(grid,again)
        self.assertEqual(grid['native_origin_ijk'],[1,0,0])
        self.assertEqual(grid['native_shape_kji'],[6,6,6])
        self.assertEqual(grid['model_axis_directions_ijk'],['R','A','S'])
        np.testing.assert_array_equal(np.asarray(grid['affine_ras_ijk'])[:3,:3],np.eye(3))
        # Model center(0,0,0) corresponds to highest native i,j of this crop.
        expected=np.asarray(self.f.evidence['orientation']['affine_ras_ijk'])@[6,5,0,1]
        np.testing.assert_allclose(np.asarray(grid['affine_ras_ijk'])@[0,0,0,1],expected)

    # Purpose: Verify image and supervision flip native i/j axes together.
    # Inputs: Isotropic native fixture read on a canonical RAS grid.
    # Outputs: Normalized image and all masks match joint flips; added/removed masks preserve net identity.
    def test_image_and_all_masks_flip_i_j_together(self):
        self.isotropic();grid=self.grid();volume=self.volume(grid)
        result=volume.read_patch([0,0,0],grid['shape_kji'])
        raw=np.fromfile(self.f.row['source_atn']['path'],'<f4').reshape(self.f.shape)
        native=raw*10.;native[1:5,1:6,1:7]=self.f.overlay
        hu=(1000.*(native/np.float32(self.f.water)-1.)).astype(np.int32)
        expected=normalize_hu(hu[:6,:6,1:7][:,::-1,::-1])
        np.testing.assert_allclose(result['image'][0],expected,rtol=0,atol=1e-6)
        provider=ThoracicCoreProvider(self.f.row,self.selection,self.f.evidence)
        masks=provider.read_block([1,0,0],[6,6,6])
        for key in ('target','baseline','domain','changed'):
            np.testing.assert_array_equal(result[key],masks[key][:,::-1,::-1])
        np.testing.assert_array_equal(result['added'],result['target']&~result['baseline'])
        np.testing.assert_array_equal(result['removed'],result['baseline']&~result['target'])

    # Purpose: Verify native HU integer conversion occurs before interpolation.
    # Inputs: Anisotropic fixture sampled at 1.5 mm with an explicit eight-corner reference.
    # Outputs: Interior normalized intensity matches integer-HU-first physical interpolation.
    def test_native_hu_integer_truncation_precedes_linear_interpolation(self):
        grid=self.grid(spacing=1.5);volume=self.volume(grid)
        result=volume.read_patch([0,0,0],grid['shape_kji'])['image'][0]
        raw=np.fromfile(self.f.row['source_atn']['path'],'<f4').reshape(self.f.shape)*10.
        raw[1:5,1:6,1:7]=self.f.overlay
        native_hu=(1000.*(raw/np.float32(self.f.water)-1.)).astype(np.int32)
        # Independent eight-corner physical interpolation at one interior center.
        point=np.asarray([2,2,2]);low=np.asarray(grid['native_origin_ijk'])
        high=low+np.asarray(grid['native_shape_kji'])[::-1]
        signs=np.asarray([-1,-1,1]);sp=np.asarray(self.f.row['spacing_ijk_mm'])
        pijk=np.where(signs>0,low+(point[::-1]+.5)*1.5/sp-.5,
                       high-(point[::-1]+.5)*1.5/sp-.5)
        coord=pijk[::-1];a=np.floor(coord).astype(int);b=np.minimum(a+1,np.asarray(native_hu.shape)-1)
        t=coord-a;expected=0.
        for corner in np.ndindex((2,2,2)):
            q=np.where(corner,b,a);weight=np.prod(np.where(corner,t,1-t))
            expected+=native_hu[tuple(q)]*weight
        expected_norm=(np.clip(int(expected),-101,813)-CT_MEAN)/CT_STD
        self.assertAlmostEqual(float(result[tuple(point)]),expected_norm,places=5)

    # Purpose: Verify normalization and padded patches preserve inputs.
    # Inputs: HU values -200.7, 0.9, 900.4 and an offset 8-cube patch around a 6-cube grid.
    # Outputs: Correct clipped normalized values, unchanged input and zero invalid image/target voxels.
    def test_normalization_and_padding_do_not_mutate_input(self):
        values=np.asarray([-200.7,.9,900.4],np.float32);old=values.copy()
        expected=(np.asarray([-101.,0.,813.])-CT_MEAN)/CT_STD
        np.testing.assert_allclose(normalize_hu(values),expected,rtol=1e-6)
        np.testing.assert_array_equal(values,old)
        self.isotropic();volume=self.volume();patch=volume.read_patch([-1,-1,-1],[8,8,8])
        self.assertEqual(patch['valid'].sum(),6**3)
        self.assertTrue(np.all(patch['image'][0][patch['valid']==0]==0))
        self.assertTrue(np.all(patch['target'][patch['valid']==0]==0))

    # Purpose: Verify cached native-core extents equal direct provider counts.
    # Inputs: Baseline template and mixed edited sibling.
    # Outputs: One addition/removal, correct physical change volume and immutable reused provider.
    def test_native_core_counts_cache_matches_direct_provider(self):
        template=NativeCoreTemplate(self.baseline(),self.selection,self.f.evidence)
        cached=template.extent(self.f.row)
        direct=native_core_extent(self.f.row,self.selection,self.f.evidence)
        self.assertEqual(cached,direct)
        self.assertEqual(cached['added_voxels'],1)
        self.assertEqual(cached['removed_voxels'],1)
        self.assertEqual(cached['change_kind'],'mixed')
        self.assertAlmostEqual(cached['changed_mm3'],2*1.*1.2*2.)
        provider=template.provider(self.f.row)
        self.assertFalse(provider.current_edit_core.flags.writeable)
        self.assertIs(provider,template.provider(self.f.row))

    # Purpose: Verify model-to-native round trips undo axis flips without losing targets.
    # Inputs: Isotropic target masks on the fixed canonical grid.
    # Outputs: Exact native mask, global/local Dice ceilings 1 and zero lost truth voxels.
    def test_model_to_native_roundtrip_undoes_both_axis_flips(self):
        self.isotropic();grid=self.grid();volume=self.volume(grid)
        model=volume.masks_full()['target']
        native=model_to_native_mask(model,grid)
        reference=volume.provider.read_block(grid['native_origin_ijk'],grid['native_shape_kji'])['target']
        np.testing.assert_array_equal(native,reference)
        audit=audit_resampling_ceiling(volume)
        self.assertEqual(audit['global_dice_ceiling'],1.)
        self.assertEqual(audit['local_change_dice_ceiling'],1.)
        self.assertEqual(audit['native_gt_lost_voxels'],0)

    # Purpose: Verify baseline sampling avoids absent changes and inference tiles cover the grid.
    # Inputs: Unchanged baseline, positive_fraction 1 and deterministic overlapping 4-cube tiles.
    # Outputs: Target-based samples contain foreground; fixed tiles cover every grid voxel.
    def test_training_sampler_avoids_absent_changes_and_fixed_tiles_cover_grid(self):
        self.isotropic();grid=self.grid();baseline=self.baseline()
        volume=PretrainedCaseVolume(baseline,grid,self.selection,self.f.evidence)
        self.addCleanup(volume.close)
        rng=np.random.default_rng(42)
        for _ in range(8):
            draw=volume.sample_patch_offset(rng,[4,4,4],positive_fraction=1.)
            self.assertEqual(draw['sampling_mode'],'target')
            self.assertGreater(volume.read_masks(draw['offset_kji'],[4,4,4])['target'].sum(),0)
        tiles=inference_tiles(grid,[4,4,4],overlap=2)
        coverage=np.zeros(grid['shape_kji'],np.uint8)
        for tile in tiles:
            k,j,i=tile['offset_kji'];coverage[k:k+4,j:j+4,i:i+4]+=1
        self.assertTrue(np.all(coverage>0))
        self.assertEqual(tiles,inference_tiles(grid,[4,4,4],overlap=2))

    # Purpose: Verify shared native exports have exact mask extent and affine.
    # Inputs: Baseline template, edited sibling and two-slice native export with baseline enabled.
    # Outputs: One addition/removal, matching image/truth shape and affine plus baseline NIfTI.
    def test_shared_native_reference_export_has_exact_extent_and_affine(self):
        self.isotropic();grid=self.grid();template=NativeCoreTemplate(self.baseline(),self.selection,self.f.evidence)
        record=export_native_reference(self.f.row,grid,self.selection,self.f.evidence,
            self.f.root/'evaluation',template=template,slab_slices=2,export_baseline=True)
        self.assertEqual(record['added_voxels'],1);self.assertEqual(record['removed_voxels'],1)
        image,affine,_=decoded(record['input_nifti'])
        truth,truth_affine,_=decoded(record['truth_nifti'])
        self.assertEqual(int(truth.sum()),record['current_core_voxels'])
        self.assertEqual(image.shape,tuple(grid['native_shape_kji']))
        np.testing.assert_array_equal(affine,truth_affine)
        np.testing.assert_allclose(affine,grid['native_affine_ras_ijk'],atol=1e-6)
        self.assertTrue(Path(record['baseline_nifti']).exists())

    # Purpose: Verify balanced training rosters are nested and evaluation siblings disjoint.
    # Inputs: Baseline plus 20 addition-only, removal-only and mixed siblings each.
    # Outputs: 17 training/13 evaluation IDs share only baseline; balanced nested subsets deterministic.
    def test_balanced_rosters_are_nested_and_disjoint_from_evaluation(self):
        rows=[{'sample_id':'c:baseline','case_id':'c','baseline':True}]
        extents={'c:baseline':{'change_kind':'none','changed_voxels':0}}
        for kind in ('addition_only','removal_only','mixed'):
            for i in range(20):
                sid=f'c:{kind}:{i:02d}'
                rows.append({'sample_id':sid,'case_id':'c','baseline':False,'duplicate_of':None})
                extents[sid]={'change_kind':kind,'changed_voxels':i+1}
        result=choose_nested_rosters(rows,extents)
        train=result['training_siblings']['c'];evaluate=result['evaluation']['c']
        self.assertEqual(len(train),17);self.assertEqual(len(evaluate),13)
        self.assertEqual(set(train)&set(evaluate),{'c:baseline'})
        self.assertEqual(result['training_nested']['c']['4'],train[:5])
        self.assertEqual(result['training_nested']['c']['8'],train[:9])
        kinds=[extents[s]['change_kind'] for s in train[1:]]
        self.assertEqual(kinds.count('addition_only'),4);self.assertEqual(kinds.count('removal_only'),4)
        self.assertEqual(kinds.count('mixed'),8)
        self.assertEqual(result,choose_nested_rosters(list(reversed(rows)),extents))


if __name__=='__main__':unittest.main()
