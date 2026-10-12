"""Regression tests for sampling-induced target loss and disconnection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.ndimage import generate_binary_structure, label

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import fakect_context_grid as grid


class ContextGridTests(unittest.TestCase):
    def audit(self, mask, native=(1.,1.,1.), target=(2.,2.,2.), origin=(0,0,0)):
        return grid._audit_array(mask, origin, native, [100,100,100], target)

    # Purpose: Verify rectangular targets preserve physical volume during downsampling.
    # Inputs: 6-cube body inside a 10-cube mask resampled from 1 to 2 mm.
    # Outputs: Ready audit, 216 mm3 at both resolutions and one model component.
    def test_preserved_rectangular_body(self):
        mask=np.zeros((10,10,10),bool);mask[2:8,2:8,2:8]=True
        result=self.audit(mask)
        self.assertTrue(result['ready'])
        self.assertEqual(result['native']['volume_mm3'],216)
        self.assertEqual(result['model']['volume_mm3'],216)
        self.assertEqual(result['model']['count'],1)

    # Purpose: Verify coarse sampling detects a newly broken native bridge.
    # Inputs: Two bodies joined by a thin bridge omitted by 2 mm sampling.
    # Outputs: Native one/model two components, failed readiness and material-fragment flags.
    def test_native_bridge_breaks_at_coarser_spacing(self):
        mask=np.zeros((12,8,8),bool)
        mask[1:5,1:6,1:6]=True
        mask[7:11,1:6,1:6]=True
        mask[5:7,2,2]=True  # Connected at native spacing; even j/i omitted at 2 mm.
        result=self.audit(mask)
        self.assertEqual(result['native']['count'],1)
        self.assertEqual(result['model']['count'],2)
        self.assertFalse(result['ready'])
        self.assertIn('material_new_disconnection',result['flags'])
        self.assertIn('introduced_fragment_gt_5_mm3',result['flags'])
        self.assertEqual(result['outside_largest_fraction_change'],.5)

    # Purpose: Verify negligible sampling fragments remain permissible.
    # Inputs: Large body and tiny joined tip sampled from 0.5 to 1 mm.
    # Outputs: Ready despite two model components, under 1 percent separation and no fragment above 5 mm3.
    def test_tiny_new_fragment_below_both_material_thresholds_is_allowed(self):
        mask=np.zeros((26,24,24),bool)
        mask[1:21,1:21,1:21]=True
        mask[21:23,2,2]=True
        mask[23:25,1:3,1:3]=True
        result=self.audit(mask,native=(.5,.5,.5),target=(1,1,1))
        self.assertEqual(result['native']['count'],1)
        self.assertEqual(result['model']['count'],2)
        self.assertTrue(result['ready'])
        self.assertLess(result['outside_largest_fraction_change'],.01)
        self.assertEqual(result['introduced_fragments_gt_5_mm3_count'],0)

    # Purpose: Verify coarse sampling detects disappearance of material native components.
    # Inputs: Main target body and separate 6 mm3 component omitted by the coarse grid.
    # Outputs: Audit rejects lost material component and records its disappearance.
    def test_native_component_above_five_mm3_can_disappear(self):
        mask=np.zeros((10,10,10),bool)
        mask[4:8,4:8,4:8]=True
        mask[0,1:4,1:3]=True  # Six mm3; k=0 is never sampled on 2 mm grid.
        result=self.audit(mask)
        self.assertFalse(result['ready'])
        self.assertIn('native_component_lost_gt_5_mm3',result['flags'])
        self.assertEqual(result['native_components_lost_gt_5_mm3'],[6.])

    # Purpose: Verify an entirely lost target fails model-grid readiness.
    # Inputs: Single native foreground voxel omitted by 2 mm sampling.
    # Outputs: Not ready, empty_model_target flag and zero model foreground voxels.
    def test_empty_sampled_target_is_rejected(self):
        mask=np.zeros((3,3,3),bool);mask[0,0,0]=True
        result=self.audit(mask)
        self.assertFalse(result['ready'])
        self.assertIn('empty_model_target',result['flags'])
        self.assertEqual(result['model']['foreground_voxels'],0)

    # Purpose: Verify preexisting disconnection is not reported as a new fragment.
    # Inputs: Two disconnected 3-cube bodies sampled without changing spacing.
    # Outputs: Ready, zero separation change and preexisting-disconnection warning without new fragments.
    def test_preexisting_disconnection_is_not_called_new_fragment(self):
        mask=np.zeros((10,10,10),bool)
        mask[1:4,1:4,1:4]=True;mask[6:9,6:9,6:9]=True
        result=self.audit(mask,target=(1,1,1))
        self.assertTrue(result['ready'])
        self.assertEqual(result['outside_largest_fraction_change'],0)
        self.assertIn('model_material_disconnection_including_preexisting',result['warnings'])
        self.assertFalse(result['introduced_fragments_gt_5_mm3'])

    # Purpose: Verify anisotropic label sampling uses the global loader center convention.
    # Inputs: Indexed 5x6x7 labels, offset origin and distinct native/model spacing.
    # Outputs: Sampled labels equal the explicit center-to-nearest-index calculation.
    def test_anisotropic_alignment_matches_global_loader_center_convention(self):
        values=np.arange(5*6*7,dtype=np.int32).reshape(5,6,7)+1
        origin=np.array([5,7,9]);native=np.array([.7,.9,1.2]);target=np.array([1.1,.6,1.3])
        sampled,low=grid._sample_labels(values,origin,native,[40,40,40],target)
        expected=np.zeros_like(sampled)
        scale=target[::-1]/native[::-1]
        for point in np.ndindex(sampled.shape):
            coord=(low+point+.5)*scale-.5
            source=np.floor(coord+.5).astype(int)-origin[::-1]
            if np.all((source>=0)&(source<values.shape)):
                expected[point]=values[tuple(source)]
        np.testing.assert_array_equal(sampled,expected)
        self.assertGreater(sampled.sum(),0)

    # Purpose: Verify fine-grid replication matches explicit connected-component statistics.
    # Inputs: Two components with anisotropic finer spacing and model voxel budget one.
    # Outputs: Exact replicated statistics without 3-D budget use; coarse-budget request rejected.
    def test_fine_replication_matches_explicit_component_counts_and_avoids_3d_budget(self):
        mask=np.zeros((7,8,9),bool)
        mask[1:3,1:4,1:4]=True;mask[4:6,5:7,5:8]=True
        origin=[7,5,3];native=[2.,1.5,1.];target=[.7,.4,.6]
        result=grid._audit_array(mask,origin,native,[50,50,50],target,max_model_voxels=1)
        explicit,_=grid._sample_labels(mask,origin,native,[50,50,50],target)
        ids,n=label(explicit,structure=generate_binary_structure(3,1))
        counts=np.bincount(ids.ravel(),minlength=n+1);counts[0]=0
        self.assertEqual(result['model'],grid._statistics(counts,np.prod(target)))
        self.assertIn('replication',result['method'])
        self.assertEqual(result['native']['count'],result['model']['count'])
        with self.assertRaisesRegex(ValueError,'exceeds'):
            grid._audit_array(mask,origin,native,[50,50,50],[3,3,3],max_model_voxels=1)

    # Purpose: Verify source-edge validity and nearest-neighbor tie handling.
    # Inputs: Offset indexed labels with 0.9 mm native and 0.6 mm target spacing.
    # Outputs: Sampled labels equal explicit clipped indices and physical validity masks.
    def test_source_edge_and_tie_phase_follow_loader_formula(self):
        values=np.arange(3*4*5,dtype=np.int32).reshape(3,4,5)+1
        origin=np.array([5,6,7]);native=np.array([.9,.9,.9]);target=np.array([.6,.6,.6])
        sampled,low=grid._sample_labels(values,origin,native,[10,10,10],target)
        axes=[];valid=[]
        scale=target[::-1]/native[::-1]
        for axis,n in enumerate(sampled.shape):
            coords=(low[axis]+np.arange(n,dtype=float)+.5)*scale[axis]-.5
            nearest=np.floor(coords+.5).astype(int)-origin[::-1][axis]
            axes.append(np.clip(nearest,0,values.shape[axis]-1))
            valid.append((coords>=-.5)&(coords<9.5)&(nearest>=0)&(nearest<values.shape[axis]))
        expected=values[np.ix_(*axes)]*valid[0][:,None,None]*valid[1][None,:,None]*valid[2][None,None,:]
        np.testing.assert_array_equal(sampled,expected)

    # Purpose: Verify model field of view follows multiply-then-divide shape arithmetic.
    # Inputs: Single edge voxel in a 300-cube source at 0.9-to-0.6 mm spacing.
    # Outputs: Expected model shape 450 cubed and sampled block within that field of view.
    def test_model_field_of_view_uses_loader_multiply_then_divide_shape(self):
        values=np.ones((1,1,1),np.int32)
        sampled,low=grid._sample_labels(values,[299,299,299],[.9,.9,.9],[300,300,300],[.6,.6,.6])
        expected_shape=np.ceil(np.array([300,300,300])*np.array([.9,.9,.9])/np.array([.6,.6,.6])).astype(int)
        np.testing.assert_array_equal(expected_shape,[450,450,450])
        self.assertTrue(np.all(low+sampled.shape<=expected_shape))

    # Purpose: Verify target-grid audits need only masks and enforce identity checks.
    # Inputs: Hashed mask-only row, then tampered NPZ and duplicated sample ID.
    # Outputs: Ready JSON-safe audit, checksum failure identifying sample and duplicate-ID rejection.
    def test_mask_only_rows_audit_has_no_attenuation_dependency_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mask.npz'
            mask=np.ones((4,4,4),bool)
            np.savez_compressed(path,after=mask,crop_origin_ijk=[0,0,0],spacing_ijk_mm=[1.,1.,1.],
                                sample_id=np.asarray('case:one'))
            row={'sample_id':'case:one','case_id':'case','mask_npz':str(path),
                 'mask_npz_sha256':grid.sha256(path),'spacing_ijk_mm':[1.,1.,1.],
                 'source_shape_kji':[10,10,10]}
            result=grid.audit_target_grid([row],[1,1,1])
            self.assertTrue(result['ready']);self.assertEqual(result['summary']['ready_sample_count'],1)
            json.dumps(result,allow_nan=False)
            path.write_bytes(path.read_bytes()+b'changed')
            result=grid.audit_target_grid([row],[1,1,1])
            self.assertFalse(result['ready'])
            self.assertIn('checksum changed',result['errors'][0])
            self.assertEqual(result['summary']['failed_sample_ids'],['case:one'])
            with self.assertRaisesRegex(ValueError,'unique'):
                grid.audit_target_grid([row,row],[1,1,1])


if __name__=='__main__':unittest.main()
