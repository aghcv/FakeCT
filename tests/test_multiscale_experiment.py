"""Pilot joins, family-safe selection, geometry and gates."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import fakect_multiscale_experiment as e


def row(case, variant, score, baseline=False, dilation=1, split='train'):
    return {'sample_id':case+':'+variant, 'case_id':case, 'anatomy_family':case,
            'baseline':baseline, 'split':split,
            'parameters':{} if baseline else {'arch_outer_expand':{'distance_mm':dilation}},
            'annotation':{'coa_benchmark_positive':score>=.2, e.SCORE:score}}


class ExperimentTests(unittest.TestCase):
    # Purpose: Verify multiscale configs reject unchecked budgets and label acceptance settings.
    # Inputs: Archived pilot INI with oversized chunk budget, unknown model key, or silent acceptance.
    # Outputs: Expected default detail shape and ValueError for every invalid config.
    def test_config_rejects_unknown_parameter_and_invalid_budget(self):
        import tempfile
        p=Path(__file__).resolve().parents[1]/'archive/research/configs/models/coa-multiscale-pilot.ini'
        c=e.load_config(p)
        self.assertEqual(c['input.detail']['shape_kji'], [216,160,160])
        for text in [p.read_text().replace('max_chunk_voxels = 4000000','max_chunk_voxels = 8000000'),
                     p.read_text().replace('[model]','[model]\nunknown = 3'),
                     p.read_text().replace('accepted_engineering_proxy','silent')]:
            with tempfile.TemporaryDirectory() as d:
                f=Path(d)/'config.ini';f.write_text(text)
                with self.assertRaises(ValueError):e.load_config(f)

    # Purpose: Verify annotation joins require unique matching identities and finite targets.
    # Inputs: One correct annotation then missing/duplicate/family-mismatched/unknown/NaN labels.
    # Outputs: One joined row and ValueError for each invalid annotation panel.
    def test_join_requires_exact_ids_case_family_and_known_labels(self):
        rows=[row('a','x',.3)]
        labels=[dict(rows[0]['annotation'],sample_id='a:x',case_id='a',anatomy_family='a')]
        self.assertEqual(len(e.join_annotations(rows,labels)),1)
        for broken in [[],labels+labels,[dict(labels[0],anatomy_family='b')],
                       [dict(labels[0],coa_benchmark_positive=None)],
                       [dict(labels[0],**{e.SCORE:float('nan')})]]:
            with self.assertRaises(ValueError):e.join_annotations(rows,broken)

    # Purpose: Verify pilot selection balances classes with matched companion edits.
    # Inputs: TRAIN/validation/test originals and siblings at two dilation levels.
    # Outputs: Eight unique development samples, four positives, fixed dilation level, and no test rows.
    def test_selection_balances_and_matches_companion_without_test_samples(self):
        rows=[]
        for case,split in [('a','train'),('b','validation'),('c','test')]:
            rows.append(row(case,'original',0,True,split=split))
            for dilation in (1,3):
                for index,score in enumerate((.1,.25,.5)):
                    rows.append(row(case,f'{dilation}-{index}',score,dilation=dilation,split=split))
        selected=e.select_pilot(rows)
        self.assertEqual(len(selected),8)
        self.assertEqual(sum(r['annotation']['coa_benchmark_positive'] for r in selected),4)
        self.assertFalse(any(r['split']=='test' for r in selected))
        self.assertTrue(all(r['baseline'] or r['parameters']['arch_outer_expand']['distance_mm']==1 for r in selected))
        self.assertEqual(len({r['sample_id'] for r in selected}),len(selected))

    # Purpose: Verify box targets use native voxel faces and padded physical overview extent.
    # Inputs: Offset anisotropic 4x5x6 target mask and 10x12x14 overview.
    # Outputs: Normalized box center and extent match physical face coordinates.
    def test_bbox_uses_padded_physical_extent_and_voxel_faces(self):
        mask=np.zeros((4,5,6),np.uint8);mask[1:3,2:4,1:5]=1
        v=SimpleNamespace(mask_after=mask,mask_origin_ijk=np.array([4,5,6]),spacing_ijk_mm=(.5,1.,2.))
        ov={'image':np.zeros((10,12,14)), 'spacing_ijk_mm':(1.5,3.,8.)}
        b=e.bbox_target(v,ov)
        lo=np.array([7,7,5])*[2,1,.5];hi=np.array([9,9,9])*[2,1,.5]
        extent=np.array([80,36,21])
        np.testing.assert_allclose(b[:3],(lo+hi)/2/extent)
        np.testing.assert_allclose(b[3:],(hi-lo)/extent)

    # Purpose: Verify predicted boxes produce physically consistent image-only crop origins.
    # Inputs: Normalized predicted box and anisotropic overview/detail grids.
    # Outputs: Box centers/extents roundtrip to origins and ordered longitudinal proposals.
    def test_proposals_roundtrip_physical_origins_no_mask_input(self):
        ov={'image':np.zeros((10,12,14)), 'spacing_ijk_mm':(1.5,3.,8.)}
        c={'input.detail':{'shape_kji':[16,16,16], 'spacing_ijk_mm':[.5,.75,1.]}}
        origins,boxes=e.proposal_origins([.5,.5,.5,.4,.2,.2],ov,c)
        extent=np.array([80,36,21])
        for o,b in zip(origins,boxes):
            np.testing.assert_allclose(b[:3]*extent,(np.array(o)+8)*[1,.75,.5],atol=1e-6)
            np.testing.assert_allclose(b[3:]*extent,16*np.array([1,.75,.5]),atol=1e-6)
        self.assertLess(origins[0][0],origins[1][0])

    # Purpose: Verify coverage separately measures removed targets across proposal unions.
    # Inputs: Mostly dilated 4-cubed mask with distant removed voxel and partial/full proposal unions.
    # Outputs: Partial removal coverage zero and full union removal/change coverage one.
    def test_coverage_union_and_removed_cannot_hide_in_dilation(self):
        before=np.zeros((4,4,4),np.uint8);before[0,0,0]=1;before[3,3,3]=1
        after=np.ones_like(before);after[3,3,3]=0
        v=SimpleNamespace(mask_before=before,mask_after=after,mask_origin_ijk=np.zeros(3,int),spacing_ijk_mm=(1,1,1))
        c={'input.detail':{'shape_kji':[3,4,4], 'spacing_ijk_mm':[1,1,1]}}
        cov=e.coverage(v,[[0,0,0],[0,0,0]],c)
        self.assertEqual(cov['removed_coverage'],0.)
        self.assertGreater(cov['changed_coverage'],.5)
        full=e.coverage(v,[[0,0,0],[1,0,0]],c)
        self.assertEqual(full['removed_coverage'],1.)
        self.assertEqual(full['changed_coverage'],1.)

    # Purpose: Verify unchanged baselines do not supply change-coverage evidence.
    # Inputs: Equal 2-cubed before/after masks and full proposal.
    # Outputs: changed_coverage is None.
    def test_empty_baseline_changes_are_not_coverage_evidence(self):
        a=np.ones((2,2,2),np.uint8)
        v=SimpleNamespace(mask_before=a,mask_after=a,mask_origin_ijk=np.zeros(3,int),spacing_ijk_mm=(1,1,1))
        c={'input.detail':{'shape_kji':[2,2,2], 'spacing_ijk_mm':[1,1,1]}}
        self.assertIsNone(e.coverage(v,[[0,0,0]],c)['changed_coverage'])

    # Purpose: Verify per-step edit accounting matches final net changes and order.
    # Inputs: Valid expansion/two-erosion steps then changed removal count or reversed order.
    # Outputs: Valid proof accepted and inconsistent/unsupported recipe evidence rejected.
    def test_no_cancellation_proof_rejects_unsupported_recipe(self):
        b=np.ones((2,2,2),np.uint8);a=b.copy();a[0,0,0]=0;a[1,1,1]=0
        v=SimpleNamespace(mask_before=b,mask_after=a)
        steps=[dict(step_name=n,operation=op,iteration=1,status='applied',added_voxels=0,removed_voxels=count)
               for n,op,count in [('arch_outer_expand','dilation',0),('arch_inner_narrow','erosion',1),('coa_narrow','erosion',1)]]
        r={'baseline':False,'step_outcomes':steps}
        e.verify_no_cancellation(v,r)
        broken=copy.deepcopy(r);broken['step_outcomes'][2]['removed_voxels']=2
        with self.assertRaises(ValueError):e.verify_no_cancellation(v,broken)
        broken=copy.deepcopy(r);broken['step_outcomes'].reverse()
        with self.assertRaises(ValueError):e.verify_no_cancellation(v,broken)


if __name__=='__main__':unittest.main()
