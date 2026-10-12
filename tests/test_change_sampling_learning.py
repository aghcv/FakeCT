"""Diagnostic gates must test both classes and whole-focus normal preservation."""
import unittest
import numpy as np
from fakect_change_sampling_learning import diagnostic_gate
from fakect_change_report import change_metrics


class GateTests(unittest.TestCase):
    def setUp(self):
        self.c={'diagnostic':{'case_ids':['adult','child']},'validation':
            {'minimum_dice':.6,'minimum_precision':.5,'minimum_recall':.5,'maximum_control_change_fraction':.01}}
        self.patches=[]
        for case in ('adult','child'):
            labels=np.zeros((4,4,4),np.uint8);labels[1,1,1]=1;labels[2,2,2]=2
            self.patches.append({'case_id':case,'baseline':False,
                'metrics':change_metrics(labels,labels,np.ones_like(labels),[1]*3)})
        self.whole=[{'case_id':case,'baseline':True,'original_aorta_voxels':100,
                     'control_change_fraction':0.} for case in ('adult','child')]

    # Purpose: Verify diagnostics require both anatomies and both learned change classes.
    # Inputs: Perfect added/removed metrics for adult and child, then only one case.
    # Outputs: Complete diagnostics pass; missing anatomy fails the gate.
    def test_both_anatomies_and_classes_must_learn(self):
        self.assertTrue(diagnostic_gate(self.c,self.patches,self.whole)['passed'])
        self.assertFalse(diagnostic_gate(self.c,self.patches[:1],self.whole)['passed'])

    # Purpose: Verify empty targets cannot satisfy change-learning diagnostics.
    # Inputs: Adult and child patches with unchanged truth and predictions.
    # Outputs: Diagnostic gate fails despite perfect empty predictions.
    def test_empty_targets_cannot_artificially_pass(self):
        labels=np.zeros((4,4,4),np.uint8)
        patches=[{'case_id':case,'baseline':False,'metrics':
            change_metrics(labels,labels,np.ones_like(labels),[1]*3)} for case in ('adult','child')]
        self.assertFalse(diagnostic_gate(self.c,patches,self.whole)['passed'])

    # Purpose: Verify control preservation uses the full-focus target denominator.
    # Inputs: Passing diagnostics with adult control_change_fraction changed to 0.02.
    # Outputs: Gate fails the configured maximum control change fraction of 0.01.
    def test_full_focus_control_denominator_not_background_size(self):
        self.whole[0]['control_change_fraction']=.02
        self.assertFalse(diagnostic_gate(self.c,self.patches,self.whole)['passed'])


if __name__=='__main__':unittest.main()
