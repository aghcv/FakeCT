"""Contracts at the focused-change workflow boundary, including overlap scoring."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fakect_change_experiment as exp
import fakect_change_learning as learning


class ConfigTests(unittest.TestCase):
    def load(self, old, new):
        text = (exp.ROOT/'archive/research/configs/models/aorta-focused-change.ini').read_text().replace(old,new)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.ini'
            path.write_text(text)
            return exp.load_config(path)

    # Purpose: Reject model settings outside supported training bounds.
    # Inputs: Change weight 65, Dice weight 11, or inference overlap 96.
    # Outputs: Configuration raises ValueError before GPU work.
    def test_model_loss_bounds_fail_before_gpu(self):
        for old,new in [('change_weight = 8','change_weight = 65'),('dice_weight = 1','dice_weight = 11'),
                        ('inference_overlap = 24','inference_overlap = 96')]:
            with self.subTest(new=new),self.assertRaises(ValueError):
                self.load(old,new)

    # Purpose: Verify the fixed training and family checkpoint policy.
    # Inputs: Archived focused-change INI and an unsupported family replacement.
    # Outputs: Defaults specify 64 epochs, 96-cube patches, five families; the replacement raises ValueError.
    def test_checkpoint_policy_is_fixed_and_inputs_simple(self):
        c=exp.load_config(exp.ROOT/'archive/research/configs/models/aorta-focused-change.ini')
        self.assertEqual(c['training']['epochs'],64)
        self.assertEqual(c['training']['patch_shape_kji'],[96]*3)
        self.assertEqual(len(c['evaluation']['families']),5)
        with self.assertRaises(ValueError):
            self.load('families = xcat_reference_10yr,','families = xcat_reference_15yr,')


class FakeTF:
    float32=np.float32
    @staticmethod
    def convert_to_tensor(value,dtype): return np.asarray(value,dtype=dtype)
    class nn:
        @staticmethod
        def softmax(logits,axis):
            e=np.exp(logits-logits.max(axis=axis,keepdims=True))
            return e/e.sum(axis=axis,keepdims=True)


class FakeSteps:
    def predict(self,x):
        labels=np.rint(x[...,0]*2).astype(int)
        return np.eye(3,dtype=np.float32)[labels]*20


class FakeReader:
    origins_kji=np.array([[0,0,0],[2,0,0]])
    shape_kji=np.array([4,4,4])
    spacing_ijk_mm=np.array([1.,1.,1.])
    labels=np.zeros((6,4,4),np.uint8)
    labels[2,1,1]=1
    labels[3,2,2]=2
    before=np.ones((6,4,4),np.uint8)
    before[2,1,1]=0
    after=before.copy()
    after[2,1,1]=1
    after[3,2,2]=0
    def __init__(self,*args,**kwargs): pass
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def inference_tiles(self,*args):
        for i in range(2):
            yield {'proposal_index':i,'offset_kji':(0,0,0),'shape_kji':(4,4,4)}
    def read_patch(self,proposal_index,offset_kji,shape_kji):
        start=2*proposal_index
        labels=self.labels[start:start+4]
        return {'x':np.stack((labels/2,np.ones_like(labels)),axis=-1).astype(np.float32)}
    def stats(self): return {}


def focus_masks(reader,index):
    sl=slice(index*2,index*2+4)
    return reader.labels[sl],np.ones((4,4,4),np.uint8),reader.before[sl],reader.after[sl]


class EvaluationTests(unittest.TestCase):
    # Purpose: Verify overlapping predictions are scored once and persisted.
    # Inputs: Two overlapping 4-cube proposals, fake reader/model, and perfect added/removed logits.
    # Outputs: Each change class has one true positive and Dice 1; saved labels are 6x4x4 with float16 probabilities.
    def test_overlap_is_scored_once_and_probabilities_saved(self):
        row={'sample_id':'p:v','case_id':'p','anatomy_family':'family','baseline':False}
        proposal={'origins_kji':[[0,0,0],[2,0,0]],'coverage_passed':True,'added_coverage':1.,'removed_coverage':1.}
        c={'training':{'patch_shape_kji':[4]*3,'inference_overlap':0}}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'predictions';out.mkdir()
            with patch.object(learning,'FocusedChangeVolume',FakeReader),patch.object(learning,'_sample_focus_masks',focus_masks):
                record=learning.evaluate_sample(FakeTF,FakeSteps(),row,proposal,{},c,out)
            for name in ('added','removed'):
                m=record['metrics']['classes'][name]
                self.assertEqual(m['target_voxels'],1)
                self.assertEqual(m['tp'],1)
                self.assertEqual(m['dice'],1.)
            with np.load(Path(directory)/record['prediction_npz']) as arrays:
                self.assertEqual(arrays['labels'].shape,(6,4,4))
                self.assertEqual(arrays['added_probability'].dtype,np.float16)
                np.testing.assert_array_equal(arrays['predicted'],arrays['labels'])

    # Purpose: Require baseline controls and both change classes for validation.
    # Inputs: Perfect edited prediction and a baseline with change fractions 0.02 then 0.
    # Outputs: Gate rejects absent/noisy controls and accepts the clean baseline.
    def test_gate_requires_controls_and_both_change_classes(self):
        from fakect_change_report import change_metrics
        labels=FakeReader.labels
        r={'sample_id':'p:v','case_id':'p','baseline':False,'locator_coverage_passed':True,
           'metrics':change_metrics(labels,labels,np.ones_like(labels),[1]*3)}
        c={'validation':{'minimum_dice':.6,'minimum_precision':.5,'minimum_recall':.5,'maximum_control_change_fraction':.01}}
        self.assertFalse(learning.validation_gate(c,[r])['passed'])
        control={'sample_id':'p:baseline','case_id':'p','baseline':True,'original_aorta_voxels':10,
                 'control_change_fraction':.02,'metrics':change_metrics(labels*0,labels*0,np.ones_like(labels),[1]*3)}
        self.assertFalse(learning.validation_gate(c,[r,control])['passed'])
        control['control_change_fraction']=0.
        self.assertTrue(learning.validation_gate(c,[r,control])['passed'])


if __name__=='__main__':
    unittest.main()
