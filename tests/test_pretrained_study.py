"""Control schedule and exact sparse caches used in the population comparison."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import fakect_pretrained_study as study


class StudyTests(unittest.TestCase):
 # Purpose: Verify pretrained comparison schedules share balanced TRAIN case/patch draws.
 # Inputs: Two-case cache, 64 updates, seed 42, foreground 0.75, and control 0.25.
 # Outputs: Reproducible balanced TRAIN draws, baseline-only controls, sibling variation, and padded i offsets.
 def test_shared_case_patch_schedule_and_training_only_cases(self):
  c={'training':{'seed':42,'updates':64,'foreground_fraction':.75,'control_fraction':.25}}
  class Cache: pass
  cache=Cache();cache.baselines={};cache.records={}
  for case in ('one','two'):
   cache.baselines[case]={'image':np.zeros((132,130,96),np.float32),'points':np.array([[70,65,45],[80,66,50]])}
   cache.records[case]={'baseline':case+':baseline'}
  lock={'rosters':{'training_siblings':{case:[case+':baseline',case+':a',case+':b'] for case in ('one','two')}}}
  fold={'train_cases':['one','two'],'test_cases':['heldout']}
  first=study.schedule(c,lock,cache,fold);second=study.schedule(c,lock,cache,fold)
  self.assertEqual(first,second)
  self.assertEqual({r['case'] for r in first},{'one','two'})
  self.assertEqual(sum(r['case']=='one' for r in first),32)
  self.assertTrue(any(r['siblings']!=r['baseline_only'] for r in first))
  self.assertTrue(all(r['offset_kji'][2]==-16 for r in first))
  self.assertTrue(all(r['baseline_only'].endswith(':baseline') for r in first))

 # Purpose: Verify sparse model caches restore exact edits and restrict supervision domains.
 # Inputs: 2-cubed baseline, two float32 delta values, one excluded domain voxel, and changed delta file.
 # Outputs: Exact edited values, preserved baseline, two targets, excluded loss voxel, and digest rejection.
 def test_sparse_new_values_are_exact_and_loss_domain_excludes_context(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);folder=out/'cases'/'one';folder.mkdir(parents=True)
   (out/'study-lock.json').write_text('{}')
   image=np.arange(8,dtype=np.float32).reshape(2,2,2);target=np.zeros_like(image,np.uint8)
   domain=np.ones_like(target);domain[1,1,1]=0
   np.savez_compressed(folder/'baseline-model.npz',image=image,target=target,domain=domain,valid=np.ones_like(target),points=np.array([[0,0,0]]))
   delta=folder/'delta.npz';newvalues=np.array([.00005,.49999997],np.float32)
   np.savez_compressed(delta,indices=np.array([3,7]),image=newvalues,target=np.array([1,1],np.uint8))
   record={'study_lock_sha256':study.sha256(out/'study-lock.json'),'baseline_model_sha256':study.sha256(folder/'baseline-model.npz'),
           'baseline':'one:baseline','samples':{'one:sibling':{'model_delta':str(delta),'model_delta_sha256':study.sha256(delta)}}}
   with patch.object(study,'checked_json',return_value=record):
    cache=study.ModelCache({'experiment':{'output_directory':str(out)}},{'cases':['one','heldout']},['one'])
   restored,after,valid=cache.volume('one:sibling')
   np.testing.assert_array_equal(restored.ravel()[[3,7]],newvalues)
   np.testing.assert_array_equal(restored.ravel()[[0,1,2,4,5,6]],image.ravel()[[0,1,2,4,5,6]])
   self.assertEqual(after.sum(),2);self.assertFalse(valid[1,1,1])
   np.testing.assert_array_equal(cache.baselines['one']['image'],image)
   with self.assertRaises(KeyError):cache.volume('heldout:sibling')
   delta.write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'delta changed'):cache.volume('one:sibling')

 # Purpose: Verify pretrained protocol fixes its reviewed model-grid resolution.
 # Inputs: Finetune INI with spacing changed from 1.5 to 0.6 mm.
 # Outputs: ValueError stating the protocol fixes 1.5 mm spacing.
 def test_unsupported_resolution_requires_new_protocol(self):
  source=Path(study.ROOT/'archive/research/configs/models/aorta-pretrained-finetune.ini').read_text()
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'input.ini';p.write_text(source.replace('spacing_mm = 1.5','spacing_mm = 0.6'))
   with self.assertRaisesRegex(ValueError,'fixes 1.5'):study.load_config(p)

 # Purpose: Verify branch finetuning uses named targets and a separate frozen reference.
 # Inputs: Archived branches finetune configuration.
 # Outputs: Connected-branch policy, common-box extent, named trunk, and absolute distinct reference output.
 def test_branch_policy_uses_named_foreground_and_frozen_reference(self):
  c=study.load_config(study.ROOT/'archive/research/configs/models/aorta-pretrained-finetune-branches.ini')
  self.assertEqual(c['target']['policy'],'thoracic_aorta_and_connected_branches')
  self.assertEqual(c['target']['branch_extent'],'common_box')
  self.assertIn('brachiocephalic_trunk',c['target']['foreground_labels'])
  self.assertTrue(Path(c['target']['reference_study']).is_absolute())
  self.assertNotEqual(c['experiment']['output_directory'],str(Path(c['target']['reference_study']).parent))

 # Purpose: Verify core pretrained settings retain their historical aorta-only target.
 # Inputs: Archived core finetune configuration without target section.
 # Outputs: No target section and foreground label list containing only aorta.
 def test_legacy_core_policy_still_uses_only_aorta(self):
  c=study.load_config(study.ROOT/'archive/research/configs/models/aorta-pretrained-finetune.ini')
  self.assertNotIn('target',c)
  self.assertEqual(study.target_settings(c)['foreground_labels'],['aorta'])


if __name__=='__main__':unittest.main()
