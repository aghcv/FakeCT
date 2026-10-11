"""Contracts around matched batching, stage separation and fixed refitting."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from fakect_coa_rotation_learning import matched_pairs,StageFineInputs
from fakect_coa_rotation_experiment import select_refit_budgets,require_smoke


def row(sample,case,label,split='train',companion=1):
    return {'sample_id':sample,'case_id':case,'anatomy_family':'family-'+case,
            'split':split,'baseline':False,'parameters':{'arch_outer_expand':{'distance_mm':companion},
            'coa_narrow':{'distance_mm':1 if not label else 2}},
            'annotation':{'coa_benchmark_positive':label}}


class RotationLearningTests(unittest.TestCase):
    def test_production_smoke_checks_geometry_resident_cache_and_gpu_evidence(self):
        c={'experiment':{'output_directory':'/tmp/fixture'},'training':{'cache_limit_mb':1,
            'host_memory_limit_mb':1024,'gpu_memory_limit_mb':512,'require_gpu':True}}
        lock={'runtime_geometry':{'input.overview':{'shape_kji':[16,16,16]},
            'input.detail':{'shape_kji':[16,16,16],'proposals':2}}}
        smoke={'status':'passed','checkpoint_parity':True,'source_lock_sha256':'lock',
            'overview_shape':[16,16,16],'detail_shape':[16,16,16],'traces':{'train':1,'predict':1},
            'locator_gradient_norm':1.,'classifier_gradient_norm':1.,'gpu_visible':True,
            'host_peak_rss_mib':800.,'gpu_peak_mib':300.,'full_cache_memory_probe':{
                'source_lock_sha256':'lock','physical_pages_touched':True,'sample_count':16,
                'resident_bytes':1048576,'passed':True,'host_guard_passed':True,'model_update_while_resident':True}}
        with patch('fakect_coa_rotation_experiment.base.checked_json',side_effect=lambda _:smoke),patch('fakect_coa_rotation_experiment.base.sha256',return_value='lock'):
            self.assertEqual(require_smoke(c,lock)['status'],'passed')
            for key,bad in (('gpu_visible',False),('detail_shape',[8,8,8]),
                            ('classifier_gradient_norm',0.),('locator_gradient_norm',float('nan')),
                            ('host_peak_rss_mib',2048.),('traces',{'train':2,'predict':1})):
                old=smoke[key];smoke[key]=bad
                with self.assertRaises(ValueError):require_smoke(c,lock)
                smoke[key]=old
            smoke['full_cache_memory_probe']['physical_pages_touched']=False
            with self.assertRaises(ValueError):require_smoke(c,lock)
    def test_matching_uses_companion_edits_and_not_list_order(self):
        rows=[row('b+','b',True),row('a-','a',False),row('a+','a',True),row('b-','b',False)]
        pairs=matched_pairs(rows)
        self.assertEqual([[r['sample_id'] for r in pair] for pair in pairs],[['a-','a+'],['b-','b+']])
        for invalid in ([rows[0]],rows+[rows[0]],
                        [row('x','a',False),row('y','a',False)],
                        [row('x','a',False),row('y','a',True,companion=2)],
                        [row('x','a',False),row('y','a',True,split='validation')]):
            with self.assertRaises(ValueError):matched_pairs(invalid)

    def test_allowlist_rejects_reserved_samples_before_native_reader(self):
        allowed=row('x','a',False,split='final_test')
        inputs=StageFineInputs({},256,[allowed])
        with patch('fakect_coa_pilot_learning.FineInputs.classifier',return_value={'detail_0':np.zeros((1,16,16,16,2))}) as read:
            result=inputs.classifier(allowed,{})
            self.assertIn('detail_0',result)
            self.assertEqual(read.call_args.args[0]['split'],'validation')
            for bad in (row('reserved','r',False,split='final_test'),{**allowed,'split':'train'}):
                with self.assertRaises(ValueError):inputs.classifier(bad,{})
            self.assertEqual(read.call_count,1)

    def test_refit_budget_uses_all_inner_validation_results_and_no_test(self):
        stages=[{'id':f'i{i}','outer_fold_id':'outer','stage':'inner_selection','seed':42} for i in range(4)]
        lock={'stages':stages}
        stage={'outer_fold_id':'outer','stage':'outer_refit','seed':42,
               'parent_inner_stage_ids':[r['id'] for r in stages]}
        docs={r['id']:{'status':'completed','stage_id':r['id'],'source_lock_sha256':'lock',
            'final_test_evaluated':False,'locator_selected_epoch':10+i,
            'classifier_selected_epoch':20+i} for i,r in enumerate(stages)}
        def checked(path):return docs[path.parent.name]
        with patch('fakect_coa_rotation_experiment.base.checked_json',side_effect=checked),patch('fakect_coa_rotation_experiment.base.sha256',return_value='lock'):
            result=select_refit_budgets(lock,stage,'/tmp/fixture')
            self.assertEqual((result['locator_epochs'],result['classifier_epochs']),(12,22))
            docs['i0']['final_test_evaluated']=True
            with self.assertRaises(ValueError):select_refit_budgets(lock,stage,'/tmp/fixture')
            docs['i0']['final_test_evaluated']=False
            docs['i0']['source_lock_sha256']='other'
            with self.assertRaises(ValueError):select_refit_budgets(lock,stage,'/tmp/fixture')


if __name__=='__main__':unittest.main()
