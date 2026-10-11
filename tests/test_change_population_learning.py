"""Expanded sibling coverage, controls and anatomy-balanced update accounting."""
import unittest
from collections import Counter
from pathlib import Path
import tempfile
from unittest.mock import patch,Mock
import numpy as np
from fakect_change_population_learning import PopulationSchedule,training_proposals


def rows(count=256):
    return [r for case in range(8) for r in
        [{'sample_id':f'{case}:baseline','case_id':str(case),'baseline':True}]+
        [{'sample_id':f'{case}:sibling-{i}','case_id':str(case),'baseline':False} for i in range(count)]]


class ScheduleTests(unittest.TestCase):
    def test_all_variants_seen_and_old_control_budget_preserved(self):
        schedule=PopulationSchedule(rows(),[str(i) for i in range(8)],42)
        by_id={r['sample_id']:r for r in rows()}
        for _ in range(64):
            block=schedule.block()
            per_case=Counter()
            for sid,count in block: per_case[by_id[sid]['case_id']]+=count
            self.assertEqual(set(per_case.values()),{28})
        r=schedule.receipt()
        self.assertEqual(r['edited_updates'],12288)
        self.assertEqual(r['control_updates'],2048)
        self.assertEqual((r['minimum_edited_visits'],r['maximum_edited_visits']),(6,6))
        self.assertEqual(set(r['controls_per_case'].values()),{256})

    def test_order_reproducible_without_replacement(self):
        first=PopulationSchedule(rows(),[str(i) for i in range(8)],42)
        second=PopulationSchedule(rows(),[str(i) for i in range(8)],42)
        self.assertEqual(first.block(),second.block())
        values=[first._next('0') for _ in range(244)]
        self.assertEqual(len(values),len(set(values)))

    def test_missing_control_and_wrong_case_count_rejected(self):
        with self.assertRaises(ValueError):PopulationSchedule(rows(),['0'],42)
        with self.assertRaises(ValueError):PopulationSchedule([r for r in rows() if r['sample_id']!='0:baseline'],[str(i) for i in range(8)],42)

    def test_new_net_targets_do_not_require_original_three_active_steps(self):
        import fakect_multiscale_network_v3 as locator
        import fakect_multiscale_learning_v3 as legacy
        import fakect_multiscale_experiment_v3 as geometry
        import fakect_context_volume as context
        runtime={'experiment':{'output_directory':''},'training':{'minimum_target_coverage':.9},
                 'pilot':{'minimum_changed_coverage':.9}}
        coverage={'target_coverage':1.,'before_coverage':1.,'changed_coverage':1.,
                  'added_coverage':1.,'removed_coverage':None,'changed_voxels':3}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'locator.keras';checkpoint.write_bytes(b'frozen locator')
            from fakect_multiscale_experiment import sha256
            stage={'locator_checkpoint':str(checkpoint),'locator_checkpoint_sha256':sha256(checkpoint)}
            inputs=Mock();inputs.coarse.return_value=({'image':np.zeros((2,2,2))},np.zeros((1,2,2,2,2)))
            volume=Mock();volume.__enter__=Mock(return_value=volume);volume.__exit__=Mock(return_value=False)
            with patch.object(locator,'load_model',return_value=Mock()),patch.object(locator,'predict_boxes',return_value=np.zeros((1,6))),\
                 patch.object(legacy,'Inputs',return_value=inputs),patch.object(legacy,'RuntimeGuard',return_value=Mock()),\
                 patch.object(geometry,'proposal_origins',return_value=([[1,2,3],[4,5,6]],np.zeros((2,6)))),\
                 patch.object(geometry,'coverage',return_value=coverage),patch.object(context,'ContextVolume',return_value=volume),\
                 patch.object(geometry,'verify_no_cancellation',side_effect=AssertionError('old per-condition proof must not run')):
                proposals=training_proposals(Mock(),{'runtime':runtime},stage,
                    [{'sample_id':'0:new-enlargement-only','split':'train','step_outcomes':[]}],root)
            self.assertTrue(proposals['0:new-enlargement-only']['coverage_passed'])
            self.assertEqual(proposals['0:new-enlargement-only']['origins_kji'],[[1,2,3],[4,5,6]])


if __name__=='__main__':unittest.main()
