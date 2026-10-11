"""Derived comparison reports must not turn incomplete or corrupt runs into scores."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('comparison_report',ROOT/'scripts/research/report_context_comparison.py')
reporter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class ComparisonReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'input.ini').write_text('frozen input')
        (self.root/'dataset.json').write_text('{}')
        source=self.root/'source/src/backend.py';source.parent.mkdir(parents=True);source.write_text('code')
        self.code={'backend.py':reporter.sha256(source)}
        self.settings=dict(epochs=2,patch_size_kji=[8,8,8],target_spacing_mm=[.6,.6,.6])
        self.comparison=dict(updates_per_epoch=8,validation_every_epochs=2,volume_cache_size=2)
        self.lock=dict(schema_version='fakect.context-comparison-lock/1',diagnostic=False,
            model_settings=self.settings,comparison=self.comparison,
            samples=[dict(sample_id=case+':'+variant,case_id=case,anatomy_family=case,
                split=split,baseline=variant=='baseline')
                for split,case in [('train','a'),('validation','b'),('test','c')]
                for variant in ('baseline','sibling')],
            input_sha256=reporter.sha256(self.root/'input.ini'),
            dataset_manifest=str(self.root/'dataset.json'),dataset_manifest_sha256=reporter.sha256(self.root/'dataset.json'),
            code_sha256={'src/backend.py':self.code['backend.py']})
        self.signed(self.root/'experiment-lock.json',self.lock)

    def signed(self,path,value):
        path.write_text(json.dumps(value,indent=2)+'\n')
        Path(str(path)+'.sha256').write_text(reporter.sha256(path)+'\n')

    def complete(self,arm,dice=.7):
        model=self.root/arm/'model';evaluation=self.root/arm/'evaluation'
        model.mkdir(parents=True);evaluation.mkdir()
        (model/'checkpoint.keras').write_bytes(b'checkpoint')
        validation=dict(epoch=2,arm=arm,summary={'family_macro':{'dice':.6}})
        self.signed(model/'validation-epoch-002.json',validation)
        history=[dict(epoch=i,train_loss=1/i,train_batches=8,
            physical_patch_events_sha256='1'*64,validation_performed=i==2) for i in (1,2)]
        history[-1].update(validation_family_macro_dice=.6,validation_file='validation-epoch-002.json',
            validation_sha256=reporter.sha256(model/'validation-epoch-002.json'))
        (model/'history.json').write_text(json.dumps(history))
        training=dict(schema_version='fakect.context-comparison-model/1',complete=True,arm=arm,
            lock_fingerprint=reporter.fingerprint(self.lock),settings=self.settings,comparison=self.comparison,
            component_sha256=self.code,model_sha256=reporter.sha256(model/'checkpoint.keras'),
            history_sha256=reporter.sha256(model/'history.json'),epochs_completed=2,optimizer_updates=16,
            initial_weights_sha256='2'*64,best_epoch=2,best_selection_value=.6,
            test_payloads_loaded=False,loaded_samples=['a:baseline','b:baseline','b:sibling'])
        self.signed(model/'training.json',training)
        records=[dict(sample_id='c:'+v,case_id='c',anatomy_family='c',split='test',
            metrics=dict(dice=dice,precision=.8,recall=.9,false_positive_mm3=4.,false_negative_mm3=3.))
            for v in ('baseline','sibling')]
        result=dict(schema_version='fakect.context-comparison-results/1',complete=True,arm=arm,split='test',
            training_metadata_sha256=reporter.sha256(model/'training.json'),test_payloads_loaded=True,
            loaded_samples=[r['sample_id']for r in records],records=records,summary=reporter.summarize(records))
        result.update({k:training[k]for k in ('lock_fingerprint','settings','comparison','component_sha256','model_sha256')})
        self.signed(evaluation/'results.json',result)
        return training,result

    def arm(self,name='baseline_only'):
        return reporter.inspect_arm(self.root,self.lock,name)

    def test_unstarted_report_contains_no_claim_of_accuracy(self):
        path=reporter.report(self.root)
        text=path.read_text()
        self.assertIn('Comparison incomplete',text)
        self.assertIn('No verified held-out metrics yet',text)
        self.assertNotIn('Test family Dice:',text)

    def test_live_history_is_unverified(self):
        model=self.root/'baseline_only/model';model.mkdir(parents=True)
        (model/'INCOMPLETE').write_text('running')
        (model/'history.json').write_text(json.dumps([dict(epoch=1,train_loss=.7,train_batches=8)]))
        row=self.arm()
        self.assertFalse(row['history_verified'])
        self.assertIsNone(row['training'])
        text=reporter.report(self.root).read_text()
        self.assertIn('Live history is provisional',text)
        self.assertIn('stroke-dasharray',text)

    def test_complete_matched_report(self):
        for arm,dice in zip(reporter.ARMS,(.7,.8)):
            self.complete(arm,dice)
        rows=[self.arm(a)for a in reporter.ARMS]
        self.assertTrue(all(r['evaluation']for r in rows))
        self.assertTrue(all(c['passed']for c in reporter.matching_checks(rows)))
        path=reporter.report(self.root);text=path.read_text()
        self.assertIn('Comparison complete',text)
        self.assertIn('+0.1000',text)
        self.assertIn('Test family Dice: 0.8000',text)
        self.assertNotIn('<script',text)

    def test_checkpoint_corruption_hides_entire_arm(self):
        self.complete('baseline_only')
        (self.root/'baseline_only/model/checkpoint.keras').write_bytes(b'changed')
        row=self.arm()
        self.assertIsNone(row['training']);self.assertIsNone(row['evaluation'])
        self.assertFalse(row['history'])
        self.assertIn('metrics hidden',row['status'])

    def test_result_corruption_retains_verified_fit_only(self):
        self.complete('baseline_only')
        (self.root/'baseline_only/evaluation/results.json').write_text('{}')
        row=self.arm()
        self.assertIsNotNone(row['training']);self.assertIsNone(row['evaluation'])
        self.assertIn('test metrics hidden',row['status'])

    def test_resigned_incorrect_aggregate_is_hidden(self):
        _,result=self.complete('baseline_only')
        result['summary']['family_macro']['dice']=.99
        self.signed(self.root/'baseline_only/evaluation/results.json',result)
        row=self.arm();self.assertIsNone(row['evaluation'])
        self.assertIn('aggregate',row['errors'][0])

    def test_resigned_wrong_test_identity_is_hidden(self):
        _,result=self.complete('baseline_only')
        result['records'][0]['case_id']='a'
        self.signed(self.root/'baseline_only/evaluation/results.json',result)
        row=self.arm();self.assertIsNone(row['evaluation'])
        self.assertIn('family/case/split',row['errors'][0])

    def test_changed_ini_replaces_stale_report_with_no_metrics(self):
        self.complete('baseline_only')
        reporter.report(self.root)
        (self.root/'input.ini').write_text('modified')
        text=reporter.report(self.root).read_text()
        self.assertIn('All metrics hidden',text)
        self.assertNotIn('Test family Dice:',text)

    def test_diagnostic_lock_not_reported_as_heldout(self):
        self.lock['diagnostic']=True
        self.signed(self.root/'experiment-lock.json',self.lock)
        text=reporter.report(self.root).read_text()
        self.assertIn('report_context_diagnostic.py',text)
        self.assertIn('All metrics hidden',text)

    def test_schedule_difference_prevents_matched_claim(self):
        for arm in reporter.ARMS:
            self.complete(arm)
        directory=self.root/'baseline_plus_siblings/model'
        history=json.loads((directory/'history.json').read_text());history[0]['physical_patch_events_sha256']='3'*64
        (directory/'history.json').write_text(json.dumps(history))
        meta=json.loads((directory/'training.json').read_text());meta['history_sha256']=reporter.sha256(directory/'history.json')
        self.signed(directory/'training.json',meta)
        evaluated=self.root/'baseline_plus_siblings/evaluation/results.json'
        result=json.loads(evaluated.read_text());result['training_metadata_sha256']=reporter.sha256(directory/'training.json')
        self.signed(evaluated,result)
        text=reporter.report(self.root).read_text()
        self.assertIn('Matching checks failed',text)


if __name__=='__main__':
    unittest.main()
