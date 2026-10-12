"""Selective finite repair and release of an individually verified old panel."""
from contextlib import redirect_stdout
from copy import deepcopy
import configparser
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import fakect_sibling_repair as repair
import fakect_sibling_population as population
from fakect_context_volume import ContextVolume
from fakect_multiscale_experiment import write_json,sha256
import test_sibling_population


class BackoffTests(unittest.TestCase):
    def setUp(self):
        self.before=np.zeros((5,3,3),bool)
        self.before[:,1,1]=True
        self.arrays={'selected':self.before,'act':np.where(self.before,100,20).astype(np.int32)}
        self.config={'recipe':{'steps':['grow','arch','coa']},'edits':{
            'grow':{'operation':'dilation','distance_mm':3.,'iterations':1,'direction':'outer','angular_width_deg':90},
            'arch':{'operation':'erosion','distance_mm':2.,'iterations':1,'path_percent':[20,55],
                    'profile':'gaussian','shape_k':6.,'preserve_connectivity':True},
            'coa':{'operation':'erosion','distance_mm':1.5,'iterations':1,'path_percent':[60,90],
                   'min_volume_ratio':.1,'preserve_connectivity':True}}}

    def runner(self,culprit,threshold):
        def run(arrays,resolved,c,on_step):
            current=self.before.copy()
            for index,name in enumerate(c['recipe']['steps'],1):
                if name==culprit and c['edits'][name]['distance_mm']>threshold:
                    current[2,1,1]=False
                origins=np.where(current,np.arange(current.size).reshape(current.shape),-1)
                on_step({'step_name':name,'iteration':1,'index':index,
                    'config':{'edit':c['edits'][name]},
                    'result':{'target_origin_index':origins,'summary':{'erosion_safeguard':None}}})
            return {'result_mask':current,'target_origin_index':origins}
        return run

    # Purpose: Verify selective repair reduces only the first failing erosion distance.
    # Inputs: Mock recipe disconnecting coa above 0.5 mm with unrelated growth/arch settings.
    # Outputs: Accepted coa 0.375 after two reductions, preserved other settings/config, and connected final mask.
    def test_only_first_failing_erosion_changes_and_other_guard_settings_survive(self):
        original=deepcopy(self.config)
        with patch.object(repair,'apply_recipe',self.runner('coa',.5)):
            result,accepted,audit=repair.repair_recipe(self.arrays,{},self.config)
        self.assertEqual(self.config,original)
        self.assertEqual(accepted['edits']['grow'],original['edits']['grow'])
        self.assertEqual(accepted['edits']['arch'],original['edits']['arch'])
        self.assertEqual(accepted['edits']['coa']['distance_mm'],.375)
        self.assertEqual(accepted['edits']['coa']['path_percent'],[60,90])
        self.assertEqual(accepted['edits']['coa']['min_volume_ratio'],.1)
        self.assertEqual(audit['distance_reductions'],{'coa':2})
        self.assertEqual(len(audit['attempts']),3)
        self.assertTrue(audit['attempts'][-1]['accepted'])
        self.assertTrue(np.array_equal(result['result_mask'],self.before))

    # Purpose: Verify arch backoff leaves coa unchanged and respects finite retry bounds.
    # Inputs: Arch failure above 0.75 mm then always-failing arch with one retry allowed.
    # Outputs: Accepted arch 0.5, unchanged coa, and RepairFailure with two audited attempts on exhaustion.
    def test_arch_repair_does_not_reduce_coa_and_has_finite_failure(self):
        with patch.object(repair,'apply_recipe',self.runner('arch',.75)):
            _,accepted,audit=repair.repair_recipe(self.arrays,{},self.config)
        self.assertEqual(accepted['edits']['arch']['distance_mm'],.5)
        self.assertEqual(accepted['edits']['coa'],self.config['edits']['coa'])
        self.assertEqual(audit['attempts'][0]['first_failing_step'],'arch')
        with patch.object(repair,'apply_recipe',self.runner('arch',0)):
            with self.assertRaisesRegex(repair.RepairFailure,'backoff exhausted') as failure:
                repair.repair_recipe(self.arrays,{},self.config,max_backoff_steps=1)
        self.assertFalse(failure.exception.audit['passed'])
        self.assertEqual(len(failure.exception.audit['attempts']),2)
        self.assertEqual(failure.exception.audit['distance_reductions'],{'arch':1})

    # Purpose: Verify selective erosion repair refuses growth failures.
    # Inputs: Mock disconnection during grow step.
    # Outputs: Non-erosion RepairFailure after one attempt and no distance reductions.
    def test_dilation_failure_is_not_silently_changed_or_allowed(self):
        with patch.object(repair,'apply_recipe',self.runner('grow',0)):
            with self.assertRaisesRegex(repair.RepairFailure,'non-erosion') as failure:
                repair.repair_recipe(self.arrays,{},self.config)
        self.assertEqual(len(failure.exception.audit['attempts']),1)
        self.assertEqual(failure.exception.audit['distance_reductions'],{})


class RepairReleaseTests(unittest.TestCase):
    def setUp(self):
        fixture=test_sibling_population.PopulationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root,self.original_c=fixture.root,fixture.c
        self.original_c['population']['variants_per_case']=9
        parser=configparser.ConfigParser(interpolation=None)
        for section,values in self.original_c.items():
            parser[section]={key:','.join(map(str,value)) if isinstance(value,list) else str(value)
                             for key,value in values.items()}
        stream=io.StringIO();parser.write(stream)
        self.content=stream.getvalue().encode()
        lock=population.freeze(self.original_c,self.content)
        with redirect_stdout(io.StringIO()):
            document=population.prepare_case(self.original_c,lock,'001')
        self.parent=Path(self.original_c['population']['output_directory'])
        self.case_path=self.parent/'cases/001/dataset-manifest.json'
        self.c={'schema_version':repair.SCHEMA,'parent_directory':str(self.parent),
                'output_directory':str(self.root/'repair'),'backoff_factor':.5,'max_backoff_steps':8}
        self.document=document

    def make_incomplete_diagnostic(self):
        d=deepcopy(self.document)
        record=d['records'][0]
        sid=record['sample_id']
        record.update(status='failed',error_type='ValueError',
                      error='Native edited trunk disconnected or completely removed')
        d['samples']=[r for r in d['samples'] if r['sample_id']!=sid]
        d.update(complete=False,successful_siblings=8,failed_siblings=1)
        write_json(self.case_path,d)
        (self.case_path.parent/'INCOMPLETE').touch()
        return sid

    # Purpose: Verify frozen repair reuses good siblings and releases a new repaired identity.
    # Inputs: Nine-sibling panel with one recorded disconnect failure and eight verified good siblings.
    # Outputs: Eight reused/one repaired, complete ten-sample population, preserved parent, and overwrite refusal.
    def test_frozen_repair_reuses_good_data_and_exports_new_accepted_identity(self):
        requested_sid=self.make_incomplete_diagnostic()
        old_receipt_hash=sha256(self.case_path)
        lock=repair.freeze(self.c,b'# test repair input\n')
        self.assertEqual(lock['reused_siblings'],8)
        self.assertEqual(lock['repair_siblings'],1)
        lock=repair.inspect_lock(self.c,b'# test repair input\n')
        with redirect_stdout(io.StringIO()):
            case=repair.repair_case(self.c,lock,'001')
        self.assertTrue(case['complete'])
        self.assertEqual(case['reused_siblings'],8)
        self.assertEqual(case['repaired_siblings'],1)
        fixed=next(r for r in case['samples'] if r.get('requested_sample_id')==requested_sid)
        self.assertNotEqual(fixed['sample_id'],requested_sid)
        self.assertIn(':repair-',fixed['sample_id'])
        with ContextVolume(fixed) as v:
            out=v.read_block(fixed['edit_crop_origin_ijk'][::-1],fixed['edit_crop_shape_kji'],fixed['spacing_ijk_mm'])
            self.assertTrue(np.isfinite(out['image']).all())
        population_document=repair.collect(self.c,lock)
        self.assertTrue(population_document['complete'])
        self.assertEqual(population_document['edited_siblings'],9)
        self.assertEqual(population_document['original_baselines'],1)
        new_case_path=Path(self.c['output_directory'])/'cases/001/dataset-manifest.json'
        for r in population_document['samples']:
            if not r['baseline']:
                self.assertEqual(r['prepared_manifest'],str(new_case_path))
                self.assertNotEqual(r['prepared_manifest'],str(self.case_path))
        self.assertEqual(sha256(self.case_path),old_receipt_hash)
        self.assertTrue((self.case_path.parent/'INCOMPLETE').exists())
        reused=[r for r in case['samples'] if not r['baseline'] and 'requested_sample_id' not in r]
        self.assertTrue(all(str(self.parent) in r['variant_npz'] for r in reused))
        with self.assertRaises(FileExistsError):
            repair.collect(self.c,lock)

    # Purpose: Verify repairs require a known native failure and verified reusable payloads.
    # Inputs: Failed record changed to unassigned attenuation then good variant bytes altered.
    # Outputs: ValueError for unsupported failure reason and changed good payload.
    def test_unknown_native_failures_or_mutated_good_bytes_are_not_reused(self):
        self.make_incomplete_diagnostic()
        d=repair._read_parent_case(self.case_path)
        next(r for r in d['records'] if r['status']=='failed')['error']='Unassigned attenuation'
        write_json(self.case_path,d)
        with self.assertRaisesRegex(ValueError,'only recorded'):
            repair.plan(self.c)
        self.make_incomplete_diagnostic()
        d=repair._read_parent_case(self.case_path)
        row=next(r for r in d['samples'] if not r['baseline'])
        Path(row['variant_npz']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'payload changed'):
            repair.plan(self.c)


if __name__=='__main__':
    unittest.main()
