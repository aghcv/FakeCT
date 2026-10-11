"""Case-balanced expanded siblings at the original localization update budget."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import gc
import resource
import time

import numpy as np

import fakect_change_learning as reference_learning
import fakect_change_network as network
from fakect_change_data import FocusedChangeVolume
from fakect_multiscale_experiment import write_json, sha256


class PopulationSchedule:
    """Two patches per edited draw; four per unchanged control, case balanced.

    Every block uses12edited draws and onecontrol percase, shuffled as groups.
    Edited queues traverse all sibling IDs without replacement before repeating.
    With256siblings and64blocks every sibling has exactly6patch visits; controls
    receive256visits. No severity or achieved response chooses sample order.
    """
    def __init__(self, rows, case_ids, seed):
        self.rng = np.random.default_rng(seed)
        self.cases = sorted(case_ids)
        if len(self.cases) != 8:
            raise ValueError('Expanded comparison requires eight training cases')
        self.edited, self.controls = {}, {}
        self.orders, self.positions, self.visits = {}, {}, Counter()
        for case in self.cases:
            group = [r for r in rows if r['case_id']==case]
            controls = [r['sample_id'] for r in group if r['baseline']]
            edited = sorted(r['sample_id'] for r in group if not r['baseline'])
            if len(controls)!=1 or not edited or len(set(edited))!=len(edited):
                raise ValueError('Every case needs unique edited siblings and one unchanged control')
            self.controls[case], self.edited[case] = controls[0], edited
            self.orders[case],self.positions[case] = [],0

    def _next(self,case):
        if self.positions[case]==len(self.orders[case]):
            self.orders[case] = list(self.rng.permutation(self.edited[case]))
            self.positions[case] = 0
        value = str(self.orders[case][self.positions[case]])
        self.positions[case] += 1
        return value

    def block(self):
        result=[]
        for case in self.rng.permutation(self.cases):
            groups=[(self.controls[case],4)]+[(self._next(case),2) for _ in range(12)]
            for index in self.rng.permutation(len(groups)):
                sid,count=groups[int(index)]
                self.visits[sid]+=count
                result.append((sid,count))
        if sum(count for _,count in result)!=224:
            raise RuntimeError('Reporting block update count changed')
        return result

    def receipt(self):
        edited=[self.visits[sid] for ids in self.edited.values() for sid in ids]
        return {'sample_patch_visits':dict(sorted(self.visits.items())),
                'edited_updates':sum(edited),'control_updates':sum(self.visits[sid] for sid in self.controls.values()),
                'minimum_edited_visits':min(edited),'maximum_edited_visits':max(edited),
                'controls_per_case':{case:self.visits[sid] for case,sid in self.controls.items()},
                'reporting_block_is_complete_population_epoch':False}


def training_proposals(tf,lock,stage,rows,output):
    """Frozen locator on new CT images; target coverage never recenters a crop."""
    import fakect_multiscale_network_v3 as locator_network
    import fakect_multiscale_learning_v3 as legacy
    import fakect_multiscale_experiment_v3 as geometry
    from fakect_context_volume import ContextVolume
    runtime=deepcopy(lock['runtime'])
    runtime['experiment']['output_directory']=str(output)
    locator=locator_network.load_model(stage['locator_checkpoint'])
    if sha256(stage['locator_checkpoint'])!=stage['locator_checkpoint_sha256']:
        raise ValueError('Frozen locator checkpoint changed')
    guard=legacy.RuntimeGuard(runtime,tf)
    inputs=legacy.Inputs(runtime)
    proposals,audit={},[]
    for row in rows:
        guard.check('population-proposal-before')
        ov,image=inputs.coarse(row)
        box=locator_network.predict_boxes(locator,image)[0]
        origins,boxes=geometry.proposal_origins(box,ov,runtime)
        with ContextVolume(row,verify_hashes=True) as volume:
            # ContextVolume verifies exact before/after/net-mask identities.
            # Cancellation and inactive steps are valid for this net-change task;
            # the legacy per-condition proof requires the old three-edit recipe.
            coverage=geometry.coverage(volume,origins,runtime)
        record={'sample_id':row['sample_id'],'split':row['split'],
                'predicted_box':box.tolist(),'origins_kji':origins,
                'proposal_boxes':boxes.tolist(),
                'coverage_passed':geometry.coverage_gate(coverage,runtime),**coverage}
        audit.append(record)
        proposals[row['sample_id']]=record
        guard.check('population-proposal-after')
    write_json(Path(output)/'proposal-audit.json',{'passed':all(r['coverage_passed'] for r in audit),
        'samples':audit,'selection_allowed':False,
        'contract':'CT-only frozen locator; native net-mask coverage only; inactive/cancelled steps are valid; QA never recenters.'})
    del locator
    gc.collect()
    if not all(row['coverage_passed'] for row in audit):
        failed=[row['sample_id'] for row in audit if not row['coverage_passed']]
        raise ValueError('New training crops failed coverage without recentering: '+','.join(failed[:20]))
    return proposals


def smoke(c,lock):
    tf=reference_learning._tensorflow()
    stage=lock['stages'][0]
    by_id={r['sample_id']:r for r in lock['samples']}
    candidates=[by_id[sid] for sid in stage['train_sample_ids'] if not by_id[sid]['baseline']
                and by_id[sid]['mask_counts'].get('added',0)>0 and by_id[sid]['mask_counts'].get('removed',0)>0]
    if not candidates:
        raise ValueError('Production smoke needs one new sibling with both achieved change classes')
    row={**candidates[0],'split':'train'}
    out=Path(c['experiment']['output_directory'])/'smoke'
    proposals=training_proposals(tf,lock,stage,[row],out)
    adapted=deepcopy(lock)
    adapted['stages']=[{**stage,'train_sample_ids':[row['sample_id']]}]
    adapted['proposals']={stage['id']:proposals}
    result=reference_learning.smoke(c,adapted)
    result.update(new_population_sample_exercised=True,matched_optimizer_budget=14336)
    write_json(Path(c['experiment']['output_directory'])/'runtime-smoke.json',result)
    return result


def run_stage(c,lock,stage):
    tf=reference_learning._tensorflow()
    t=c['training']
    out=Path(c['experiment']['output_directory'])/'stages'/stage['id']
    out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').touch()
    rows={r['sample_id']:r for r in lock['samples']}
    train=[{**rows[sid],'split':'train'} for sid in stage['train_sample_ids']]
    evaluate=[{**rows[sid],'split':'final_test'} for sid in stage['final_test_sample_ids']]
    by_id={r['sample_id']:r for r in train}
    write_json(out/'stage-input.json',{'stage':stage,'config':c,
        'source_lock_sha256':sha256(Path(c['experiment']['output_directory'])/'experiment-lock.json'),
        'classifier_started':False,'comparison':'Expanded training siblings with fixed held-out panel and update budget'})
    edited=[r for r in train if not r['baseline']]
    proposals=deepcopy(lock['proposals'][stage['id']])
    proposals.update(training_proposals(tf,lock,stage,edited,out))
    write_json(out/'all-training-proposals.json',{'samples':[proposals[r['sample_id']] for r in train],
        'selection_allowed':False,'baseline_control_policy':'Original fixed paired focus unchanged'})
    model=network.build_model(tuple(t['patch_shape_kji']),t['base_filters'],t['seed'])
    steps=reference_learning._steps(c,model)
    schedule=PopulationSchedule(train,stage['train_case_ids'],t['seed'])
    rng=np.random.default_rng(t['seed'])
    verified,history={},[]
    for block in range(1,t['reporting_blocks']+1):
        started,metrics=time.monotonic(),[]
        order=schedule.block()
        for sid,count in order:
            row=by_id[sid]
            with FocusedChangeVolume(row,lock['runtime'],proposals[sid]['origins_kji'],
                                     verify_hashes=sid not in verified) as reader:
                stats=(reader.volume._variant_stat,reader.volume._mask_stat)
                if sid in verified and verified[sid]!=stats:
                    raise ValueError('Pinned population payload changed between visits')
                verified[sid]=stats
                for patch_index in range(count):
                    mode=('added' if patch_index%2==0 else 'removed') if rng.random()<t['changed_fraction'] else 'uniform'
                    tile=reader.training_patch(t['patch_shape_kji'],rng,mode=mode)
                    data=reader.read_patch(**tile)
                    x,y,v=reference_learning._tensors(tf,data)
                    value=steps.train(x,y,v)
                    metrics.append({key:float(np.asarray(val)) for key,val in value.items()})
            del reader,data,x,y,v,value
        steps.assert_stable_traces()
        if int(steps.optimizer.iterations.numpy())!=block*t['updates_per_block']:
            raise ValueError('Population optimizer update budget changed, possibly invalid patches')
        record={'epoch':block,'reporting_block':block,'loss':float(np.mean([r['loss'] for r in metrics])),
                'ce':float(np.mean([r['cross_entropy'] for r in metrics])),
                'dice_loss':float(np.mean([r['foreground_dice_loss'] for r in metrics])),
                'updates':int(steps.optimizer.iterations.numpy()),'seconds':time.monotonic()-started,
                'full_population_pass':False,'edited_patch_updates':192,'control_patch_updates':32}
        history.append(record)
        write_json(out/'history.json',{'epochs':history,'held_out_scored_during_training':False,
            'epoch_field_means_reporting_block':True,'population_sampling':schedule.receipt()})
        print(f"{stage['outer_fold_id']} block {block}/64 loss={record['loss']:.5f} updates={record['updates']}",flush=True)
    sampling=schedule.receipt()
    if sampling['edited_updates']!=12288 or sampling['control_updates']!=2048:
        raise ValueError('Edited/control exposure no longer matches the reference')
    if (sampling['minimum_edited_visits']!=stage['training_schedule']['minimum_edited_visits']
            or sampling['maximum_edited_visits']!=stage['training_schedule']['maximum_edited_visits']):
        raise ValueError('Expanded population coverage differs from the declared schedule')
    model.save(out/'change-model.keras')
    sample=train[0]
    with FocusedChangeVolume(sample,lock['runtime'],proposals[sample['sample_id']]['origins_kji']) as reader:
        data=reader.read_patch(0,[0,0,0],t['patch_shape_kji'])
        x=tf.convert_to_tensor(data['x'][None],tf.float32)
        expected=np.asarray(steps.predict(x))
        restored=network.load_model(out/'change-model.keras')
        if not np.allclose(expected,np.asarray(restored(x,training=False)),rtol=1e-6,atol=1e-6):
            raise ValueError('Population model reload parity failed')
    del restored,expected,reader,data,x
    (out/'predictions').mkdir()
    records=[reference_learning.evaluate_sample(tf,steps,row,proposals[row['sample_id']],lock['runtime'],c,out/'predictions')
             for row in evaluate]
    for sid,(variant_stat,mask_stat) in verified.items():
        for key,expected in [('variant_npz',variant_stat),('mask_npz',mask_stat)]:
            stat=Path(by_id[sid][key]).stat()
            if (stat.st_size,stat.st_mtime_ns)!=expected:
                raise ValueError('Training payload changed before final publication')
    result={'status':'completed','stage_id':stage['id'],'config':c,'records':records,
        'gate':reference_learning.validation_gate(c,records),'reporting_blocks_completed':64,
        'epochs_completed':64,'epoch_field_means_reporting_block':True,
        'optimizer_updates':int(steps.optimizer.iterations.numpy()),'training_schedule':sampling,
        'training_samples':len(train),'evaluation_samples':len(evaluate),
        'training_case_ids':stage['train_case_ids'],'held_out_case_ids':stage['final_test_case_ids'],
        'reserved_test_evaluated':False,'classifier_started':False,
        'source_lock_sha256':sha256(Path(c['experiment']['output_directory'])/'experiment-lock.json'),
        'checkpoint':str(out/'change-model.keras'),'checkpoint_sha256':sha256(out/'change-model.keras'),
        'checkpoint_selection':'declared final14336updates only',
        'scope':'Expanded morphology diversity; same CT-only network and fixed original held-out panel',
        'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'gpu_peak_mib':tf.config.experimental.get_memory_info('GPU:0')['peak']/1024**2}
    write_json(out/'results.json',result)
    (out/'INCOMPLETE').unlink()
    return result
