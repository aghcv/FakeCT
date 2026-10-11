"""Small fixed TRAIN-patch learning check with present-class/burden balancing."""
from pathlib import Path
import json
import resource
import time

import numpy as np

import fakect_change_network as network
import fakect_change_learning as previous
from fakect_change_data import FocusedChangeVolume
from fakect_change_sampling import strict_training_patch
from fakect_change_report import change_metrics
from fakect_multiscale_experiment import sha256,write_json,checked_json


def _masks(reader,tile,block):
    origin=reader.origins_kji[tile['proposal_index']]+tile['offset_kji']
    scale=reader.spacing_ijk_mm[::-1]/reader.volume._native_spacing_kji
    coords=[(int(o)+np.arange(int(n),dtype=np.float64)+.5)*s-.5
            for o,n,s in zip(origin,tile['shape_kji'],scale)]
    after=reader.volume._nearest_mask([np.floor(c+.5).astype(np.int64) for c in coords])*block['valid']
    before=((after.astype(bool)&(block['labels']!=1))|(block['labels']==2)).astype(np.uint8)
    return before,after


def prepare_patches(c,lock):
    """Freeze supervised fixed patch positions only within the declared TRAIN panel."""
    out=Path(c['experiment']['output_directory']);manifest=out/'patch-manifest.json'
    if manifest.exists():
        doc=checked_json(manifest)
        if doc['source_lock_sha256']!=sha256(out/'experiment-lock.json'):
            raise ValueError('Fixed patch cache belongs to another source lock')
        for row in doc['patches']:
            if sha256(out/row['path'])!=row['sha256']:
                raise ValueError('Cached diagnostic patch changed')
        return doc
    folder=out/'fixed-patches';folder.mkdir(exist_ok=False)
    stage=lock['stages'][0];rows={r['sample_id']:r for r in lock['samples']}
    proposals=lock['proposals'][stage['id']]
    t=c['training'];rng=np.random.default_rng(t['seed'])
    entries=[dict(s,control=False) for s in stage['diagnostic_strata']]
    for sid in stage['train_sample_ids']:
        if rows[sid]['baseline']:
            entries.extend({'sample_id':sid,'case_id':rows[sid]['case_id'],'mode':'uniform',
                            'burden_bin':None,'control':True,'proposal_index':index}
                           for index in range(lock['runtime']['input.detail']['proposals']))
    patches=[]
    for index,entry in enumerate(entries):
        row=rows[entry['sample_id']]
        if row['case_id'] not in stage['train_case_ids']:
            raise ValueError('Diagnostic cache cannot read held-out anatomy')
        with FocusedChangeVolume(row,lock['runtime'],proposals[row['sample_id']]['origins_kji']) as reader:
            if entry['control']:
                offset=(reader.shape_kji-np.asarray(t['patch_shape_kji']))//2
                tile={'proposal_index':entry['proposal_index'],'offset_kji':tuple(map(int,offset)),
                      'shape_kji':tuple(t['patch_shape_kji'])}
                block=reader.read_patch(**tile)
                audit={'control':True,'target_guided':False,'centering_policy':'center of each frozen image-predicted TRAIN proposal'}
                if np.any(block['labels']):
                    raise ValueError('Original control contains net change targets')
            else:
                chosen=strict_training_patch(reader,t['patch_shape_kji'],rng,entry['mode'],
                                             changed_fraction=1.,centered=True)
                tile,block,audit=chosen['tile'],chosen['data'],chosen['audit']
            before,after=_masks(reader,tile,block)
            identity=row['sample_id']+(':diagnostic-control-'+str(entry['proposal_index']) if entry['control']
                                      else ':diagnostic-'+entry['mode']+'-'+str(entry['burden_bin']))
            name=f'fixed-patches/{index:04d}.npz'
            with (out/name).open('xb') as stream:
                np.savez_compressed(stream,x=block['x'],labels=block['labels'],valid=block['valid'],
                                    before=before,after=after,spacing_ijk_mm=reader.spacing_ijk_mm,
                                    origin_kji=block['metadata']['origin_kji'],sample_id=np.asarray(identity))
            patches.append({'sample_id':identity,'base_sample_id':row['sample_id'],'case_id':row['case_id'],
                'anatomy_family':row['anatomy_family'],'baseline':row['baseline'],'mode':entry['mode'],
                'burden_bin':entry['burden_bin'],'tile':tile,'sampling_audit':audit,'path':name,
                'sha256':sha256(out/name),'original_aorta_voxels':int(before.sum()),
                'target_added_voxels':int((block['labels']==1).sum()),
                'target_removed_voxels':int((block['labels']==2).sum())})
    doc={'patches':patches,'source_lock_sha256':sha256(out/'experiment-lock.json'),
         'train_only':True,'held_out_evaluated':False,'edited_requested_positive_fraction':1.,
         'edited_cached_patches':12,'healthy_cached_patches':4,
         'scope':'Fixed supervised change-centred TRAIN memorization exercise; not image-only inference-crop selection'}
    write_json(manifest,doc)
    return doc


def _load(c,doc):
    out=Path(c['experiment']['output_directory']);arrays=[];total=0
    for row in doc['patches']:
        if sha256(out/row['path'])!=row['sha256']:
            raise ValueError('Diagnostic patch bytes changed')
        with np.load(out/row['path'],allow_pickle=False) as data:
            item={k:np.asarray(data[k]) for k in ('x','labels','valid','before','after','spacing_ijk_mm','origin_kji')}
        total+=sum(a.nbytes for a in item.values())
        if total>1024**3:
            raise ValueError('Fixed diagnostic cache exceeds one GiB')
        arrays.append(item)
    return arrays


def _evaluate(tf,steps,c,doc,arrays,phase):
    out=Path(c['experiment']['output_directory']);folder=out/'diagnostic-predictions'/phase
    folder.mkdir(parents=True,exist_ok=False);records=[]
    for index,(row,data) in enumerate(zip(doc['patches'],arrays)):
        logits=steps.predict(tf.convert_to_tensor(data['x'][None],tf.float32))
        predicted=np.asarray(tf.argmax(logits,axis=-1))[0].astype(np.uint8)*data['valid']
        metrics=change_metrics(data['labels'],predicted,data['valid'],data['spacing_ijk_mm'])
        name=f'diagnostic-predictions/{phase}/{index:04d}.npz'
        with (out/name).open('xb') as stream:
            np.savez_compressed(stream,before=data['before'],after=data['after'],labels=data['labels'],
                predicted=predicted,valid=data['valid'],spacing_ijk_mm=data['spacing_ijk_mm'],
                origin_kji=data['origin_kji'],sample_id=np.asarray(row['sample_id']))
        records.append({**row,'metrics':metrics,'prediction_npz':name,'prediction_sha256':sha256(out/name),
                        'phase':phase,'training_only':True})
    return records


def diagnostic_gate(c,patches,whole):
    thresholds=c['validation'];checks={};scores={}
    edited=[r for r in patches if not r['baseline']]
    for name in ('added','removed'):
        per_case={}
        for row in edited:
            m=row['metrics']['classes'][name]
            if not m['target_empty']:
                per_case.setdefault(row['case_id'],[]).append(m)
        scores[name]={}
        for metric in ('dice','precision','recall'):
            values=[np.mean([m[metric] if m[metric] is not None else 0. for m in rows]) for rows in per_case.values()]
            scores[name][metric]=float(np.mean(values)) if values else None
            checks[name+'_'+metric]=bool(values) and scores[name][metric]>=thresholds['minimum_'+metric]
        checks[name+'_both_cases_detected']=len(per_case)==len(c['diagnostic']['case_ids']) and all(
            sum(m['tp'] for m in rows)>0 for rows in per_case.values())
    controls=[r for r in whole if r['baseline']]
    checks['full_focus_controls']=len(controls)==len(c['diagnostic']['case_ids']) and all(
        r['original_aorta_voxels']>0 and r['control_change_fraction']<=thresholds['maximum_control_change_fraction']
        for r in controls)
    return {'passed':bool(all(checks.values())),'checks':checks,'case_macro_patch_metrics':scores,
            'thresholds':thresholds,'scope':'TRAIN-only fixed-patch learning gate plus whole-focus original-control check',
            'held_out_evaluated':False,'classifier_started':False,'full_population_training_started':False}


def smoke(c,lock):
    tf=previous._tensorflow();out=Path(c['experiment']['output_directory'])
    doc=prepare_patches(c,lock);arrays=_load(c,doc)
    model=network.build_model(tuple(c['training']['patch_shape_kji']),c['training']['base_filters'],c['training']['seed'])
    steps=previous._steps(c,model)
    indices=[next(i for i,r in enumerate(doc['patches']) if r['mode']==mode) for mode in ('added','removed')]
    indices.append(next(i for i,r in enumerate(doc['patches']) if r['baseline']))
    metrics=[]
    for index in indices:
        x,y,v=previous._tensors(tf,arrays[index]);value=steps.train(x,y,v);steps.predict(x)
        metrics.append({k:float(np.asarray(val)) for k,val in value.items()})
    model.save(out/'smoke/model.keras');restored=network.load_model(out/'smoke/model.keras')
    parity=bool(np.allclose(np.asarray(steps.predict(x)),np.asarray(restored(x,training=False)),rtol=1e-6,atol=1e-6))
    result={'passed':bool(parity and all(m['gradient_global_norm']>0 for m in metrics)),
            'gpu_visible':True,'reload_parity':parity,'train_traces':steps.train.experimental_get_tracing_count(),
            'predict_traces':steps.predict.experimental_get_tracing_count(),
            'source_lock_sha256':sha256(out/'experiment-lock.json'),'patch_count':len(arrays),
            'patch_manifest_sha256':sha256(out/'patch-manifest.json'),
            'gpu_peak_mib':tf.config.experimental.get_memory_info('GPU:0')['peak']/1024**2,
            'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'tensorflow_version':tf.__version__,'metrics':metrics,'held_out_evaluated':False}
    if result['gpu_peak_mib']>12288 or result['host_peak_rss_mib']>16384:result['passed']=False
    write_json(out/'runtime-smoke.json',result)
    return result


def run_stage(c,lock,stage):
    tf=previous._tensorflow();out=Path(c['experiment']['output_directory'])
    smoke_receipt=checked_json(out/'runtime-smoke.json')
    if sha256(out/'patch-manifest.json')!=smoke_receipt.get('patch_manifest_sha256'):
        raise ValueError('Fixed patch manifest changed after the verified smoke')
    folder=out/'stages'/stage['id'];folder.mkdir(parents=True,exist_ok=False);(folder/'INCOMPLETE').touch()
    doc=prepare_patches(c,lock);arrays=_load(c,doc);t=c['training'];rng=np.random.default_rng(t['seed'])
    model=network.build_model(tuple(t['patch_shape_kji']),t['base_filters'],t['seed']);steps=previous._steps(c,model)
    initial=_evaluate(tf,steps,c,doc,arrays,'initial');history=[]
    edited=[i for i,r in enumerate(doc['patches']) if not r['baseline']]
    controls=[i for i,r in enumerate(doc['patches']) if r['baseline']]
    order_base=edited*2+controls
    if len(order_base)!=t['steps_per_epoch'] or len(controls)!=4:
        raise ValueError('Fixed pilot update/control accounting changed')
    for epoch in range(1,t['epochs']+1):
        started=time.monotonic();values=[]
        for index in rng.permutation(order_base):
            x,y,v=previous._tensors(tf,arrays[int(index)])
            value=steps.train(x,y,v);values.append({k:float(np.asarray(val)) for k,val in value.items()})
        steps.assert_stable_traces()
        if int(steps.optimizer.iterations.numpy())!=epoch*t['steps_per_epoch']:
            raise ValueError('Diagnostic optimizer count changed')
        record={'epoch':epoch,'loss':float(np.mean([m['loss'] for m in values])),
                'ce':float(np.mean([m['cross_entropy'] for m in values])),
                'dice_loss':float(np.mean([m['foreground_dice_loss'] for m in values])),
                'updates':int(steps.optimizer.iterations.numpy()),'seconds':time.monotonic()-started}
        history.append(record);write_json(out/'history.json',{'epochs':history,'training_only':True})
        print(f"Sampling diagnostic epoch {epoch}/{t['epochs']} loss={record['loss']:.4f}",flush=True)
    model.save(folder/'change-model.keras');restored=network.load_model(folder/'change-model.keras')
    if not np.allclose(np.asarray(steps.predict(x)),np.asarray(restored(x,training=False)),rtol=1e-6,atol=1e-6):
        raise ValueError('Pilot reload parity failed')
    del restored
    final=_evaluate(tf,steps,c,doc,arrays,'final')
    rows={r['sample_id']:r for r in lock['samples']};proposals=lock['proposals'][stage['id']]
    (folder/'predictions').mkdir()
    whole=[]
    for sid in stage['train_sample_ids']:
        row={**rows[sid],'split':'train'}
        result=previous.evaluate_sample(tf,steps,row,proposals[sid],lock['runtime'],c,folder/'predictions')
        result['prediction_npz']=str((folder/result['prediction_npz']).relative_to(out))
        result['training_only']=True;whole.append(result)
    gate=diagnostic_gate(c,final,whole)
    result={'status':'completed','stage_id':stage['id'],'config':c,'initial_patch_records':initial,
            'final_patch_records':final,'whole_focus_training_records':whole,'gate':gate,
            'optimizer_updates':int(steps.optimizer.iterations.numpy()),'epochs_completed':t['epochs'],
            'source_lock_sha256':sha256(out/'experiment-lock.json'),'checkpoint':str(folder/'change-model.keras'),
            'checkpoint_sha256':sha256(folder/'change-model.keras'),'training_case_ids':stage['train_case_ids'],
            'held_out_evaluated':False,'reserved_test_evaluated':False,'classifier_started':False,
            'full_population_training_started':False,'edited_fixed_patch_sampling_fraction':1.,
            'edited_updates_per_epoch':24,'healthy_updates_per_epoch':4,
            'scope':'Same-cache TRAIN memorization check; whole-focus TRAIN scores separately descriptive; no generalization claim'}
    write_json(folder/'results.json',result);(folder/'INCOMPLETE').unlink();write_json(out/'results.json',result)
    return result
