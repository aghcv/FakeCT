#!/usr/bin/env python3
"""Read-only trained-checkpoint recovery; no optimizer updates or source rewrites."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
import fakect_change_sampling_experiment as exp
import fakect_change_sampling_learning as learning
import fakect_change_network as network
from fakect_multiscale_experiment import sha256,write_json,checked_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    config_path=Path(args.config);c=exp.load_config(config_path);lock=exp.inspect_lock(c,config_path.read_bytes())
    original=Path(c['experiment']['output_directory']);stage=lock['stages'][0]
    checkpoint=original/'stages'/stage['id']/'change-model.keras'
    model_sha=sha256(checkpoint);history=checked_json(original/'history.json')
    t=c['training']
    if len(history['epochs'])!=t['epochs'] or history['epochs'][-1]['updates']!=t['epochs']*t['steps_per_epoch']:
        raise ValueError('Recovery requires completed fixed-budget training history')
    smoke=checked_json(original/'runtime-smoke.json')
    if sha256(original/'patch-manifest.json')!=smoke['patch_manifest_sha256']:
        raise ValueError('Original fixed cache changed after smoke')
    doc=learning.prepare_patches(c,lock);arrays=learning._load(c,doc)
    out=original/'recovery-r1';out.mkdir(exist_ok=False)
    (out/'INCOMPLETE').touch()
    # The exact source lock is copied for reporter identity checks; inputs stay
    # at their original paths and are referenced by hashes below.
    (out/'experiment-lock.json').write_bytes((original/'experiment-lock.json').read_bytes())
    write_json(out/'history.json',history)
    rc=deepcopy(c);rc['experiment']['output_directory']=str(out)
    tf=learning.previous._tensorflow()
    model=network.load_model(checkpoint);steps=learning.previous._steps(c,model)
    with tempfile.TemporaryDirectory(prefix='fakect-sampling-reload-',dir='/tmp') as temp:
        saved=Path(temp)/'model.keras';model.save(saved)
        other=network.load_model(saved)
        if not all(np.array_equal(a,b) for a,b in zip(model.get_weights(),other.get_weights())):
            raise ValueError('Read-only recovery changed saved weights')
        other_steps=learning.previous._steps(c,other)
        reports=[]
        for i,data in enumerate(arrays):
            x=tf.convert_to_tensor(data['x'][None],tf.float32)
            a=np.asarray(steps.predict(x));b=np.asarray(other_steps.predict(x));e=np.asarray(other(x,training=False))
            pa=np.asarray(tf.nn.softmax(a,axis=-1));pb=np.asarray(tf.nn.softmax(b,axis=-1));pe=np.asarray(tf.nn.softmax(e,axis=-1))
            record={'patch':doc['patches'][i]['sample_id'],
                'graph_roundtrip_max_logit_error':float(np.max(np.abs(a-b))),
                'graph_roundtrip_max_probability_error':float(np.max(np.abs(pa-pb))),
                'graph_roundtrip_changed_classes':int(np.count_nonzero(np.argmax(a,-1)!=np.argmax(b,-1))),
                'graph_eager_max_logit_error':float(np.max(np.abs(a-e))),
                'graph_eager_max_probability_error':float(np.max(np.abs(pa-pe))),
                'graph_eager_changed_classes':int(np.count_nonzero(np.argmax(a,-1)!=np.argmax(e,-1)))}
            if not np.allclose(a,b,rtol=1e-6,atol=1e-6) or record['graph_roundtrip_changed_classes']:
                raise ValueError('Equivalent graph inference differs after exact weight roundtrip')
            reports.append(record)
        write_json(out/'reload-audit.json',{'weight_arrays_bitwise_equal':True,'patches':reports,
            'original_pre_save_gpu_variables_available':False,
            'interpretation':'Equivalent graph roundtrip checked; eager-path differences measured separately, not silently ignored.'})
    final=learning._evaluate(tf,steps,rc,doc,arrays,'final')
    initial_model=network.build_model(tuple(t['patch_shape_kji']),t['base_filters'],t['seed'])
    initial_steps=learning.previous._steps(c,initial_model)
    initial=learning._evaluate(tf,initial_steps,rc,doc,arrays,'initial')
    # Verify reproduced initial decisions against the failed run's saved initial
    # arrays, avoiding a silently changed initial comparison.
    for index,row in enumerate(initial):
        old=original/'diagnostic-predictions/initial'/f'{index:04d}.npz'
        with np.load(old,allow_pickle=False) as a,np.load(out/row['prediction_npz'],allow_pickle=False) as b:
            if not all(np.array_equal(a[k],b[k]) for k in ('predicted','labels','valid','before','after')):
                raise ValueError('Recovered initial comparison differs from saved original arrays')
    del initial_model,initial_steps
    rows={r['sample_id']:r for r in lock['samples']};proposals=lock['proposals'][stage['id']]
    (out/'predictions').mkdir();whole=[]
    for sid in stage['train_sample_ids']:
        row={**rows[sid],'split':'train'}
        value=learning.previous.evaluate_sample(tf,steps,row,proposals[sid],lock['runtime'],rc,out/'predictions')
        value['training_only']=True;whole.append(value)
    gate=learning.diagnostic_gate(c,final,whole)
    result={'status':'completed_recovery','config':rc,'initial_patch_records':initial,'final_patch_records':final,
        'whole_focus_training_records':whole,'gate':gate,'optimizer_updates':history['epochs'][-1]['updates'],
        'epochs_completed':t['epochs'],'source_lock_sha256':sha256(original/'experiment-lock.json'),
        'checkpoint':str(checkpoint),'checkpoint_sha256':model_sha,'training_case_ids':stage['train_case_ids'],
        'held_out_evaluated':False,'reserved_test_evaluated':False,'classifier_started':False,
        'full_population_training_started':False,'weights_updated_during_recovery':False,
        'scope':'TRAIN-only derived recovery of saved final checkpoint; original failed export retained',
        'recovery_source_sha256':sha256(Path(__file__))}
    if sha256(checkpoint)!=model_sha:raise ValueError('Original trained archive changed during recovery')
    write_json(out/'results.json',result)
    (out/'INCOMPLETE').unlink()
    from fakect_change_sampling_report import write_report
    print(json.dumps(write_report(rc,lock),indent=2))


if __name__=='__main__':main()
