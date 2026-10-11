#!/usr/bin/env python3
"""Freeze and validate focused CT-only enlargement/narrowing localization."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import fakect_change_experiment as exp
from fakect_multiscale_experiment import write_json


def report(c,lock):
    from fakect_change_report import write_report
    out = Path(c['experiment']['output_directory'])
    links, records, gates = [], [], []
    for stage in lock['stages']:
        folder = out/'stages'/stage['id']
        status = 'pending'
        if (folder/'failure.json').exists():
            status = 'failed'
        elif (folder/'results.json').exists() and not (folder/'INCOMPLETE').exists():
            result = exp.base.checked_json(folder/'results.json')
            if result['source_lock_sha256'] != exp.base.sha256(out/'experiment-lock.json'):
                raise ValueError('Result belongs to another source lock')
            for row in result['records']:
                if exp.base.sha256(folder/row['prediction_npz']) != row['prediction_sha256']:
                    raise ValueError('Saved prediction changed')
            write_report(folder)
            status = 'completed'
            records += result['records']
            gates.append(result['gate']['passed'])
        link = str((folder/'report.html').relative_to(out))
        links.append((stage['outer_fold_id'],status,link))
    import html
    complete = all(status=='completed' for _,status,_ in links)
    document = {'status':'completed' if complete else 'pending_or_failed',
                'folds':[{'family':f,'status':s,'report':p} for f,s,p in links],
                'all_localization_gates_passed':bool(complete and all(gates)),
                'classifier_started':False,'reserved_test_evaluated':False}
    write_json(out/'evaluation-status.json',document)
    lines = ['<!doctype html><meta charset="utf-8"><title>Focused aorta change localization</title>',
             '<h1>Focused CT → enlargement / narrowing maps</h1>',
             '<p>Baseline masks are supervision only. Each fold uses a fresh U-Net and frozen image-predicted crops. ',
             'Unchanged controls use paired fixed crops. Classifier training has not started.</p>',
             '<p>Development family holdout, not clinical validation; reserved infants remain unopened.</p>']
    for family,status,path in links:
        lines.append('<p>'+html.escape(family)+': '+html.escape(status)+
            (' · <a href="'+html.escape(path)+'">metrics and interactive 3D</a>' if status=='completed' else '')+'</p>')
    lines.append('<p>All declared localization gates passed: '+str(document['all_localization_gates_passed'])+'</p>')
    (out/'report.html').write_text('\n'.join(lines))
    return {**document,'report':str(out/'report.html')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--stage',choices=('plan','list','smoke','fit','report'),default='plan')
    p.add_argument('--run-id')
    p.add_argument('--ids-only',action='store_true')
    args = p.parse_args()
    started = None
    try:
        path = Path(args.config)
        c,content = exp.load_config(path),path.read_bytes()
        out = Path(c['experiment']['output_directory'])
        if args.validate_only:
            lock = exp.plan(c)
            result = {'ready':True,'folds':len(lock['stages']),'samples':len(lock['samples']),
                      'reserved_case_ids':lock['reserved_case_ids'],'classifier_started':False}
        elif args.stage=='plan':
            lock = exp.freeze(c,content)
            result = report(c,lock)
        else:
            lock = exp.inspect_lock(c,content)
            if args.stage=='list':
                if args.ids_only:
                    print('\n'.join(s['id'] for s in lock['stages']))
                    return 0
                result = {'stages':[s['id'] for s in lock['stages']]}
            elif args.stage=='report':
                result = report(c,lock)
            elif args.stage=='smoke':
                from fakect_change_learning import smoke
                started = out/'smoke'
                if started.exists():
                    started=None
                    raise FileExistsError('Smoke already started; preserve and use a fresh revision')
                started.mkdir()
                result = smoke(c,lock)
                exp.inspect_lock(c,content)
                exp.require_smoke(c,lock)
            else:
                exp.require_smoke(c,lock)
                stages = [s for s in lock['stages'] if s['id']==args.run_id]
                if len(stages)!=1:
                    raise ValueError('Use a --run-id from --stage list')
                started = out/'stages'/stages[0]['id']
                if started.exists():
                    started=None
                    raise FileExistsError('Fold already started; preserve and use a fresh revision')
                from fakect_change_learning import run_stage
                result = run_stage(c,lock,stages[0])
                exp.inspect_lock(c,content)
                result = {k:v for k,v in result.items() if k!='records'}
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except Exception as error:
        if started is not None:
            started.mkdir(parents=True,exist_ok=True)
            write_json(started/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        print('error: '+str(error),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
