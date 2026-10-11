#!/usr/bin/env python3
"""Plan, fit isolated stages and report rotating focused CoA evaluation."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import fakect_coa_rotation_experiment as exp
from fakect_multiscale_experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--stage',choices=('plan','smoke','fit','report','list'),default='plan')
    parser.add_argument('--run-id',help='Exact stage ID from --stage list; fitting never changes the plan')
    parser.add_argument('--ids-only',action='store_true',help='Print stage IDs one per line for Slurm orchestration')
    parser.add_argument('--group',choices=('primary_inner_remaining','outer_refit','secondary_eligible'),
                        help='Frozen eligible stage group for dependency-controlled Slurm arrays')
    args = parser.parse_args()
    c,started = None,None
    try:
        path = Path(args.config)
        c = exp.load_config(path);content = path.read_bytes()
        out = Path(c['experiment']['output_directory'])
        if args.validate_only:
            lock = exp.plan(c)
            result = {'ready':True,'stages':len(lock['stages']),
                      'eligible_stages':sum(s['execution_eligible'] for s in lock['stages']),
                      'reserved_case_ids':lock['reserved_case_ids'],
                      'selected_samples':len(lock['samples']), 'fresh_locator_required':True}
        elif args.stage == 'plan':
            lock = exp.freeze(c,content)
            result = {'ready':True,'stages':len(lock['stages']),'report':exp.report(c,lock)}
        else:
            lock = exp.inspect_lock(c,content)
            if args.stage == 'list':
                stages = lock['stages']
                if args.group:
                    stages = [s for s in stages if s['execution_eligible']]
                    if args.group == 'primary_inner_remaining':
                        stages = [s for s in stages if s['stage']=='inner_selection'][1:]
                    elif args.group == 'outer_refit':
                        stages = [s for s in stages if s['stage']=='outer_refit']
                    else:
                        stages = [s for s in stages if s['stage']=='secondary_selection']
                if args.ids_only:
                    print('\n'.join(stage['id'] for stage in stages))
                    return 0
                result = {'stages':[{k:stage.get(k) for k in ('id','protocol','stage','counts')}
                                    for stage in stages]}
            elif args.stage == 'report':
                result = exp.report(c,lock)
            elif args.stage == 'smoke':
                if (out/'runtime-smoke.json').exists() or (out/'smoke/failure.json').exists():
                    raise ValueError('Smoke already started; use a new experiment revision')
                started = out/'smoke';started.mkdir(exist_ok=False)
                from fakect_coa_rotation_learning import smoke
                result = smoke(c,lock)
            else:
                if not args.run_id:
                    raise ValueError('--stage fit requires --run-id from --stage list')
                exp.require_smoke(c,lock)
                stage = exp.stage_by_id(lock,args.run_id)
                if not stage['execution_eligible']:
                    raise ValueError('Stage excluded by isolation checks: '+stage['exclusion_reason'])
                if (out/'stages'/stage['id']).exists():
                    raise ValueError('Stage already started; preserve it and use a new revision')
                if stage['stage']=='outer_refit':
                    exp.select_refit_budgets(lock,stage,out)
                started = out/'stages'/stage['id']
                from fakect_coa_rotation_learning import run_stage
                result = run_stage(c,lock,stage)
                # Reports are assembled by the separate report stage, avoiding
                # concurrent fold jobs overwriting a shared gallery.
                result = {k:v for k,v in result.items() if k not in ('final_predictions','locator','classifier')}
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except Exception as error:
        if started is not None:
            started.mkdir(parents=True,exist_ok=True)
            write_json(started/'failure.json', {'type':type(error).__name__, 'message':str(error),
                        'traceback':traceback.format_exc()})
        print('error: '+str(error),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
