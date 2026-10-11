#!/usr/bin/env python3
"""Freeze and run a fresh classification-only 200-epoch control."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import fakect_coa_classification_experiment as exp
from fakect_multiscale_experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stage', choices=('plan', 'smoke', 'fit', 'report', 'list'), default='plan')
    parser.add_argument('--run-id')
    parser.add_argument('--ids-only', action='store_true')
    args = parser.parse_args()
    started = None
    try:
        path = Path(args.config)
        c, content = exp.load_config(path), path.read_bytes()
        out = Path(c['experiment']['output_directory'])
        if args.validate_only:
            lock = exp.plan(c)
            result = {'ready':True, 'stages':len(lock['stages']),
                      'classifier_epochs':c['training']['classifier_epochs'],
                      'reserved_case_ids':lock['reserved_case_ids'], 'locator_retraining':False, 'regression_weight':0.0, 'severity_trained':False}
        elif args.stage == 'plan':
            lock = exp.freeze(c, content)
            from fakect_coa_classification_report import write_report
            result = {'ready':True, 'stages':len(lock['stages']), 'report':write_report(out, lock)}
        else:
            lock = exp.inspect_lock(c, content)
            if args.stage == 'list':
                if args.ids_only:
                    print('\n'.join(s['id'] for s in lock['stages']))
                    return 0
                result = {'stages':[s['id'] for s in lock['stages']]}
            elif args.stage == 'report':
                from fakect_coa_classification_report import write_report
                result = write_report(out, lock)
            elif args.stage == 'smoke':
                if (out/'runtime-smoke.json').exists() or (out/'smoke').exists():
                    raise ValueError('Smoke already started; preserve it and use a new revision')
                started = out/'smoke'
                started.mkdir()
                result = exp.smoke(c, lock)
                exp.inspect_lock(c, content)
                exp.require_smoke(c, lock)
            else:
                exp.require_smoke(c, lock)
                stages = [s for s in lock['stages'] if s['id'] == args.run_id]
                if len(stages) != 1:
                    raise ValueError('--stage fit requires a frozen --run-id from --stage list')
                started = out/'stages'/stages[0]['id']
                if started.exists():
                    started = None
                    raise ValueError('Fold already started; preserve it and use a new revision')
                result = exp.run_stage(c, lock, stages[0])
                result = {k:v for k,v in result.items() if k not in ('classifier', 'final_predictions')}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        if started is not None:
            started.mkdir(parents=True, exist_ok=True)
            write_json(started/'failure.json', {'type':type(error).__name__, 'message':str(error),
                                               'traceback':traceback.format_exc()})
        print('error: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
