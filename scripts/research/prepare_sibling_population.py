#!/usr/bin/env python3
"""Freeze, trial, generate and collect independently edited aorta siblings."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import fakect_sibling_population as population
from fakect_multiscale_experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--stage',choices=('plan','list','trial','prepare','collect'),default='plan')
    parser.add_argument('--case-id')
    parser.add_argument('--trial-count',type=int,default=7)
    parser.add_argument('--ids-only',action='store_true')
    args = parser.parse_args()
    try:
        path = Path(args.config)
        c,content = population.load_config(path),path.read_bytes()
        if args.validate_only:
            lock = population.plan(c)
            result = {'metadata_validated':True,'cases':len(lock['cases']),
                      'requested_siblings':sum(len(r['recipes']) for r in lock['cases'].values()),
                      'native_execution_complete':False,'reserved_test_evaluated':False}
        elif args.stage=='plan':
            lock = population.freeze(c,content)
            result = {'population_lock':str(Path(c['population']['output_directory'])/'population-lock.json'),
                      'cases':list(lock['cases']),'classifier_started':False}
        else:
            lock = population.inspect_lock(c,content)
            if args.stage=='list':
                if args.ids_only:
                    print('\n'.join(lock['cases']))
                    return 0
                result = {'case_ids':list(lock['cases'])}
            elif args.stage=='trial':
                if args.case_id is None:
                    raise ValueError('--stage trial requires --case-id')
                result = population.trial_case(c,lock,args.case_id,args.trial_count)
            elif args.stage=='prepare':
                if args.case_id is None:
                    raise ValueError('--stage prepare requires --case-id')
                result = population.prepare_case(c,lock,args.case_id)
            else:
                result = population.collect(c,lock)
            population.inspect_lock(c,content)
        print(json.dumps({k:v for k,v in result.items() if k not in ('samples','records')},indent=2,allow_nan=False))
        return 0
    except Exception as error:
        print('error: '+str(error),file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__=='__main__':
    raise SystemExit(main())
