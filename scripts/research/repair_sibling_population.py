#!/usr/bin/env python3
"""Freeze conservative repairs, repair cases, then release the full population."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import fakect_sibling_repair as repair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--stage',choices=('plan','list','repair','collect'),default='plan')
    parser.add_argument('--case-id')
    parser.add_argument('--ids-only',action='store_true')
    args=parser.parse_args()
    try:
        path=Path(args.config)
        c,content=repair.load_config(path),path.read_bytes()
        if args.validate_only:
            lock=repair.plan(c)
            result={'metadata_validated':True,'reused_siblings':lock['reused_siblings'],
                    'repair_siblings':lock['repair_siblings'],'native_repair_complete':False}
        elif args.stage=='plan':
            lock=repair.freeze(c,content)
            result={'repair_lock':str(Path(c['output_directory'])/'repair-lock.json'),
                    'case_ids':list(lock['cases']),'reused_siblings':lock['reused_siblings'],
                    'repair_siblings':lock['repair_siblings']}
        else:
            lock=repair.inspect_lock(c,content)
            if args.stage=='list':
                if args.ids_only:
                    print('\n'.join(lock['cases']))
                    return 0
                result={'case_ids':list(lock['cases'])}
            elif args.stage=='repair':
                if args.case_id is None:
                    raise ValueError('--stage repair requires --case-id')
                result=repair.repair_case(c,lock,args.case_id)
            else:
                result=repair.collect(c,lock)
            repair.inspect_lock(c,content)
        print(json.dumps({k:v for k,v in result.items() if k not in ('samples','records','repair_records')},
                         indent=2,allow_nan=False))
        return 0
    except Exception as error:
        print('error: '+str(error),file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__=='__main__':
    raise SystemExit(main())
