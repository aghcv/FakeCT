#!/usr/bin/env python3
"""Validate, freeze, train or report a CoA classifier using a reviewed locator."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import fakect_coa_classifier_experiment as exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('plan','fit','report'),default='plan')
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    c,authorized_fit = None,False
    try:
        path = Path(args.config); content = path.read_bytes()
        c = exp.load_config(path)
        if args.validate_only:
            result = exp.plan(c)
        elif args.stage=='plan':
            result = exp.freeze(c,content)
            from fakect_coa_classifier_report import write_report
            write_report(c['experiment']['output_directory'])
        else:
            lock = exp.inspect_lock(c,content)
            if args.stage=='report':
                from fakect_coa_classifier_report import write_report
                result = {'report':str(write_report(c['experiment']['output_directory']))}
            else:
                # Protect an existing run's results and failure records from
                # accidental repeated submissions or changed configurations.
                out = Path(c['experiment']['output_directory'])
                if any((out/name).exists() for name in ('model','results.json','failure.json')):
                    raise ValueError('Classifier run already started; select a new revision')
                authorized_fit = True
                from fakect_coa_classifier_learning import fit
                result = fit(c,lock)
        omitted = {'samples','proposals','config','parent_config','design','code_sha256',
                   'dependency_sha256','predictions','reserved_test_sample_ids'}
        print(json.dumps({k:v for k,v in result.items() if k not in omitted},indent=2))
        return 0
    except Exception as error:
        if authorized_fit:
            out = Path(c['experiment']['output_directory'])
            exp.write_json(out/'failure.json',{'type':type(error).__name__,'message':str(error),
                'traceback':traceback.format_exc(),'classifier_only':True})
            try:
                from fakect_coa_classifier_report import write_report
                write_report(out)
            except Exception:
                pass  # Preserve the training exception for the scheduler log.
        print('error: '+str(error),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
