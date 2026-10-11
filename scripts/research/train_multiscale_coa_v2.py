#!/usr/bin/env python3
"""Plan, prepare and run the bounded CoA multiresolution engineering pilot."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import fakect_multiscale_experiment_v2 as exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('plan','prepare','fit','all','report','smoke'), default='plan')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    c = None
    try:
        path = Path(args.config)
        content = path.read_bytes()
        c = exp.load_config(path)
        if args.validate_only:
            result = exp.plan(c)
        elif args.stage == 'plan':
            result = exp.freeze(c, content)
            exp.report(c)
        else:
            lock = exp.inspect_lock(c, content)
            if args.stage == 'report':
                result = {'report':exp.report(c)}
            else:
                import fakect_multiscale_learning_v2 as learning
                if args.stage == 'smoke':
                    result = learning.runtime_smoke(c)
                if args.stage in ('prepare', 'all'):
                    result = learning.prepare(c, lock)
                if args.stage in ('fit', 'all'):
                    result = learning.fit(c, lock)
        print(json.dumps({k:v for k,v in result.items() if k not in ('samples','config','design','predictions','code_sha256')}, indent=2))
        return 0
    except Exception as error:
        if c and args.stage in ('prepare','fit','all','smoke') and not args.validate_only:
            out = Path(c['experiment']['output_directory'])
            if out.is_dir():
                exp.write_json(out/'failure.json', {'type':type(error).__name__, 'message':str(error),
                                                   'traceback':traceback.format_exc()})
                if (out/'experiment-lock.json').exists():
                    exp.report(c)
        print('error: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
