#!/usr/bin/env python3
"""Plan, fit and evaluate a larger-context 3D attenuation-to-aorta model."""
import argparse
import configparser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_context_experiment import load_config, plan, freeze, inspect_lock, fit, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stage', choices=('plan','fit','test','all'), default='plan')
    args = parser.parse_args()
    try:
        raw = args.config.read_bytes()
        config = load_config(args.config)
        if args.validate_only:
            result = plan(config)
        elif args.stage == 'plan':
            result = freeze(config, raw)
        elif args.stage == 'fit':
            result = fit(config, raw)
        elif args.stage == 'test':
            result = evaluate(config, raw)
        else:
            directory = Path(config['experiment']['output_directory'])
            if (directory/'experiment-lock.json').exists():
                inspect_lock(config, raw)
            else:
                freeze(config, raw)
            fit(config, raw)
            result = evaluate(config, raw)
        if args.config.read_bytes() != raw:
            raise ValueError('INI changed during execution')
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}, indent=2, allow_nan=False))
        return 0 if result.get('ready',True) else 2
    except (OSError,ValueError,KeyError,RuntimeError,configparser.Error) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
