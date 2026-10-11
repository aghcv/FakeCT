#!/usr/bin/env python3
"""Plan and run matched baseline-only and synthetic-sibling segmentation arms."""
import argparse
import configparser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_context_comparison import ARMS, load_config, plan, freeze, inspect_lock, fit, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', choices=('plan', 'fit', 'test', 'all'), default='plan')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        raw = args.config.read_bytes()
        config = load_config(args.config)
        if args.validate_only:
            result = plan(config)
        elif args.stage == 'plan':
            result = freeze(config, raw)
        else:
            if args.arm is None:
                raise ValueError('--arm is required for fit, test and all')
            # Freeze the common plan once, before launching independent arm jobs.
            inspect_lock(config, raw)
            if args.stage in ('fit', 'all'):
                result = fit(config, raw, args.arm)
            if args.stage in ('test', 'all'):
                result = evaluate(config, raw, args.arm)
        if args.config.read_bytes() != raw:
            raise ValueError('INI changed during execution')
        print(json.dumps({key: value for key, value in result.items()
            if key not in ('samples', 'records')}, indent=2, allow_nan=False))
        return 0 if result.get('ready', True) else 2
    except (OSError, ValueError, KeyError, RuntimeError, configparser.Error) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
