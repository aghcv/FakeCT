#!/usr/bin/env python3
"""Train/test reference reconstruction from supplied masks, without Model 1."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_mask_model import load_config, plan, freeze, inspect_lock, fit, evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('plan', 'fit', 'test', 'report', 'all'), default='plan')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        path = Path(args.config).expanduser().resolve()
        raw = path.read_bytes()
        config = load_config(path)
        if args.validate_only:
            result = plan(config)
        elif args.stage == 'plan':
            result = freeze(config, raw)
        elif args.stage == 'fit':
            result = fit(config, raw)
        elif args.stage == 'test':
            result = evaluate(config, raw)
        elif args.stage == 'report':
            from fakect_two_stage_report import render_report
            inspect_lock(config, raw)
            result = render_report(config['experiment']['output_directory'])
        else:
            directory = Path(config['experiment']['output_directory'])
            if (directory/'experiment-lock.json').exists():
                inspect_lock(config, raw)
            else:
                freeze(config, raw)
            fit(config, raw)
            result = evaluate(config, raw)
            from fakect_two_stage_report import render_report
            result['report'] = render_report(directory)
        if path.read_bytes() != raw:
            raise ValueError('INI changed during execution')
        visible = {key: value for key, value in result.items() if key != 'samples'}
        print(json.dumps(visible, indent=2, allow_nan=False))
        return 0 if result.get('ready', True) else 2
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
