#!/usr/bin/env python3
"""Prepare, train and test the two-model synthetic aorta engineering pilot."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_two_stage_experiment import load_config, prepare, plan, freeze, inspect_lock, fit, evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', default='plan', choices=(
        'prepare', 'plan', 'fit-segmentation', 'fit-reconstruction', 'test', 'all', 'report'))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        path = Path(args.config).expanduser().resolve()
        raw = path.read_bytes()
        config = load_config(path)
        if args.validate_only:
            result = plan(config)
        elif args.stage == 'prepare':
            result = prepare(config)
        elif args.stage == 'plan':
            result = freeze(config, raw)
        elif args.stage == 'fit-segmentation':
            result = fit(config, raw, 1)
        elif args.stage == 'fit-reconstruction':
            result = fit(config, raw, 2)
        elif args.stage == 'test':
            result = evaluate(config, raw)
        elif args.stage == 'report':
            from fakect_two_stage_report import render_report
            inspect_lock(config, raw)
            result = render_report(config['experiment']['output_directory'])
        else:
            prepare(config)
            directory = Path(config['experiment']['output_directory'])
            if (directory/'experiment-lock.json').exists():
                inspect_lock(config, raw)
            else:
                freeze(config, raw)
            fit(config, raw, 1)
            fit(config, raw, 2)
            result = evaluate(config, raw)
            from fakect_two_stage_report import render_report
            result['report'] = render_report(directory)
        if path.read_bytes() != raw:
            raise ValueError('INI changed during execution')
        # Manifests/locks remain on disk; avoid dumping hundreds of sample rows.
        visible = {key: value for key, value in result.items() if key not in ('samples', 'dataset_metadata')}
        print(json.dumps(visible, indent=2, allow_nan=False))
        return 0 if result.get('ready', True) else 2
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
