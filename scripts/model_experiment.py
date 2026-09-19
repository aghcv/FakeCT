#!/usr/bin/env python3
"""Validate, freeze or fit a model-only experiment over registered datasets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_model_experiment import (fit_model_experiment, freeze_model_experiment,
                                     load_model_experiment_config, plan_model_experiment)


def run(config_path, *, stage='plan', validate_only=False):
    path = Path(config_path).expanduser().resolve()
    original = path.read_bytes()
    config = load_model_experiment_config(path)
    if path.read_bytes() != original:
        raise ValueError('Model input changed while being parsed')
    if validate_only:
        result = plan_model_experiment(config)
        result.pop('samples', None)
    elif stage == 'plan':
        result = freeze_model_experiment(config, original)
    elif stage == 'fit':
        result = fit_model_experiment(config, original)
    else:
        raise ValueError('Model stage must be plan or fit')
    if path.read_bytes() != original:
        raise ValueError('Model input changed during execution; frozen outputs must be reviewed')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('plan', 'fit'), default='plan')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run(args.config, stage=args.stage, validate_only=args.validate_only)
    except (OSError, ValueError, KeyError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get('ready', True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
