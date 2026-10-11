#!/usr/bin/env python3
"""Freeze and execute the TRAIN-only aorta change-sampling diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import fakect_change_sampling_experiment as exp


def report(config, lock):
    from fakect_change_sampling_report import write_report
    return write_report(config, lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stage', choices=('plan', 'list', 'smoke', 'fit', 'report'), default='plan')
    parser.add_argument('--run-id')
    parser.add_argument('--ids-only', action='store_true')
    args = parser.parse_args()
    started = None
    try:
        path = Path(args.config)
        config, content = exp.load_config(path), path.read_bytes()
        output = Path(config['experiment']['output_directory'])
        if args.validate_only:
            lock = exp.plan(config)
            stage = lock['stages'][0]
            result = {'ready': True, 'diagnostic_samples': len(lock['samples']),
                      'training_case_ids': stage['train_case_ids'],
                      'optimizer_updates': stage['training_schedule']['total_optimizer_updates'],
                      'held_out_evaluated': False, 'reserved_test_evaluated': False,
                      'classifier_started': False}
        elif args.stage == 'plan':
            lock = exp.freeze(config, content)
            result = report(config, lock)
        else:
            lock = exp.inspect_lock(config, content)
            if args.stage == 'list':
                if args.ids_only:
                    print('\n'.join(stage['id'] for stage in lock['stages']))
                    return 0
                result = {'stages': [stage['id'] for stage in lock['stages']]}
            elif args.stage == 'report':
                result = report(config, lock)
            elif args.stage == 'smoke':
                from fakect_change_sampling_learning import smoke
                started = output/'smoke'
                if started.exists():
                    started = None
                    raise FileExistsError('Smoke already started; use a fresh revision')
                started.mkdir()
                result = smoke(config, lock)
                exp.inspect_lock(config, content)
                exp.require_smoke(config, lock)
            else:
                exp.require_smoke(config, lock)
                matches = [stage for stage in lock['stages'] if stage['id'] == args.run_id]
                if len(matches) != 1:
                    raise ValueError('Use a --run-id from --stage list')
                started = output/'stages'/matches[0]['id']
                if started.exists():
                    started = None
                    raise FileExistsError('Diagnostic already started; use a fresh revision')
                from fakect_change_sampling_learning import run_stage
                result = run_stage(config, lock, matches[0])
                exp.inspect_lock(config, content)
                result = {key: value for key, value in result.items() if key != 'records'}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        if started is not None:
            started.mkdir(parents=True, exist_ok=True)
            exp.base.write_json(started/'failure.json', {'type': type(error).__name__,
                'message': str(error), 'traceback': traceback.format_exc()})
        print('error: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
