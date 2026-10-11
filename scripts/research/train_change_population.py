#!/usr/bin/env python3
"""Expand TRAIN sibling diversity; preserve focused-change holdouts and budget."""
import argparse
import html
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import fakect_change_population_experiment as exp
from fakect_multiscale_experiment import write_json


def report(c, lock):
    from fakect_change_report import write_report
    out = Path(c['experiment']['output_directory'])
    folds, gates = [], []
    for stage in lock['stages']:
        folder = out/'stages'/stage['id']
        status = 'pending'
        if (folder/'failure.json').exists():
            status = 'failed'
        elif (folder/'results.json').exists() and not (folder/'INCOMPLETE').exists():
            result = exp.base.checked_json(folder/'results.json')
            if result['source_lock_sha256'] != exp.base.sha256(out/'experiment-lock.json'):
                raise ValueError('Result belongs to another source lock')
            expected = set(stage['final_test_sample_ids'])
            if ({row['sample_id'] for row in result['records']} != expected
                    or len(result['records']) != len(expected)):
                raise ValueError('Expanded experiment changed the fixed held-out panel')
            for row in result['records']:
                if exp.base.sha256(folder/row['prediction_npz']) != row['prediction_sha256']:
                    raise ValueError('Saved prediction changed')
            write_report(folder)
            status = 'completed'
            gates.append(result['gate']['passed'])
        folds.append({'family': stage['outer_fold_id'], 'status': status,
                      'report': str((folder/'report.html').relative_to(out))})
    complete = all(f['status'] == 'completed' for f in folds)
    document = {'status': 'completed' if complete else 'pending_or_failed', 'folds': folds,
                'all_localization_gates_passed': bool(complete and all(gates)),
                'classifier_started': False, 'reserved_test_evaluated': False,
                'comparison_contract': lock['comparison_contract'],
                'reporting_block_is_complete_population_epoch': False}
    write_json(out/'evaluation-status.json', document)
    lines = ['<!doctype html><meta charset="utf-8"><title>Expanded sibling change localization</title>',
             '<h1>Expanded TRAIN siblings → enlargement / narrowing maps</h1>',
             '<p>The held-out 60 modified scans and 10 original controls are unchanged. ',
             'Same CT-only U-Net, loss, resolution and 14,336 optimizer updates per fold.</p>',
             '<p>64 reporting blocks of 224 updates are not 64 complete cohort passes. ',
             'Case-balanced edited sampling visits every selected sibling before repeating; ',
             'unchanged controls retain 1/7 of updates. Baseline is supervision only.</p>',
             '<p>Development family holdout; reserved infants remain unopened; classifier has not started.</p>']
    for fold in folds:
        line = '<p>'+html.escape(fold['family'])+': '+html.escape(fold['status'])
        if fold['status'] == 'completed':
            line += ' · <a href="'+html.escape(fold['report'])+'">Metrics and interactive 3D</a>'
        lines.append(line+'</p>')
    lines.append('<p>All declared localization gates passed: '+str(document['all_localization_gates_passed'])+'</p>')
    (out/'report.html').write_text('\n'.join(lines))
    return {**document, 'report': str(out/'report.html')}


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
        c, content = exp.load_config(path), path.read_bytes()
        out = Path(c['experiment']['output_directory'])
        if args.validate_only:
            lock = exp.plan(c)
            result = {'ready': True, 'folds': len(lock['stages']), 'samples': len(lock['samples']),
                      'training_siblings_per_case': c['population']['siblings_per_case'],
                      'optimizer_updates_per_fold': 14336, 'fixed_held_out_edited': 60,
                      'fixed_held_out_controls': 10, 'reserved_case_ids': lock['reserved_case_ids'],
                      'classifier_started': False}
        elif args.stage == 'plan':
            lock = exp.freeze(c, content)
            result = report(c, lock)
        else:
            lock = exp.inspect_lock(c, content)
            if args.stage == 'list':
                if args.ids_only:
                    print('\n'.join(stage['id'] for stage in lock['stages']))
                    return 0
                result = {'stages': [stage['id'] for stage in lock['stages']]}
            elif args.stage == 'report':
                result = report(c, lock)
            elif args.stage == 'smoke':
                from fakect_change_population_learning import smoke
                started = out/'smoke'
                if started.exists():
                    started = None
                    raise FileExistsError('Smoke already started; preserve and use a fresh revision')
                started.mkdir()
                result = smoke(c, lock)
                exp.inspect_lock(c, content)
                exp.require_smoke(c, lock)
            else:
                exp.require_smoke(c, lock)
                stages = [stage for stage in lock['stages'] if stage['id'] == args.run_id]
                if len(stages) != 1:
                    raise ValueError('Use a --run-id from --stage list')
                started = out/'stages'/stages[0]['id']
                if started.exists():
                    started = None
                    raise FileExistsError('Fold already started; preserve and use a fresh revision')
                from fakect_change_population_learning import run_stage
                result = run_stage(c, lock, stages[0])
                exp.inspect_lock(c, content)
                result = {key: value for key, value in result.items() if key != 'records'}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        if started is not None:
            started.mkdir(parents=True, exist_ok=True)
            write_json(started/'failure.json', {'type': type(error).__name__, 'message': str(error),
                                              'traceback': traceback.format_exc()})
        print('error: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
