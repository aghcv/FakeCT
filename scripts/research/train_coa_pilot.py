#!/usr/bin/env python3
"""Plan and train a focused CoA development model from a passed diagnostic."""
import argparse
import html
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import fakect_coa_pilot_experiment as exp


def write_index(output_directory):
    """One report entry point for the memory check and development learning."""
    out = Path(output_directory)
    lock_path = out/'experiment-lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    counts = lock.get('pilot_counts', {})
    budget = lock.get('config', {}).get('training', {}).get('epochs', 'configured')
    parts = ['<h1>Focused CoA development pilot</h1>',
        '<p>The passed training diagnostic supports this new classifier fit. '
        'The image-only locator supplies two 0.6-mm attenuation and validity crops. '
        'The successful architecture and paired optimization are retained.</p>',
        '<p>Fresh weights train for up to '+html.escape(str(budget))+' epochs. '
        'Validation binary cross-entropy selects the checkpoint. Related siblings '
        'retain their phantom-family partition; reserved test images remain excluded. '
        'Severity estimates are secondary development measurements.</p>',
        '<p><a href="input.ini">Frozen INI</a> · '
        '<a href="experiment-lock.json">Frozen experiment metadata</a></p>',
        '<pre>'+html.escape(json.dumps(counts, indent=2))+'</pre>']
    for stage, title, record_path in [
            ('smoke', 'Production-size GPU and full-cache memory check', out/'runtime-smoke.json'),
            ('pilot', 'Training and development validation', out/'pilot/results.json')]:
        parts.append('<h2>'+title+'</h2>')
        failure = out/stage/'failure.json'
        history_path = out/stage/'classifier-history.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            parts.append('<p>Status: <strong>'+html.escape(str(record.get('status', 'recorded')))+
                '</strong></p>')
            if stage == 'pilot':
                parts.append('<p>Completed epochs: '+str(record.get('epochs_completed', '?'))+
                    '; selected epoch: '+str(record.get('selected_epoch', '?'))+'</p>')
            else:
                parts.append('<p><a href="runtime-smoke.json">GPU and cache measurements</a></p>')
        elif failure.exists():
            parts.append('<p>Stopped. <a href="'+stage+'/failure.json">Failure details</a></p>')
        elif history_path.exists():
            history = json.loads(history_path.read_text())
            parts.append('<p>Training in progress: '+str(len(history.get('epochs', [])))+
                ' epochs recorded.</p>')
        else:
            parts.append('<p>Awaiting this stage.</p>')
        if (out/stage/'report.html').exists():
            parts.append('<p><a href="'+stage+'/report.html">Open detailed development report</a></p>')
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Focused CoA development pilot</title><style>body{font:16px system-ui;'
        'max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.6}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}'
        'a{color:#006179}</style></head><body>'+''.join(parts)+'</body></html>')
    (out/'report.html').write_text(document)
    return str(out/'report.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stage', choices=('plan', 'smoke', 'fit', 'report'), default='plan')
    args = parser.parse_args()
    c, authorized_stage = None, None
    try:
        path = Path(args.config)
        c = exp.load_config(path)
        content = path.read_bytes()
        out = Path(c['experiment']['output_directory'])
        if args.validate_only:
            result = exp.plan(c)
        elif args.stage == 'plan':
            result = exp.freeze(c, content)
            write_index(out)
        else:
            lock = exp.inspect_lock(c, content)
            if args.stage == 'report':
                from fakect_coa_pilot_learning import write_report
                if (out/'pilot/classifier-history.json').exists():
                    write_report(out/'pilot')
                result = {'report': write_index(out)}
            else:
                stage = 'smoke' if args.stage == 'smoke' else 'pilot'
                existing = [out/stage/name for name in ('model', 'results.json', 'failure.json')]
                if stage == 'smoke':
                    existing.append(out/'runtime-smoke.json')
                if any(p.exists() for p in existing):
                    raise ValueError('Pilot '+stage+' stage already started; use a new revision')
                exp.require_diagnostic(c, lock)
                if stage == 'pilot':
                    exp.require_smoke(c, lock)
                authorized_stage = stage
                from fakect_coa_pilot_learning import smoke, fit
                result = smoke(c, lock) if stage == 'smoke' else fit(c, lock)
                write_index(out)
        omitted = {'config', 'parent_config', 'samples', 'proposals', 'design', 'dependency_sha256',
            'code_sha256', 'reserved_test_sample_ids', 'predictions', 'reference_config'}
        print(json.dumps({k:v for k,v in result.items() if k not in omitted}, indent=2))
        return 0
    except Exception as error:
        if authorized_stage:
            folder = Path(c['experiment']['output_directory'])/authorized_stage
            folder.mkdir(parents=True, exist_ok=True)
            exp.write_json(folder/'failure.json', {'type':type(error).__name__,
                'message':str(error), 'traceback':traceback.format_exc(), 'stage':authorized_stage})
            try:
                write_index(c['experiment']['output_directory'])
            except Exception:
                pass
        print('error: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
