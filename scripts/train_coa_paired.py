#!/usr/bin/env python3
"""Freeze and run a focused 3D CoA classifier on reviewed image-derived crops."""
import argparse
import html
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import fakect_coa_paired_experiment as exp


def write_index(out):
    out = Path(out)
    sections = ['<h1>Focused CoA classifier: paired training</h1>',
                '<p>The verified locator supplies two 0.6 mm thoracic-aorta image blocks. '
                'The focused classifier retains features at 2.4 mm spacing and a spatial feature grid. '
                'Its inputs are attenuation and acquisition validity.</p>',
                '<p>Each optimizer update averages the mild and severe sibling losses at the same weights before clipping. '
                'The first stage is a matched-sibling training diagnostic. '
                'A passing diagnostic supports development-pilot training with fresh initialization; '
                'it does not measure generalization. Reserved test phantoms are excluded.</p>',
                '<p><a href="input.ini">Frozen INI</a> · <a href="experiment-lock.json">Frozen metadata</a></p>']
    for folder, title in [('smoke', 'Production-size GPU check'), ('diagnostic', 'Matched-sibling training diagnostic'),
                          ('pilot', '32-training / 8-validation development pilot')]:
        sections.append('<h2>'+title+'</h2>')
        path = out/'runtime-smoke.json' if folder == 'smoke' else out/folder/'results.json'
        history = out/folder/'classifier-history.json'
        failure = out/folder/'failure.json'
        if path.exists():
            record = json.loads(path.read_text())
            sections.append('<p>Status: <b>'+html.escape(record.get('status', 'recorded'))+'</b></p>')
            if folder != 'smoke':
                sections.append('<p>Completed epochs: '+str(record.get('epochs_completed', '?'))+'</p>')
        elif failure.exists():
            sections.append('<p>Run stopped. <a href="'+folder+'/failure.json">Failure details</a></p>')
        elif history.exists():
            record = json.loads(history.read_text())
            sections.append('<p>Training in progress; '+str(len(record.get('epochs', [])))+' epochs recorded.</p>')
        else:
            sections.append('<p>Awaiting this stage.</p>')
        if (out/folder/'report.html').exists():
            sections.append('<p><a href="'+folder+'/report.html">Open detailed HTML report</a></p>')
        for suffix, title in [('initial', 'Initial image and layer sensitivity'),
                              ('selected', 'Selected-checkpoint image and layer sensitivity')]:
            filename = 'sibling-sensitivity-'+suffix+'.json'
            if folder != 'smoke' and (out/folder/filename).exists():
                sections.append('<p><a href="'+folder+'/'+filename+'">'+title+'</a></p>')
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Focused CoA classifier: paired training</title>'
    document += '<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.6}h2{margin-top:2rem}a{color:#075ba3}</style></head><body>'
    document += ''.join(sections)+'</body></html>'
    (out/'report.html').write_text(document)
    return str(out/'report.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stage', choices=('plan', 'smoke', 'diagnostic', 'fit', 'report'), default='plan')
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
                from fakect_coa_paired_learning import write_report
                for name in ('diagnostic', 'pilot'):
                    if (out/name/'classifier-history.json').exists() or (out/name/'results.json').exists():
                        write_report(out/name)
                result = {'report': write_index(out)}
            else:
                folder = 'smoke' if args.stage == 'smoke' else ('diagnostic' if args.stage == 'diagnostic' else 'pilot')
                existing = [out/folder/name for name in ('model', 'results.json', 'failure.json')]
                if args.stage == 'smoke':
                    existing.append(out/'runtime-smoke.json')
                if any(p.exists() for p in existing):
                    raise ValueError('Focused '+folder+' stage already started; use a new revision')
                if args.stage != 'smoke':
                    exp.require_smoke(c, lock)
                if args.stage == 'fit':
                    exp.require_diagnostic(c, lock)
                authorized_stage = folder
                from fakect_coa_paired_learning import smoke, fit
                result = smoke(c, lock) if args.stage == 'smoke' else fit(c, lock, diagnostic=args.stage == 'diagnostic')
                write_index(out)
        omitted = {'config', 'parent_config', 'samples', 'proposals', 'design', 'dependency_sha256',
                   'code_sha256', 'reserved_test_sample_ids', 'predictions'}
        print(json.dumps({k: v for k, v in result.items() if k not in omitted}, indent=2))
        return 0
    except Exception as error:
        if authorized_stage:
            folder = Path(c['experiment']['output_directory'])/authorized_stage
            folder.mkdir(parents=True, exist_ok=True)
            exp.write_json(folder/'failure.json', {'type': type(error).__name__, 'message': str(error),
                                                   'traceback': traceback.format_exc(), 'stage': authorized_stage})
            try:
                write_index(c['experiment']['output_directory'])
            except Exception:
                pass
        print('error: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
