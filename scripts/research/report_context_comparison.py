#!/usr/bin/env python3
"""Render a standalone, checksum-aware baseline-versus-sibling status report.

This reader does not import TensorFlow or open dataset volumes. Regeneration
replaces only its derived HTML/JSON report, never experiment artifacts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path


ARMS = ('baseline_only', 'baseline_plus_siblings')
NAMES = {'baseline_only': 'Baseline only', 'baseline_plus_siblings': 'Baseline + siblings'}
COLORS = {'baseline_only': '#1463ad', 'baseline_plus_siblings': '#b5406b'}
FIELDS = ('dice', 'precision', 'recall', 'false_positive_mm3', 'false_negative_mm3')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def read_signed(path):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != Path(str(path)+'.sha256').read_text().strip():
        raise ValueError(f'{path.name}: checksum mismatch')
    return json.loads(raw)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_history(history):
    require(isinstance(history, list), 'History is not a list')
    for index, row in enumerate(history, 1):
        require(row.get('epoch') == index, 'History epochs are missing or unordered')
        require(finite(row.get('train_loss')), 'History loss is not finite')
        require(isinstance(row.get('train_batches'), int) and row['train_batches'] > 0,
                'History update count is invalid')
        if row.get('validation_performed'):
            value = row.get('validation_family_macro_dice')
            require(finite(value) and 0 <= value <= 1, 'Validation Dice is invalid')
    return history


def checked_file(directory, name, digest):
    path = (directory/name).resolve()
    require(path.parent == directory.resolve(), 'Artifact path leaves its directory')
    require(sha256(path) == digest, f'{name}: checksum mismatch')
    return path


def inspect_arm(experiment, lock, arm):
    result = dict(arm=arm, status='Not started', history=[], training=None, evaluation=None,
                  errors=[], history_verified=False)
    model = experiment/arm/'model'
    evaluation = experiment/arm/'evaluation'
    history_path, training_path = model/'history.json', model/'training.json'
    if not model.exists():
        return result
    result['status'] = 'Incomplete fit; may be running or stopped'
    if (model/'INCOMPLETE').exists() or not training_path.exists():
        if history_path.exists():
            try:
                result['history'] = check_history(json.loads(history_path.read_text()))
            except (OSError, ValueError, KeyError, TypeError) as error:
                result['errors'].append('Live history unavailable: '+str(error))
        return result
    try:
        training = read_signed(training_path)
        require(training.get('schema_version') == 'fakect.context-comparison-model/1' and
                training.get('complete') is True and training.get('arm') == arm,
                'Training metadata is not a completed comparison arm')
        require(not training.get('diagnostic', False) and not lock.get('diagnostic', False),
                'Diagnostic results cannot be presented as unseen-family results')
        require(training.get('lock_fingerprint') == fingerprint(lock), 'Training lock fingerprint differs')
        require(training.get('settings') == lock['model_settings'], 'Model settings differ from the shared lock')
        expected_comparison = lock.get('comparison_settings', lock.get('comparison',
            lock.get('config', {}).get('comparison')))
        require(training.get('comparison') == expected_comparison, 'Comparison settings differ from the lock')
        initial=training.get('initial_weights_sha256')
        require(isinstance(initial,str) and len(initial)==64 and all(c in '0123456789abcdef' for c in initial),
                'Initial-weights digest is missing or malformed')
        require(bool(training.get('component_sha256')), 'Training component hashes are missing')
        for name, digest in training['component_sha256'].items():
            require(lock['code_sha256'].get('src/'+name) == digest, 'Training implementation differs from the lock')
        checked_file(model, 'checkpoint.keras', training['model_sha256'])
        checked_file(model, 'history.json', training['history_sha256'])
        history = check_history(json.loads(history_path.read_text()))
        require(training['epochs_completed'] == len(history) == training['settings']['epochs'],
                'Training epoch count differs')
        require(training['optimizer_updates'] == sum(r['train_batches'] for r in history),
                'Training update count differs')
        require(all(r['train_batches'] == training['comparison']['updates_per_epoch'] for r in history),
                'Per-epoch update budget differs')
        require(training.get('test_payloads_loaded') is False, 'Training loaded test payloads')
        forbidden = {r['sample_id'] for r in lock['samples'] if r['split'] == 'test' or
            (arm == 'baseline_only' and r['split'] == 'train' and not r.get('baseline'))}
        require(not forbidden.intersection(training['loaded_samples']), 'Forbidden training payload recorded')
        allowed={r['sample_id'] for r in lock['samples']}-forbidden
        require(set(training['loaded_samples']) <= allowed, 'Unknown training payload recorded')
        for row in history:
            if row.get('validation_performed'):
                checked_file(model, row['validation_file'], row['validation_sha256'])
                validation = read_signed(model/row['validation_file'])
                require(validation['epoch'] == row['epoch'] and validation['arm'] == arm and
                    math.isclose(validation['summary']['family_macro']['dice'],
                        row['validation_family_macro_dice'],rel_tol=1e-9,abs_tol=1e-12),
                    'Validation selection evidence disagrees with history')
        eligible=[r for r in history if r.get('validation_performed')]
        require(bool(eligible),'No validation checkpoint-selection evidence')
        best=max(eligible,key=lambda r:r['validation_family_macro_dice'])
        require(training['best_epoch'] == best['epoch'] and
            math.isclose(training['best_selection_value'],best['validation_family_macro_dice'],
                         rel_tol=1e-9,abs_tol=1e-12),'Selected checkpoint differs from validation history')
        result.update(training=training, history=history, history_verified=True,
                      status='Fit complete; held-out evaluation not completed')
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status='Invalid training artifacts; metrics hidden', errors=[str(error)])
        return result
    results_path = evaluation/'results.json'
    if not results_path.exists() or (evaluation/'INCOMPLETE').exists():
        if evaluation.exists():
            result['status'] = 'Fit complete; evaluation incomplete (running or stopped)'
        return result
    try:
        scored = read_signed(results_path)
        require(scored.get('schema_version') == 'fakect.context-comparison-results/1' and
                scored.get('complete') is True and scored.get('arm') == arm and scored.get('split') == 'test',
                'Evaluation is not a completed held-out test arm')
        require(not scored.get('diagnostic', False), 'Diagnostic evaluation is not held-out performance')
        require(scored.get('training_metadata_sha256') == sha256(training_path), 'Evaluation training hash differs')
        for key in ('lock_fingerprint', 'model_sha256', 'settings', 'comparison', 'component_sha256'):
            require(scored.get(key) == training.get(key), f'Evaluation {key} differs from training')
        expected = {r['sample_id']: r for r in lock['samples'] if r['split'] == 'test'}
        records = scored['records']
        require(len(records) == len(expected) and {r['sample_id'] for r in records} == set(expected),
                'Evaluation samples differ from the complete frozen test set')
        require(scored.get('test_payloads_loaded') is True and set(scored['loaded_samples']) == set(expected),
                'Evaluation payload accounting differs from the frozen test set')
        for record in records:
            source = expected[record['sample_id']]
            require(all(record.get(k) == source[k] for k in ('case_id','anatomy_family','split')),
                    'Evaluation family/case/split differs')
            for key in FIELDS:
                value = record['metrics'][key]
                require(finite(value) and value >= 0 and (value <= 1 if key in FIELDS[:3] else True),
                        'Evaluation metric is invalid')
        recalculated = summarize(records)
        require(equal_numbers(scored['summary'], recalculated), 'Evaluation aggregate metrics disagree with records')
        result.update(evaluation=scored, status='Complete; held-out metrics verified')
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status='Invalid evaluation artifacts; test metrics hidden')
        result['errors'].append(str(error))
    return result


def summarize(records):
    cases = {}
    for row in records:
        cases.setdefault(row['case_id'], []).append(row)
    mean = lambda rows: {k:sum(r['metrics'][k] for r in rows)/len(rows) for k in FIELDS}
    case_means = {c:mean(rows) for c,rows in cases.items()}
    families = {}
    for case, rows in cases.items():
        families.setdefault(rows[0]['anatomy_family'], []).append(dict(metrics=case_means[case]))
    family_means = {f:mean(rows) for f,rows in families.items()}
    return dict(sample_macro=mean(records), case_means=case_means, family_means=family_means,
                family_macro=mean([dict(metrics=r) for r in family_means.values()]))


def equal_numbers(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equal_numbers(a[k], b[k]) for k in a)
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def matching_checks(arms):
    if any(not a['training'] for a in arms):
        return []
    a,b = arms
    return [dict(check=title, passed=bool(value)) for title,value in [
        ('Identical initialized weights', a['training']['initial_weights_sha256'] == b['training']['initial_weights_sha256']),
        ('Identical model settings', a['training']['settings'] == b['training']['settings']),
        ('Identical physical patch order and augmentation seeds',
         [r.get('physical_patch_events_sha256') for r in a['history']] ==
         [r.get('physical_patch_events_sha256') for r in b['history']] and
         all(isinstance(r.get('physical_patch_events_sha256'),str) and len(r['physical_patch_events_sha256'])==64
             for r in a['history']+b['history'])),
        ('Identical optimizer update budget', a['training']['optimizer_updates'] == b['training']['optimizer_updates'] and
         [r['train_batches'] for r in a['history']] == [r['train_batches'] for r in b['history']]),
        ('Identical validation schedule', [r['epoch'] for r in a['history'] if r.get('validation_performed')] ==
         [r['epoch'] for r in b['history'] if r.get('validation_performed')]),
    ]]


def chart(arms, key, title, validation=False):
    series = [(a, [(r['epoch'],r[key]) for r in a['history'] if key in r and finite(r[key])]) for a in arms]
    points = [p for _,values in series for p in values]
    if not points:
        return '<p>No '+html.escape(title.lower())+' recorded yet.</p>'
    maximum = max(p[0] for p in points)
    low,high = (0.,1.) if validation else (0.,max(p[1] for p in points)*1.1 or 1.)
    sx=lambda x:55+(x-1)/max(1,maximum-1)*565
    sy=lambda y:215-(y-low)/(high-low)*170
    bits=[f'<svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 650 270">',
          f'<text x="55" y="24">{html.escape(title)}</text>',
          '<path d="M55 40V215H630" fill="none" stroke="#9baabd"/>']
    for y in (low,(low+high)/2,high):
        bits.append(f'<text x="5" y="{sy(y)+4:.1f}" font-size="12">{y:.3f}</text>')
    bits += [f'<text x="55" y="238" font-size="12">Epoch 1</text>',
             f'<text x="575" y="238" font-size="12">{maximum}</text>']
    for a,values in series:
        if not values:
            continue
        color=COLORS[a['arm']]; coords=' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in values)
        dash='' if a['history_verified'] else ' stroke-dasharray="5 4"'
        bits.append(f'<polyline points="{coords}" stroke="{color}" fill="none" stroke-width="2"{dash}/>')
        for x,y in values:
            bits.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{color}"><title>Epoch {x}: {y:.6f}</title></circle>')
    bits.append('</svg>')
    return ''.join(bits)


def metric_table(arms, group):
    available = {a['arm']:a['evaluation']['summary'][group] for a in arms if a['evaluation']}
    names = sorted({key for values in available.values() for key in values})
    if not names:
        return '<p>No verified held-out metrics yet.</p>'
    rows=['<table><tr><th>'+('Case' if group=='case_means' else 'Family')+'</th><th>Baseline Dice</th><th>Sibling Dice</th><th>Difference</th></tr>']
    for name in names:
        values=[available.get(arm,{}).get(name,{}).get('dice') for arm in ARMS]
        delta=values[1]-values[0] if all(v is not None for v in values) else None
        rows.append('<tr><th>'+html.escape(name)+'</th>'+''.join('<td>'+('Pending' if v is None else f'{v:.4f}')+'</td>' for v in values)+
                    '<td>'+('Pending' if delta is None else f'{delta:+.4f}')+'</td></tr>')
    return ''.join(rows)+'</table>'


def report(experiment, output=None):
    experiment=Path(experiment).resolve()
    output=Path(output).resolve() if output else experiment/'comparison-report.html'
    require(output not in (experiment/'experiment-lock.json',experiment/'input.ini') and output.suffix.lower()=='.html',
            'Report output must be a separate HTML file')
    try:
        lock=read_signed(experiment/'experiment-lock.json')
        require(not lock.get('diagnostic',False), 'Use report_context_diagnostic.py for a diagnostic experiment')
        require(lock.get('schema_version') == 'fakect.context-comparison-lock/1', 'Not a comparison experiment lock')
        require('samples' in lock and 'model_settings' in lock, 'Comparison lock is missing required fields')
        require(sha256(experiment/'input.ini') == lock['input_sha256'], 'Frozen INI checksum differs')
        require(sha256(lock['dataset_manifest']) == lock['dataset_manifest_sha256'], 'Dataset manifest checksum differs')
        for name,digest in lock['code_sha256'].items():
            source=(experiment/'source'/name).resolve()
            require((experiment/'source').resolve() in source.parents, 'Snapshot path leaves source directory')
            require(sha256(source) == digest, 'Frozen implementation checksum differs')
    except (OSError,ValueError,KeyError,TypeError) as error:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Comparison unavailable</title>'
            '<h1>Comparison integrity check failed</h1><p>All metrics hidden.</p><p>'+html.escape(str(error))+'</p></html>')
        output.with_suffix('.json').write_text(json.dumps(dict(schema_version='fakect.context-comparison-report/1',
            status='Invalid experiment; all metrics hidden',error=str(error),html_sha256=sha256(output)),indent=2)+'\n')
        return output
    arms=[inspect_arm(experiment,lock,arm) for arm in ARMS]
    checks=matching_checks(arms)
    matched=bool(checks) and all(c['passed'] for c in checks)
    completed=all(a['evaluation'] for a in arms)
    status='Comparison complete' if completed and matched else 'Comparison incomplete'
    if checks and not matched:
        status='Matching checks failed; do not interpret an augmentation effect'
    counts={part:dict(samples=sum(r['split']==part for r in lock['samples']),
        cases=len({r['case_id'] for r in lock['samples'] if r['split']==part}),
        families=len({r['anatomy_family'] for r in lock['samples'] if r['split']==part}))
        for part in ('train','validation','test')}
    for part in counts:
        counts[part]['case_ids']=sorted({r['case_id'] for r in lock['samples'] if r['split']==part})
    timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    escaped=lambda v:html.escape(str(v))
    cards=[]
    for a in arms:
        t=a['training']; e=a['evaluation']; title=NAMES[a['arm']]
        body=f'<h2>{title}</h2><p><strong>{escaped(a["status"])}</strong></p>'
        if a['history']:
            body+=f'<p>{len(a["history"])} epochs recorded · {sum(r["train_batches"] for r in a["history"]):,} updates.</p>'
            if not a['history_verified']:
                body+='<p class="notice">Live history is provisional and not finalized or checksummed. A directory alone does not prove a job is still running.</p>'
        if t:
            body+=f'<p>Selected checkpoint: epoch {t["best_epoch"]}; validation family Dice {t["best_selection_value"]:.4f}.</p>'
        if e:
            summary=e['summary']; body+=f'<p class="metric">Test family Dice: {summary["family_macro"]["dice"]:.4f}</p>'
            body+=f'<p>Test sample Dice: {summary["sample_macro"]["dice"]:.4f}. '
            body+=f'Family precision/recall: {summary["family_macro"]["precision"]:.4f} / {summary["family_macro"]["recall"]:.4f}.</p>'
        if a['errors']:
            body+='<ul>'+''.join('<li>'+escaped(v)+'</li>' for v in a['errors'])+'</ul>'
        cards.append('<article>'+body+'</article>')
    split_table='<table><tr><th>Partition</th><th>Cases</th><th>Families</th><th>Scenarios</th><th>Case IDs</th></tr>'
    for part,row in counts.items():
        split_table+=f'<tr><th>{part}</th><td>{row["cases"]}</td><td>{row["families"]}</td><td>{row["samples"]}</td><td>{escaped(", ".join(row["case_ids"]))}</td></tr>'
    split_table+='</table>'
    check_html='<p>Matching checks become available after both fits complete.</p>'
    if checks:
        check_html='<ul>'+''.join('<li>'+('PASS' if c['passed'] else 'FAIL')+' — '+escaped(c['check'])+'</li>' for c in checks)+'</ul>'
    settings=lock['model_settings']; extents=[n*s for n,s in zip(settings['patch_size_kji'],settings['target_spacing_mm'][::-1])]
    caption=' · '.join(f'<span style="color:{COLORS[arm]}">{NAMES[arm]}</span>' for arm in ARMS)
    markup=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aorta context comparison</title>
<style>body{{font:16px system-ui,sans-serif;color:#203044;background:#f8fafc;max-width:1200px;margin:2rem auto;padding:0 1rem}}h1,h2,h3{{line-height:1.2}}.notice{{background:#fff2ce;padding:1rem;border-radius:6px}}.cards,.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}}article{{background:white;border:1px solid #d5deea;border-radius:8px;padding:1rem}}table{{border-collapse:collapse;background:white;width:100%;font-size:.93rem}}td,th{{padding:.55rem;text-align:left;border:1px solid #d5deea}}.metric{{font-size:1.35rem;font-weight:600}}svg{{width:100%;background:white;border:1px solid #d5deea}}code{{overflow-wrap:anywhere}}small{{color:#526174}}.scroll{{overflow:auto}}</style>
<h1>Aorta context: baseline versus siblings</h1><p><strong>{escaped(status)}</strong> · generated {escaped(timestamp)}</p>
<p class="notice">Unseen-family synthetic segmentation experiment. Provisional atlas-derived thoracic aorta and proximal branches; no clinical validation. The earlier fixed-training-patch diagnostic is not included as a test result. Siblings are correlated examples, not independent patients.</p>
<div class="cards">{''.join(cards)}</div>
<h2>Shared experiment</h2><p>Image input: attenuation only. Patch extent (k,j,i): {escaped(extents)} mm; model spacing (i,j,k): {escaped(settings['target_spacing_mm'])} mm. Validation selects the checkpoint; the held-out test set is used after fitting.</p>
<div class="scroll">{split_table}</div><p>The train count lists all frozen scenarios available to the comparison. Baseline-only fitting uses one unedited sample per training case; both arms use the same physical sampling schedule and update budget. Inference covers the whole available scan without a target-derived crop.</p>
<h2>Matching evidence</h2>{check_html}
<h2>Learning curves</h2><p>{caption}. Dashed curves are unverified live history. Validation points represent complete whole-scan evaluations.</p><div class="charts">{chart(arms,'train_loss','Training loss')}{chart(arms,'validation_family_macro_dice','Validation family Dice',True)}</div>
<h2>Held-out cases</h2><p>Each case is averaged across its scenarios.</p><div class="scroll">{metric_table(arms,'case_means')}</div>
<h2>Held-out families</h2><p>Each family is averaged across its cases. Difference = sibling-trained minus baseline-trained Dice; this is descriptive, not a significance test.</p><div class="scroll">{metric_table(arms,'family_means')}</div>
<h2>Integrity and interpretation</h2><p>Completed metrics are shown only after artifact checksums, model/history/lock links, split identities and aggregate arithmetic pass verification. Matching failures prevent an augmentation-effect conclusion. Live history can change until fitting finishes; refresh by regenerating this report. This page has no external scripts or assets.</p>
<p><small>Experiment: <code>{escaped(experiment)}</code><br>Lock SHA-256: <code>{sha256(experiment/'experiment-lock.json')}</code><br>Renderer SHA-256: <code>{sha256(__file__)}</code></small></p></html>'''
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+'.tmp');temp.write_text(markup);temp.replace(output)
    provenance=dict(schema_version='fakect.context-comparison-report/1',generated_utc=timestamp,
        experiment=str(experiment),lock_sha256=sha256(experiment/'experiment-lock.json'),
        renderer_sha256=sha256(__file__),html_sha256=sha256(output),status=status,
        both_test_evaluations_complete=completed,matching_checks=checks,split_counts=counts,
        arms=[{k:a[k] for k in ('arm','status','errors','history_verified')} for a in arms])
    path=output.with_suffix('.json');tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment',type=Path)
    group.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        if args.config:
            import sys
            sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
            from fakect_context_comparison import load_config
            experiment=load_config(args.config)['experiment']['output_directory']
        else:
            experiment=args.experiment
        print(report(experiment,args.output))
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(2,f'error: {error}\n')
