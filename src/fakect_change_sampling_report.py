"""Saved TRAIN-only learning check for the corrected change sampler.

Fixed, truth-centred patches test whether the unchanged network and loss can
learn both change classes. They are deliberately supervised TRAIN examples,
not held-out inference. Complete-focus TRAIN measurements remain separate.
This module reads saved scores/arrays and never trains or selects a checkpoint.
"""
from html import escape
import json
from pathlib import Path

import numpy as np

import fakect_change_report as reference


SCHEMA = 'fakect.change-sampling-report/1'
SCOPE = ('Training-only learning diagnostic. Fixed patch locations use TRAIN '
         'targets. Scores measure fitting of these supervised examples; they '
         'do not measure unseen-anatomy performance or clinical generalization.')


def _records(document, key):
    rows = document.get(key, [])
    if not isinstance(rows, list):
        raise ValueError(key + ' must be a list of saved records')
    if rows:
        reference.summarize(rows)
    return rows


def _identity(row):
    return row['sample_id'], row.get('proposal_index', row.get('focus_index'))


def _paired(initial, final):
    first = {_identity(row): row for row in initial}
    last = {_identity(row): row for row in final}
    if not first or set(first) != set(last):
        raise ValueError('Initial and final measurements require the same fixed TRAIN patches')
    for key in first:
        before, after = first[key], last[key]
        for field in ('base_sample_id', 'case_id', 'anatomy_family', 'baseline',
                      'mode', 'burden_bin', 'tile', 'offset_kji', 'shape_kji'):
            if before.get(field) != after.get(field):
                raise ValueError('Fixed patch identity or geometry changed: ' + field)
        a, b = before['metrics'], after['metrics']
        if a['valid_voxels'] != b['valid_voxels'] or a['spacing_ijk_mm'] != b['spacing_ijk_mm']:
            raise ValueError('Fixed patch validity or spacing changed')
        if any(a['classes'][name]['target_voxels'] != b['classes'][name]['target_voxels']
               for name in reference.CLASSES):
            raise ValueError('Fixed patch target counts changed')


def _summary(rows):
    return reference.summarize(rows) if rows else None


def summarize(document):
    initial = _records(document, 'initial_patch_records')
    final = _records(document, 'final_patch_records')
    whole = _records(document, 'whole_focus_training_records')
    _paired(initial, final)
    return dict(schema_version=SCHEMA, scope=SCOPE,
        initial_fixed_train_patches=_summary(initial),
        final_fixed_train_patches=_summary(final),
        complete_focus_train=_summary(whole),
        gate=document.get('gate'), classifier_started=False,
        held_out_evaluated=False,
        patch_control_policy='Report false-positive mm3 separately; complete-focus controls use original-aorta volume as their denominator.')


def _num(value):
    return '—' if value is None else f'{float(value):.3f}'


def _score(summary, name, key):
    if summary is None:
        return None
    return summary['classes'][name]['case_macro'][key]


def _plot(directory, summary, history):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    epochs = history.get('epochs', [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for field, label in (('loss', 'Total loss'), ('ce', 'Weighted CE'), ('dice_loss', 'Foreground Dice loss')):
        entries = [entry for entry in epochs if field in entry]
        if entries:
            axes[0].plot([entry.get('updates', entry.get('epoch')) for entry in entries],
                         [entry[field] for entry in entries], label=label)
    axes[0].set(xlabel='Optimizer updates', ylabel='TRAIN loss', title='Fixed supervised TRAIN patches')
    if axes[0].lines:
        axes[0].legend(frameon=False)
    x = np.arange(2)
    for delta, key, label, color in ((-.18, 'initial_fixed_train_patches', 'Initial', '#7187a3'),
                                   (.18, 'final_fixed_train_patches', 'Final', '#147df5')):
        scores = [_score(summary[key], name, 'dice') for name in ('added', 'removed')]
        axes[1].bar(x + delta, [np.nan if value is None else value for value in scores],
                    width=.36, label=label, color=color)
    axes[1].set(xticks=x, xticklabels=['Enlargement', 'Narrowing'], ylabel='Case-average Dice',
                ylim=(0, 1), title='Same fixed TRAIN patches')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    paths = []
    for suffix in ('png', 'pdf', 'svg'):
        path = directory / ('change-sampling-diagnostic.' + suffix)
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def _table(rows):
    values = []
    for row in rows:
        metrics = row['metrics']
        false_volume = sum(metrics['classes'][name]['false_positive_mm3'] for name in ('added', 'removed'))
        fields = [escape(str(row['sample_id'])), escape(str(row.get('case_id', ''))),
                  _num(metrics['classes']['added']['dice']), _num(metrics['classes']['removed']['dice']),
                  _num(false_volume), _num(row.get('control_change_fraction'))]
        values.append('<tr>' + ''.join('<td>' + value + '</td>' for value in fields) + '</tr>')
    return ('<div class="table"><table><tr><th>TRAIN sample / patch</th><th>Case</th>'
            '<th>Enlargement Dice</th><th>Narrowing Dice</th><th>False change mm³</th>'
            '<th>Control changes / original aorta</th></tr>' + ''.join(values) + '</table></div>')


def write_report(c, lock):
    """Write a separate report from the saved learning check, preserving inputs.

    ``c['experiment']['output_directory']`` locates results.json. Prediction
    paths use the same bounded relative-path and checksum contract as the
    existing change report. ``lock`` is accepted by the experiment CLI; its
    immutable on-disk checksum is checked when saved results declare it.
    """
    directory = Path(c['experiment']['output_directory']).resolve()
    source = directory / 'results.json'
    if not source.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / 'report.html'
        stages = lock.get('stages', [])
        cases = stages[0].get('train_case_ids', []) if stages else []
        report_path.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>TRAIN sampling check · pending</title><body><h1>Training-only change sampling check</h1>'
            '<p>Results pending. No learning gate has been evaluated.</p><p>' + escape(SCOPE) + '</p>'
            '<p>Source TRAIN cases: ' + escape(', '.join(map(str, cases)))
            + '. No held-out anatomy is evaluated and Model 2 has not started.</p></body></html>', encoding='utf-8')
        lock_path = directory / 'experiment-lock.json'
        inputs = {'experiment-lock.json': reference._hash(lock_path)} if lock_path.is_file() else {}
        provenance = dict(schema_version=SCHEMA, status='pending', scope=SCOPE,
            inputs_sha256=inputs, reporter_sha256=reference._hash(__file__),
            outputs_sha256={'report.html': reference._hash(report_path)}, interactive_3d=None)
        (directory / 'change-sampling-report-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return dict(status='pending', report=str(report_path), summary=None, interactive_3d=None,
                    classifier_started=False, gate=None)
    document = json.loads(source.read_text())
    if document.get('classifier_started', False) or document.get('held_out_evaluated', False):
        raise ValueError('This report is restricted to TRAIN-only localization diagnostics')
    summary = summarize(document)
    inputs = {'results.json': reference._hash(source)}
    lock_path = directory / 'experiment-lock.json'
    if lock_path.is_file():
        inputs['experiment-lock.json'] = reference._hash(lock_path)
        expected = document.get('source_lock_sha256')
        if expected and inputs['experiment-lock.json'] != expected:
            raise ValueError('Saved source lock changed before report generation')
    elif document.get('source_lock_sha256'):
        raise ValueError('Saved results require the immutable experiment lock')
    history_path = directory / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.is_file() else {'epochs': []}
    if history_path.is_file():
        inputs['history.json'] = reference._hash(history_path)
    initial = document['initial_patch_records']
    final = document['final_patch_records']
    whole = document.get('whole_focus_training_records', [])
    paths = {}
    for section, rows in (('initial', initial), ('final', final), ('whole', whole)):
        for row in rows:
            path = reference._prediction_path(directory, row)
            if path is not None:
                paths[(section, _identity(row))] = path
                inputs[str(path.relative_to(directory))] = reference._hash(path)
    # Count agreement alone cannot establish that the cached supervision stayed
    # fixed. Compare exact saved truth/validity arrays for paired patch archives.
    for row in final:
        key = _identity(row)
        a, b = paths.get(('initial', key)), paths.get(('final', key))
        if a is not None and b is not None:
            with np.load(a, allow_pickle=False) as before, np.load(b, allow_pickle=False) as after:
                for field in ('labels', 'valid', 'before', 'after', 'spacing_ijk_mm', 'origin_kji'):
                    if field in before or field in after:
                        if field not in before or field not in after or not np.array_equal(before[field], after[field]):
                            raise ValueError('Saved fixed patch supervision changed: ' + field)
    plots = _plot(directory, summary, history)
    scenes = [(row, paths[('final', _identity(row))]) for row in final if ('final', _identity(row)) in paths]
    viewer = None
    if scenes:
        from fakect_two_stage_3d import render_viewer
        viewer = render_viewer(directory / 'report-3d', (reference._scene(path, row) for row, path in scenes),
            title='TRAIN-only fixed patch fitting · final saved predictions', show_metrics=False,
            footer_note=SCOPE + ' Original masks and true change maps are display truth, never predictor inputs.')
    comparisons = []
    for name, label in (('added', 'Enlargement'), ('removed', 'Narrowing')):
        for metric in ('dice', 'precision', 'recall'):
            comparisons.append('<tr><td>' + label + '</td><td>' + metric + '</td><td>'
                + _num(_score(summary['initial_fixed_train_patches'], name, metric)) + '</td><td>'
                + _num(_score(summary['final_fixed_train_patches'], name, metric)) + '</td></tr>')
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>TRAIN-only change sampling check</title><style>body{font:16px/1.5 system-ui,sans-serif;background:#f5f7fa;color:#17212b;max-width:1250px;margin:2rem auto;padding:0 1.5rem}section{background:white;padding:1.2rem;margin:1rem 0;border:1px solid #d8dee4}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:.5rem;border:1px solid #d8dee4;text-align:left}img{width:100%}iframe{width:100%;height:1100px;border:0}.table{overflow:auto}.notice{border-left:5px solid #d55e00}</style>'
        '<h1>Training-only check of the corrected change sampler</h1><section class="notice"><p>' + escape(SCOPE) + '</p>'
        '<p>The predictor receives current attenuation and acquisition validity only. Baseline differences supervise added and removed classes. The architecture, loss and image grid remain unchanged. Model 2 has not started; passing this check does not launch it.</p></section>'
        '<section><h2>Same fixed patches: initial versus final</h2><p>Case averages give each anatomy equal weight. Empty-target overlap scores are undefined and excluded; controls do not inflate positive-class Dice.</p>'
        '<table><tr><th>Class</th><th>Metric</th><th>Initial</th><th>Final</th></tr>' + ''.join(comparisons) + '</table>'
        '<img src="change-sampling-diagnostic.png" alt="Fixed TRAIN patch fitting history and initial versus final Dice">'
        '<p>Saved learning gate: <code>' + escape(json.dumps(document.get('gate'), allow_nan=False)) + '</code></p></section>'
        '<section><h2>Final fixed-patch measurements</h2><p>Patch controls report false-positive volume. Full-focus unchanged controls use predicted changes divided by original-aorta volume, not background volume.</p>' + _table(final) + '</section>'
        '<section><h2>Complete-focus TRAIN diagnostic</h2><p>These separate measurements use target-independent complete tiling within the frozen locator focus. The model was trained on these anatomy cases. This is not held-out validation, and complete-focus scores are not combined with fixed-patch fitting scores.</p>'
        + (_table(whole) if whole else '<p>Complete-focus TRAIN measurements were not saved.</p>') + '</section>'
        + ('<section><h2>Final fixed-patch 3D review</h2><p>Rotate and zoom; toggle truth and predictions independently at 15%, 45% or 80% opacity.</p><a href="report-3d/index.html">Open TRAIN patch viewer</a><iframe src="report-3d/index.html" title="TRAIN-only patch change surfaces"></iframe></section>' if viewer else '')
        + '</html>')
    report_path = directory / 'report.html'
    report_path.write_text(html, encoding='utf-8')
    summary_path = directory / 'change-sampling-summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    output_paths = [report_path, summary_path] + plots
    if viewer:
        output_paths.extend(path for path in (directory / 'report-3d').rglob('*') if path.is_file())
    provenance = dict(schema_version=SCHEMA, scope=SCOPE, inputs_sha256=inputs,
        reporter_sha256=reference._hash(__file__), reference_reporter_sha256=reference._hash(reference.__file__),
        outputs_sha256={str(path.relative_to(directory)): reference._hash(path) for path in output_paths},
        interactive_3d=viewer)
    (directory / 'change-sampling-report-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return dict(report=str(report_path), summary=str(summary_path), interactive_3d=viewer,
        initial_patch_records=len(initial), final_patch_records=len(final), whole_focus_training_records=len(whole))
