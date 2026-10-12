"""Portable HTML review of saved two-stage measurements and mask predictions.

Rendering reads completed results only. It never fits a model, changes a split,
or rewrites model weights or measurements. Slice figures are embedded in HTML;
interactive surfaces load one scenario at a time from local report-3d assets.
"""
import base64
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np


_STYLE = """
body{margin:0;background:#f2f5f8;color:#14212c;font:16px/1.55 system-ui,sans-serif}
main{max-width:1250px;margin:auto;padding:28px}h1,h2,h3{line-height:1.2}
h1{font-size:2rem}h2{margin-top:0}section{padding:24px;background:white;border:1px solid #d6dee6;border-radius:12px;margin:24px 0}
nav{display:flex;flex-wrap:wrap;gap:18px}a{color:#1457a2}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:9px;text-align:left;border-bottom:1px solid #dde4e9;vertical-align:top}th{background:#f1f5f8}
.table{overflow:auto}.muted,figcaption{color:#4b6171;font-size:.9rem}.notice{border-left:4px solid #c78612;padding:10px 18px;background:#fff7e7}
figure{margin:18px 0}img{width:100%;height:auto}code{overflow-wrap:anywhere}details{margin:16px 0}summary{cursor:pointer;font-weight:600}
.tag{display:inline-block;padding:3px 10px;background:#e6eef7;border-radius:10px;font-size:.85rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}
@media print{body{background:white}section{break-inside:avoid;border:0}nav{display:none}details{display:block}}
"""
_CHANNELS = {
    'aorta': 'Observed aorta', 'stage1_aorta': 'Model 1 observed aorta',
    'oracle_reference': 'Reference reconstruction · exact mask input',
    'oracle_added': 'Expansion · exact mask input',
    'oracle_removed': 'Narrowing · exact mask input',
    'identity_reference': 'Identity baseline · keep input unchanged',
    'identity_added': 'Identity baseline · expansion',
    'identity_removed': 'Identity baseline · narrowing',
    'chained_reference': 'Reference reconstruction · Model 1 input',
    'chained_added': 'Expansion versus known synthetic additions',
    'chained_removed': 'Narrowing versus known synthetic removals',
    'chained_residual_added': 'Restoration residual · input outside reference',
    'chained_residual_removed': 'Restoration residual · reference missing from input',
}


def _text(value):
    return escape(str(value), quote=True)


def _number(value, digits=3):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return '—'
    if not math.isfinite(float(value)):
        return '—'
    return f'{int(value):,}' if digits == 0 else f'{float(value):.{digits}f}'


def _table(headers, rows):
    return ('<div class="table"><table><thead><tr>' + ''.join('<th>'+_text(h)+'</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+str(cell)+'</td>' for cell in row)+'</tr>' for row in rows)
            + '</tbody></table></div>')


def _read(path):
    return json.loads(Path(path).read_text())


def _figure_image(figure, caption):
    from matplotlib import pyplot as plt
    stream = BytesIO()
    figure.savefig(stream, format='png', dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    payload = base64.b64encode(stream.getvalue()).decode('ascii')
    return '<figure><img alt="'+_text(caption)+'" src="data:image/png;base64,'+payload+'"><figcaption>'+_text(caption)+'</figcaption></figure>'


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    return plt


def _history(directory, stage):
    path = directory/f'model{stage}'/'history.json'
    if not path.is_file():
        return '<p class="muted">Training history was not saved.</p>'
    rows = _read(path)
    if not rows:
        return '<p class="muted">Training history is empty.</p>'
    plt = _plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, label, color in [('train_loss', 'Training', '#1764aa'), ('validation_loss', 'Validation', '#ca6500')]:
        pairs = [(r.get('epoch'), r.get(name)) for r in rows]
        pairs = [(x, y) for x, y in pairs if isinstance(x, (float, int)) and isinstance(y, (float, int)) and math.isfinite(x) and math.isfinite(y)]
        if pairs:
            axes[0].plot(*zip(*pairs), marker='o', label=label, color=color)
    axes[0].set_ylabel('BCE + Dice loss')
    if axes[0].lines:
        axes[0].legend(frameon=False)
    pairs = [(r.get('epoch'), r.get('validation_dice')) for r in rows]
    pairs = [(x, y) for x, y in pairs if isinstance(x, (float, int)) and isinstance(y, (float, int)) and math.isfinite(x) and math.isfinite(y)]
    if pairs:
        axes[1].plot(*zip(*pairs), marker='o', color='#16815a')
    axes[1].set_ylabel('Validation Dice'); axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    fig.tight_layout()
    return _figure_image(fig, f'Model {stage} training and validation history. Checkpoint selection uses validation loss; test data do not select an epoch.')


def _model_description(directory, stage):
    path = directory/f'model{stage}'/'model-metadata.json'
    if not path.is_file():
        return ''
    metadata = _read(path)
    settings = metadata.get('settings', {})
    items = [('Architecture', settings.get('architecture')), ('Patch dimensions (k,j,i)', settings.get('patch_size_kji')),
             ('Model spacing (i,j,k), mm', settings.get('target_spacing_mm')),
             ('Base filters / depth', [settings.get('base_filters'), settings.get('depth')]),
             ('Epochs completed / selected', [metadata.get('epochs_completed'), metadata.get('best_epoch')]),
             ('Probability threshold', settings.get('threshold')), ('TensorFlow version', metadata.get('tensorflow_version')),
             ('Weight SHA256', metadata.get('model_sha256'))]
    return _table(['Training setting', 'Saved value'], [[_text(name), _text(value)] for name, value in items if value is not None])


def _metrics(summary, prefixes):
    rows = []
    for channel, entry in summary.items():
        if not any(channel == prefix or channel.startswith(prefix+'_') for prefix in prefixes):
            continue
        metrics = entry.get('family_macro', {})
        rows.append([_text(_CHANNELS.get(channel, channel.replace('_', ' '))),
                     *[_number(metrics.get(key)) for key in ('dice', 'precision', 'recall', 'false_positive_mm3', 'false_negative_mm3')]])
    if not rows:
        return '<p class="muted">No measurements are available for this group.</p>'
    return _table(['Output / comparison', 'Dice', 'Precision', 'Recall', 'False positive mm³', 'False negative mm³'], rows)


def _splits(results, lock):
    design = results.get('design', lock.get('design', {}))
    families = design.get('families', {})
    folds = design.get('folds', [])
    assignment = folds[0].get('family_assignment', {}) if folds else {}
    rows = []
    for split in ('train', 'validation', 'test'):
        groups = sorted(f for f, value in assignment.items() if value == split)
        cases = sorted(str(case) for family in groups for case in families.get(family, []))
        count = results.get('split_counts', lock.get('split_counts', {})).get(split)
        rows.append([_text(split.title()), _text(', '.join(cases) or 'Not recorded'),
                     '<br>'.join(_text(f) for f in groups) or 'Not recorded', _number(count, 0)])
    return _table(['Partition', 'Phantom cases', 'Provisional anatomy groups', 'Scenarios'], rows)


def _representatives(records):
    """Select by true native intervention, never by prediction performance."""
    cases = {}
    for row in records:
        cases.setdefault(str(row['case_id']), []).append(row)
    chosen = []
    for case in sorted(cases):
        rows = sorted(cases[case], key=lambda r: str(r['sample_id']))
        controls = [r for r in rows if r.get('is_control')]
        edited = [r for r in rows if not r.get('is_control')]
        if controls:
            chosen.append(controls[0]['sample_id'])
        if edited:
            chosen.append(max(edited, key=lambda r: (r.get('native_added_voxels', r.get('added_truth_voxels', 0))
                                                    + r.get('native_removed_voxels', r.get('removed_truth_voxels', 0)), str(r['sample_id'])))['sample_id'])
    return chosen


def _prediction_path(directory, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Prediction path must be relative to its evaluation directory')
    root = directory.resolve()
    path = (root/name).resolve()
    if not path.is_relative_to(root) or path.suffix != '.npz':
        raise ValueError('Prediction path escapes its evaluation directory')
    return path


def _slice_prediction(path, sample_id):
    with np.load(path, allow_pickle=False) as data:
        if 'sample_id' in data and str(data['sample_id'].item()) != sample_id:
            raise ValueError('Saved prediction belongs to another sample')
        after = data['after']
        if after.ndim != 3 or not after.size or after.size > 16_777_216:
            raise ValueError('Prediction requires a bounded three-dimensional model grid')
        shape = after.shape
        changed = np.asarray(data['added'], bool) | np.asarray(data['removed'], bool)
        if changed.shape != shape:
            raise ValueError('Prediction masks use inconsistent grids')
        weights = changed.sum(axis=(1, 2))
        if not weights.any():
            weights = after.sum(axis=(1, 2))
        k = int(np.argmax(weights))
        spacing = np.asarray(data['spacing_ijk_mm'], float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError('Invalid model-grid spacing')
        fields = {}
        for name in data.files:
            if name in {'before', 'after', 'added', 'removed', 'segmentation_error_mask'} or name.endswith(('_probability', '_mask')):
                array = data[name]
                if array.shape != shape:
                    raise ValueError('Prediction fields use inconsistent grids')
                plane = array[k].copy()
                if not np.isfinite(plane).all():
                    raise ValueError('Nonfinite saved prediction')
                fields[name] = plane
        return fields, spacing, k


def _overlay(ax, truth, prediction, spacing, title):
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    colors = np.array([[16, 25, 35], [0, 213, 240], [255, 172, 38], [232, 240, 247]], dtype=np.uint8)
    codes = truth.astype(np.uint8) + 2*prediction.astype(np.uint8)
    ax.imshow(colors[codes], origin='lower', interpolation='nearest', extent=(0, truth.shape[1]*spacing[0], 0, truth.shape[0]*spacing[1]))
    ax.set_title(title, fontsize=10)


def _change_view(ax, added, removed, spacing, title):
    added, removed = np.asarray(added, bool), np.asarray(removed, bool)
    colors = np.array([[16, 25, 35], [246, 80, 178], [53, 186, 255], [255, 255, 255]], dtype=np.uint8)
    codes = added.astype(np.uint8) + 2*removed.astype(np.uint8)
    ax.imshow(colors[codes], origin='lower', interpolation='nearest', extent=(0, added.shape[1]*spacing[0], 0, added.shape[0]*spacing[1]))
    ax.set_title(title, fontsize=10)


def _sample_figure(decoded, row, mode, threshold, first_threshold):
    fields, spacing, k = decoded
    before, after = fields['before'].astype(bool), fields['after'].astype(bool)
    plt = _plotting()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    if mode == 'segmentation':
        predicted = fields['aorta_probability'] >= threshold
        _overlay(axes[0], before, after, spacing, 'Original / observed geometry')
        _overlay(axes[1], after, predicted, spacing, 'Observed truth / Model 1')
        _change_view(axes[2], fields['added'], fields['removed'], spacing, 'Known synthetic additions / removals')
        _change_view(axes[3], predicted & ~after, after & ~predicted, spacing, 'Segmentation false positive / false negative')
    else:
        prefix = 'oracle' if mode == 'oracle' else 'chained'
        observed = after if mode == 'oracle' else fields['stage1_aorta_probability'] >= first_threshold
        reference = fields[prefix+'_reference_probability'] >= threshold
        add = fields.get(prefix+'_added_mask', observed & ~reference)
        remove = fields.get(prefix+'_removed_mask', reference & ~observed)
        _overlay(axes[0], before if mode == 'oracle' else after, observed, spacing,
                 'Original / observed input' if mode == 'oracle' else 'Observed truth / Model 1 input')
        _overlay(axes[1], before, reference, spacing, 'Original truth / estimated reference')
        _change_view(axes[2], fields['added'], fields['removed'], spacing, 'Known synthetic additions / removals')
        _change_view(axes[3], add, remove, spacing, 'Estimated expansion / narrowing')
    # Fix the visible field to ground-truth anatomy plus 8 mm, independent of
    # prediction quality. Predictions outside this field remain in full metrics.
    support = np.argwhere(before | after)
    if support.size:
        j0, i0 = support.min(axis=0); j1, i1 = support.max(axis=0)+1
        bounds = (max(0, i0*spacing[0]-8), min(after.shape[1]*spacing[0], i1*spacing[0]+8),
                  max(0, j0*spacing[1]-8), min(after.shape[0]*spacing[1], j1*spacing[1]+8))
        for ax in axes:
            ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:])
    for ax in axes:
        ax.set_xlabel('i direction · mm from crop face'); ax.set_ylabel('j direction · mm from crop face')
    fig.suptitle(f'{row["sample_id"]}\nModel grid k={k}; z={(k+.5)*spacing[2]:.2f} mm from lower crop face', fontsize=12)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.text(.5, .01, 'Geometry: cyan = first mask only; amber = second only; white = overlap.\nChanges: pink = added / false positive; blue = removed / false negative.', ha='center', fontsize=9)
    caption = ('Representative axial slice selected by greatest known change count, or greatest observed-mask area for controls. '
               'Images are zoomed to true anatomy plus 8 mm; metrics use the entire crop. Coordinates are model-grid distances, not native XCAT indices. '
               'End-to-end differences can include segmentation errors.' if mode == 'chained' else
               'Representative axial slice selected by greatest known change count, or greatest observed-mask area for controls. '
               'Images are zoomed to true anatomy plus 8 mm; metrics use the entire crop. Coordinates are model-grid distances, not native XCAT indices.')
    caption = f'Model grid k={k}; z={(k+.5)*spacing[2]:.2f} mm from lower crop face. ' + caption
    return _figure_image(fig, caption)


def _examples(directory, report, sample_ids, mode, first_threshold, cache):
    records = {row['sample_id']: row for row in report.get('records', [])}
    figures, embedded = [], 0
    for sample_id in sample_ids:
        row = records.get(sample_id)
        if row is None:
            continue
        if not row.get('prediction_file'):
            figures.append('<p class="muted">'+_text(sample_id)+': predictions were not saved; metrics remain available.</p>')
            continue
        try:
            path = _prediction_path(directory, row['prediction_file'])
            if path not in cache:
                cache[path] = _slice_prediction(path, sample_id)
            content = _sample_figure(cache[path], row, mode, report.get('threshold', .5), first_threshold)
            embedded += 1
        except (OSError, ValueError, KeyError) as exc:
            content = '<p class="notice">Preview unavailable: '+_text(exc)+'</p>'
        label = sample_id + (' · unmodified control' if row.get('is_control') else ' · largest native change in this case')
        figures.append('<details><summary>'+_text(label)+'</summary>'+content+'</details>')
    return ''.join(figures) or '<p class="muted">No held-out predictions were recorded.</p>', embedded


def _counts(report, lock):
    samples = {row['sample_id']: row for row in lock.get('samples', [])}
    rows = []
    for row in report.get('records', []):
        source = samples.get(row['sample_id'], {})
        native_spacing = source.get('geometry', {}).get('spacing_ijk_mm', [])
        native_cell = float(np.prod(native_spacing)) if len(native_spacing) == 3 else None
        model_spacing = report.get('target_spacing_ijk_mm', [])
        model_cell = float(np.prod(model_spacing)) if len(model_spacing) == 3 else None
        values = [row.get(key) for key in ('native_added_voxels', 'native_removed_voxels', 'added_truth_voxels', 'removed_truth_voxels')]
        native_volume = sum(values[:2])*native_cell if None not in values[:2] and native_cell is not None else None
        model_volume = sum(values[2:])*model_cell if None not in values[2:] and model_cell is not None else None
        rows.append([_text(row['sample_id']), 'Control' if row.get('is_control') else 'Edited',
                     *[_number(v, 0) for v in values], _number(native_volume), _number(model_volume)])
    return _table(['Sample', 'Type', 'Native added voxels', 'Native removed voxels',
                   'Model added voxels', 'Model removed voxels', 'Native changed mm³', 'Model changed mm³'], rows)


def render_report(experiment_directory):
    """Render/refresh report.html from saved results; return its absolute path."""
    directory = Path(experiment_directory).resolve()
    results = _read(directory/'results.json')
    mask_only = results.get('schema_version') == 'fakect.mask-model-results/1'
    if results.get('schema_version') not in {'fakect.two-stage-results/1', 'fakect.mask-model-results/1'}:
        raise ValueError('Expected completed two-stage results')
    lock_path = directory/'experiment-lock.json'
    lock = _read(lock_path) if lock_path.is_file() else {}
    first, second = results.get('stage1', {}), results.get('stage2', {})
    warnings = results.get('warnings', [])
    heading = 'Estimate the reference shape from a segmented aorta' if mask_only else 'Segment the aorta, estimate its reference shape'
    description = ('Model 2 maps a supplied aorta mask to an estimated unedited reference. ' if mask_only else
                   'Model 1 maps attenuation to the observed aorta. Model 2 maps an aorta mask to an estimated unedited reference. ')
    fragments = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
                 '<title>syncoCT · two-model training review</title><style>'+_STYLE+'</style></head><body><main>',
                 '<header><span class="tag">Engineering pilot · provisional anatomy groups</span><h1>'+heading+'</h1>',
                 '<p>'+description+
                 'Reference differences identify synthetic expansion and narrowing; they are not established clinical diagnoses.</p>',
                 '<nav><a href="#review3d">Interactive 3D</a><a href="#design">Data and split</a>'+
                 ('' if mask_only else '<a href="#model1">Model 1</a><a href="#pipeline">End-to-end</a>')+
                 '<a href="#model2">Model 2 · exact masks</a><a href="#grid">Model grid</a></nav></header>',
                 '<section id="design"><h2>Frozen data and evaluation split</h2>', _splits(results, lock),
                 '<p class="notice">'+'<br>'.join(_text(w) for w in warnings)+'</p>' if warnings else '',
                 '<p>Every sibling of a phantom stays in the same partition for both models. The unedited phantoms are reference-normal shapes under the study assumption. '
                 'The independent-patient hypothesis remains untested by this engineering holdout.</p>',
                 '<p class="muted">Dataset manifest SHA256: <code>'+_text(results.get('dataset_manifest_sha256', 'Not recorded'))+'</code></p></section>']
    from fakect_two_stage_3d import render_prediction_viewer
    try:
        viewer = render_prediction_viewer(directory, results)
        if viewer:
            fragments.extend(['<section id="review3d"><h2>Interactive 3D · all saved test scenarios</h2>',
                '<p>Select a scenario and comparison. Rotate, zoom, hide labels, and set each opacity independently. '
                'The full saved model crop includes distant false positives. Start with exact masks to isolate Model 2.</p>',
                '<p><a href="report-3d/index.html" target="_blank" rel="noopener">Open full-screen 3D viewer</a> · '+
                str(viewer['scenarios'])+' scenarios · '+_number(viewer['scene_bytes']/1024**2, 1)+' MiB of meshes loaded one scenario at a time.</p>',
                '<iframe title="Interactive aorta truth and prediction surfaces" src="report-3d/index.html" loading="lazy" '
                'style="width:100%;height:1160px;border:1px solid #d6dee6;border-radius:8px"></iframe>',
                '<p class="muted">To share this report, copy report.html together with the complete report-3d folder. '
                'It opens offline without a server. These are exact saved model-grid surfaces, not new model predictions.</p></section>'])
        else:
            fragments.append('<section id="review3d"><h2>Interactive 3D</h2><p>Predictions were not saved; 3D surfaces are unavailable.</p></section>')
    except (OSError, ValueError, KeyError) as exc:
        viewer = None
        fragments.append('<section id="review3d"><h2>Interactive 3D</h2><p class="notice">3D preview unavailable: '+_text(exc)+'</p></section>')
    ids = _representatives(second.get('records') or first.get('records', []))
    cache, image_count = {}, 0
    sections = [('model1', 'Model 1 · attenuation to observed aorta', first, ('aorta',), 'segmentation', 1,
                 'Segmentation is compared with the final edited target mask.'),
                ('model2', 'Model 2 · exact observed-mask input', second, ('oracle', 'identity'), 'oracle', 2,
                 'This isolates reference reconstruction from segmentation errors. The identity baseline keeps the observed mask unchanged and predicts no changes.'),
                ('pipeline', 'End-to-end · Model 1 prediction into Model 2', second, ('chained', 'stage1_aorta'), 'chained', None,
                 'Known synthetic change metrics compare with A minus B and B minus A. Restoration residual metrics compare with S minus B and B minus S, '
                 'where S is Model 1 output. Their differences expose segmentation error rather than concealing it.')]
    if mask_only:
        sections = [section for section in sections if section[0] == 'model2']
    for anchor, title, report, prefixes, mode, history_stage, explanation in sections:
        fragments.extend(['<section id="'+anchor+'"><h2>'+title+'</h2><p>'+explanation+'</p>',
                          _metrics(report.get('summary', {}), prefixes),
                          '<p class="muted">Tables use the saved equal-family mean. Perfect empty/empty Dice is 1; controls are shown separately below. '
                          'One held-out group cannot describe population variability.</p>'])
        if history_stage:
            fragments.extend([_model_description(directory, history_stage), _history(directory, history_stage)])
        fragments.append('<details><summary>Unmodified controls · preservation and false-positive changes</summary>'+_metrics(report.get('controls_summary', {}), prefixes)+'</details>')
        fragments.append('<p>Examples: the unmodified control and greatest native added-plus-removed voxel count per held-out case, selected independently of model scores.</p>')
        examples, count = _examples(directory/('test-model1' if mode == 'segmentation' else 'test-model2'), report, ids, mode,
                                    first.get('threshold', .5), cache)
        fragments.append(examples); image_count += count
        if mode == 'chained':
            rows = [[_text(r['sample_id']), *[_number(r.get(key), 0) for key in (
                'segmentation_error_voxels', 'chained_added_on_segmentation_error_voxels', 'chained_removed_on_segmentation_error_voxels')]]
                for r in report.get('records', [])]
            fragments.append('<details><summary>Segmentation-error overlap</summary>'+_table(
                ['Sample', 'Segmentation error voxels', 'Predicted additions on error voxels', 'Predicted removals on error voxels'], rows)+'</details>')
        fragments.append('</section>')
    fragments.extend(['<section id="grid"><h2>Native changes and the model grid</h2>',
                      '<p>Masks use nearest-neighbor physical resampling and attenuation uses linear interpolation. '
                      'Voxel counts on different grids are not directly comparable; changed volume is also shown in mm³. '
                      'Small edits may be reduced or disappear on the model grid. Control status is determined on the native grid.</p>',
                      _counts(second or first, lock), '</section>',
                      '<footer class="muted">This report embeds its 2D plots. Copy the report-3d folder alongside it to include interactive surfaces. It reads saved predictions and measurements; '
                      'it does not retrain models or change scores. Representative views show masks, not CT intensity slices.</footer></main></body></html>'])
    destination = directory/'report.html'
    temporary = directory/'.report.html.tmp'
    temporary.write_text(''.join(fragments), encoding='utf-8')
    temporary.replace(destination)
    return {'report': str(destination), 'representative_samples': ids, 'embedded_prediction_figures': image_count,
            'interactive_3d': viewer}
