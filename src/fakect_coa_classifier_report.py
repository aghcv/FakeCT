"""Numerical evaluation and portable reports for the CoA classifier pilot.

This module consumes prediction/experiment JSON only. It never loads image,
mask, annotation or checkpoint payloads. Metric evaluation needs only the
standard library; plotting imports Matplotlib lazily.
"""
import base64
import html
import json
import math
from pathlib import Path


DECISION_THRESHOLD = 0.5
LOSS_LABEL_THRESHOLD = 0.20
NEAR_THRESHOLD = (0.18, 0.22)
_BCE_EPS = 2.0 ** -23  # Match the original float32 epsilon clipping contract.


def _checked(records):
    records = list(records)
    for row in records:
        if row['label'] not in (0, 1):
            raise ValueError('Classifier labels must be zero or one')
        values = [float(row[key]) for key in ('probability', 'score_label', 'score_prediction')]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Classifier predictions and labels must be finite')
        if not 0 <= values[0] <= 1:
            raise ValueError('Classifier probability must be between zero and one')
    return records


def _auroc(records):
    """Mann--Whitney rank AUROC with average ranks for tied probabilities."""
    ordered = sorted(records, key=lambda row: float(row['probability']))
    positives = sum(int(row['label']) for row in ordered)
    negatives = len(ordered) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end]['probability'] == ordered[start]['probability']:
            end += 1
        mean_rank = (start + 1 + end) / 2.0
        rank_sum += mean_rank * sum(int(row['label']) for row in ordered[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def summarize(records):
    """Fixed-0.5 classification and continuous local-volume-loss evaluation.

    Empty groups return None, matching the original pilot metrics API. Other
    undefined quantities (e.g. one-class AUROC) are None, never NaN. The score
    error/RMSE use fraction units; severity MAE also has percentage-point units.
    These are descriptive scenario-level metrics, not independent-patient CIs.
    """
    rows = _checked(records)
    if not rows:
        return None
    positive = [row for row in rows if row['label'] == 1]
    negative = [row for row in rows if row['label'] == 0]
    tp = sum(row['probability'] >= DECISION_THRESHOLD for row in positive)
    fp = sum(row['probability'] >= DECISION_THRESHOLD for row in negative)
    tn, fn = len(negative) - fp, len(positive) - tp
    sensitivity = tp / len(positive) if positive else None
    specificity = tn / len(negative) if negative else None
    errors = [float(row['score_prediction']) - float(row['score_label']) for row in rows]
    mae = math.fsum(abs(value) for value in errors) / len(rows)
    def clipped(value):
        return min(1 - _BCE_EPS, max(_BCE_EPS, value))
    bce = math.fsum(-row['label'] * math.log(clipped(row['probability']))
                   - (1 - row['label']) * math.log(clipped(1 - row['probability']))
                   for row in rows) / len(rows)
    auc = _auroc(rows)
    return {
        'samples': len(rows), 'bce': bce, 'accuracy': (tp + tn) / len(rows),
        'sensitivity': sensitivity, 'specificity': specificity, 'score_mae': mae,
        'positive_probability_mean': (math.fsum(row['probability'] for row in positive) / len(positive)
                                      if positive else None),
        'negative_probability_mean': (math.fsum(row['probability'] for row in negative) / len(negative)
                                      if negative else None),
        'auroc': auc, 'roc_auc': auc,
        'balanced_accuracy': ((sensitivity + specificity) / 2 if positive and negative else None),
        'precision': tp / (tp + fp) if tp + fp else None,
        'confusion_counts': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
        'severity_mae_percentage_points': 100 * mae,
        'regression_rmse': math.sqrt(math.fsum(value * value for value in errors) / len(rows))}


def _read(directory, name):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else {}


def _text(value):
    return html.escape(str(value), quote=True)


def _number(value, digits=3):
    return 'N/A' if value is None else f'{float(value):.{digits}f}'


def _table(headers, rows, classes=None):
    body = []
    for index, row in enumerate(rows):
        class_name = classes[index] if classes else ''
        body.append('<tr class="' + _text(class_name) + '">' + ''.join(
            '<td>' + _text(value) + '</td>' for value in row) + '</tr>')
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(
        '<th>' + _text(value) + '</th>' for value in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'


def _is_original(row):
    return row.get('role') in ('original', 'baseline')


def _near_threshold(row):
    return bool(row.get('near_threshold', False)) or NEAR_THRESHOLD[0] <= row['score_label'] <= NEAR_THRESHOLD[1]


def _split_rows(rows, split):
    aliases = {'training': ('train', 'training'), 'validation': ('validation', 'valid')}
    return [row for row in rows if row.get('split') in aliases[split]]


def _metrics_rows(rows, result):
    groups = []
    for split in ('training', 'validation'):
        selected = _split_rows(rows, split)
        groups.append((split.title(), summarize(selected) if rows else result.get(split)))
        groups.append((split.title() + ', edited only', summarize([row for row in selected if not _is_original(row)])
                       if rows else result.get(split + '_edited_only')))
        groups.append((split.title() + ', originals', summarize([row for row in selected if _is_original(row)])))
        groups.append((split.title() + ', edited negatives', summarize([
            row for row in selected if not _is_original(row) and row['label'] == 0])))
        groups.append((split.title() + ', 18–22% loss', summarize([row for row in selected if _near_threshold(row)])))
    return groups


def _plotting(directory, history, rows, initialization):
    """Write reusable PNGs and return their self-contained HTML embeds."""
    if not history.get('epochs') and not rows:
        return ''
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plots = directory / 'figures'
    plots.mkdir(exist_ok=True)
    sections = []

    def save(fig, name, caption):
        target = plots / name
        try:
            fig.savefig(target, dpi=130, bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)
        encoded = base64.b64encode(target.read_bytes()).decode('ascii')
        sections.append('<figure><img alt="' + _text(caption) + '" src="data:image/png;base64,' + encoded + '"><figcaption>' + _text(caption) + '</figcaption></figure>')

    epochs = history.get('epochs', [])
    if epochs:
        points = [{'epoch': 0, 'training': history.get('initial_training') or initialization.get('training'),
                   'validation': history.get('initial_validation') or initialization.get('validation')}] + epochs
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, key, title in zip(axes.ravel(), ('bce', 'accuracy', 'auroc', 'score_mae'),
                                  ('Binary cross-entropy', 'Accuracy at probability 0.5', 'AUROC', 'Local-volume-loss MAE (percentage points)')):
            for split, color in [('training', '#1976ad'), ('validation', '#c65927')]:
                valid = []
                for point in points:
                    metric = (point.get(split) or {}).get(key)
                    if metric is None and key == 'auroc':
                        predictions = point.get('predictions', [])
                        predictions = predictions or point.get('train_predictions' if split == 'training' else 'validation_predictions', [])
                        selected = _split_rows(predictions, split)
                        metric = (summarize(selected) or {}).get(key)
                    if metric is not None:
                        valid.append((point['epoch'], metric * (100 if key == 'score_mae' else 1)))
                if valid:
                    ax.plot([v[0] for v in valid], [v[1] for v in valid], 'o-', ms=3, color=color, label=split.title())
            ax.set(title=title, xlabel='Epoch (0 = initialization)')
            if key in ('accuracy', 'auroc'):
                ax.set_ylim(-.03, 1.03)
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8)
        save(fig, 'learning-curves.png', 'Learning curves. Validation selects the checkpoint; it is not an independent test set.')

    if rows:
        colors = {'train': '#1976ad', 'training': '#1976ad', 'validation': '#c65927', 'valid': '#c65927'}
        roles = [('original', 'o', 'Original negative'), ('edited_negative', 's', 'Edited negative'),
                 ('lower_positive', '^', 'Lower positive'), ('higher_positive', 'D', 'Higher positive')]
        for kind in ('probability', 'severity'):
            fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
            for split in ('training', 'validation'):
                selected = _split_rows(rows, split)
                for role, marker, label in roles:
                    group = [row for row in selected if row.get('role') == role]
                    if not group:
                        continue
                    ax.scatter([100 * row['score_label'] for row in group],
                               [row['probability'] if kind == 'probability' else 100 * row['score_prediction'] for row in group],
                               c=colors[split], marker=marker, s=55, edgecolors='white', linewidths=.6,
                               label=split.title() + ': ' + label, alpha=.85)
                other = [row for row in selected if row.get('role') not in {r[0] for r in roles}]
                if other:
                    ax.scatter([100 * row['score_label'] for row in other],
                               [row['probability'] if kind == 'probability' else 100 * row['score_prediction'] for row in other],
                               c=colors[split], marker='x', label=split.title() + ': other roles')
            ax.axvspan(18, 22, color='#e2b44b', alpha=.15, label='18–22% near threshold')
            ax.axvline(20, color='#777', linestyle=':', linewidth=1)
            if kind == 'probability':
                ax.axhline(.5, color='#777', linestyle='--', linewidth=1)
                ax.set_ylim(-.03, 1.03)
                ax.set_ylabel('Predicted CoA probability')
                caption = 'Probability against achieved local target volume loss. The vertical dotted line is the 20% label threshold; the horizontal dashed line is the 0.5 decision threshold.'
            else:
                upper = max([100 * max(row['score_label'], row['score_prediction']) for row in rows] + [25]) * 1.05
                lower = min([100 * min(row['score_label'], row['score_prediction']) for row in rows] + [0])
                ax.plot([lower, upper], [lower, upper], color='#777', linestyle='--', linewidth=1, label='Perfect prediction')
                ax.set_ylabel('Predicted local-volume-loss proxy (%)')
                caption = 'Severity-proxy parity. Scores describe achieved local target volume loss, not clinical severity or minimum lumen-area reduction.'
            ax.set_xlabel('Achieved local target volume loss (%)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='best', ncol=2)
            save(fig, kind + '-vs-achieved-loss.png', caption)
    return ''.join(sections)


def write_report(output_directory):
    """Render report.html from saved numeric artifacts, including plan-only runs.

    PNGs are saved under figures/ and embedded in HTML, so report.html is
    portable on its own. No experiment metadata is modified.
    """
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    lock = _read(out, 'experiment-lock.json')
    result = _read(out, 'results.json')
    history = _read(out, 'classifier-history.json')
    initialization = _read(out, 'initialization.json')
    prediction_file = _read(out, 'classifier-predictions.json')
    rows = result.get('predictions') or prediction_file.get('predictions') or []
    # While a run is active, show the latest completed epoch and label it as
    # provisional. Never silently substitute initialization for trained results.
    provisional = False
    if not rows and history.get('epochs'):
        latest = history['epochs'][-1]
        rows = latest.get('predictions') or (latest.get('train_predictions', []) + latest.get('validation_predictions', []))
        provisional = bool(rows)
    rows = _checked(rows)
    status = result.get('status', 'Training in progress' if history.get('epochs') else
                        'Predictions saved; final run status unavailable' if rows else
                        'Planned; classifier results not available')
    sections = ['<h1>CoA classifier: development evaluation</h1>',
                '<p class="status"><strong>' + _text(status) + '</strong></p>',
                '<p>This engineering panel is designed for <strong>32 training and 8 validation scenarios</strong> from 10 development phantoms. Siblings stay in the same phantom-family partition. The reserved test families are outside this report.</p>',
                '<p><strong>Validation selects the checkpoint; these are not held-out test results.</strong> Scenario counts do not equal independent anatomy counts. The class label uses <strong>20% achieved local target volume loss</strong> during the CoA edit. This is an engineering proxy, not a clinical diagnosis, clinical severity grade, or lumen-area threshold.</p>',
                '<p>Decisions use probability ≥ 0.5. Loss of 18–22% is highlighted for threshold sensitivity; these cases retain their original labels. Original negatives and mildly edited negatives are reported separately.</p>']
    if provisional:
        sections.append('<p class="notice">Predictions below are from the latest completed epoch, not a finalized selected checkpoint.</p>')
    selected_epoch = result.get('selected_epoch', prediction_file.get('selected_epoch'))
    checkpoint = lock.get('locator_checkpoint', {})
    if selected_epoch is not None:
        label = 'Selected classifier epoch' if result else 'Saved prediction checkpoint epoch (run not finalized)'
        sections.append('<p>' + label + ': <strong>' + _text(selected_epoch) + '</strong>.</p>')
    if checkpoint:
        sections.append('<p>Frozen locator epoch: ' + _text(checkpoint.get('selected_epoch', 'recorded in parent metadata')) + '. Locator checkpoint SHA-256: <code>' + _text(checkpoint.get('sha256', 'not recorded')) + '</code>.</p>')

    heading = ('Performance at the selected checkpoint' if result and selected_epoch is not None else
               'Latest completed epoch: provisional performance' if provisional else
               'Saved prediction performance' if rows else 'Awaiting classifier performance')
    sections.append('<h2>' + heading + '</h2>')
    metric_rows = []
    for group, metrics in _metrics_rows(rows, result):
        m = metrics or {}
        confusion = m.get('confusion_counts', {})
        metric_rows.append([group, m.get('samples', 0), _number(m.get('bce')),
                            _number(m.get('auroc', m.get('roc_auc'))), _number(m.get('accuracy')),
                            _number(m.get('balanced_accuracy')), _number(m.get('sensitivity')),
                            _number(m.get('specificity')), _number(m.get('precision')),
                            '/'.join(str(confusion.get(key, '—')) for key in ('TP', 'TN', 'FP', 'FN')),
                            _number(m.get('severity_mae_percentage_points')), _number(m.get('regression_rmse'))])
    sections.append(_table(['Group', 'N', 'BCE', 'AUROC', 'Accuracy', 'Balanced accuracy', 'Sensitivity',
                            'Specificity', 'Precision', 'TP/TN/FP/FN', 'Loss MAE (pp)', 'Loss RMSE (fraction)'], metric_rows))
    sections.append('<p class="muted">N/A denotes an undefined metric or an empty group. AUROC handles tied probabilities with average ranks. Regression targets and RMSE are fractions; MAE is also expressed in percentage points (pp).</p>')
    sections.append(_plotting(out, history, rows, initialization))

    if rows:
        errors = [row for row in rows if (row['probability'] >= .5) != bool(row['label'])]
        sections.append('<h2>False positives and false negatives</h2>')
        if errors:
            sections.append(_table(['Scenario', 'Split', 'Role', 'Error', 'Probability', 'Achieved loss (%)'], [
                [row['sample_id'], row.get('split', ''), row.get('role', ''),
                 'False negative' if row['label'] else 'False positive', _number(row['probability']),
                 _number(100 * row['score_label'], 2)] for row in errors]))
        else:
            sections.append('<p>No misclassifications at probability 0.5 in these development predictions.</p>')
        sections.append('<h2>Every development scenario</h2>')
        table_rows, classes = [], []
        for row in sorted(rows, key=lambda item: (item.get('split', ''), str(item.get('case_id', '')), item['score_label'])):
            near = _near_threshold(row)
            classes.append('near' if near else '')
            table_rows.append([row['sample_id'], row.get('case_id', ''), row.get('family', ''),
                               row.get('split', ''), row.get('role', ''), row['label'],
                               _number(row['probability']), int(row['probability'] >= .5),
                               _number(100 * row['score_label'], 2), _number(100 * row['score_prediction'], 2),
                               _number(100 * abs(row['score_prediction'] - row['score_label']), 2), '18–22%' if near else ''])
        sections.append(_table(['Scenario', 'Phantom', 'Family', 'Split', 'Role', 'Label', 'Probability',
                                'Predicted class', 'Achieved loss (%)', 'Predicted loss (%)', 'Absolute error (pp)', 'Near threshold'], table_rows, classes))
    else:
        sections.append('<p>No completed classifier predictions are available yet.</p>')

    for name, title in [('initialization.json', 'Initialization evaluation'), ('results.json', 'Saved numerical results'),
                        ('failure.json', 'Run failure'), ('report-errors.json', 'Earlier report errors')]:
        value = _read(out, name)
        if value:
            if name == 'results.json':
                value = {key: item for key, item in value.items() if key != 'predictions'}
            sections.append('<details><summary>' + _text(title) + '</summary><pre>' + _text(json.dumps(value, indent=2)) + '</pre></details>')
    frozen = {key: lock[key] for key in ('config', 'parent_config', 'locator_checkpoint', 'pilot_counts',
                                         'schema_version', 'parent_lock_sha256') if key in lock}
    if frozen:
        sections.append('<details><summary>Frozen experiment provenance</summary><pre>' + _text(json.dumps(frozen, indent=2)) + '</pre></details>')
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoA classifier development evaluation</title><style>body{font:16px system-ui,sans-serif;line-height:1.5;margin:2rem auto;max-width:1250px;padding:0 1rem;color:#172434}h1,h2{line-height:1.2}h2{margin-top:2rem}.status,.notice{background:#eaf1f7;padding:1rem;border-left:4px solid #1976ad}.muted,figcaption{color:#506174;font-size:.9rem}.table-scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:.5rem;border:1px solid #ccd6df;text-align:left;vertical-align:top}th{background:#eaf1f7}.near{background:#fff3cb}img{max-width:100%;height:auto}figure{margin:1.5rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f2f5f7;padding:1rem}code{overflow-wrap:anywhere}details{margin:1rem 0}summary{cursor:pointer}</style></head><body>' + ''.join(sections) + '</body></html>'
    path = out / 'report.html'
    path.write_text(document, encoding='utf-8')
    return str(path)
