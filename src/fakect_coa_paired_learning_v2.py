"""Balanced same-weight sibling updates inside immutable locator crops.

The diagnostic uses matched mild/severe siblings from training families only.
It answers whether local image evidence can be learned; it cannot establish
generalization. A later fresh development fit requires that diagnostic to pass.
"""
from collections import OrderedDict
import gc
import html
import json
from pathlib import Path
import tempfile
import time

import numpy as np

import fakect_coa_paired_experiment_v2 as exp
import fakect_multiscale_experiment_v3 as geometry
import fakect_multiscale_learning_v3 as legacy
from fakect_context_volume import ContextVolume
from fakect_coa_classifier_learning import (checkpoint_prediction_parity,
    learning_gate, verify_reviewed_proposals, weight_fingerprint)
from fakect_coa_classifier_report import summarize


class FineInputs:
    """Bounded LRU cache of normalized image/validity crops, never targets.

    Four diagnostic samples occupy about 486 MiB at the current geometry. The
    complete panel is streamed when it exceeds the INI cache budget. No learned
    features, whole-body sibling images or class-guided crop locations are cached.
    """
    def __init__(self, runtime, limit_mb):
        if isinstance(limit_mb, bool) or not isinstance(limit_mb, (int, float)) or limit_mb <= 0:
            raise ValueError('Image cache limit must be positive')
        self.runtime = runtime
        self.limit_bytes = int(limit_mb * 1024 ** 2)
        self.cache = OrderedDict()
        self.bytes = 0
        self.peak_bytes = 0
        self.loads = 0
        self.hits = 0

    def classifier(self, row, proposal, *, cache=True):
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Focused classifier cannot read reserved test samples')
        detail = self.runtime['input.detail']
        origins = np.asarray(proposal['origins_kji'])
        count = detail['proposals']
        if origins.shape != (count, 3) or not np.issubdtype(origins.dtype, np.integer):
            raise ValueError('Fine inputs require exact reviewed integer crop origins')
        key = (row['sample_id'], tuple(int(v) for v in origins.ravel()))
        if cache and key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        needed = count * int(np.prod(detail['shape_kji'])) * 2 * np.dtype(np.float32).itemsize
        if needed > self.limit_bytes:
            raise ValueError('A single fine image sample exceeds the configured cache budget')
        # Make room before allocating the new sample, including uncached audit
        # crops. RuntimeGuard separately measures all live/transient allocations.
        while self.cache and self.bytes + needed > self.limit_bytes:
            _, old = self.cache.popitem(last=False)
            self.bytes -= sum(value.nbytes for value in old.values())
        volume = ContextVolume(row, verify_hashes=True)
        values = {}
        try:
            for index, origin in enumerate(origins):
                block = volume.read_block(origin, detail['shape_kji'], detail['spacing_ijk_mm'])
                normalized = geometry.normalized_input(block['image'], block['valid'], self.runtime)[None]
                if normalized.dtype != np.float32 or not np.isfinite(normalized).all():
                    raise ValueError('Invalid normalized fine image')
                normalized.flags.writeable = False
                values['detail_' + str(index)] = normalized
                del block
        finally:
            volume.close()
        observed = sum(value.nbytes for value in values.values())
        if observed != needed:
            raise RuntimeError('Normalized fine input size disagrees with the image-cache estimate')
        self.loads += 1
        self.peak_bytes = max(self.peak_bytes, self.bytes + observed)
        if cache:
            self.cache[key] = values
            self.bytes += observed
        return values

    def stats(self):
        return {'limit_mib': self.limit_bytes / 1024 ** 2,
            'resident_mib': self.bytes / 1024 ** 2, 'peak_mib': self.peak_bytes / 1024 ** 2,
            'samples_resident': len(self.cache), 'loads': self.loads, 'hits': self.hits,
            'payload': 'normalized attenuation plus image validity only; no target or learned-feature cache'}


def diagnostic_gate(c, records):
    """Require correct, separated mild/severe predictions in every train case."""
    rows = list(records)
    settings = c['diagnostic']
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('The memorization diagnostic requires training samples only')
    cases = sorted({str(row['case_id']) for row in rows})
    expected = sorted(str(case) for case in settings['case_ids'])
    if cases != expected or len(rows) != 2 * len(cases):
        raise ValueError('The diagnostic requires one matched binary pair per configured training case')
    pairs = []
    for case in cases:
        pair = [row for row in rows if str(row['case_id']) == case]
        if sorted(row['label'] for row in pair) != [0, 1]:
            raise ValueError('Each diagnostic training case needs both binary classes')
        negative = next(row for row in pair if not row['label'])
        positive = next(row for row in pair if row['label'])
        pairs.append({'case_id': case, 'negative_sample_id': negative['sample_id'],
            'positive_sample_id': positive['sample_id'],
            'bce': summarize(pair)['bce'],
            'probability_gap': float(positive['probability'] - negative['probability'])})
    metrics = summarize(rows)
    checks = {'sensitivity_equal_one': metrics['sensitivity'] == 1.,
        'specificity_equal_one': metrics['specificity'] == 1.,
        'bce_at_most_maximum': metrics['bce'] <= settings['maximum_bce'],
        'pooled_probability_gap_at_least_minimum': (
            metrics['positive_probability_mean'] - metrics['negative_probability_mean']
            >= settings['minimum_probability_gap']),
        'every_case_probability_gap_at_least_minimum': all(
            pair['probability_gap'] >= settings['minimum_probability_gap'] for pair in pairs),
        'every_case_bce_at_most_maximum': all(
            pair['bce'] <= settings['maximum_bce'] for pair in pairs)}
    return {'passed': all(checks.values()), 'checks': checks, 'matched_pairs': pairs,
        'maximum_bce': settings['maximum_bce'],
        'minimum_probability_gap': settings['minimum_probability_gap'], 'decision_threshold': .5,
        'scope': 'Training-only matched-sibling memorization diagnostic; no generalization claim.'}


def pair_rows(rows):
    """Prespecified negative/positive pairs within one case and partition.

    The diagnostic pairs the edited negative with its higher-positive sibling.
    The full development panel adds original/lower-positive pairs. These are
    training/evaluation batches only; each image has its own scalar prediction.
    """
    rows = list(rows)
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Sibling batching requires a nonempty unique sample panel')
    if any(row['split'] not in ('train', 'validation') for row in rows):
        raise ValueError('Sibling batching cannot consume reserved test samples')
    pairs = []
    for case_id in sorted({str(row['case_id']) for row in rows}):
        group = [row for row in rows if str(row['case_id']) == case_id]
        if len({row['split'] for row in group}) != 1 or len({row['anatomy_family'] for row in group}) != 1:
            raise ValueError('Sibling pairs must share one anatomy case, family and partition')
        roles = {row['pilot_role']: row for row in group}
        if len(roles) != len(group):
            raise ValueError('Duplicate sibling roles cannot define a balanced pair')
        ordered = [('edited_negative', 'higher_positive')]
        if set(roles) == {'original', 'edited_negative', 'lower_positive', 'higher_positive'}:
            ordered.insert(0, ('original', 'lower_positive'))
        elif set(roles) != {'edited_negative', 'higher_positive'}:
            raise ValueError('Sibling batching requires the prescribed two or four case roles')
        for negative_role, positive_role in ordered:
            negative, positive = roles[negative_role], roles[positive_role]
            if (negative['annotation']['coa_benchmark_positive'] is not False or
                    positive['annotation']['coa_benchmark_positive'] is not True):
                raise ValueError('Every sibling batch must contain one negative and one positive')
            if negative_role == 'edited_negative' and (
                    negative.get('pilot_companion_parameters') != positive.get('pilot_companion_parameters') or
                    {key: value for key, value in negative['parameters'].items() if key != 'coa_narrow'} !=
                    {key: value for key, value in positive['parameters'].items() if key != 'coa_narrow'}):
                raise ValueError('Matched edited sibling batches must have identical companion edits')
            pairs.append((negative, positive))
    return pairs


def pair_inputs(inputs, negative, positive, proposals):
    """Concatenate two independent image crops; no image differences enter."""
    first = inputs.classifier(negative, proposals[negative['sample_id']])
    second = inputs.classifier(positive, proposals[positive['sample_id']])
    if set(first) != set(second):
        raise ValueError('Paired image dictionaries must share their fine input keys')
    return {key: np.concatenate((first[key], second[key]), axis=0) for key in first}


def _score(tf, steps, rows, inputs, proposals, guard):
    records = []
    for pair in pair_rows(rows):
        guard.check('focused-evaluation-before', steps=steps)
        values = legacy._tensor_inputs(tf, pair_inputs(inputs, *pair, proposals))
        output = steps.predict(values)
        logits = np.asarray(output['coa_logit']).ravel()
        scores = np.asarray(output['local_volume_loss']).ravel()
        if logits.shape != (2,) or scores.shape != (2,) or not np.isfinite([logits, scores]).all():
            raise ValueError('Nonfinite focused classifier prediction')
        for index, row in enumerate(pair):
            logit, score = float(logits[index]), float(scores[index])
            records.append({'sample_id': row['sample_id'], 'case_id': row['case_id'],
                'family': row['anatomy_family'], 'split': row['split'], 'role': row['pilot_role'],
                'label': int(row['annotation']['coa_benchmark_positive']), 'logit': logit,
                'probability': float(1 / (1 + np.exp(-np.clip(logit, -80, 80)))),
                'score_label': float(row['annotation'][exp.SCORE]), 'score_prediction': score,
                'near_threshold': bool(row['annotation'].get('review_near_threshold', False))})
        guard.check('focused-evaluation-after', steps=steps)
    return records


def difference_summary(before, after):
    """Compact finite difference measures for inputs, maps or pooled vectors."""
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Sibling sensitivity requires finite equal-shaped arrays')
    difference = np.subtract(after, before, dtype=np.float32)
    return {'shape': list(before.shape),
        'rms_difference': float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
        'maximum_absolute_difference': float(np.max(np.abs(difference))),
        'different_fraction': float(np.mean(difference != 0)),
        'reference_rms': float(np.sqrt(np.mean(before.astype(np.float64) ** 2)))}


def sibling_sensitivity(network, classifier, rows, inputs, proposals, guard):
    """Trace same-case signals on both inference crops and one fixed image grid.

    The fixed grid is the negative sibling's IMAGE-PREDICTED crop geometry, used
    only for this explanatory audit. It cannot influence training, checkpoint
    selection or inference. Comparison on the actual per-sibling grids separately
    reveals whether locator movement changes the framing.
    """
    probe = network.feature_probe(classifier)
    results = []
    for case in sorted({str(row['case_id']) for row in rows}):
        group = [row for row in rows if str(row['case_id']) == case]
        negatives = [row for row in group if row['pilot_role'] == 'edited_negative']
        positives = [row for row in group if row['pilot_role'] == 'higher_positive']
        if len(negatives) != 1 or len(positives) != 1:
            continue
        negative, positive = negatives[0], positives[0]
        pn, pp = proposals[negative['sample_id']], proposals[positive['sample_id']]
        xn, xp = inputs.classifier(negative, pn), inputs.classifier(positive, pp)
        actual = {key: {'attenuation': difference_summary(xn[key][..., 0], xp[key][..., 0]),
            'validity': difference_summary(xn[key][..., 1], xp[key][..., 1])} for key in xn}
        origin_difference = np.asarray(pp['origins_kji']) - np.asarray(pn['origins_kji'])
        # If origins coincide, avoid duplicating 128 MiB unnecessarily.
        aligned = xp if not np.any(origin_difference) else inputs.classifier(positive, pn, cache=False)
        common = {key: {'attenuation': difference_summary(xn[key][..., 0], aligned[key][..., 0]),
            'validity': difference_summary(xn[key][..., 1], aligned[key][..., 1])} for key in xn}
        guard.check('sibling-feature-audit-before')
        a = probe(xn, training=False)
        b = probe(aligned, training=False)
        layers = {key: difference_summary(a[key], b[key]) for key in a}
        guard.check('sibling-feature-audit-after')
        results.append({'case_id': case, 'negative_sample_id': negative['sample_id'],
            'positive_sample_id': positive['sample_id'],
            'inference_origin_difference_kji': origin_difference.tolist(),
            'inference_crop_differences': actual, 'same_grid_input_differences': common,
            'same_grid_feature_differences': layers})
        del xn, xp, aligned, a, b
    return {'pairs': results, 'mask_guided': False, 'test_evaluated': False,
        'same_grid_scope': 'Audit only: both siblings sampled at the negative sibling image-predicted origins.',
        'interpretation': 'Nonzero differences show signal survival, not successful disease recognition or causal attribution.'}


def write_report(output_directory):
    """Reuse the established plots, with accurate focused-stage scope text."""
    from fakect_coa_classifier_report import write_report as existing_report
    out = Path(output_directory)
    if out.name not in ('diagnostic', 'pilot'):
        lock_path = out / 'experiment-lock.json'
        lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        sections = ['<h1>Paired CoA classifier</h1>',
            '<p>The frozen locator supplies two 0.6-mm attenuation crops. The new classifier '
            'retains a stride-four spatial feature map, local evidence and ordered spatial bins. '
            'Predictor channels contain attenuation and image validity only.</p>',
            '<p>This revision averages the unchanged binary cross-entropy and severity loss '
            'over a matched negative/positive batch at one set of model weights, then clips '
            'the averaged gradient and performs one optimizer update. Architecture, seed '
            'and learning rate match the preceding single-image diagnostic.</p>',
            '<p>The first stage trains matched mild/severe siblings from training families. '
            'Its checkpoint is selected using training BCE. A subsequent fresh development '
            'fit is allowed only after that feasibility diagnostic passes. Reserved test images remain separate.</p>']
        for stage in ('diagnostic', 'pilot'):
            folder = out / stage
            result_path = folder / 'results.json'
            history_path = folder / 'classifier-history.json'
            failure_path = folder / 'failure.json'
            status = 'Not started'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                status = result['status'] + ' · selected epoch ' + str(result['selected_epoch'])
            elif failure_path.exists():
                status = 'Failed; see saved stage failure'
            elif history_path.exists():
                history = json.loads(history_path.read_text())
                status = 'In progress · ' + str(len(history.get('epochs', []))) + ' completed epochs'
            sections.append('<h2>' + stage.title() + '</h2><p>' + html.escape(status) + '</p>')
            if (folder / 'report.html').exists():
                sections.append('<p><a href="' + stage + '/report.html">Open ' + stage + ' report</a></p>')
        for filename, title in [('runtime-smoke.json', 'Full-size synthetic GPU smoke'),
                                ('failure.json', 'Root-stage failure')]:
            source = out / filename
            if source.exists():
                sections.append('<details><summary>' + title + '</summary><pre>' +
                    html.escape(source.read_text()) + '</pre></details>')
        sections.append('<details><summary>Frozen input and diagnostic counts</summary><pre>' + html.escape(
            json.dumps({key: lock[key] for key in ('config', 'diagnostic_counts', 'training_contract')
                if key in lock}, indent=2)) + '</pre></details>')
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'report.html'
        path.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Paired CoA classifier</title><style>body{font:16px system-ui;max-width:1100px;'
            'margin:2rem auto;padding:1rem;line-height:1.5}pre{white-space:pre-wrap;overflow-wrap:anywhere;'
            'font-size:12px;background:#f2f5f7;padding:1rem}summary{cursor:pointer}</style><body>' +
            ''.join(sections) + '</body></html>')
        return str(path)
    path = Path(existing_report(out))
    document = path.read_text()
    diagnostic = out.name == 'diagnostic'
    heading = 'Paired CoA classifier: ' + ('training diagnostic' if diagnostic else 'development evaluation')
    document = document.replace('CoA classifier: development evaluation', heading)
    lock_path = out / 'experiment-lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    counts = lock.get('diagnostic_counts', {})
    sample_count, case_count = counts.get('samples', 'configured'), counts.get('cases', 'configured')
    panel = (f'This diagnostic uses <strong>{sample_count} edited training siblings from {case_count} training phantoms</strong>, '
             'with matched companion edits and one mild/severe pair per phantom. It measures training '
             'memorization only; validation and reserved test images are not loaded.' if diagnostic else
             'This development panel uses <strong>32 training and 8 validation scenarios</strong>. '
             'A passed training-only diagnostic authorizes this fresh classifier fit; reserved test families remain separate.')
    original = ('This engineering panel is designed for <strong>32 training and 8 validation scenarios</strong> '
                'from 10 development phantoms. Siblings stay in the same phantom-family partition. '
                'The reserved test families are outside this report.')
    document = document.replace(original, panel)
    document = document.replace('</h1>', '</h1><p>Each optimizer update averages a same-case '
        'negative/positive batch at the same model weights before clipping. Individual image '
        'predictions use attenuation and validity only; sibling differences are explanatory '
        'audits and do not enter the predictor.</p>', 1)
    if diagnostic:
        document = document.replace('Validation selects the checkpoint; these are not held-out test results.',
            'Training BCE selects the diagnostic checkpoint; these are memorized training cases.')
        document = document.replace('Validation selects the checkpoint; it is not an independent test set.',
            'Training BCE selects this diagnostic checkpoint; no validation or test cases are loaded.')
    audits = []
    for filename, title in [('runtime-smoke.json', 'Full-size GPU smoke'),
        ('continuation-audit.json', 'Image-only input and frozen-locator audit'),
        ('sibling-sensitivity-initial.json', 'Signal differences at initialization'),
        ('sibling-sensitivity-selected.json', 'Signal differences at the selected checkpoint')]:
        source = out / filename
        if source.exists():
            audits.append('<details><summary>' + html.escape(title) + '</summary><pre>' +
                html.escape(source.read_text()) + '</pre></details>')
    document = document.replace('</body>', ''.join(audits) + '</body>')
    path.write_text(document)
    return str(path)


def _render(out):
    try:
        write_report(out)
        if Path(out).name in ('diagnostic', 'pilot'):
            write_report(Path(out).parent)
    except Exception as error:
        exp.write_json(out / 'report-errors.json', {'report_generated': False,
            'numeric_results_preserved': True, 'error_type': type(error).__name__, 'error': str(error)})
        print(json.dumps({'stage': 'report', 'status': 'failed', 'error': str(error)}), flush=True)


def _save_sensitivity(out, filename, network, classifier, rows, inputs, proposals, guard):
    """Keep explanatory feature audits separate from durable learning metrics."""
    try:
        value = sibling_sensitivity(network, classifier, rows, inputs, proposals, guard)
        value['status'] = 'completed'
    except RuntimeError:
        # A RuntimeGuard violation is a training resource failure, not an
        # optional visualization failure. It must stop the experiment.
        raise
    except guard.tf.errors.ResourceExhaustedError:
        raise
    except Exception as error:
        value = {'status': 'audit_failed', 'numeric_learning_results_preserved': True,
            'error_type': type(error).__name__, 'error': str(error), 'test_evaluated': False}
        print(json.dumps({'stage': 'sibling-sensitivity', **value}), flush=True)
    exp.write_json(out / filename, value)


def smoke(c, lock):
    """Full-size synthetic GPU forward/backward and checkpoint safety check."""
    import fakect_coa_paired_steps as network
    root = Path(c['experiment']['output_directory'])
    input_bytes = (root / 'input.ini').read_bytes()
    lock = exp.inspect_lock(c, input_bytes)
    runtime = exp.runtime_config(c, lock)
    tf, _ = legacy.configure_tensorflow(runtime)
    guard = legacy.RuntimeGuard(runtime, tf)
    shape = runtime['input.detail']['shape_kji']
    count = runtime['input.detail']['proposals']
    model = network.build_classifier(shape, num_proposals=count,
        base_filters=c['model']['base_filters'], seed=c['training']['seed'])
    steps = network.make_steps(model, shape, num_proposals=count,
        learning_rate=c['training']['learning_rate'], regression_weight=c['model']['regression_weight'])
    values = {}
    for index in range(count):
        attenuation = tf.random.stateless_uniform((2, *shape, 1), seed=(c['training']['seed'], index))
        values['detail_' + str(index)] = tf.concat((attenuation, tf.ones_like(attenuation)), axis=-1)
    samples = []
    for iteration in range(8):
        guard.check('focused-smoke-before', steps=steps)
        target = {'coa_logit': tf.constant([[0.], [1.]], tf.float32),
            'local_volume_loss': tf.constant([[.1], [.4]], tf.float32)}
        result = steps.train(values, target)
        steps.predict(values)
        steps.assert_stable_traces()
        sample = {'step': iteration + 1, 'loss': float(result['loss']),
            'gradient_global_norm': float(result['gradient_global_norm']),
            **guard.check('focused-smoke-after', steps=steps, save=True)}
        if not np.isfinite([sample['loss'], sample['gradient_global_norm']]).all():
            raise ValueError('Nonfinite full-size focused smoke')
        samples.append(sample)
        print(json.dumps({'stage': 'focused-smoke', **sample}), flush=True)
    if not any(row['gradient_global_norm'] > 0 for row in samples):
        raise ValueError('Full-size focused smoke produced no classifier gradient')
    growth = samples[-1]['rss_mib'] - samples[3]['rss_mib']
    if growth > 512:
        raise RuntimeError('Focused smoke RSS grew more than 512 MiB after warmup')
    before = steps.predict(values)
    with tempfile.TemporaryDirectory(prefix='fakect-focused-smoke-', dir=str(root)) as directory:
        path = Path(directory) / 'smoke.keras'
        model.save(path)
        restored = network.load_model(path)
        after = restored(values, training=False)
        for key in before:
            np.testing.assert_allclose(before[key], after[key], atol=1e-6, rtol=1e-5)
    exp.inspect_lock(c, input_bytes)
    result = {'status': 'passed', 'synthetic_data_only': True, 'checkpoint_parity': True,
        'source_lock_sha256': exp.sha256(root / 'experiment-lock.json'),
        'detail_shape_kji': list(shape), 'proposals': count, 'batch_size': 2, 'samples': samples,
        'traces': steps.tracing_counts(),
        'classifier_traces': steps.tracing_counts(), 'training_traces': steps.tracing_counts(),
        'tracecounts': steps.tracing_counts(), 'warmup_rss_growth_mib': growth,
        'tensorflow_version': tf.__version__, 'gpu_visible': bool(tf.config.list_logical_devices('GPU')),
        **legacy.memory_snapshot(tf)}
    exp.write_json(root / 'runtime-smoke.json', result)
    return result


def fit(c, lock, *, diagnostic=True):
    """Run a matched-sibling diagnostic or an independently initialized fit."""
    import fakect_coa_paired_steps as network
    root = Path(c['experiment']['output_directory'])
    input_bytes = (root / 'input.ini').read_bytes()
    lock = exp.inspect_lock(c, input_bytes)
    exp.require_smoke(c, lock)
    dependency = None if diagnostic else exp.require_diagnostic(c, lock)
    stage = 'diagnostic' if diagnostic else 'pilot'
    out = root / stage
    out.mkdir(exist_ok=True)
    model_dir = out / 'model'
    model_dir.mkdir(exist_ok=False)
    exp.write_json(out / 'experiment-lock.json', lock)
    exp.write_json(out / 'stage.json', {'stage': stage, 'source_lock_sha256': exp.sha256(root / 'experiment-lock.json')})
    if dependency is not None:
        exp.write_json(out / 'diagnostic-dependency.json', dependency)
    runtime = exp.runtime_config(c, lock)
    runtime['experiment']['output_directory'] = str(out)
    selected_ids = set(lock['diagnostic_sample_ids']) if diagnostic else {row['sample_id'] for row in lock['samples']}
    rows = [row for row in lock['samples'] if row['sample_id'] in selected_ids]
    if len(rows) != len(selected_ids) or any(row['split'] not in ('train', 'validation') for row in rows):
        raise ValueError('Selected focused samples do not join the frozen development panel')
    train = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'validation']
    if not train or (diagnostic and validation) or (not diagnostic and not validation):
        raise ValueError('Invalid focused diagnostic/development sample partition')
    if diagnostic and any(row['pilot_role'] not in ('edited_negative', 'higher_positive') for row in train):
        raise ValueError('Diagnostic may train only the prescribed matched edited siblings')
    train_pairs = pair_rows(train)
    checkpoint = lock['locator_checkpoint']
    if exp.sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
        raise ValueError('Locator checkpoint changed before focused training')
    started = time.monotonic()
    tf, locator_network = legacy.configure_tensorflow(runtime)
    guard = legacy.RuntimeGuard(runtime, tf)
    guard.check('focused-configured', save=True)
    locator = locator_network.load_model(checkpoint['path'])
    locator.trainable = False
    locator_before = weight_fingerprint(locator)
    overview_inputs = legacy.Inputs(runtime)
    reviewed = {row['sample_id']: lock['proposals'][row['sample_id']] for row in rows}
    proposals, replay = verify_reviewed_proposals(locator, locator_network, rows, overview_inputs,
        reviewed, runtime, guard)
    exp.write_json(out / 'proposal-replay.json', replay)
    overview_inputs.cache.clear()
    del overview_inputs
    gc.collect()
    inputs = FineInputs(runtime, c['diagnostic']['cache_limit_mb'])
    seed = c['training']['seed']
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    detail = runtime['input.detail']
    classifier = network.build_classifier(detail['shape_kji'], num_proposals=detail['proposals'],
        base_filters=c['model']['base_filters'], seed=seed)
    initial_weight_hash = weight_fingerprint(classifier)
    reference_initial = lock.get('reference_initial_classifier_weights_sha256')
    if reference_initial is not None and initial_weight_hash != reference_initial:
        raise ValueError('Paired control must reproduce the preceding diagnostic classifier initialization')
    settings = c['diagnostic'] if diagnostic else c['training']
    steps = network.make_steps(classifier, detail['shape_kji'], num_proposals=detail['proposals'],
        learning_rate=settings['learning_rate'], regression_weight=c['model']['regression_weight'])
    provenance = {'stage': stage, 'locator_checkpoint': checkpoint, 'locator_retrained': False,
        'source_lock_sha256': exp.sha256(root / 'experiment-lock.json'),
        'locator_weights_before_sha256': locator_before, 'initialization_seed': seed, 'shuffle_seed': seed,
        'initial_classifier_weights_sha256': initial_weight_hash,
        'reference_initial_classifier_weights_sha256': reference_initial,
        'same_classifier_initialization_as_reference': (
            initial_weight_hash == reference_initial if reference_initial is not None else None),
        'random_state_policy': 'Fresh classifier weights and independent NumPy shuffle generator.',
        'input_contract': 'Two 0.6-mm normalized attenuation and validity crops; no overview, crop coordinates or segmentation-mask predictor channels.',
        'maximum_encoder_stride': 4, 'training_sample_count': len(train),
        'batch_size': 2, 'training_pairs_per_epoch': len(train_pairs),
        'update_contract': 'Same-case negative and positive images at the same model weights; average BCE and Huber before one gradient clipping and Adam update.',
        'loss_contract': 'Unchanged BCE plus regression_weight times Huber(delta=0.1); no ranking loss.',
        'validation_sample_count': len(validation), 'test_evaluated': False,
        'proposal_replay_passed': True, 'checkpoint_selection': (
            'Minimum training binary cross entropy; training-only diagnostic.' if diagnostic else
            'Minimum validation binary cross entropy; fresh initialization after a passed diagnostic.'),
        'tensorflow_version': tf.__version__}
    exp.write_json(out / 'continuation-audit.json', provenance)
    initial_records = _score(tf, steps, train + validation, inputs, proposals, guard)
    initial_train = [row for row in initial_records if row['split'] == 'train']
    initial_validation = [row for row in initial_records if row['split'] == 'validation']
    exp.write_json(out / 'initialization.json', {'training': summarize(initial_train),
        'validation': summarize(initial_validation), 'predictions': initial_records,
        'seed': seed, 'weights_sha256': provenance['initial_classifier_weights_sha256']})
    _save_sensitivity(out, 'sibling-sensitivity-initial.json', network, classifier,
        train, inputs, proposals, guard)
    history, best, best_epoch = [], float('inf'), None
    for epoch in range(settings['epochs']):
        losses, norms, bces, hubers, order, pair_order = [], [], [], [], [], []
        for index in rng.permutation(len(train_pairs)):
            pair = train_pairs[int(index)]
            pair_ids = [row['sample_id'] for row in pair]
            order.extend(pair_ids)
            pair_order.append(pair_ids)
            guard.check('focused-train-before', steps=steps)
            values = legacy._tensor_inputs(tf, pair_inputs(inputs, *pair, proposals))
            target = legacy._tensor_inputs(tf, {
                'coa_logit': np.array([[row['annotation']['coa_benchmark_positive']] for row in pair], np.float32),
                'local_volume_loss': np.array([[row['annotation'][exp.SCORE]] for row in pair], np.float32)})
            value = steps.train(values, target)
            losses.append(float(value['loss']))
            norms.append(float(value['gradient_global_norm']))
            bces.append(float(value['binary_crossentropy']))
            hubers.append(float(value['huber']))
            guard.check('focused-train-after', steps=steps)
        del values, target, value
        if weight_fingerprint(locator) != locator_before or locator.trainable_variables:
            raise RuntimeError('Frozen locator changed during focused classifier training')
        records = _score(tf, steps, train + validation, inputs, proposals, guard)
        train_records = [row for row in records if row['split'] == 'train']
        validation_records = [row for row in records if row['split'] == 'validation']
        selection = summarize(train_records if diagnostic else validation_records)['bce']
        record = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
            'train_binary_crossentropy': float(np.mean(bces)), 'train_huber': float(np.mean(hubers)),
            'gradient_norm_mean': float(np.mean(norms)), 'training': summarize(train_records),
            'validation': summarize(validation_records), 'checkpoint_selection_bce': selection,
            'training_sample_order': order, 'training_pair_order': pair_order,
            'batch_size': 2, 'optimizer_updates_this_epoch': len(train_pairs),
            'predictions': records, 'image_cache': inputs.stats(),
            'runtime': guard.check('focused-epoch', steps=steps, save=True)}
        if not np.isfinite([record['train_loss'], selection]).all():
            raise ValueError('Nonfinite focused classifier learning')
        if selection < best:
            best, best_epoch = selection, epoch + 1
            classifier.save(model_dir / 'classifier.keras')
        record['selected_checkpoint_epoch_so_far'] = best_epoch
        history.append(record)
        exp.write_json(out / 'classifier-history.json', {'initial_training': summarize(initial_train),
            'initial_validation': summarize(initial_validation), 'epochs': history, 'selected_epoch': best_epoch,
            'checkpoint_selection': provenance['checkpoint_selection']})
        _render(out)
        print(json.dumps({'stage': stage, **{key: val for key, val in record.items()
            if key not in ('predictions', 'training_sample_order', 'training_pair_order')}}), flush=True)
    trace_summary = {'classifier': steps.tracing_counts()}
    optimizer_steps = int(np.asarray(steps.optimizer.iterations))
    final_weight_hash = weight_fingerprint(classifier)
    if final_weight_hash == provenance['initial_classifier_weights_sha256']:
        raise RuntimeError('Focused classifier weights did not change')
    del steps, classifier
    gc.collect()
    classifier = network.load_model(model_dir / 'classifier.keras')
    selected = network.make_steps(classifier, detail['shape_kji'], num_proposals=detail['proposals'],
        learning_rate=settings['learning_rate'], regression_weight=c['model']['regression_weight'])
    records = _score(tf, selected, train + validation, inputs, proposals, guard)
    parity = checkpoint_prediction_parity(history[best_epoch - 1]['predictions'], records)
    training_records = [row for row in records if row['split'] == 'train']
    validation_records = [row for row in records if row['split'] == 'validation']
    gate = diagnostic_gate(c, training_records) if diagnostic else learning_gate(initial_train, training_records)
    trace_summary['selected_classifier'] = selected.tracing_counts()
    checkpoint_hash = exp.sha256(model_dir / 'classifier.keras')
    exp.write_json(out / 'classifier-predictions.json', {'selected_epoch': best_epoch,
        'checkpoint_sha256': checkpoint_hash, 'source_lock_sha256': provenance['source_lock_sha256'],
        'predictions': records, 'test_evaluated': False})
    unchanged = weight_fingerprint(locator) == locator_before and not locator.trainable_variables
    if not unchanged:
        raise RuntimeError('Locator changed during focused classifier learning')
    provenance.update(locator_weights_after_sha256=weight_fingerprint(locator), locator_unchanged=True,
        selected_epoch=best_epoch, selected_classifier_checkpoint_sha256=checkpoint_hash,
        optimizer_steps=optimizer_steps, training_traces=trace_summary, image_cache=inputs.stats(),
        selected_checkpoint_prediction_parity=parity)
    exp.write_json(out / 'continuation-audit.json', provenance)
    result = {'stage': stage, 'status': ('diagnostic_passed' if gate['passed'] else 'diagnostic_failed')
        if diagnostic else ('learning_check_passed' if gate['passed'] else 'learning_check_failed'),
        'source_lock_sha256': provenance['source_lock_sha256'], 'classifier_trained': True,
        'locator_retrained': False, 'locator_unchanged': True, 'test_evaluated': False,
        'selected_epoch': best_epoch, 'epochs_completed': len(history), 'learning_gate': gate,
        'training': summarize(training_records), 'validation': summarize(validation_records),
        'training_edited_only': summarize([row for row in training_records if row['role'] != 'original']),
        'validation_edited_only': summarize([row for row in validation_records if row['role'] != 'original']),
        'initial_training': summarize(initial_train), 'initial_validation': summarize(initial_validation),
        'predictions': records, 'checkpoint_sha256': {'locator.keras': checkpoint['sha256'],
            'classifier.keras': checkpoint_hash}, 'selected_checkpoint_prediction_parity': parity,
        'training_traces': trace_summary, 'optimizer_steps': optimizer_steps,
        'batch_size': 2, 'training_pairs_per_epoch': len(train_pairs),
        'sample_presentations': optimizer_steps * 2,
        'checkpoint_selection': provenance['checkpoint_selection'], 'image_cache': inputs.stats(),
        'elapsed_seconds': time.monotonic() - started, 'tensorflow_version': tf.__version__,
        **legacy.memory_snapshot(tf), 'severity_definition': 'Achieved local CoA volume-loss fraction; engineering proxy.',
        'classification_decision_threshold': .5, 'maximum_encoder_stride': 4}
    guard.check('focused-complete', steps=selected, save=True)
    exp.inspect_lock(c, input_bytes)
    if dependency is not None and exp.require_diagnostic(c, lock) != dependency:
        raise ValueError('Diagnostic artifacts changed during the subsequent development fit')
    if exp.sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
        raise ValueError('Locator checkpoint changed during focused learning')
    exp.write_json(out / 'results.json', result)
    _save_sensitivity(out, 'sibling-sensitivity-selected.json', network, classifier,
        train, inputs, proposals, guard)
    _render(out)
    return result
