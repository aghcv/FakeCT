"""Classifier-only continuation of a reviewed, immutable spatial locator.

The locator is loaded, never trained. Its image-derived proposals must replay
the reviewed crop geometry for every development sample before any gradient is
applied. Classifier initialization and shuffle order start from an explicit new
seed; this is not a resume of the interrupted locator job's random state.
"""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import fakect_coa_classifier_experiment as exp
import fakect_multiscale_experiment_v3 as geometry
import fakect_multiscale_learning_v3 as legacy


def weight_fingerprint(model):
    """Hash ordered numeric weights, independent of Keras variable path names."""
    digest = hashlib.sha256()
    for index, variable in enumerate(model.weights):
        array = np.ascontiguousarray(np.asarray(variable))
        header = json.dumps({'index': index, 'shape': list(array.shape),
                             'dtype': array.dtype.str}, sort_keys=True).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def compare_proposal(reviewed, predicted_box, origins, proposal_boxes, *, atol=1e-6):
    """Fail closed on any changed native crop or nonfinite box prediction.

    Boxes allow floating point roundoff between CPU review and GPU inference;
    the native crop origins must be exactly equal, so no data window can move.
    """
    observed = np.asarray(predicted_box, dtype=np.float64)
    expected = np.asarray(reviewed['predicted_box'], dtype=np.float64)
    observed_boxes = np.asarray(proposal_boxes, dtype=np.float64)
    expected_boxes = np.asarray(reviewed['proposal_boxes'], dtype=np.float64)
    observed_origins = np.asarray(origins)
    expected_origins = np.asarray(reviewed['origins_kji'])
    if (observed.shape != (6,) or expected.shape != (6,)
            or observed_boxes.shape != expected_boxes.shape
            or observed_boxes.ndim != 2 or observed_boxes.shape[1] != 6
            or observed_origins.shape != expected_origins.shape
            or observed_origins.shape != (len(observed_boxes), 3)
            or not np.isfinite(observed).all() or not np.isfinite(expected).all()
            or not np.isfinite(observed_boxes).all() or not np.isfinite(expected_boxes).all()
            or not np.issubdtype(observed_origins.dtype, np.integer)
            or not np.issubdtype(expected_origins.dtype, np.integer)):
        raise ValueError('Invalid recomputed or reviewed proposal geometry')
    if not np.array_equal(observed_origins, expected_origins):
        raise ValueError('Recomputed native crop origins differ from the reviewed proposals')
    if (not np.allclose(observed, expected, atol=atol, rtol=0)
            or not np.allclose(observed_boxes, expected_boxes, atol=atol, rtol=0)):
        raise ValueError('Recomputed locator boxes differ from the reviewed proposals')
    return {'maximum_box_absolute_error': float(np.max(np.abs(observed-expected))),
            'maximum_proposal_box_absolute_error': float(np.max(np.abs(observed_boxes-expected_boxes))),
            'native_origins_identical': True}


def verify_reviewed_proposals(locator, network, rows, inputs, reviewed, c, guard):
    """Replay all development images; no mask or condition target guides a crop."""
    if (len({r['sample_id'] for r in rows}) != len(rows)
            or set(reviewed) != {r['sample_id'] for r in rows}):
        raise ValueError('Reviewed proposals must join development samples exactly')
    records, proposals = [], {}
    for row in rows:
        key = row['sample_id']
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Classifier continuation cannot read test samples')
        if reviewed[key].get('coverage_passed') is not True:
            raise ValueError('Reviewed proposal failed coverage: ' + key)
        guard.check('proposal-replay-before')
        ov, image = inputs.coarse(row)
        predicted = network.predict_boxes(locator, image)[0]
        origins, boxes = geometry.proposal_origins(predicted, ov, c)
        try:
            error = compare_proposal(reviewed[key], predicted, origins, boxes)
        except ValueError as failure:
            raise ValueError(key + ': ' + str(failure)) from failure
        records.append({'sample_id': key, 'split': row['split'], **error})
        # Use the reviewed representation once its image-only replay is proven,
        # avoiding tiny CPU/GPU box roundoff in otherwise identical fine inputs.
        proposals[key] = reviewed[key]
        guard.check('proposal-replay-after')
        print(json.dumps({'stage': 'proposal-replay', 'completed': len(records),
                          'total': len(rows), 'sample_id': key}), flush=True)
    return proposals, {'passed': True, 'samples': records,
        'absolute_box_tolerance': 1e-6, 'native_origins_exact': True,
        'test_evaluated': False,
        'contract': 'All image-only predictions reproduce reviewed boxes and exact native crop origins before classifier gradients.'}


def _score(tf, steps, rows, inputs, proposals, guard):
    records = []
    for row in rows:
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Classifier continuation cannot score test samples')
        guard.check('classifier-evaluation-before', steps=steps)
        values = legacy._tensor_inputs(tf, inputs.classifier(row, proposals[row['sample_id']]))
        output = steps.predict(values)
        logit = float(np.asarray(output['coa_logit']).ravel()[0])
        score = float(np.asarray(output['local_volume_loss']).ravel()[0])
        if not np.isfinite([logit, score]).all():
            raise ValueError('Nonfinite classifier output')
        record = {'sample_id': row['sample_id'], 'case_id': row['case_id'],
            'family': row['anatomy_family'], 'split': row['split'], 'role': row['pilot_role'],
            'label': int(row['annotation']['coa_benchmark_positive']), 'logit': logit,
            'probability': float(1/(1+np.exp(-np.clip(logit, -80, 80)))),
            'score_label': row['annotation'][exp.SCORE], 'score_prediction': score}
        if 'review_near_threshold' in row['annotation']:
            record['near_threshold'] = bool(row['annotation']['review_near_threshold'])
        records.append(record)
        guard.check('classifier-evaluation-after', steps=steps)
    return records


def learning_gate(initial_training, selected_training):
    """Keep the original training-only, edited-negative engineering check."""
    initial = legacy.metrics(initial_training)
    training = legacy.metrics(selected_training)
    edited = legacy.metrics([r for r in selected_training if r['role'] != 'original'])
    required = ('sensitivity', 'specificity', 'positive_probability_mean', 'negative_probability_mean')
    if (initial is None or training is None or edited is None
            or any(edited.get(k) is None for k in required)):
        raise ValueError('Learning gate requires original and edited binary classes')
    checks = {'training_bce_reduced_by_20_percent': bool(training['bce'] < .8*initial['bce']),
              'edited_sensitivity_at_least_0_75': bool(edited['sensitivity'] >= .75),
              'edited_specificity_at_least_0_75': bool(edited['specificity'] >= .75),
              'edited_probability_separation_at_least_0_15': bool(
                  edited['positive_probability_mean']-edited['negative_probability_mean'] >= .15)}
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'Training engineering check only; validation and test performance are separate.',
            'decision_threshold': .5}


def _unchanged_encoder(encoder, expected):
    actual = weight_fingerprint(encoder)
    if encoder.trainable or encoder.trainable_variables or actual != expected:
        raise RuntimeError('Frozen locator overview encoder changed during classifier training')
    return actual


def checkpoint_prediction_parity(saved_records, reloaded_records):
    """Verify the selected checkpoint recreates the scores used to select it."""
    previous = {r['sample_id']: r for r in saved_records}
    current = {r['sample_id']: r for r in reloaded_records}
    if (len(previous) != len(saved_records) or len(current) != len(reloaded_records)
            or set(previous) != set(current)):
        raise ValueError('Selected checkpoint predictions do not join epoch predictions')
    maximum_error = 0.
    for key, old in previous.items():
        new = current[key]
        expected = np.array([old[k] for k in ('logit', 'probability', 'score_prediction')])
        observed = np.array([new[k] for k in ('logit', 'probability', 'score_prediction')])
        if not np.allclose(expected, observed, atol=1e-6, rtol=1e-5):
            raise ValueError('Selected classifier checkpoint prediction mismatch: ' + key)
        maximum_error = max(maximum_error, float(np.max(np.abs(expected-observed))))
    return {'passed': True, 'maximum_absolute_error': maximum_error,
            'absolute_tolerance': 1e-6, 'relative_tolerance': 1e-5}


def _render(out):
    """Keep a rendering failure separate from saved quantitative results."""
    try:
        from fakect_coa_classifier_report import write_report
        write_report(out)
    except Exception as error:
        exp.write_json(out/'report-errors.json', {'report_generated': False,
            'numeric_results_preserved': True, 'error_type': type(error).__name__, 'error': str(error)})
        print(json.dumps({'stage': 'report', 'status': 'failed', 'error': str(error)}), flush=True)


def fit(c, lock):
    """Train a fresh classifier from an immutable reviewed locator checkpoint."""
    from fakect_coa_classifier_report import summarize
    from fakect_tensorflow_steps import make_classifier_steps

    out = Path(c['experiment']['output_directory'])
    input_bytes = (out/'input.ini').read_bytes()
    # Consume the verified on-disk object, never a separately supplied stale
    # object that happens to accompany a valid frozen experiment directory.
    lock = exp.inspect_lock(c, input_bytes)
    runtime = exp.runtime_config(c, lock)
    rows = lock['samples']
    if any(r['split'] not in ('train', 'validation') for r in rows):
        raise ValueError('Classifier continuation cannot consume test samples')
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    if not train or not validation:
        raise ValueError('Classifier needs training and validation development samples')
    checkpoint = lock['locator_checkpoint']
    if exp.sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
        raise ValueError('Locator checkpoint changed before classifier continuation')
    model_dir = out/'model'
    model_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    tf, network = legacy.configure_tensorflow(runtime)
    guard = legacy.RuntimeGuard(runtime, tf)
    guard.check('tensorflow-configured', save=True)
    inputs = legacy.Inputs(runtime)
    locator = network.load_model(checkpoint['path'])
    locator.trainable = False
    locator_before = weight_fingerprint(locator)
    encoder_before = weight_fingerprint(locator.get_layer('overview_encoder'))
    proposals, replay = verify_reviewed_proposals(locator, network, rows, inputs,
        lock['proposals'], runtime, guard)
    exp.write_json(out/'proposal-replay.json', replay)
    seed = int(c['training']['seed'])
    shape, detail = runtime['input.overview']['shape_kji'], runtime['input.detail']
    # Explicit reset: the preceding checkpoint load/replay must not define the
    # initial classifier weights or the separate NumPy shuffle stream.
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    classifier = network.build_classifier(shape, detail['shape_kji'],
        num_proposals=detail['proposals'], base_filters=runtime['model']['base_filters'],
        seed=seed, locator=locator)
    _unchanged_encoder(classifier.get_layer('overview_encoder'), encoder_before)
    steps = make_classifier_steps(classifier, shape, detail['shape_kji'],
        num_proposals=detail['proposals'], learning_rate=c['training']['learning_rate'])
    provenance = {'locator_checkpoint': checkpoint, 'locator_retrained': False,
        'initialization_seed': seed, 'shuffle_seed': seed,
        'random_state_policy': 'Fresh classifier initialization and independent default_rng shuffle stream; not the interrupted job RNG state.',
        'initial_classifier_weights_sha256': weight_fingerprint(classifier),
        'locator_weights_before_sha256': locator_before,
        'overview_encoder_before_sha256': encoder_before,
        'coarse_encoder_frozen': True, 'proposal_replay_passed': True,
        'training_sample_count': len(train), 'validation_sample_count': len(validation),
        'test_evaluated': False, 'tensorflow_version': tf.__version__,
        'checkpoint_selection': 'Minimum native validation binary cross entropy; fixed probability decision threshold 0.5.',
        'input_contract': 'Attenuation + validity + frozen image-predicted crop coordinates only. Condition labels are loss targets, not input channels.'}
    exp.write_json(out/'continuation-audit.json', provenance)
    initial_records = _score(tf, steps, train+validation, inputs, proposals, guard)
    initial_train = [r for r in initial_records if r['split'] == 'train']
    initial_validation = [r for r in initial_records if r['split'] == 'validation']
    initial_metrics = {'training': summarize(initial_train), 'validation': summarize(initial_validation)}
    exp.write_json(out/'initialization.json', {**initial_metrics, 'predictions': initial_records,
        'weights_sha256': provenance['initial_classifier_weights_sha256'], 'seed': seed})
    history, best, best_epoch = [], float('inf'), None
    for epoch in range(c['training']['epochs']):
        losses, norms, bces, hubers = [], [], [], []
        order = []
        for index in rng.permutation(len(train)):
            row = train[int(index)]
            order.append(row['sample_id'])
            guard.check('classifier-train-before', steps=steps)
            values = legacy._tensor_inputs(tf, inputs.classifier(row, proposals[row['sample_id']]))
            target = legacy._tensor_inputs(tf, {
                'coa_logit': np.array([[row['annotation']['coa_benchmark_positive']]], np.float32),
                'local_volume_loss': np.array([[row['annotation'][exp.SCORE]]], np.float32)})
            result = steps.train(values, target)
            losses.append(float(result['loss']))
            norms.append(float(result['gradient_global_norm']))
            bces.append(float(result['binary_crossentropy']))
            hubers.append(float(result['huber']))
            guard.check('classifier-train-after', steps=steps)
        del values, target, result
        encoder_after = _unchanged_encoder(classifier.get_layer('overview_encoder'), encoder_before)
        records = _score(tf, steps, train+validation, inputs, proposals, guard)
        training_records = [r for r in records if r['split'] == 'train']
        valid_records = [r for r in records if r['split'] == 'validation']
        selection_bce = legacy.metrics(valid_records)['bce']
        record = {'epoch': epoch+1, 'train_loss': float(np.mean(losses)),
            'train_binary_crossentropy': float(np.mean(bces)), 'train_huber': float(np.mean(hubers)),
            'gradient_norm_mean': float(np.mean(norms)), 'training': summarize(training_records),
            'validation': summarize(valid_records), 'checkpoint_selection_bce': selection_bce,
            'training_sample_order': order, 'predictions': records,
            'overview_encoder_sha256': encoder_after,
            'runtime': guard.check('classifier-epoch', steps=steps, save=True)}
        if not np.isfinite([record['train_loss'], selection_bce]).all():
            raise ValueError('Nonfinite classifier training')
        if selection_bce < best:
            best, best_epoch = selection_bce, epoch+1
            classifier.save(model_dir/'classifier.keras')
        record['selected_checkpoint_epoch_so_far'] = best_epoch
        history.append(record)
        exp.write_json(out/'classifier-history.json', {'initial_training': initial_metrics['training'],
            'initial_validation': initial_metrics['validation'], 'epochs': history,
            'selected_epoch': best_epoch, 'checkpoint_selection': provenance['checkpoint_selection']})
        # Render only after epoch data and any improved model are durable. The
        # running report remains useful even if a later epoch or job fails.
        _render(out)
        print(json.dumps({'stage': 'classifier', **{k: v for k, v in record.items()
                          if k not in ('predictions', 'training_sample_order')}}), flush=True)
    if weight_fingerprint(locator) != locator_before:
        raise RuntimeError('Locator weights changed during classifier continuation')
    trace_summary = {'classifier': steps.tracing_counts()}
    optimizer_iterations = int(np.asarray(steps.optimizer.iterations))
    del steps, classifier
    gc.collect()
    classifier = network.load_model(model_dir/'classifier.keras')
    _unchanged_encoder(classifier.get_layer('overview_encoder'), encoder_before)
    selected = make_classifier_steps(classifier, shape, detail['shape_kji'],
        num_proposals=detail['proposals'], learning_rate=c['training']['learning_rate'])
    final_records = _score(tf, selected, train+validation, inputs, proposals, guard)
    parity = checkpoint_prediction_parity(history[best_epoch-1]['predictions'], final_records)
    training_records = [r for r in final_records if r['split'] == 'train']
    valid_records = [r for r in final_records if r['split'] == 'validation']
    gate = learning_gate(initial_train, training_records)
    trace_summary['selected_classifier'] = selected.tracing_counts()
    # Numeric predictions are independently accessible even if report rendering
    # fails, and include both selected epoch and checkpoint content hash.
    classifier_hash = exp.sha256(model_dir/'classifier.keras')
    exp.write_json(out/'classifier-predictions.json', {'selected_epoch': best_epoch,
        'checkpoint_sha256': classifier_hash, 'predictions': final_records, 'test_evaluated': False})
    provenance.update(overview_encoder_after_sha256=weight_fingerprint(classifier.get_layer('overview_encoder')),
        locator_weights_after_sha256=weight_fingerprint(locator),
        selected_classifier_checkpoint_sha256=classifier_hash, selected_epoch=best_epoch,
        optimizer_steps=optimizer_iterations, training_traces=trace_summary,
        selected_checkpoint_prediction_parity=parity)
    exp.write_json(out/'continuation-audit.json', provenance)
    result = {'status': 'learning_check_passed' if gate['passed'] else 'learning_check_failed',
        'classifier_trained': True, 'locator_retrained': False, 'test_evaluated': False,
        'selected_epoch': best_epoch, 'epochs_completed': len(history),
        'training': summarize(training_records),
        'training_edited_only': summarize([r for r in training_records if r['role'] != 'original']),
        'validation': summarize(valid_records),
        'validation_edited_only': summarize([r for r in valid_records if r['role'] != 'original']),
        'initial_training': initial_metrics['training'], 'initial_validation': initial_metrics['validation'],
        'learning_gate': gate, 'predictions': final_records,
        'elapsed_seconds': time.monotonic()-started, 'tensorflow_version': tf.__version__,
        **legacy.memory_snapshot(tf), 'gpu_limit_mib': runtime['pilot']['gpu_memory_limit_mb'],
        'host_limit_mib': runtime['training']['host_memory_limit_mb'], 'training_traces': trace_summary,
        'checkpoint_sha256': {'locator.keras': checkpoint['sha256'], 'classifier.keras': classifier_hash},
        'checkpoint_selection': provenance['checkpoint_selection'],
        'severity_definition': 'Achieved local CoA volume-loss fraction; engineering proxy, not a clinical severity category.',
        'classification_decision_threshold': .5, 'optimizer_steps': optimizer_iterations,
        'overview_encoder_unchanged': True, 'proposal_replay_passed': True,
        'selected_checkpoint_prediction_parity': parity}
    guard.check('classifier-complete', steps=selected, save=True)
    exp.inspect_lock(c, input_bytes)
    if exp.sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
        raise ValueError('Locator checkpoint changed during classifier continuation')
    exp.write_json(out/'results.json', result)
    _render(out)
    return result
