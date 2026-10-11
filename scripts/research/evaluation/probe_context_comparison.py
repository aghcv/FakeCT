#!/usr/bin/env python3
"""Read-only checkpoint diagnostic on selected training/validation patches.

This is an explanatory probe, not whole-volume performance or threshold tuning.
Frozen experiment source is imported after its hashes are checked. No optimizer
is constructed and no training, model, experiment, or dataset file is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(path, expected=None):
    path = Path(path)
    actual = sha(path)
    expected = expected or Path(str(path) + '.sha256').read_text().strip()
    if actual != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return actual


def signed_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    Path(str(path) + '.sha256').write_text(sha(path) + '\n')


def stats(array):
    value = np.asarray(array)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite diagnostic values')
    if not value.size:
        return {'count': 0}
    return dict(count=int(value.size), minimum=float(value.min()),
        maximum=float(value.max()), mean=float(value.mean()), std=float(value.std()),
        quantiles=dict(zip(('q00', 'q01', 'q25', 'q50', 'q75', 'q99', 'q100'),
            map(float, np.percentile(value, [0, 1, 25, 50, 75, 99, 100])))))


def verify_experiment(directory, comparison):
    lock_path = directory / 'experiment-lock.json'
    checked(lock_path)
    lock = json.loads(lock_path.read_text())
    checked(directory / 'input.ini', lock['input_sha256'])
    checked(lock['dataset_manifest'], lock['dataset_manifest_sha256'])
    for name, expected in lock['code_sha256'].items():
        checked(directory / 'source' / name, expected)
    arms = ['baseline_only', 'baseline_plus_siblings'] if comparison else ['positive_control']
    metadata, checkpoint_hashes = {}, {}
    for arm in arms:
        model_dir = directory / arm / 'model' if comparison else directory / 'model'
        path = model_dir / 'training.json'
        checked(path)
        meta = json.loads(path.read_text())
        checkpoint_hashes[arm] = checked(model_dir / 'checkpoint.keras', meta['model_sha256'])
        if meta.get('history_sha256'):
            checked(model_dir / 'history.json', meta['history_sha256'])
        if comparison:
            canonical = hashlib.sha256(json.dumps(lock, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if meta['lock_fingerprint'] != canonical or not meta['complete']:
                raise ValueError('Training lock identity/completion mismatch')
            for name, expected in meta['component_sha256'].items():
                checked(directory / 'source' / 'src' / name, expected)
        metadata[arm] = meta
    return lock, metadata, checkpoint_hashes


def origins_for(volume, settings, validation):
    points = volume.foreground_points(settings['target_spacing_mm'], max_points=100000, seed=42)
    median = np.median(points, axis=0)
    shape = np.asarray(volume.model_shape(settings['target_spacing_mm']))
    patch = np.asarray(settings['patch_size_kji'])
    origins = []
    for quantile in (.25, .75):
        desired = median.copy()
        desired[0] = np.quantile(points[:, 0], quantile)
        point = points[np.argmin(np.sum((points - desired)**2, axis=1))]
        origin = np.clip(point - patch // 2, 0, np.maximum(0, shape - patch))
        origins.append((f'foreground_k{int(quantile*100)}', tuple(map(int, origin))))
    if validation:
        point = points[np.argmin(np.sum((points - median)**2, axis=1))]
        origin = point // patch * patch
        origins.append(('whole_fov_grid_target_tile', tuple(map(int, origin))))
    return origins


def prediction_probe(model, x, truth, valid, learning, check_training_mode=False):
    tf = learning._tensorflow()
    inference = learning._inference_model(model)
    probability, logits = learning._predict_pair(inference, x)
    fg, bg = truth.astype(bool) & valid.astype(bool), ~truth.astype(bool) & valid.astype(bool)
    output = dict(probability_foreground=stats(probability[fg]),
        probability_background=stats(probability[bg]), logits_foreground=stats(logits[fg]),
        logits_background=stats(logits[bg]), thresholds={})
    for threshold in (.1, .25, .5):
        counts = learning.counts_for(truth, probability >= threshold, valid)
        output['thresholds'][str(threshold)] = learning.metrics_from_counts(counts)
    if check_training_mode:
        p_train, l_train = inference(x, training=True)
        output['training_mode_max_probability_difference'] = float(np.max(np.abs(
            probability - np.asarray(p_train)[0, ..., 0])))
        output['training_mode_max_logit_difference'] = float(np.max(np.abs(
            logits - np.asarray(l_train)[0, ..., 0])))
        features = tf.keras.Model(model.inputs[0], model.get_layer('aorta_logits').input)
        values = np.asarray(features(x, training=False))[0]
        output['final_decoder_features'] = dict(zero_fraction=float(np.mean(values == 0)),
            all_zero_channels=int(np.all(values == 0, axis=(0, 1, 2)).sum()),
            channels=int(values.shape[-1]),
            channel_std=[float(v) for v in values.std(axis=(0, 1, 2))],
            channel_maximum=[float(v) for v in values.max(axis=(0, 1, 2))])
    return output, probability


def save_known_figure(path, image, truth, probabilities):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    center = np.median(np.argwhere(truth), axis=0).astype(int)
    figure, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    names = ['Input + target', 'Baseline only', 'Baseline + siblings', 'Successful patch control']
    for axis in range(3):
        selection = [slice(None)] * 3
        selection[axis] = int(center[axis])
        selection = tuple(selection)
        axes[axis, 0].imshow(image[selection], cmap='gray', vmin=0, vmax=.5)
        axes[axis, 0].contour(truth[selection], levels=[.5], colors=['cyan'], linewidths=.7)
        for col, arm in enumerate(('baseline_only', 'baseline_plus_siblings', 'positive_control'), 1):
            axes[axis, col].imshow(probabilities[arm][selection], cmap='magma', vmin=0, vmax=1)
            axes[axis, col].contour(truth[selection], levels=[.5], colors=['cyan'], linewidths=.7)
        for col in range(4):
            axes[axis, col].set_axis_off()
            if axis == 0:
                axes[axis, col].set_title(names[col])
    figure.suptitle('Known training patch: probability scale 0–1; cyan = target\n'
        'Selected cohort checkpoints compared with the one-patch diagnostic control')
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run(args):
    directory, diagnostic, output = (p.resolve() for p in (args.experiment, args.diagnostic, args.output))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').write_text('Read-only checkpoint probe incomplete\n')
    started = time.monotonic()
    lock, metadata, checkpoint_hashes = verify_experiment(directory, True)
    diagnostic_lock, control_metadata, control_hashes = verify_experiment(diagnostic, False)
    sys.path.insert(0, str(directory / 'source' / 'src'))
    import fakect_context_learning as learning
    from fakect_context_volume import ContextVolume
    settings = learning.settings_for({'model': lock['model_settings']})
    for key in ('patch_size_kji', 'target_spacing_mm', 'clip_min', 'clip_max'):
        if settings[key] != diagnostic_lock['model_settings'][key]:
            raise ValueError('Positive control input normalization or grid differs')
    tf = learning._tensorflow()
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('GPU allocation required')
    tf.keras.utils.set_random_seed(settings['seed'])
    tf.config.experimental.enable_op_determinism()
    models = {arm: tf.keras.models.load_model(directory / arm / 'model/checkpoint.keras', compile=False)
        for arm in metadata}
    models['positive_control'] = tf.keras.models.load_model(diagnostic / 'model/checkpoint.keras', compile=False)
    weights = {}
    for arm, model in models.items():
        arrays = model.get_weights()
        weights[arm] = dict(all_finite=all(bool(np.isfinite(v).all()) for v in arrays),
            minimum=min(float(v.min()) for v in arrays), maximum=max(float(v.max()) for v in arrays),
            output_head_kernel=stats(model.get_layer('aorta_logits').get_weights()[0]),
            output_head_bias=model.get_layer('aorta_logits').get_weights()[1].tolist(),
            batch_normalization_or_dropout_layers=[l.name for l in model.layers
                if isinstance(l, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout))])
    report = dict(schema_version='fakect.context-comparison-checkpoint-probe/1', complete=False,
        scope='Selected positive patches only; not whole-volume performance or threshold selection',
        test_payloads_loaded=False, tensorflow_version=tf.__version__,
        source_directory=str(directory / 'source' / 'src'), script_sha256=sha(__file__),
        lock_sha256=sha(directory / 'experiment-lock.json'), checkpoint_sha256=checkpoint_hashes,
        positive_control_checkpoint_sha256=control_hashes['positive_control'],
        settings=settings, model_weights=weights, records=[])
    known = diagnostic / 'model/diagnostic-prediction.npz'
    checked(known, control_metadata['positive_control']['diagnostic_prediction_sha256'])
    with np.load(known, allow_pickle=False) as data:
        image, truth, valid = (data[k].copy() for k in ('image', 'mask', 'valid'))
        known_id = str(data['sample_id'])
        known_origin = data['origin_kji'].tolist()
    known_x, _, _ = learning.prepare_block(image, truth, valid, settings)
    control_probabilities = {}
    for arm, model in models.items():
        observation, probability = prediction_probe(model, known_x, truth, valid, learning, True)
        control_probabilities[arm] = probability
        record = dict(arm=arm, sample_id=known_id, split='train', patch_kind='known_diagnostic_patch',
            origin_kji=known_origin, input_normalized=stats(known_x),
            true_voxels=int(truth.sum()), **observation)
        report['records'].append(record)
        print(json.dumps({'phase':'known_patch', **record}, allow_nan=False), flush=True)
    save_known_figure(output / 'known-training-patch.png', image, truth, control_probabilities)
    del control_probabilities, image, truth, valid, known_x
    baselines = [row for row in lock['samples'] if row['baseline'] and row['split'] in ('train', 'validation')]
    if len(baselines) != 10 or sum(row['split'] == 'train' for row in baselines) != 8:
        raise ValueError('Unexpected diagnostic sample partition')
    for row in baselines:
        with ContextVolume(row) as volume:
            for kind, origin in origins_for(volume, settings, row['split'] == 'validation'):
                arrays = volume.read_block(origin, settings['patch_size_kji'], settings['target_spacing_mm'])
                truth, valid = arrays['mask'], arrays['valid']
                if not np.any(truth & valid):
                    raise ValueError('Positive probe missed target')
                x, _, _ = learning.prepare_block(arrays['image'], truth, valid, settings)
                input_sha = hashlib.sha256(x.tobytes()).hexdigest()
                for arm in metadata:
                    observation, _ = prediction_probe(models[arm], x, truth, valid, learning)
                    record = dict(arm=arm, sample_id=row['sample_id'], split=row['split'],
                        patch_kind=kind, origin_kji=list(origin), input_sha256=input_sha,
                        input_normalized=stats(x), image_foreground=stats(arrays['image'][truth.astype(bool)]),
                        true_voxels=int(truth.sum()), **observation)
                    report['records'].append(record)
                    print(json.dumps({'phase':'selected_patch', **record}, allow_nan=False), flush=True)
                signed_json(output / 'probe-progress.json', report)
    _, _, after_hashes = verify_experiment(directory, True)
    _, _, control_after = verify_experiment(diagnostic, False)
    if after_hashes != checkpoint_hashes or control_after != control_hashes:
        raise ValueError('Checkpoint changed during read-only probe')
    report.update(complete=True, all_source_and_checkpoint_hashes_verified=True,
        elapsed_seconds=time.monotonic()-started, gpu_allocator_memory=learning.gpu_allocator_memory(tf))
    signed_json(output / 'probe.json', report)
    (output / 'INCOMPLETE').unlink()
    print(json.dumps(dict(phase='probe_complete', output=str(output), records=len(report['records']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--diagnostic', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
