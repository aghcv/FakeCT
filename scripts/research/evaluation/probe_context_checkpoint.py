#!/usr/bin/env python3
"""Read-only GPU probe of a saved context checkpoint and known diagnostic patch.

Run on an allocated GPU. This does not fit, alter, complete or bless an
experiment; it reports tensor structure, finite/range checks and reload behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def structure(value):
    if isinstance(value, (tuple, list)):
        return {'type': type(value).__name__, 'items': [structure(v) for v in value]}
    return {'type': type(value).__name__, 'shape': list(value.shape),
            'dtype': str(value.dtype)}


def statistics(value):
    array = np.asarray(value)
    finite = np.isfinite(array)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'finite': bool(finite.all()), 'nonfinite_count': int((~finite).sum()),
            'minimum': float(array[finite].min()) if finite.any() else None,
            'maximum': float(array[finite].max()) if finite.any() else None,
            'below_zero': int((array < 0).sum()), 'above_one': int((array > 1).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--patch', required=True, type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 5:
        parser.error('--repeats must be between 1 and 5')
    if args.output_json.exists():
        parser.error('Use a fresh output-json path')
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise SystemExit('This probe requires an allocated GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_op_determinism()
    directory = args.experiment.resolve()
    lock_path = directory/'experiment-lock.json'
    lock = json.loads(lock_path.read_text())
    if sha(lock_path) != Path(str(lock_path)+'.sha256').read_text().strip():
        raise ValueError('Experiment lock checksum differs')
    settings = lock['model_settings']
    tf.keras.utils.set_random_seed(settings['seed'])
    model_path = directory/'model/checkpoint.keras'
    checkpoint_before, patch_before = sha(model_path), sha(args.patch)
    with np.load(args.patch, allow_pickle=False) as data:
        image, truth, valid = data['image'], data['mask'], data['valid']
        sample_id = str(data['sample_id'])
        spacing, origin = data['spacing_ijk_mm'], data['origin_kji']
    if (sample_id != settings['diagnostic_sample_id'] or list(image.shape) != settings['patch_size_kji']
            or not np.array_equal(spacing, settings['target_spacing_mm'])):
        raise ValueError('Known patch identity/grid differs from frozen diagnostic settings')
    image = (np.clip(image, settings['clip_min'], settings['clip_max'])-settings['clip_min'])/(settings['clip_max']-settings['clip_min'])
    image = np.ascontiguousarray((image*valid)[None, ..., None], np.float32)
    model = tf.keras.models.load_model(model_path, compile=False)
    weights = [np.asarray(w) for w in model.get_weights()]
    report = {'tensorflow_version': tf.__version__, 'gpu': [d.name for d in devices],
              'checkpoint_sha256': checkpoint_before, 'patch_sha256': patch_before,
              'sample_id': sample_id, 'patch_origin_kji': origin.tolist(),
              'input': statistics(image), 'model_input': structure(model.input),
              'model_output': structure(model.output), 'flat_outputs': structure(model.outputs),
              'weight_arrays': len(weights), 'weights_finite': all(np.isfinite(w).all() for w in weights),
              'weights_minimum': min(float(w.min()) for w in weights),
              'weights_maximum': max(float(w.max()) for w in weights), 'repeats': []}
    print(json.dumps({'probe_setup': report}, allow_nan=False), flush=True)
    legacy_aux = tf.keras.Model(model.input, [model.output, model.get_layer('aorta_logits').output])
    flat_aux = tf.keras.Model(model.inputs[0], [model.outputs[0], model.get_layer('aorta_logits').output])
    previous = None
    for index in range(args.repeats):
        public = model(image, training=False)
        old = legacy_aux(image, training=False)
        probability_tensor, logits_tensor = flat_aux(image, training=False)
        probability, logits = np.asarray(probability_tensor), np.asarray(logits_tensor)
        sigmoid = np.asarray(tf.math.sigmoid(logits_tensor))
        observation = {'index': index, 'public_structure': structure(public),
            'public_array': statistics(public), 'original_aux_structure': structure(old),
            'original_aux_probability_array': statistics(old[0]),
            'original_aux_logit_array': statistics(old[1]),
            'flat_aux_probability': statistics(probability), 'flat_aux_logits': statistics(logits),
            'sigmoid_of_logits': statistics(sigmoid),
            'probability_vs_sigmoid_max_abs': float(np.max(np.abs(probability-sigmoid))),
            'repeat_probability_max_abs': float(np.max(np.abs(probability-previous))) if previous is not None else None}
        if probability.shape == image.shape:
            prediction = probability[0, ..., 0] >= settings['threshold']
            mask = truth.astype(bool); validity = valid.astype(bool)
            tp = int(np.count_nonzero(prediction & mask & validity))
            fp = int(np.count_nonzero(prediction & ~mask & validity))
            fn = int(np.count_nonzero(~prediction & mask & validity))
            observation['metrics'] = {'true_positive': tp, 'false_positive': fp,
                'false_negative': fn, 'dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.}
        report['repeats'].append(observation)
        print(json.dumps({'probe_repeat': observation}, allow_nan=False), flush=True)
        previous = probability.copy()
    report['sources_unchanged'] = checkpoint_before == sha(model_path) and patch_before == sha(args.patch)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    Path(str(args.output_json)+'.sha256').write_text(sha(args.output_json)+'\n')
    print(str(args.output_json), flush=True)


if __name__ == '__main__':
    main()
