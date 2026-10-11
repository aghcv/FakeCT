"""Attenuation-only 3D segmentation with streamed, physically larger context.

Training targets come from the atlas upstream. Neither atlas labels nor a
ground-truth localization crop are input features. Regular validation and test
use the same nonoverlapping whole-volume tiling, excluding padded voxels.
The explicit diagnostic mode fits one fixed training patch; its results are
learning diagnostics, not estimates of generalization.
"""
import configparser
import hashlib
from importlib.metadata import version as package_version
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np


MODEL_SCHEMA = 'fakect.context-segmentation-model/1'
RESULT_SCHEMA = 'fakect.context-segmentation-results/1'
DEFAULTS = dict(architecture='unet3d', target_spacing_mm=[1., 1., 1.],
    patch_size_kji=[128, 96, 96], base_filters=8, depth=3, epochs=20,
    patches_per_volume=4, batch_size=1, learning_rate=.001, threshold=.5,
    seed=42, clip_min=0., clip_max=.5, foreground_fraction=.5,
    augmentation_flip=True, intensity_scale_range=[.9, 1.1],
    save_predictions=False, diagnostic_steps_per_epoch=8,
    diagnostic_sample_id='', max_train_batches_per_epoch=0)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _signed_write(path, document):
    _write(path, document)
    Path(str(path) + '.sha256').write_text(sha256(path) + '\n')


def _fingerprint(lock):
    return hashlib.sha256(json.dumps(lock, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _int(value, name, minimum=1, maximum=None):
    if isinstance(value, str):
        if not value.strip().isdigit():
            raise ValueError(f'{name} must be an integer')
        value = int(value.strip())
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum or maximum is not None and value > maximum):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum or "unbounded"}]')
    return int(value)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be a finite number')
    return result


def _boolean(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError(f'{name} must be true or false')


def _sequence(value, name, length):
    if isinstance(value, str):
        value = [v.strip() for v in value.split(',')]
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f'{name} needs {length} comma-separated values')
    return value


def settings_for(config):
    """Parse either a ConfigParser or a typed mapping with a [model] section."""
    section = dict(config['model']) if isinstance(config, configparser.ConfigParser) else dict(config.get('model', {}))
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown model settings: {sorted(unknown)}')
    result = {**DEFAULTS, **section}
    if result['architecture'] != 'unet3d':
        raise ValueError('architecture must be unet3d')
    for key, low, high in [('depth', 1, 4), ('base_filters', 2, 32),
                          ('epochs', 1, None), ('patches_per_volume', 1, None),
                          ('batch_size', 1, 1), ('seed', 0, None),
                          ('diagnostic_steps_per_epoch', 1, None),
                          ('max_train_batches_per_epoch', 0, None)]:
        result[key] = _int(result[key], key, low, high)
    result['patch_size_kji'] = [_int(v, 'patch_size_kji', 8, 256)
        for v in _sequence(result['patch_size_kji'], 'patch_size_kji', 3)]
    if any(v % 2**result['depth'] for v in result['patch_size_kji']):
        raise ValueError('Each patch dimension must be divisible by 2**depth')
    if math.prod(result['patch_size_kji']) > 8_388_608:
        raise ValueError('Patch exceeds the 8,388,608-voxel memory budget')
    result['target_spacing_mm'] = [_number(v, 'target_spacing_mm')
        for v in _sequence(result['target_spacing_mm'], 'target_spacing_mm', 3)]
    if any(v <= 0 for v in result['target_spacing_mm']):
        raise ValueError('target_spacing_mm must be positive (i,j,k order)')
    for key in ('learning_rate', 'threshold', 'clip_min', 'clip_max', 'foreground_fraction'):
        result[key] = _number(result[key], key)
    if result['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    if not 0 < result['threshold'] < 1:
        raise ValueError('threshold must lie strictly between 0 and 1')
    if result['clip_min'] >= result['clip_max']:
        raise ValueError('clip_min must be smaller than clip_max')
    if not 0 < result['foreground_fraction'] < 1:
        raise ValueError('foreground_fraction must lie strictly between 0 and 1')
    result['intensity_scale_range'] = [_number(v, 'intensity_scale_range')
        for v in _sequence(result['intensity_scale_range'], 'intensity_scale_range', 2)]
    if not 0 < result['intensity_scale_range'][0] <= result['intensity_scale_range'][1]:
        raise ValueError('intensity_scale_range must be positive and increasing')
    for key in ('augmentation_flip', 'save_predictions'):
        result[key] = _boolean(result[key], key)
    if result['save_predictions']:
        raise ValueError('Full-volume prediction storage is disabled; diagnostic predictions are saved automatically')
    if not isinstance(result['diagnostic_sample_id'], str):
        raise ValueError('diagnostic_sample_id must be a string or blank')
    result['diagnostic_sample_id'] = result['diagnostic_sample_id'].strip()
    return result


def _shape(value, name):
    if len(value) != 3:
        raise ValueError(f'{name} needs three dimensions')
    return tuple(_int(v, name) for v in value)


def tile_origins(shape, patch):
    """Every model-grid voxel once; edge patches are padded, never shifted."""
    shape, patch = _shape(shape, 'shape'), _shape(patch, 'patch')
    return itertools.product(*(range(0, n, width) for n, width in zip(shape, patch)))


def tile_count(shape, patch):
    return math.prod(math.ceil(n / p) for n, p in zip(_shape(shape, 'shape'), _shape(patch, 'patch')))


def sample_origins(shape, patch, foreground_points, count, rng, foreground_fraction=.5):
    """Foreground-centered and spatially uniform patches; no lesion sampling.

    The default with four patches gives exactly two of each. With odd counts
    the foreground quota is stochastically rounded to avoid a systematic bias.
    Uniform patches may contain foreground; they sample the entire input FOV.
    """
    shape, patch = _shape(shape, 'shape'), _shape(patch, 'patch')
    count = _int(count, 'count')
    points = np.asarray(foreground_points)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
            or not np.issubdtype(points.dtype, np.integer)
            or np.any(points < 0) or np.any(points >= np.asarray(shape))):
        raise ValueError('foreground_points must be nonempty integer model-grid k,j,i coordinates')
    if not 0 < foreground_fraction < 1:
        raise ValueError('foreground_fraction must lie between zero and one')
    desired = count * foreground_fraction
    foreground_count = math.floor(desired) + int(rng.random() < desired % 1)
    flags = np.arange(count) < foreground_count
    rng.shuffle(flags)
    result = []
    for foreground in flags:
        if foreground:
            center = points[int(rng.integers(len(points)))]
            # Jitter by a quarter patch, keeping the chosen voxel in the block.
            jitter = [int(rng.integers(-p//4, p//4 + 1)) for p in patch]
            origin = tuple(max(0, min(max(0, n-p), int(c)-p//2+j))
                for n, p, c, j in zip(shape, patch, center, jitter))
        else:
            origin = tuple(int(rng.integers(0, max(1, n-p+1))) for n, p in zip(shape, patch))
        result.append(origin)
    return result


def counts_for(truth, prediction, valid):
    truth, prediction, valid = (np.asarray(a) for a in (truth, prediction, valid))
    if truth.shape != prediction.shape or truth.shape != valid.shape:
        raise ValueError('Truth, prediction and validity shapes differ')
    if any(not np.all((a == 0) | (a == 1)) for a in (truth, prediction, valid)):
        raise ValueError('Truth, prediction and validity must be binary')
    truth, prediction, valid = (a.astype(bool) for a in (truth, prediction, valid))
    return dict(true_positive=int(np.count_nonzero(truth & prediction & valid)),
                false_positive=int(np.count_nonzero(~truth & prediction & valid)),
                false_negative=int(np.count_nonzero(truth & ~prediction & valid)),
                true_negative=int(np.count_nonzero(~truth & ~prediction & valid)),
                valid_voxels=int(np.count_nonzero(valid)))


def metrics_from_counts(counts, voxel_volume_mm3=1.):
    tp, fp, fn = (int(counts[k]) for k in ('true_positive', 'false_positive', 'false_negative'))
    return {**counts, 'dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        'precision': tp/(tp+fp) if tp+fp else (0. if fn else 1.),
        'recall': tp/(tp+fn) if tp+fn else 1.,
        'true_voxels': tp+fn, 'predicted_voxels': tp+fp,
        'false_positive_mm3': fp*float(voxel_volume_mm3),
        'false_negative_mm3': fn*float(voxel_volume_mm3)}


def loss_totals(truth, probability, valid, *, logits=None):
    """Additive BCE/Dice statistics; production scoring supplies raw logits.

    The probability-only fallback supports external saved-probability audits,
    but cannot recover BCE for saturated predictions. Model selection therefore
    always uses the stable ``logaddexp(0, logits) - truth * logits`` expression.
    """
    truth, probability, valid = (np.asarray(a) for a in (truth, probability, valid))
    if truth.shape != probability.shape or truth.shape != valid.shape:
        raise ValueError('Loss array shapes differ')
    if (not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
            or not np.all((truth == 0) | (truth == 1)) or not np.all((valid == 0) | (valid == 1))):
        raise ValueError('Loss needs binary truth/validity and finite probabilities in [0,1]')
    selected = valid.astype(bool)
    y = truth[selected].astype(np.float64)
    p = probability[selected].astype(np.float64)
    if logits is None:
        q = np.clip(p, 1e-6, 1-1e-6)
        bce = -(y*np.log(q)+(1-y)*np.log1p(-q))
    else:
        logits = np.asarray(logits)
        if logits.shape != probability.shape or not np.isfinite(logits).all():
            raise ValueError('BCE needs finite logits matching probability shape')
        z = logits[selected].astype(np.float64)
        # This equivalent form avoids cancellation for a correctly predicted
        # foreground voxel with a very large positive logit.
        bce = np.maximum(z, 0)-y*z+np.log1p(np.exp(-np.abs(z)))
    return dict(valid_voxels=int(selected.sum()),
        bce_sum=float(np.sum(bce)),
        intersection_sum=float(np.sum(y*p)), truth_sum=float(y.sum()),
        probability_sum=float(p.sum()))


def loss_from_totals(totals):
    if totals['valid_voxels'] <= 0:
        raise ValueError('Cannot score a block without valid voxels')
    return (totals['bce_sum']/totals['valid_voxels'] + 1 -
            (2*totals['intersection_sum']+1)/(totals['truth_sum']+totals['probability_sum']+1))


def prepare_block(image, mask, valid, settings, *, rng=None):
    image, mask, valid = (np.asarray(a) for a in (image, mask, valid))
    shape = tuple(settings['patch_size_kji'])
    if any(a.shape != shape for a in (image, mask, valid)):
        raise ValueError('Loader block shape disagrees with requested patch')
    if not np.isfinite(image).all() or not np.all((mask == 0) | (mask == 1)) or not np.all((valid == 0) | (valid == 1)):
        raise ValueError('Loader returned invalid attenuation, target or validity')
    if not np.any(valid):
        raise ValueError('Loader returned an entirely padded block')
    if np.any(mask.astype(bool) & ~valid.astype(bool)):
        raise ValueError('Target cannot occupy invalid padded voxels')
    image = image.astype(np.float32, copy=True)
    mask, valid = mask.astype(np.float32), valid.astype(np.float32)
    if rng is not None:
        image *= rng.uniform(*settings['intensity_scale_range'])
        if settings['augmentation_flip']:
            # Axial reversal would invert superior/inferior anatomy; retain k.
            for axis in (2,):
                if rng.random() < .5:
                    image, mask, valid = (np.flip(a, axis) for a in (image, mask, valid))
    image = (np.clip(image, settings['clip_min'], settings['clip_max']) - settings['clip_min']) / (settings['clip_max']-settings['clip_min'])
    image *= valid
    return (np.ascontiguousarray(image[None, ..., None]),
            np.ascontiguousarray(mask[None, ..., None]),
            np.ascontiguousarray(valid[None, ..., None]))


def _tensorflow():
    import tensorflow as tf
    for device in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass  # A caller may have initialized the device already.
    return tf


def gpu_allocator_memory(tf):
    """Best-effort TensorFlow allocator snapshot; not total device occupancy."""
    records = []
    for device in tf.config.list_logical_devices():
        name = device.name
        if getattr(device, 'device_type', '') != 'GPU' and '/device:GPU:' not in name:
            continue
        query_name = name.rsplit('device:', 1)[-1]
        record = dict(device=name, allocator_device=query_name)
        try:
            values = tf.config.experimental.get_memory_info(query_name)
            current, peak = int(values['current']), int(values['peak'])
            if current < 0 or peak < current:
                raise ValueError('Invalid allocator byte counts')
            record.update(status='available', current_bytes=current, peak_bytes=peak)
        except Exception as error:
            # Unsupported allocator queries must not invalidate a completed fit.
            record.update(status='unavailable', reason=f'{type(error).__name__}: {error}')
        records.append(record)
    return records


def build_model(settings):
    """Compact three-dimensional U-Net; only attenuation is an input."""
    settings = settings_for({'model': settings})
    tf = _tensorflow()
    layers = tf.keras.layers
    inputs = layers.Input(tuple(settings['patch_size_kji'])+(1,), name='attenuation')
    def block(x, filters):
        for _ in range(2):
            x = layers.Conv3D(filters, 3, padding='same', activation='relu',
                              kernel_initializer='he_normal')(x)
        return x
    x, skips = inputs, []
    for level in range(settings['depth']):
        filters = settings['base_filters']*2**level
        x = block(x, filters)
        skips.append(x)
        x = layers.Conv3D(filters*2, 2, strides=2, activation='relu',
                          kernel_initializer='he_normal')(x)
    x = block(x, settings['base_filters']*2**settings['depth'])
    for level in reversed(range(settings['depth'])):
        x = layers.UpSampling3D(2)(x)
        x = layers.Concatenate()([x, skips[level]])
        x = block(x, settings['base_filters']*2**level)
    logits = layers.Conv3D(1, 1, name='aorta_logits')(x)
    probability = layers.Activation('sigmoid', name='aorta_probability')(logits)
    return tf.keras.Model(inputs, probability, name='fakect_context_unet3d')


def _logit_loss(tf, truth, logits, valid):
    """Stable BCE retains corrective gradients for very wrong predictions."""
    probability = tf.math.sigmoid(logits)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=logits)
    bce = tf.reduce_sum(bce*valid)/tf.maximum(tf.reduce_sum(valid), 1.)
    dice = (2*tf.reduce_sum(truth*probability*valid)+1)/(tf.reduce_sum((truth+probability)*valid)+1)
    return bce+1-dice


def _single_io(model):
    """Use canonical flat tensors across Keras save/reload structure changes."""
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError('Context segmentation requires exactly one image input and one probability output')
    return model.inputs[0], model.outputs[0]


def _training_step(model, settings):
    tf = _tensorflow()
    image_input, _ = _single_io(model)
    logits_model = tf.keras.Model(image_input, model.get_layer('aorta_logits').output)
    optimizer = tf.keras.optimizers.Adam(settings['learning_rate'])
    optimizer.build(model.trainable_variables)
    spec = tf.TensorSpec((1,)+tuple(settings['patch_size_kji'])+(1,), tf.float32)
    @tf.function(input_signature=[spec, spec, spec], jit_compile=False)
    def step(image, truth, valid):
        with tf.GradientTape() as tape:
            logits = logits_model(image, training=True)
            loss = _logit_loss(tf, truth, logits, valid)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    return step


def inspect_rows(lock):
    rows = lock.get('samples', [])
    if not rows:
        raise ValueError('Experiment lock needs samples')
    seen, registries = set(), {'case_id': {}, 'anatomy_family': {}}
    for row in rows:
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError('sample_id must be nonempty and unique')
        seen.add(sid)
        split = row.get('split')
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Samples require train, validation or test split')
        for key, registry in registries.items():
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f'Sample requires {key}')
            if value in registry and registry[value] != split:
                raise ValueError(f'{key} crosses partitions; refusing leakage')
            registry[value] = split
    return rows


def _volume(row):
    from fakect_context_volume import ContextVolume
    return ContextVolume(row, verify_hashes=True)


def _close(volume):
    close = getattr(volume, 'close', None)
    if close:
        close()


def _read(volume, origin, settings):
    block = volume.read_block(origin, settings['patch_size_kji'], settings['target_spacing_mm'])
    return tuple(block[key] for key in ('image', 'mask', 'valid'))


def _predict(model, image):
    probability = np.asarray(model(image, training=False))
    if (probability.shape != image.shape or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Network returned invalid sigmoid probabilities')
    return probability[0, ..., 0]


def _inference_model(model):
    """Shared weights, one forward pass yielding probabilities and raw logits."""
    image_input, probability_output = _single_io(model)
    return _tensorflow().keras.Model(image_input,
        [probability_output, model.get_layer('aorta_logits').output])


def _predict_pair(inference_model, image):
    probability, logits = (np.asarray(a) for a in inference_model(image, training=False))
    if probability.shape != image.shape or logits.shape != image.shape:
        raise ValueError(f'Network probability/logit shape mismatch: expected {image.shape}, '
                         f'probability {probability.shape}, logits {logits.shape}')
    if not np.isfinite(probability).all():
        raise ValueError(f'Network probability has {np.count_nonzero(~np.isfinite(probability))} nonfinite values')
    if not np.isfinite(logits).all():
        raise ValueError(f'Network logits have {np.count_nonzero(~np.isfinite(logits))} nonfinite values')
    outside = (probability < 0) | (probability > 1)
    if np.any(outside):
        raise ValueError(f'Network probability has {np.count_nonzero(outside)} values outside [0,1]; '
                         f'minimum={probability.min()}, maximum={probability.max()}')
    return probability[0, ..., 0], logits[0, ..., 0]


def _block_score(model, arrays, settings):
    x, y, valid = prepare_block(*arrays, settings)
    probability, logits = _predict_pair(_inference_model(model), x)
    truth, validity = y[0, ..., 0], valid[0, ..., 0]
    counts = counts_for(truth, probability >= settings['threshold'], validity)
    totals = loss_totals(truth, probability, validity, logits=logits)
    return metrics_from_counts(counts, math.prod(settings['target_spacing_mm'])), loss_from_totals(totals), probability


def score_volume(model, volume, settings):
    """Stream every FOV block; no target-box inference or overlap averaging."""
    counts, totals, blocks = None, None, 0
    inference_model = _inference_model(model)
    shape = volume.model_shape(settings['target_spacing_mm'])
    for origin in tile_origins(shape, settings['patch_size_kji']):
        arrays = _read(volume, origin, settings)
        if not np.any(arrays[2]):
            continue
        x, y, valid = prepare_block(*arrays, settings)
        probability, logits = _predict_pair(inference_model, x)
        truth, validity = y[0, ..., 0], valid[0, ..., 0]
        current = counts_for(truth, probability >= settings['threshold'], validity)
        current_loss = loss_totals(truth, probability, validity, logits=logits)
        counts = current if counts is None else {key: counts[key]+value for key, value in current.items()}
        totals = current_loss if totals is None else {key: totals[key]+value for key, value in current_loss.items()}
        blocks += 1
    if counts is None or counts['true_positive']+counts['false_negative'] == 0:
        raise ValueError('Full-volume target is empty on the requested model grid')
    return dict(metrics=metrics_from_counts(counts, math.prod(settings['target_spacing_mm'])),
        loss=loss_from_totals(totals), tiles=blocks, model_shape_kji=list(shape))


def _identity(row):
    return {key: row[key] for key in ('sample_id', 'case_id', 'anatomy_family', 'split')}


def _score_rows(model, rows, settings, loaded):
    records = []
    for row in rows:
        volume = _volume(row)
        loaded.add(row['sample_id'])
        try:
            record = {**_identity(row), **score_volume(model, volume, settings)}
        finally:
            _close(volume)
        records.append(record)
    return records


def summarize(records):
    """Report sample, case and family means separately; siblings aren't cases."""
    if not records:
        raise ValueError('Cannot summarize empty evaluation')
    by_case, by_family = {}, {}
    for record in records:
        by_case.setdefault(record['case_id'], []).append(record['metrics'])
        by_family.setdefault(record['anatomy_family'], set()).add(record['case_id'])
    fields = ('dice', 'precision', 'recall', 'false_positive_mm3', 'false_negative_mm3')
    case_means = {case: {key: float(np.mean([m[key] for m in values])) for key in fields}
                  for case, values in by_case.items()}
    family_means = {family: {key: float(np.mean([case_means[case][key] for case in cases]))
                            for key in fields} for family, cases in by_family.items()}
    return dict(sample_macro={key: float(np.mean([r['metrics'][key] for r in records])) for key in fields},
        case_means=case_means, family_means=family_means,
        family_macro={key: float(np.mean([m[key] for m in family_means.values()])) for key in fields})


def _diagnostic_selection(rows, settings):
    candidates = [row for row in rows if row['split'] == 'train']
    requested = settings['diagnostic_sample_id']
    if requested:
        candidates = [row for row in candidates if row['sample_id'] == requested]
    if not candidates:
        raise ValueError('Diagnostic sample must exist in the training partition')
    row = sorted(candidates, key=lambda r: r['sample_id'])[0]
    volume = _volume(row)
    try:
        shape = volume.model_shape(settings['target_spacing_mm'])
        points = np.asarray(volume.foreground_points(settings['target_spacing_mm'], max_points=100_000, seed=settings['seed']))
        if not len(points):
            raise ValueError('Diagnostic target is empty on the requested model grid')
        median = np.median(points, axis=0)
        center = points[np.argmin(np.sum((points-median)**2, axis=1))]
        origin = tuple(max(0, min(max(0, n-p), int(c)-p//2))
            for n, p, c in zip(shape, settings['patch_size_kji'], center))
        arrays = _read(volume, origin, settings)
        if not np.any(arrays[1]):
            raise ValueError('Foreground-centered diagnostic patch contains no target')
    finally:
        _close(volume)
    return row, arrays, dict(sample_id=row['sample_id'], case_id=row['case_id'],
        split='train', origin_kji=list(origin), model_shape_kji=list(shape),
        patch_size_kji=settings['patch_size_kji'], spacing_ijk_mm=settings['target_spacing_mm'],
        selection='Fixed patch centered on a foreground point nearest the foreground median; training partition only')


def fit(config, lock, output_directory, *, diagnostic=False):
    settings, rows = settings_for(config), inspect_rows(lock)
    train = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'validation']
    if not train or not diagnostic and not validation:
        raise ValueError('Regular fit needs nonempty training and validation partitions')
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(settings['seed'])
    tf.config.experimental.enable_op_determinism()
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'INCOMPLETE').write_text('Training has not completed. Use a fresh output revision after failure.\n')
    model = build_model(settings)
    step = _training_step(model, settings)
    history, loaded = [], set()
    diagnostic_info, arrays, baseline = None, None, None
    if diagnostic:
        row, arrays, diagnostic_info = _diagnostic_selection(rows, settings)
        loaded.add(row['sample_id'])
        initial_metrics, initial_loss, _ = _block_score(model, arrays, settings)
        baseline = dict(metrics=initial_metrics, loss=initial_loss)
        fixed = prepare_block(*arrays, settings)
    best_value, best_epoch = math.inf if diagnostic else -math.inf, None
    start = time.monotonic()
    for epoch in range(settings['epochs']):
        epoch_start = time.monotonic()
        losses = []
        if diagnostic:
            for _ in range(settings['diagnostic_steps_per_epoch']):
                losses.append(float(step(*fixed)))
        else:
            rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], epoch]))
            order = rng.permutation(len(train))
            for index in order:
                row = train[int(index)]
                volume = _volume(row)
                loaded.add(row['sample_id'])
                try:
                    points = volume.foreground_points(settings['target_spacing_mm'], max_points=100_000, seed=settings['seed']+epoch)
                    origins = sample_origins(volume.model_shape(settings['target_spacing_mm']), settings['patch_size_kji'],
                        points, settings['patches_per_volume'], rng, settings['foreground_fraction'])
                    for origin in origins:
                        raw = _read(volume, origin, settings)
                        losses.append(float(step(*prepare_block(*raw, settings, rng=rng))))
                        if settings['max_train_batches_per_epoch'] and len(losses) >= settings['max_train_batches_per_epoch']:
                            break
                finally:
                    _close(volume)
                if settings['max_train_batches_per_epoch'] and len(losses) >= settings['max_train_batches_per_epoch']:
                    break
        if not losses or not np.isfinite(losses).all():
            raise ValueError('Training produced no finite losses')
        train_seconds = time.monotonic()-epoch_start
        if diagnostic:
            measured, score, _ = _block_score(model, arrays, settings)
            assessment = dict(diagnostic_loss=score, diagnostic_metrics=measured)
            improved = score < best_value
        else:
            records = _score_rows(model, validation, settings, loaded)
            summary = summarize(records)
            score = summary['family_macro']['dice']
            assessment = dict(validation_family_macro_dice=score,
                validation_sample_macro_dice=summary['sample_macro']['dice'],
                validation_samples=len(records), validation_tiles=sum(r['tiles'] for r in records))
            improved = score > best_value
        if not math.isfinite(score):
            raise ValueError('Checkpoint selection score is nonfinite')
        if improved:
            best_value, best_epoch = score, epoch+1
            model.save(directory/'checkpoint.keras')
        traces = int(step.experimental_get_tracing_count())
        if traces > 2:
            raise RuntimeError('Training graph retraced unexpectedly')
        record = dict(epoch=epoch+1, train_loss=float(np.mean(losses)),
            train_batches=len(losses), train_seconds=train_seconds,
            epoch_seconds=time.monotonic()-epoch_start, train_function_traces=traces, **assessment)
        history.append(record)
        _write(directory/'history.json', history)
        print(json.dumps(record, allow_nan=False), flush=True)
    best_model = tf.keras.models.load_model(directory/'checkpoint.keras', compile=False)
    diagnostic_final = None
    if diagnostic:
        final_metrics, final_loss, probability = _block_score(best_model, arrays, settings)
        diagnostic_final = dict(metrics=final_metrics, loss=final_loss)
        _save_diagnostic(directory/'diagnostic-prediction.npz', arrays, probability, diagnostic_info)
    metadata = dict(schema_version=MODEL_SCHEMA, complete=True, diagnostic=bool(diagnostic),
        scope='engineering', input='attenuation only, normalized with fixed clipping bounds',
        settings=settings, lock_fingerprint=_fingerprint(lock), source_sha256=sha256(__file__),
        volume_source_sha256=sha256(Path(__file__).with_name('fakect_context_volume.py')),
        model_sha256=sha256(directory/'checkpoint.keras'), history_sha256=sha256(directory/'history.json'),
        epochs_completed=len(history), best_epoch=best_epoch, best_selection_value=best_value,
        model_selection='minimum fixed training-patch loss' if diagnostic else 'maximum validation family-macro whole-volume Dice',
        diagnostic_patch=diagnostic_info, diagnostic_before=baseline, diagnostic_after=diagnostic_final,
        diagnostic_prediction_sha256=sha256(directory/'diagnostic-prediction.npz') if diagnostic else None,
        loaded_samples=sorted(loaded), test_payloads_loaded=False,
        model_parameters=int(model.count_params()), elapsed_seconds=time.monotonic()-start,
        tensorflow_version=tf.__version__, numpy_version=np.__version__,
        scipy_version=package_version('scipy'), keras_version=package_version('keras'),
        python_version=platform.python_version(),
        devices=[device.name for device in tf.config.list_logical_devices()],
        gpu_allocator_memory=gpu_allocator_memory(tf),
        gpu_allocator_memory_scope='TensorFlow allocator current/peak bytes at fit completion; excludes driver and other processes. Peak since allocator initialization or last external reset; this fit does not reset it.',
        optimizer='Adam', jit_compile=False, deterministic_operations=True,
        loss='valid-voxel mean sigmoid cross-entropy from logits + soft Dice loss (smoothing=1); batch size 1',
        bce_numerics='TensorFlow sigmoid_cross_entropy_with_logits for fitting; stable raw-logit BCE for checkpoint selection and evaluation; no clipped-probability BCE gradient',
        diagnostic_augmentation=False if diagnostic else None,
        training_patch_policy='One fixed training patch, no augmentation' if diagnostic else
            'Foreground-centered and whole-FOV uniform blocks; no lesion oversampling; paired i-axis flips and attenuation scaling',
        inference_policy='One fixed training patch; not generalization' if diagnostic else
            'Nonoverlapping whole-FOV tiles, identical for validation and test; no ground-truth crop; padding excluded',
        resampling=dict(array_order='kji', target_spacing_ijk_mm=settings['target_spacing_mm'],
                        grid_alignment='lower voxel face', image='linear', masks='nearest'),
        split_assignment=[_identity(row) for row in rows],
        optimizer_state_saved=False)
    _signed_write(directory/'training.json', metadata)
    (directory/'INCOMPLETE').unlink()
    return metadata


def _save_diagnostic(path, arrays, probability, diagnostic_info):
    np.savez_compressed(path, image=np.asarray(arrays[0], np.float32),
        mask=np.asarray(arrays[1], np.uint8), valid=np.asarray(arrays[2], np.uint8),
        probability=np.asarray(probability, np.float32),
        origin_kji=np.asarray(diagnostic_info['origin_kji'], np.int64),
        spacing_ijk_mm=np.asarray(diagnostic_info['spacing_ijk_mm'], np.float64),
        sample_id=np.asarray(diagnostic_info['sample_id']))


def load_model(model_directory, config, lock, *, diagnostic=False):
    directory = Path(model_directory)
    path = directory/'training.json'
    if (directory/'INCOMPLETE').exists():
        raise ValueError('Training is incomplete')
    if Path(str(path)+'.sha256').read_text().strip() != sha256(path):
        raise ValueError('Training metadata checksum differs')
    metadata = json.loads(path.read_text())
    if (metadata.get('schema_version') != MODEL_SCHEMA or not metadata.get('complete')
            or metadata.get('diagnostic') is not bool(diagnostic)
            or metadata.get('settings') != settings_for(config)
            or metadata.get('lock_fingerprint') != _fingerprint(lock)
            or metadata.get('source_sha256') != sha256(__file__)
            or metadata.get('volume_source_sha256') != sha256(Path(__file__).with_name('fakect_context_volume.py'))
            or metadata.get('model_sha256') != sha256(directory/'checkpoint.keras')
            or metadata.get('history_sha256') != sha256(directory/'history.json')):
        raise ValueError('Model provenance differs from the frozen experiment')
    if diagnostic and metadata.get('diagnostic_prediction_sha256') != sha256(directory/'diagnostic-prediction.npz'):
        raise ValueError('Diagnostic prediction checksum differs')
    model = _tensorflow().keras.models.load_model(directory/'checkpoint.keras', compile=False)
    shape = (None,)+tuple(metadata['settings']['patch_size_kji'])+(1,)
    image_input, probability_output = _single_io(model)
    if tuple(image_input.shape) != shape or tuple(probability_output.shape) != shape:
        raise ValueError('Saved model shape differs from training settings')
    return model, metadata


def evaluate(config, lock, model_directory, output_directory, *, split='test', diagnostic=False):
    rows, settings = inspect_rows(lock), settings_for(config)
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Evaluation split must be train, validation or test')
    model, metadata = load_model(model_directory, config, lock, diagnostic=diagnostic)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'INCOMPLETE').write_text('Evaluation has not completed.\n')
    loaded = set()
    if diagnostic:
        # The default split argument is ignored deliberately: diagnostic results
        # always identify themselves as training-patch measurements.
        row, arrays, info = _diagnostic_selection(rows, settings)
        if info != metadata['diagnostic_patch']:
            raise ValueError('Diagnostic patch differs from the fitted patch')
        loaded.add(row['sample_id'])
        metrics, loss, probability = _block_score(model, arrays, settings)
        _save_diagnostic(directory/'prediction.npz', arrays, probability, info)
        records = [{**_identity(row), 'metrics': metrics, 'loss': loss,
            'tiles': 1, 'diagnostic_patch': info, 'prediction_file': 'prediction.npz',
            'prediction_sha256': sha256(directory/'prediction.npz')}]
        split = 'train'
    else:
        selected = [row for row in rows if row['split'] == split]
        if not selected:
            raise ValueError(f'No {split} samples')
        records = _score_rows(model, selected, settings, loaded)
    result = dict(schema_version=RESULT_SCHEMA, complete=True, diagnostic=bool(diagnostic),
        scope='fixed training-patch learning diagnostic; not generalization' if diagnostic else 'unseen-family synthetic segmentation' if split == 'test' else f'{split} segmentation',
        split=split, settings=settings, lock_fingerprint=_fingerprint(lock),
        model_sha256=metadata['model_sha256'], training_metadata_sha256=sha256(Path(model_directory)/'training.json'),
        inference_policy=metadata['inference_policy'], records=records, summary=summarize(records),
        loaded_samples=sorted(loaded), test_payloads_loaded=not diagnostic and split == 'test',
        diagnostic_before=metadata['diagnostic_before'] if diagnostic else None)
    _signed_write(directory/'results.json', result)
    (directory/'INCOMPLETE').unlink()
    return result
