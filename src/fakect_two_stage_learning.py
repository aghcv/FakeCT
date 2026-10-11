"""Bounded 3D TensorFlow pilot: CT→observed aorta, observed→reference aorta.

Native paired data are immutable. Both stages use the same explicit physical
model grid and family partition. Model 2 only receives the observed binary
mask; reference and changed masks are supervision, never inference features.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from fakect_model_resampling import output_shape, resample_pair, spacing
from fakect_segmentation import normalize_image, validate_clip_bounds


MODEL_SCHEMA = 'fakect.two-stage-model/1'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, name, low=1, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low or (high and value > high):
        raise ValueError(f'{name} must be an integer in [{low}, {high or "unbounded"}]')
    return int(value)


def settings_for(config, stage):
    stage = _integer(stage, 'stage', 1, 2)
    source = dict(config.get('model', {}))
    source.update(config.get(f'stage{stage}', {}))
    if source.get('architecture', 'unet3d') != 'unet3d':
        raise ValueError('architecture must be unet3d')
    depth = _integer(source.get('depth', 3), 'depth', 1, 3)
    patch = source.get('patch_size_kji', [64, 64, 64])
    if not isinstance(patch, (list, tuple)) or len(patch) != 3:
        raise ValueError('patch_size_kji needs three dimensions')
    patch = [_integer(n, 'patch_size_kji', 4, 128) for n in patch]
    if any(n % (2**depth) for n in patch):
        raise ValueError('patch dimensions must be divisible by 2**depth')
    clip_min, clip_max = validate_clip_bounds(source.get('clip_min', 0.), source.get('clip_max', .5))
    result = dict(architecture='unet3d', stage=stage, depth=depth, patch_size_kji=patch,
                  base_filters=_integer(source.get('base_filters', 8), 'base_filters', 2, 32),
                  batch_size=_integer(source.get('batch_size', 1), 'batch_size', 1, 16),
                  epochs=_integer(source.get('epochs', 5), 'epochs'),
                  patches_per_volume=_integer(source.get('patches_per_volume', 4), 'patches_per_volume'),
                  max_train_batches_per_epoch=_integer(source.get('max_train_batches_per_epoch', 0), 'max_train_batches_per_epoch', 0),
                  seed=_integer(source.get('seed', 42), 'seed', 0),
                  target_spacing_mm=list(spacing(source.get('target_spacing_mm', [2., 2., 2.]))),
                  clip_min=clip_min, clip_max=clip_max,
                  save_predictions=source.get('save_predictions', True))
    if not isinstance(result['save_predictions'], bool):
        raise ValueError('save_predictions must be Boolean')
    for key, default in [('learning_rate', .001), ('change_weight', 5.), ('threshold', .5)]:
        value = float(source.get(key, default))
        if not math.isfinite(value) or value < 0 or (key == 'learning_rate' and value == 0):
            raise ValueError(f'{key} must be finite and nonnegative (learning rate positive)')
        result[key] = value
    if not 0 < result['threshold'] < 1:
        raise ValueError('threshold must be strictly between zero and one')
    return result


def inspect_manifest(manifest):
    """Metadata-only partition checks: never opens payloads, including test."""
    base = Path.cwd()
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).resolve()
        base = path.parent
        manifest = json.loads(path.read_text())
    samples = manifest.get('samples', [])
    if not samples:
        raise ValueError('Manifest needs samples')
    rows, ids, families, cases, source_splits = [], set(), {}, {}, {}
    for entry in samples:
        row = dict(entry)
        sid = row.get('sample_id') or f'{row.get("case_id", "")}/{row.get("variant_id", "")}'
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError('Every sample_id must be unique')
        ids.add(sid)
        row['sample_id'] = sid
        split = row.get('split')
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Every sample needs train, validation, or test split')
        for key, registry in [('anatomy_family', families), ('case_id', cases)]:
            value = row.get(key)
            if not value:
                raise ValueError(f'Every sample needs {key}')
            if value in registry and registry[value] != split:
                raise ValueError(f'{key} crosses splits; refusing leakage')
            registry[value] = split
        for key in ('source_npz', 'variant_npz'):
            if not row.get(key):
                raise ValueError(f'Every sample needs {key}')
            row[key] = str((base / row[key]).resolve())
        row['source_sha256'] = row.get('source_sha256', row.get('source_npz_sha256'))
        row['variant_sha256'] = row.get('variant_sha256', row.get('variant_npz_sha256'))
        source_identity = row.get('before_sha256') or row.get('before_mask_sha256') or row['source_npz']
        if source_identity in source_splits and source_splits[source_identity] != split:
            raise ValueError('Shared reference anatomy crosses splits')
        source_splits[source_identity] = split
        geometry = row.get('geometry', manifest.get('geometry', {}))
        row['spacing_ijk_mm'] = list(spacing(row.get('spacing_ijk_mm', geometry.get('spacing_ijk_mm'))))
        rows.append(row)
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    return rows, fingerprint


class VolumeLoader:
    """One resampled volume cache, payload hashes checked on first access."""
    def __init__(self, rows, settings, allowed_splits):
        self.rows, self.settings, self.allowed_splits = rows, settings, set(allowed_splits)
        self.cached_id, self.cached = None, None
        self.verified, self.loaded_samples = set(), set()

    def load(self, row):
        if row['split'] not in self.allowed_splits:
            raise ValueError('Payload split is unavailable in this operation')
        if row['sample_id'] == self.cached_id:
            return self.cached
        self.cached, self.cached_id = None, None
        for key, digest_key in [('source_npz', 'source_sha256'), ('variant_npz', 'variant_sha256')]:
            digest = row.get(digest_key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f'Every payload needs {digest_key}')
            pair = (row[key], digest)
            if pair not in self.verified:
                if _sha(row[key]) != digest:
                    raise ValueError(f'Payload hash changed: {row[key]}')
                self.verified.add(pair)
        with np.load(row['source_npz'], allow_pickle=False) as source:
            before = source['target_mask']
        with np.load(row['variant_npz'], allow_pickle=False) as variant:
            image, after = variant['image'], variant['mask']
        if image.dtype != np.float32 or image.ndim != 3 or not np.isfinite(image).all():
            raise ValueError('Image must be finite float32 k,j,i attenuation')
        if image.size > 16_777_216 or 0 in image.shape:
            raise ValueError('Native crop exceeds the bounded volume limit')
        for value in (before, after):
            if value.shape != image.shape or not np.all((value == 0) | (value == 1)):
                raise ValueError('Before/after masks must be binary and match image geometry')
        before, after = before.astype(np.uint8), after.astype(np.uint8)
        native_added = int(np.count_nonzero(after.astype(bool) & ~before.astype(bool)))
        native_removed = int(np.count_nonzero(before.astype(bool) & ~after.astype(bool)))
        native, target = row['spacing_ijk_mm'], self.settings['target_spacing_mm']
        # Apply identical nearest-neighbor coordinates to both masks. Changes
        # are recomputed after resampling to preserve exact Boolean identities.
        model_image, after = resample_pair(image, after, native, target)
        from scipy.ndimage import affine_transform
        scale = np.asarray(target[::-1]) / np.asarray(native[::-1])
        before = affine_transform(before, matrix=np.diag(scale), offset=(scale-1)/2,
                                  output_shape=model_image.shape, order=0, mode='nearest', prefilter=False,
                                  output=np.uint8)
        self.cached = dict(image=model_image, before=before, after=after,
                           added=(after.astype(bool) & ~before.astype(bool)),
                           removed=(before.astype(bool) & ~after.astype(bool)),
                           native_added_voxels=native_added, native_removed_voxels=native_removed)
        self.cached_id = row['sample_id']
        self.loaded_samples.add(row['sample_id'])
        return self.cached


def tile_origins(shape, patch, overlap=False):
    """Cover every voxel, anchoring the last tile to the far volume boundary."""
    axes = []
    for size, width in zip(shape, patch):
        step = max(1, width // 2) if overlap else width
        if overlap:
            positions = list(range(0, max(1, size-width+1), step))
            if positions[-1] != max(0, size-width):
                positions.append(max(0, size-width))
        else:
            positions = list(range(0, size, step))
        axes.append(positions)
    return list(itertools.product(*axes))


def _crop(array, origin, patch):
    result = np.zeros(tuple(patch), np.float32)
    sizes = tuple(min(p, n-o) for p, n, o in zip(patch, array.shape, origin))
    src = tuple(slice(o, o+n) for o, n in zip(origin, sizes))
    dest = tuple(slice(0, n) for n in sizes)
    result[dest] = array[src]
    valid = np.zeros_like(result)
    valid[dest] = 1
    return result, valid


def training_origins(volume, settings, rng):
    """Half change-centered, quarter target-centered, quarter uniform patches.

    Sampling includes controls and background. It does not use test anatomy.
    """
    shape, patch = volume['after'].shape, settings['patch_size_kji']
    changed = np.flatnonzero(volume['added'] | volume['removed'])
    foreground = np.flatnonzero(volume['after'] | volume['before'])
    result = []
    for index in range(settings['patches_per_volume']):
        pool = changed if index % 4 < 2 and changed.size else foreground if index % 4 < 3 else []
        if len(pool):
            center = np.unravel_index(int(rng.choice(pool)), shape)
            origin = tuple(max(0, min(n-p, int(c)-p//2)) for c, n, p in zip(center, shape, patch))
        else:
            origin = tuple(int(rng.integers(0, max(1, n-p+1))) for n, p in zip(shape, patch))
        result.append(origin)
    return result


def _batches(loader, rows, settings, split, epoch=0):
    rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], epoch]))
    selected = [row for row in rows if row['split'] == split]
    if split == 'train':
        rng.shuffle(selected)
    for row in selected:
        volume = loader.load(row)
        inputs = normalize_image(volume['image'], settings) if settings['stage'] == 1 else volume['after']
        target = volume['after'] if settings['stage'] == 1 else volume['before']
        changed = volume['added'] | volume['removed']
        origins = training_origins(volume, settings, rng) if split == 'train' else tile_origins(target.shape, settings['patch_size_kji'])
        for start in range(0, len(origins), settings['batch_size']):
            group = origins[start:start+settings['batch_size']]
            x = np.zeros((len(group),) + tuple(settings['patch_size_kji']) + (1,), np.float32)
            y = np.zeros(x.shape[:-1] + (3,), np.float32)
            for index, origin in enumerate(group):
                x[index, ..., 0], _ = _crop(inputs, origin, settings['patch_size_kji'])
                y[index, ..., 0], y[index, ..., 1] = _crop(target, origin, settings['patch_size_kji'])
                y[index, ..., 2], _ = _crop(changed, origin, settings['patch_size_kji'])
            yield x, y


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('Fit/test require TensorFlow; use the confirmed tensorflow-gpu/2.17 environment') from exc
    return tf


def _loss_function(tf, change_weight):
    def loss(y_true, y_pred):
        truth, valid, changed = y_true[..., :1], y_true[..., 1:2], y_true[..., 2:3]
        pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        weight = valid * (1 + change_weight * changed)
        axes = (1, 2, 3, 4)
        bce = -(truth*tf.math.log(pred) + (1-truth)*tf.math.log(1-pred))
        bce = tf.reduce_sum(weight*bce, axes) / tf.maximum(tf.reduce_sum(weight, axes), 1.)
        dice = (2*tf.reduce_sum(truth*pred*valid, axes)+1.) / (tf.reduce_sum((truth+pred)*valid, axes)+1.)
        return bce + 1-dice
    return loss


def build_unet3d(settings):
    tf = _tensorflow()
    layers = tf.keras.layers
    inputs = layers.Input(tuple(settings['patch_size_kji']) + (1,), name='attenuation' if settings['stage'] == 1 else 'observed_aorta')
    def block(x, filters):
        for _ in range(2):
            x = layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
        return x
    x, skips = inputs, []
    for level in range(settings['depth']):
        x = block(x, settings['base_filters'] * 2**level)
        skips.append(x)
        # TensorFlow 2.17 GPU max-pool gradients use nondeterministic
        # select-and-scatter under XLA. Learned strided convolutions preserve
        # the same spatial pyramid while supporting deterministic GPU fitting.
        x = layers.Conv3D(settings['base_filters'] * 2**level, 2, strides=2,
                          padding='same', activation='relu')(x)
    x = block(x, settings['base_filters'] * 2**settings['depth'])
    for level in reversed(range(settings['depth'])):
        x = layers.Concatenate()([layers.UpSampling3D(2)(x), skips[level]])
        x = block(x, settings['base_filters'] * 2**level)
    output = layers.Conv3D(1, 1, activation='sigmoid', name='aorta_probability' if settings['stage'] == 1 else 'reference_probability')(x)
    model = tf.keras.Model(inputs, output, name=f'fakect_stage{settings["stage"]}_unet3d')
    model.compile(optimizer=tf.keras.optimizers.Adam(settings['learning_rate']),
                  loss=_loss_function(tf, settings['change_weight'] if settings['stage'] == 2 else 0.),
                  jit_compile=False)
    return model


def build_training_step(model, settings):
    """One tensor-signature graph; avoid TF 2.17's iterator-per-batch retraces.

    Keras 3's older train_on_batch wrapper passes a fresh Python generator to
    a tf.function on every call. Those graphs retain batches and exhaust memory
    for real volumes. Calling the same public train_step with TensorSpecs keeps
    optimizer/loss/metric behavior while bounding the trace cache.
    """
    tf = _tensorflow()
    model.optimizer.build(model.trainable_variables)
    shape = (None,) + tuple(settings['patch_size_kji'])
    @tf.function(input_signature=[tf.TensorSpec(shape+(1,), tf.float32),
                                  tf.TensorSpec(shape+(3,), tf.float32)], jit_compile=False)
    def step(inputs, packed_targets):
        return model.train_step((inputs, packed_targets, None))
    return step


def predict_volume(model, volume, patch_size, batch_size=1):
    """50% overlap, average probabilities, zero padded outside the crop."""
    values = np.asarray(volume)
    if values.ndim != 3 or not np.isfinite(values).all() or 0 in values.shape:
        raise ValueError('Inference volume must be finite and nonempty')
    if values.size > 16_777_216:
        raise ValueError('Inference volume exceeds the bounded volume limit')
    batch_size = _integer(batch_size, 'batch_size')
    origins = tile_origins(values.shape, patch_size, overlap=True)
    total, count = np.zeros(values.shape, np.float32), np.zeros(values.shape, np.uint16)
    for start in range(0, len(origins), batch_size):
        group = origins[start:start+batch_size]
        x = np.stack([_crop(values, origin, patch_size)[0] for origin in group])[..., None]
        probabilities = np.asarray(model.predict_on_batch(x))
        if probabilities.shape != x.shape or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError('Invalid sigmoid prediction')
        for index, origin in enumerate(group):
            sizes = tuple(min(p, n-o) for p, n, o in zip(patch_size, values.shape, origin))
            dest = tuple(slice(o, o+n) for o, n in zip(origin, sizes))
            src = tuple(slice(0, n) for n in sizes)
            total[dest] += probabilities[index, ..., 0][src]
            count[dest] += 1
    if np.any(count == 0):
        raise RuntimeError('Inference failed to cover the whole volume')
    return total / count


def change_probabilities(observed_mask, reference_probability):
    observed = np.asarray(observed_mask)
    probability = np.asarray(reference_probability)
    if observed.shape != probability.shape or not np.all((observed == 0) | (observed == 1)):
        raise ValueError('Observed mask must be binary and match reference probability')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Reference probability must be finite in [0,1]')
    return observed * (1-probability), (1-observed.astype(np.float32)) * probability


def change_masks(observed_mask, reference_probability, threshold=.5):
    """Threshold reference once, ensuring exact reconstruction identities."""
    change_probabilities(observed_mask, reference_probability)
    observed = np.asarray(observed_mask, bool)
    reference = np.asarray(reference_probability) >= threshold
    return observed & ~reference, ~observed & reference


def binary_metrics(truth, predicted, voxel_volume_mm3):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp, fp, fn = [int(np.count_nonzero(x)) for x in (truth & predicted, ~truth & predicted, truth & ~predicted)]
    denominator = 2*tp + fp + fn
    return dict(dice=2*tp/denominator if denominator else 1.,
                precision=tp/(tp+fp) if tp+fp else (1. if not truth.any() else 0.),
                recall=tp/(tp+fn) if tp+fn else 1.,
                true_positive=tp, false_positive=fp, false_negative=fn,
                true_voxels=int(truth.sum()), predicted_voxels=int(predicted.sum()),
                false_positive_mm3=fp*voxel_volume_mm3, false_negative_mm3=fn*voxel_volume_mm3)


def fit_stage(config, manifest, stage, outdir):
    settings = settings_for(config, stage)
    rows, fingerprint = inspect_manifest(manifest)
    if not all(any(r['split'] == split for r in rows) for split in ('train', 'validation')):
        raise ValueError('Fit needs nonempty family-disjoint train and validation splits')
    directory = Path(outdir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Model output exists and is not empty')
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(settings['seed'])
    tf.config.experimental.enable_op_determinism()
    model = build_unet3d(settings)
    train_step = build_training_step(model, settings)
    loss_fn = _loss_function(tf, settings['change_weight'] if stage == 2 else 0.)
    loader = VolumeLoader(rows, settings, ['train', 'validation'])
    directory.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, history = float('inf'), None, []
    for epoch in range(settings['epochs']):
        epoch_start = time.monotonic()
        train_sum, train_count, train_batches = 0., 0, 0
        for x, y in _batches(loader, rows, settings, 'train', epoch):
            model.reset_metrics()
            loss = float(train_step(x, y)['loss'])
            if not math.isfinite(loss):
                raise ValueError('Training loss is nonfinite')
            train_sum += loss*len(x)
            train_count += len(x)
            train_batches += 1
            if settings['max_train_batches_per_epoch'] and train_batches >= settings['max_train_batches_per_epoch']:
                break
        train_seconds = time.monotonic()-epoch_start
        validation_start = time.monotonic()
        val_sum, val_count, tp, fp, fn = 0., 0, 0, 0, 0
        for x, y in _batches(loader, rows, settings, 'validation'):
            probability = np.asarray(model.predict_on_batch(x))
            losses = np.asarray(loss_fn(y, probability))
            if not np.isfinite(losses).all():
                raise ValueError('Validation loss is nonfinite')
            val_sum += float(losses.sum())
            val_count += len(x)
            truth, pred, valid = y[..., 0] > .5, probability[..., 0] >= settings['threshold'], y[..., 1] > .5
            tp += int(np.count_nonzero(truth & pred & valid))
            fp += int(np.count_nonzero(~truth & pred & valid))
            fn += int(np.count_nonzero(truth & ~pred & valid))
        val_loss = val_sum/val_count
        record = dict(epoch=epoch+1, train_loss=train_sum/train_count, train_patches=train_count,
                      train_batches=train_batches, validation_loss=val_loss, validation_patches=val_count,
                      validation_dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                      train_seconds=train_seconds, validation_seconds=time.monotonic()-validation_start,
                      epoch_seconds=time.monotonic()-epoch_start,
                      train_function_traces=train_step.experimental_get_tracing_count())
        if record['train_function_traces'] > 2:
            raise RuntimeError('Training function retraced unexpectedly; refusing unbounded graph growth')
        history.append(record)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch+1
            model.save(directory/'model.keras')
        (directory/'history.json').write_text(json.dumps(history, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(stage=stage, **record)), flush=True)
    metadata = dict(schema_version=MODEL_SCHEMA, stage=stage, settings=settings,
                    manifest_fingerprint=fingerprint, model_sha256=_sha(directory/'model.keras'),
                    history_sha256=_sha(directory/'history.json'),
                    source_sha256=_sha(__file__), best_epoch=best_epoch, best_validation_loss=best_loss,
                    epochs_completed=len(history), history=history, loaded_samples=sorted(loader.loaded_samples),
                    test_payloads_loaded=False, model_selection='minimum validation weighted-BCE plus Dice loss',
                    tensorflow_version=tf.__version__, numpy_version=np.__version__,
                    devices=[device.name for device in tf.config.list_logical_devices()],
                    downsampling='learned stride-2 Conv3D', jit_compile=False,
                    training_function='fixed tensor-signature tf.function calling model.train_step',
                    train_function_traces=train_step.experimental_get_tracing_count(),
                    split_assignment=[{key: row[key] for key in ('sample_id', 'case_id', 'anatomy_family', 'split')} for row in rows],
                    training_patch_policy='Per-volume sampled: half change-centered, quarter foreground, quarter uniform; controls retained; seeded shuffle each epoch',
                    validation_policy='Every validation voxel once in nonoverlapping tiles; padding excluded; no test selection',
                    inference_policy='50% overlapping tiles, average probabilities, zero padding outside crop',
                    resampling=dict(array_order='kji', grid_alignment='lower_voxel_face', image='linear', masks='nearest',
                                    target_spacing_ijk_mm=settings['target_spacing_mm']),
                    stage2_input='ground-truth after mask only' if stage == 2 else None)
    (directory/'model-metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    (directory/'model-metadata.json.sha256').write_text(_sha(directory/'model-metadata.json')+'\n')
    return metadata


def load_stage(model_path):
    path = Path(model_path)
    directory = path if path.is_dir() else path.parent
    checksum = directory/'model-metadata.json.sha256'
    if not checksum.is_file() or checksum.read_text().strip() != _sha(directory/'model-metadata.json'):
        raise ValueError('Model metadata checksum is absent or changed')
    metadata = json.loads((directory/'model-metadata.json').read_text())
    if metadata.get('schema_version') != MODEL_SCHEMA:
        raise ValueError('Unsupported two-stage model metadata')
    if _sha(directory/'model.keras') != metadata['model_sha256']:
        raise ValueError('Model hash differs from recorded provenance')
    if _sha(directory/'history.json') != metadata.get('history_sha256'):
        raise ValueError('Training history differs from recorded provenance')
    settings = settings_for({'model': metadata['settings']}, metadata['stage'])
    model = _tensorflow().keras.models.load_model(directory/'model.keras', compile=False)
    expected = tuple(settings['patch_size_kji']) + (1,)
    if tuple(model.input_shape[1:]) != expected or tuple(model.output_shape[1:]) != expected:
        raise ValueError('Saved model shapes disagree with metadata')
    return model, metadata


def _summarize_records(records):
    result = {}
    keys = records[0]['metrics']
    for channel in keys:
        families, cases = {}, {}
        for row in records:
            cases.setdefault(row['case_id'], []).append(row['metrics'][channel])
            families.setdefault(row['anatomy_family'], set()).add(row['case_id'])
        case_means = {case: {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
                      for case, metrics in cases.items()}
        family_means = {family: {key: float(np.mean([case_means[case][key] for case in sorted(case_ids)]))
                                 for key in next(iter(case_means.values()))}
                        for family, case_ids in families.items()}
        result[channel] = dict(case_means=case_means, family_means=family_means,
                               family_macro={key: float(np.mean([m[key] for m in family_means.values()])) for key in next(iter(family_means.values()))},
                               sample_macro={key: float(np.mean([row['metrics'][channel][key] for row in records])) for key in records[0]['metrics'][channel]})
    return result


def evaluate_stage(config, manifest, stage, model_path, outdir, stage1_model_path=None):
    """Evaluate every test crop; optional end-to-end second-stage comparison.

    Chained change masks are derived from the actual first-stage prediction S,
    not substituted oracle A. Their truth remains actual A−B and B−A, exposing
    rather than hiding segmentation-induced false positive changes.
    """
    rows, fingerprint = inspect_manifest(manifest)
    selected = [row for row in rows if row['split'] == 'test']
    if not selected:
        raise ValueError('No test samples')
    model, metadata = load_stage(model_path)
    if metadata['stage'] != stage or metadata['manifest_fingerprint'] != fingerprint:
        raise ValueError('Model stage or frozen sample partition differs from evaluation')
    settings = metadata['settings']
    requested = settings_for(config, stage)
    if requested != settings:
        raise ValueError('Evaluation settings differ from trained model; use the frozen configuration')
    first, first_meta = None, None
    if stage1_model_path:
        if stage != 2:
            raise ValueError('Chained evaluation applies to stage 2 only')
        first, first_meta = load_stage(stage1_model_path)
        if first_meta['stage'] != 1 or first_meta['manifest_fingerprint'] != fingerprint:
            raise ValueError('First-stage model has a different dataset partition')
        if first_meta['settings']['target_spacing_mm'] != settings['target_spacing_mm']:
            raise ValueError('Both stages must use the same physical model grid')
    directory = Path(outdir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Evaluation output exists and is not empty')
    directory.mkdir(parents=True, exist_ok=True)
    loader = VolumeLoader(rows, settings, ['test'])
    voxel_volume = float(np.prod(settings['target_spacing_mm']))
    records = []
    for index, row in enumerate(selected):
        volume = loader.load(row)
        probability = predict_volume(model, normalize_image(volume['image'], settings) if stage == 1 else volume['after'],
                                     settings['patch_size_kji'], settings['batch_size'])
        arrays, metrics, confounding = {}, {}, {}
        if stage == 1:
            arrays['aorta_probability'] = probability
            metrics['aorta'] = binary_metrics(volume['after'], probability >= settings['threshold'], voxel_volume)
        else:
            arrays['oracle_reference_probability'] = probability
            added, removed = change_probabilities(volume['after'], probability)
            arrays.update(oracle_added_probability=added, oracle_removed_probability=removed)
            add_mask, remove_mask = change_masks(volume['after'], probability, settings['threshold'])
            arrays.update(oracle_added_mask=add_mask, oracle_removed_mask=remove_mask)
            for prefix, reference, add, remove in [('oracle', probability, added, removed),
                                                   ('identity', volume['after'], np.zeros_like(added), np.zeros_like(removed))]:
                add_mask, remove_mask = change_masks(volume['after'], reference, settings['threshold'])
                metrics[f'{prefix}_reference'] = binary_metrics(volume['before'], reference >= settings['threshold'], voxel_volume)
                metrics[f'{prefix}_added'] = binary_metrics(volume['added'], add_mask, voxel_volume)
                metrics[f'{prefix}_removed'] = binary_metrics(volume['removed'], remove_mask, voxel_volume)
            if first is not None:
                first_settings = first_meta['settings']
                predicted_after_probability = predict_volume(first, normalize_image(volume['image'], first_settings),
                                                           first_settings['patch_size_kji'], first_settings['batch_size'])
                predicted_after = (predicted_after_probability >= first_settings['threshold']).astype(np.uint8)
                reference = predict_volume(model, predicted_after, settings['patch_size_kji'], settings['batch_size'])
                added, removed = change_probabilities(predicted_after, reference)
                add_mask, remove_mask = change_masks(predicted_after, reference, settings['threshold'])
                arrays.update(stage1_aorta_probability=predicted_after_probability,
                              chained_reference_probability=reference, chained_added_probability=added, chained_removed_probability=removed,
                              chained_added_mask=add_mask, chained_removed_mask=remove_mask)
                residual_add, residual_remove = change_masks(predicted_after, volume['before'], settings['threshold'])
                segmentation_error = predicted_after.astype(bool) != volume['after'].astype(bool)
                arrays.update(chained_residual_added_truth=residual_add, chained_residual_removed_truth=residual_remove,
                              segmentation_error_mask=segmentation_error)
                metrics['stage1_aorta'] = binary_metrics(volume['after'], predicted_after, voxel_volume)
                metrics['chained_reference'] = binary_metrics(volume['before'], reference >= settings['threshold'], voxel_volume)
                metrics['chained_added'] = binary_metrics(volume['added'], add_mask, voxel_volume)
                metrics['chained_removed'] = binary_metrics(volume['removed'], remove_mask, voxel_volume)
                metrics['chained_residual_added'] = binary_metrics(residual_add, add_mask, voxel_volume)
                metrics['chained_residual_removed'] = binary_metrics(residual_remove, remove_mask, voxel_volume)
                confounding = dict(segmentation_error_voxels=int(segmentation_error.sum()),
                                   chained_added_on_segmentation_error_voxels=int(np.count_nonzero(add_mask & segmentation_error)),
                                   chained_removed_on_segmentation_error_voxels=int(np.count_nonzero(remove_mask & segmentation_error)))
        record = {key: row[key] for key in ('sample_id', 'case_id', 'anatomy_family')}
        record.update(metrics=metrics, shape_kji=list(volume['after'].shape),
                      is_control=not bool(volume['native_added_voxels'] or volume['native_removed_voxels']),
                      native_added_voxels=volume['native_added_voxels'], native_removed_voxels=volume['native_removed_voxels'],
                      added_truth_voxels=int(volume['added'].sum()), removed_truth_voxels=int(volume['removed'].sum()))
        record.update(confounding)
        if settings['save_predictions']:
            filename = f'{index:04d}-prediction.npz'
            np.savez_compressed(directory/filename, **arrays, after=volume['after'], before=volume['before'],
                                added=volume['added'], removed=volume['removed'],
                                spacing_ijk_mm=np.asarray(settings['target_spacing_mm']),
                                sample_id=np.asarray(row['sample_id']))
            record['prediction_file'] = filename
            record['prediction_sha256'] = _sha(directory/filename)
        records.append(record)
        print(json.dumps(dict(stage=stage, evaluated=index+1, total=len(selected), sample_id=row['sample_id'])), flush=True)
    controls = [record for record in records if record['is_control']]
    report = dict(schema_version='fakect.two-stage-evaluation/1', stage=stage,
                  model_sha256=metadata['model_sha256'], manifest_fingerprint=fingerprint,
                  stage1_model_sha256=first_meta['model_sha256'] if first_meta else None,
                  grid='resampled physical model grid; not native-grid measurements',
                  target_spacing_ijk_mm=settings['target_spacing_mm'], threshold=settings['threshold'],
                  test_samples=len(records), records=records, summary=_summarize_records(records),
                  controls_summary=_summarize_records(controls) if controls else {},
                  empty_truth_and_prediction_dice=1.,
                  interpretation='Synthetic held-out anatomy families; reconstruction is a plausible learned reference, not a clinical normality determination')
    (directory/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (directory/'evaluation.json.sha256').write_text(_sha(directory/'evaluation.json')+'\n')
    return report
