"""Small reproducible 2D U-Net experiment on prepared synthetic XCAT crops.

TensorFlow is imported only when building, fitting or loading a model. Dataset
inspection and patch preprocessing use NumPy, plus SciPy for optional physical
resampling. Legacy datasets hold out scenarios from one anatomy; model-only
experiments hold out reviewed anatomy families. Test payloads are never opened
by this training runner.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MODEL_SCHEMA = 'fakect.segmentation-model/1'
MAX_VOLUME_VOXELS = 16_777_216


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def validate_clip_bounds(clip_min, clip_max):
    """Require an ordered clipping interval usable by float32 image arithmetic.

    Preserve the user's finite Python values in metadata, but reject endpoints
    that round together and ranges that overflow or disappear in float32.
    This helper imports neither TensorFlow nor image data.
    """
    try:
        lo, hi = float(clip_min), float(clip_max)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Fixed clipping requires finite clip_min < clip_max') from exc
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError('Fixed clipping requires finite clip_min < clip_max')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        lower, upper = np.float32(lo), np.float32(hi)
        span = np.float32(upper - lower)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError('Fixed clipping bounds must remain finite and distinct in float32')
    if not np.isfinite(span) or span <= 0:
        raise ValueError('Fixed clipping interval width must remain positive and finite in float32')
    return lo, hi


def model_settings(config):
    """Validate one normalization and model contract shared by training/inference."""
    source = config.get('model', {})
    if source.get('architecture', 'unet2d') != 'unet2d':
        raise ValueError('model.architecture must be unet2d')
    if source.get('slice_axis', 'k') != 'k':
        raise ValueError('model.slice_axis must be k (native axial slices)')
    if source.get('normalization', 'fixed_clip') != 'fixed_clip':
        raise ValueError('model.normalization must be fixed_clip')
    patch = source.get('patch_size', (64, 64))
    if not isinstance(patch, (list, tuple)) or len(patch) != 2:
        raise ValueError('model.patch_size must contain j,i dimensions')
    patch = tuple(_integer(n, 'model.patch_size', 4) for n in patch)
    if any(n % 4 or n > 512 for n in patch):
        raise ValueError('Each patch dimension must be a multiple of 4 and <= 512')
    lo, hi = validate_clip_bounds(source.get('clip_min', 0), source.get('clip_max', .5))
    rate = float(source.get('learning_rate', .001))
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('model.learning_rate must be positive and finite')
    return {'architecture': 'unet2d', 'patch_size': list(patch), 'slice_axis': 'k',
            'normalization': 'fixed_clip', 'clip_min': lo, 'clip_max': hi,
            'epochs': _integer(source.get('epochs', 1), 'model.epochs'),
            'batch_size': _integer(source.get('batch_size', 4), 'model.batch_size'),
            'learning_rate': rate, 'seed': _integer(source.get('seed', 1729), 'model.seed', 0)}


def normalize_image(image, settings):
    """Fixed 1/cm clipping, then [0,1] scaling; no sample-derived statistics."""
    values = np.asarray(image)
    if values.dtype.kind not in 'iuf' or not np.all(np.isfinite(values)):
        raise ValueError('Images must contain finite real attenuation values')
    lo, hi = validate_clip_bounds(settings['clip_min'], settings['clip_max'])
    lower, upper = np.float32(lo), np.float32(hi)
    clipped = np.clip(values, lower, upper).astype(np.float32)
    return ((clipped - lower) / np.float32(upper - lower)).astype(np.float32)


def inspect_manifest(manifest_path):
    """Validate split/provenance metadata without opening sample NPZ payloads."""
    path = Path(manifest_path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema_version') == 'fakect.model-experiment-lock/1':
        from fakect_model_experiment import inspect_experiment_lock
        return inspect_experiment_lock(path)
    if data.get('schema_version') != 'fakect.training-dataset/1':
        raise ValueError('Unsupported training dataset manifest schema')
    if data.get('split_mode') != 'scenario_only':
        raise ValueError('This first runner supports only the explicit scenario_only protocol')
    if not data.get('anatomy_family'):
        raise ValueError('Dataset must identify its shared anatomy_family')
    if data.get('image_units') != 'cm^-1':
        raise ValueError('Dataset image_units must be cm^-1')
    geometry = data.get('geometry', {})
    if geometry.get('array_order') != 'kji':
        raise ValueError('Dataset array_order must be kji')
    shape = geometry.get('crop_shape_kji')
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError('Dataset needs a three-dimensional crop_shape_kji')
    shape = tuple(_integer(n, 'crop_shape_kji') for n in shape)
    if math.prod(shape) > MAX_VOLUME_VOXELS:
        raise ValueError('Prepared crops exceed the bounded segmentation volume limit')
    samples = data.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('Dataset manifest has no samples')
    ids, paths, split_groups, split_masks = set(), set(), {}, {}
    split_sizes = {'train': 0, 'validation': 0, 'test': 0}
    normalized = []
    for sample in samples:
        row = dict(sample)
        variant, split = row.get('variant_id'), row.get('split')
        if not isinstance(variant, str) or not variant or variant in ids:
            raise ValueError('Every sample requires a unique nonempty variant_id')
        ids.add(variant)
        if split not in split_sizes:
            raise ValueError('Sample split must be train, validation or test')
        if row.get('anatomy_family', data['anatomy_family']) != data['anatomy_family']:
            raise ValueError('scenario_only samples must use the declared shared anatomy family')
        relative = Path(row.get('path', ''))
        if relative.is_absolute() or not str(row.get('path', '')):
            raise ValueError('Sample paths must be relative to the manifest')
        resolved = (path.parent / relative).resolve()
        try:
            resolved.relative_to(path.parent)
        except ValueError:
            raise ValueError('Sample path leaves the prepared dataset directory') from None
        if resolved in paths:
            raise ValueError('A sample file cannot appear more than once in the manifest')
        paths.add(resolved)
        for name, registry in (('geometry_group', split_groups), ('mask_sha256', split_masks)):
            key = row.get(name)
            if not isinstance(key, str) or not key:
                raise ValueError(f'Every sample requires {name} for split integrity')
            if key in registry and registry[key] != split:
                raise ValueError(f'{name} appears in multiple splits; refusing leakage between train/validation/test')
            registry[key] = split
        for name in ('sha256', 'image_sha256', 'mask_sha256'):
            digest = row.get(name)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'Sample {name} must be a lowercase SHA256 digest')
        if tuple(row.get('shape_kji', shape)) != shape:
            raise ValueError('Sample shape metadata disagrees with the dataset crop')
        split_sizes[split] += 1
        row['_path'] = resolved
        normalized.append(row)
    if not split_sizes['train'] or not split_sizes['validation']:
        raise ValueError('Training requires nonempty train and validation scenario splits')
    return {'path': path, 'sha256': _sha256(path), 'data': data, 'samples': normalized,
            'shape_kji': shape, 'split_sizes': split_sizes}


class PatchDataset:
    """Metadata-first axial patch index with a single decoded-volume cache.

    Patches tile every source slice without overlap or skipped dimensions.
    Lower/right padding is excluded from the loss and validation measurements
    using the paired valid mask. Background-only patches remain in the dataset.
    """
    def __init__(self, manifest_path, settings):
        self.manifest = inspect_manifest(manifest_path)
        self.settings = dict(settings)
        self.shape = self.manifest['shape_kji']
        self.patch_size = tuple(settings['patch_size'])
        self._cache_path = None
        self._cache = None
        self.loaded_variants = set()
        self._patches = {'train': [], 'validation': []}
        for sample_index, sample in enumerate(self.manifest['samples']):
            split = sample['split']
            if split == 'test':
                continue
            shape = tuple(sample.get('shape_kji', self.shape))
            for k in range(shape[0]):
                for j in range(0, shape[1], self.patch_size[0]):
                    for i in range(0, shape[2], self.patch_size[1]):
                        self._patches[split].append((sample_index, k, j, i))

    def patches(self, split):
        if split not in self._patches:
            raise ValueError('Only train and validation patches are available; test is held untouched')
        return tuple(self._patches[split])

    def ordered_patches(self, split, epoch=0):
        patches = self.patches(split)
        if split != 'train':
            return patches
        # Shuffle variants and patches within a variant, keeping each decoded
        # volume together rather than repeatedly decompressing all data.
        rng = np.random.default_rng(np.random.SeedSequence([self.settings['seed'], int(epoch)]))
        groups = {}
        for patch in patches:
            groups.setdefault(patch[0], []).append(patch)
        order = list(groups)
        rng.shuffle(order)
        result = []
        for index in order:
            group = groups[index]
            rng.shuffle(group)
            result.extend(group)
        return tuple(result)

    def _load(self, sample_index):
        sample = self.manifest['samples'][sample_index]
        if sample['split'] == 'test':
            raise ValueError('Test payloads are not available to the training runner')
        path = sample['_path']
        if self._cache_path == path:
            return self._cache
        # Drop the previous decoded volume before decompressing the next.
        self._cache = None
        self._cache_path = None
        if _sha256(path) != sample['sha256']:
            raise ValueError(f'Sample file hash changed: {sample["variant_id"]}')
        with np.load(path, allow_pickle=False) as archive:
            image, mask = archive['image'], archive['mask']
        shape = tuple(sample.get('source_shape_kji', sample.get('shape_kji', self.shape)))
        if image.dtype != np.float32 or image.shape != shape or not np.all(np.isfinite(image)):
            raise ValueError('Sample image must be finite float32 matching crop_shape_kji')
        if mask.dtype != np.uint8 or mask.shape != shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError('Sample mask must be uint8 {0,1} matching crop_shape_kji')
        for array, name in ((image, 'image_sha256'), (mask, 'mask_sha256')):
            if hashlib.sha256(array.tobytes(order='C')).hexdigest() != sample[name]:
                raise ValueError(f'Sample {name} disagrees with its payload')
        if 'target_spacing_ijk_mm' in sample:
            from fakect_model_resampling import resample_pair
            image, mask = resample_pair(image, mask, sample['source_spacing_ijk_mm'], sample['target_spacing_ijk_mm'])
            if image.shape != tuple(sample['shape_kji']):
                raise ValueError('Resampled shape disagrees with frozen experiment geometry')
        # Do not retain a second full normalized volume in the cache.
        self._cache = (image, mask)
        self._cache_path = path
        self.loaded_variants.add(sample['variant_id'])
        return self._cache

    def batch(self, records):
        height, width = self.patch_size
        images = np.zeros((len(records), height, width, 1), dtype=np.float32)
        masks = np.zeros_like(images)
        valid = np.zeros_like(images)
        for index, (sample_index, k, j, i) in enumerate(records):
            image, mask = self._load(sample_index)
            h, w = min(height, image.shape[1] - j), min(width, image.shape[2] - i)
            images[index, :h, :w, 0] = normalize_image(image[k, j:j+h, i:i+w], self.settings)
            masks[index, :h, :w, 0] = mask[k, j:j+h, i:i+w]
            valid[index, :h, :w, 0] = 1
            del image, mask  # Let the cache release this volume before loading another.
        return images, masks, valid

    def batches(self, split, epoch=0):
        records = self.ordered_patches(split, epoch)
        batch_size = self.settings['batch_size']
        for start in range(0, len(records), batch_size):
            yield self.batch(records[start:start+batch_size])


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('TensorFlow is required only for fit/predict. Use the confirmed tensorflow-gpu/2.17 container environment.') from exc
    return tf


def _loss_function(tf):
    @tf.keras.utils.register_keras_serializable(package='FakeCT', name='valid_bce_dice')
    def valid_bce_dice(y_true, y_pred):
        # Source masks occupy channel zero; channel one excludes padding.
        truth, valid = y_true[..., :1], y_true[..., 1:2]
        pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        axes = (1, 2, 3)
        bce = -(truth * tf.math.log(pred) + (1-truth) * tf.math.log(1-pred))
        bce = tf.reduce_sum(bce * valid, axis=axes) / tf.maximum(tf.reduce_sum(valid, axis=axes), 1.)
        numerator = 2 * tf.reduce_sum(truth * pred * valid, axis=axes) + 1.
        denominator = tf.reduce_sum((truth + pred) * valid, axis=axes) + 1.
        return bce + 1 - numerator / denominator
    return valid_bce_dice


def build_unet(settings):
    """Two pooling levels, 16/32/64 filters, one sigmoid target channel."""
    tf = _tensorflow()
    layers = tf.keras.layers
    inputs = layers.Input(shape=tuple(settings['patch_size']) + (1,), name='attenuation')
    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        return layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    skip1 = block(inputs, 16)
    skip2 = block(layers.MaxPool2D(2)(skip1), 32)
    x = block(layers.MaxPool2D(2)(skip2), 64)
    x = block(layers.Concatenate()([layers.UpSampling2D(2)(x), skip2]), 32)
    x = block(layers.Concatenate()([layers.UpSampling2D(2)(x), skip1]), 16)
    output = layers.Conv2D(1, 1, activation='sigmoid', name='target_probability')(x)
    model = tf.keras.Model(inputs, output, name='fakect_unet2d')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings['learning_rate']),
                  loss=_loss_function(tf))
    return model


def _validation(model, dataset, tf):
    loss_fn = _loss_function(tf)
    total_loss, examples, tp, fp, fn, valid_voxels = 0., 0, 0, 0, 0, 0
    for images, masks, valid in dataset.batches('validation'):
        prediction = np.asarray(model.predict_on_batch(images))
        if prediction.shape != masks.shape or not np.all(np.isfinite(prediction)):
            raise ValueError('Model produced invalid validation probabilities')
        losses = np.asarray(loss_fn(np.concatenate((masks, valid), axis=-1), prediction))
        total_loss += float(losses.sum())
        examples += len(images)
        truth, positive, keep = masks > .5, prediction >= .5, valid > .5
        tp += int(np.count_nonzero(truth & positive & keep))
        fp += int(np.count_nonzero(~truth & positive & keep))
        fn += int(np.count_nonzero(truth & ~positive & keep))
        valid_voxels += int(keep.sum())
    denominator = 2*tp + fp + fn
    return {'loss': total_loss/examples, 'dice': 2*tp/denominator if denominator else 1.,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
            'valid_voxels': valid_voxels, 'patches': examples, 'threshold': .5,
            'empty_truth_and_prediction_dice': 1.0}


def train_segmentation(config, manifest_path):
    """Fit train patches, select the best epoch using validation; never open test.

    The caller must explicitly choose the study's fit stage. This function does
    not submit jobs or prepare/alter the supplied synthetic dataset.
    """
    settings = model_settings(config)
    dataset = PatchDataset(manifest_path, settings)
    directory = config.get('train', {}).get('model_directory')
    if not directory:
        raise ValueError('train.model_directory is required')
    output = Path(directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Model directory exists and is not empty; choose a new model_directory')
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(settings['seed'])
    tf.config.experimental.enable_op_determinism()
    model = build_unet(settings)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / 'model.keras'
    history = []
    best_loss, best_epoch = float('inf'), None
    for epoch in range(settings['epochs']):
        weighted_loss, examples = 0., 0
        for images, masks, valid in dataset.batches('train', epoch):
            model.reset_metrics()
            result = model.train_on_batch(images, np.concatenate((masks, valid), axis=-1), return_dict=True)
            loss = float(result['loss'])
            if not np.isfinite(loss):
                raise ValueError('Training loss is nonfinite; no completed model metadata was written')
            weighted_loss += loss * len(images)
            examples += len(images)
        validation = _validation(model, dataset, tf)
        if not np.isfinite(validation['loss']):
            raise ValueError('Validation loss is nonfinite')
        record = {'epoch': epoch+1, 'train_loss': weighted_loss/examples, 'train_patches': examples,
                  'validation': validation}
        history.append(record)
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch+1
            model.save(model_path)
        (output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'epoch': epoch+1, 'train_loss': record['train_loss'],
                          'validation_loss': validation['loss'], 'validation_dice': validation['dice']}), flush=True)
    data = dataset.manifest['data']
    if _sha256(dataset.manifest['path']) != dataset.manifest['sha256']:
        raise ValueError('Dataset manifest changed during training')
    metadata = {'schema_version': MODEL_SCHEMA, 'architecture': settings['architecture'],
                'settings': settings, 'normalization': {'method': 'fixed_clip', 'clip_min': settings['clip_min'],
                'clip_max': settings['clip_max'], 'input_units': 'cm^-1', 'output_range': [0, 1]},
                'source_manifest': str(dataset.manifest['path']), 'source_manifest_sha256': dataset.manifest['sha256'],
                'split_mode': data['split_mode'], 'anatomy_family': data['anatomy_family'],
                'case_id': data.get('case_id'), 'frame': data.get('frame'), 'geometry': data['geometry'],
                'image_method': data.get('image_method'), 'split_sizes': dataset.manifest['split_sizes'],
                'split_assignment': [{'variant_id': s['variant_id'], 'split': s['split'],
                                     'geometry_group': s['geometry_group'], 'mask_sha256': s['mask_sha256']}
                                    for s in dataset.manifest['samples']],
                'loaded_variants': sorted(dataset.loaded_variants), 'test_payloads_loaded': False,
                'test_evaluated': False, 'model_selection': 'lowest validation BCE-plus-Dice loss; test untouched',
                'evaluation_scope': 'Synthetic scenario holdout within one shared anatomy; no independent-patient or independent-anatomy generalization claim.',
                'patch_policy': 'All axial slices; nonoverlapping j,i tiles; lower/right zero padding excluded from loss and validation metrics; no foreground-only filtering.',
                'data_order': 'Seeded variant order and per-variant patch order; one decoded volume cache; no expanded tensor shuffle buffer.',
                'determinism': 'TensorFlow seed and deterministic ops enabled; bitwise equivalence across hardware/software is not claimed.',
                'best_epoch': best_epoch, 'best_validation_loss': best_loss, 'epochs_completed': len(history),
                'history': history, 'tensorflow_version': tf.__version__, 'numpy_version': np.__version__,
                'model_file': model_path.name, 'model_sha256': _sha256(model_path),
                'module_sha256': _sha256(__file__), 'weights_include_test_selection': False}
    if data['split_mode'] == 'anatomy_family':
        metadata.update(
            evaluation_scope='Holdout of verified source anatomy families; synthetic CT proxy data, not a patient-generalization claim.',
            family_assignment=data['family_assignment'], source_datasets=data['source_datasets'],
            preprocessing=data.get('preprocessing', {}))
    (output / 'model-metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata


def predict_axial_slice(model, image_ji, settings, batch_size=None):
    """Predict a whole native axial slice with the exact training tile contract."""
    image = np.asarray(image_ji)
    if image.ndim != 2 or 0 in image.shape:
        raise ValueError('Inference requires a nonempty axial j,i image')
    height, width = settings['patch_size']
    batch_size = settings.get('batch_size', 4) if batch_size is None else _integer(batch_size, 'batch_size')
    prediction = np.empty(image.shape, dtype=np.float32)
    positions = [(j, i) for j in range(0, image.shape[0], height) for i in range(0, image.shape[1], width)]
    for start in range(0, len(positions), batch_size):
        records = positions[start:start+batch_size]
        batch = np.zeros((len(records), height, width, 1), dtype=np.float32)
        for index, (j, i) in enumerate(records):
            h, w = min(height, image.shape[0]-j), min(width, image.shape[1]-i)
            batch[index, :h, :w, 0] = normalize_image(image[j:j+h, i:i+w], settings)
        values = np.asarray(model.predict_on_batch(batch))
        if values.shape != batch.shape or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError('Model produced invalid sigmoid prediction shape or probabilities')
        for index, (j, i) in enumerate(records):
            h, w = min(height, image.shape[0]-j), min(width, image.shape[1]-i)
            prediction[j:j+h, i:i+w] = values[index, :h, :w, 0]
    return prediction


def load_segmentation(model_directory):
    """Load frozen best-validation model and its normalization for inference."""
    directory = Path(model_directory)
    metadata = json.loads((directory / 'model-metadata.json').read_text())
    if metadata.get('schema_version') != MODEL_SCHEMA or metadata.get('model_file') != 'model.keras':
        raise ValueError('Unsupported segmentation model metadata')
    settings = model_settings({'model': metadata['settings']})
    path = directory / 'model.keras'
    if _sha256(path) != metadata['model_sha256']:
        raise ValueError('Saved model hash differs from recorded model provenance')
    tf = _tensorflow()
    model = tf.keras.models.load_model(path, compile=False)
    if tuple(model.input_shape[1:]) != tuple(settings['patch_size']) + (1,) or tuple(model.output_shape[1:]) != tuple(settings['patch_size']) + (1,):
        raise ValueError('Saved model shapes disagree with preprocessing metadata')
    return model, metadata
