"""Binary anatomical supervision of the complete pretrained cardiac nnU-Net.

The original 19-channel graph and every deep-supervision head are retained.
Default supervision and extraction retain aorta channel 2. Optional grouped
foreground includes the same named major arch branches in training/inference.
Inputs are already resampled 1.5-mm CT arrays, never masks or atlas channels.
"""
from contextlib import nullcontext
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np


AORTA_CHANNEL = 2
NUM_CLASSES = 19
CARDIAC_AORTIC_NAMES = ('aorta', 'brachiocephalic_trunk',
    'subclavian_artery_right', 'subclavian_artery_left',
    'common_carotid_artery_right', 'common_carotid_artery_left')
PATCH_SIZE = (128, 128, 128)
CT_PROPERTIES = {'percentile_00_5': -101., 'percentile_99_5': 813.,
                 'mean': 142.1977119916391, 'std': 154.74032651622053}
SCHEMA = 'fakect.pretrained-cardiac-weights/1'


def _torch():
    import torch
    return torch


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def weights_sha256(model):
    """Fingerprint all state entries, including all pretrained output heads."""
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def _foreground_channels(foreground_channels):
    """Require a nonempty, distinct set of native foreground class indices."""
    try:
        channels = tuple(foreground_channels)
    except TypeError as exc:
        raise ValueError('foreground_channels must be a sequence of integer channels') from exc
    if (not channels or any(isinstance(x, (bool, np.bool_)) or
                            not isinstance(x, (int, np.integer)) for x in channels) or
            len(set(channels)) != len(channels) or
            any(x < 1 or x >= NUM_CLASSES for x in channels)):
        raise ValueError('foreground_channels must contain distinct foreground integers in 1..18')
    return tuple(sorted(map(int, channels)))


def resolve_foreground_channels(labels, foreground_names=('aorta',)):
    """Resolve exact vendor names without changing the official label ontology.

    ``labels`` is the original checkpoint's ``dataset_json['labels']`` map.
    The cardiac head supports aorta and five major arch branch names. Coronary
    arteries have a separate pretrained task, so atlas coronary ostia included
    in a broader training reference remain additional supervision rather than
    an existing coronary channel in this cardiac checkpoint.
    """
    if (not isinstance(labels, dict) or len(labels) != NUM_CLASSES or
            any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in labels.values()) or
            set(labels.values()) != set(range(NUM_CLASSES)) or labels.get('aorta') != AORTA_CHANNEL):
        raise ValueError('Require original Dataset293 labels with aorta channel 2 and 19 outputs')
    if isinstance(foreground_names, str):
        names = (foreground_names,)
    else:
        try:
            names = tuple(foreground_names)
        except TypeError as exc:
            raise ValueError('foreground_names must contain exact cardiac aortic names') from exc
    if (not names or any(not isinstance(name, str) or name not in CARDIAC_AORTIC_NAMES for name in names) or
            len(set(names)) != len(names)):
        raise ValueError('foreground_names must contain distinct exact cardiac aortic names')
    missing = set(names) - set(labels)
    if missing:
        raise ValueError('Requested foreground names are absent from checkpoint labels: ' + ', '.join(sorted(missing)))
    return _foreground_channels(labels[name] for name in names)


def foreground_metadata(foreground_channels=(AORTA_CHANNEL,), labels=None):
    """Record binary target grouping separately from the preserved source labels."""
    channels = _foreground_channels(foreground_channels)
    result = {'source_aorta_channel': AORTA_CHANNEL,
              'foreground_channels': list(channels),
              'extraction': ('19-way argmax == local aorta channel 2' if channels == (AORTA_CHANNEL,)
                             else '19-way argmax in local foreground channels ' + ','.join(map(str, channels))),
              'probability': 'sum of foreground softmax probabilities',
              'binary_logit': 'logsumexp(foreground) - logsumexp(complement)'}
    if labels is not None:
        reverse = {value: name for name, value in labels.items()}
        if any(channel not in reverse for channel in channels):
            raise ValueError('Foreground channels are absent from preserved checkpoint labels')
        result['foreground_names'] = [reverse[channel] for channel in channels]
    if len(channels) > 1:
        result['scope_note'] = ('The cardiac head has no coronary channel. Atlas coronary ostia in a broader '
                                'reference are additional supervised target anatomy, not a source coronary class.')
    return result


def model_metadata(model, *, foreground_channels=(AORTA_CHANNEL,)):
    """Public provenance for initialization, geometry and unchanged heads."""
    metadata = getattr(model, '_fakect_pretrained_metadata', None)
    if metadata is None:
        raise ValueError('Model is missing verified pretrained metadata')
    config = metadata['plans']['configurations'][metadata['configuration']]
    return {'architecture': type(model).__name__,
        'source_checkpoint_sha256': model._fakect_source_checkpoint_sha256,
        'loaded_checkpoint_sha256': model._fakect_loaded_checkpoint_sha256,
        'weights_sha256': weights_sha256(model),
        'parameters': sum(p.numel() for p in model.parameters()),
        'input_channels': 1, 'output_channels': NUM_CLASSES, 'aorta_channel': AORTA_CHANNEL,
        'spacing_mm': list(config['spacing']), 'checkpoint_patch_size': list(config['patch_size']),
        'normalization': dict(CT_PROPERTIES),
        'transpose_forward': metadata['plans'].get('transpose_forward', [0, 1, 2]),
        'source_labels': dict(metadata['dataset_json']['labels']),
        'head_shapes': {key: list(value.shape) for key, value in model.state_dict().items()
                        if '.seg_layers.' in key},
        **foreground_metadata(foreground_channels, metadata['dataset_json']['labels'])}


def normalize_hu(resampled_hu, intensity_properties=None):
    """Official cardiac CT scaling AFTER HU resampling and int32 conversion.

    Geometry/resampling are the data adapter's responsibility. This operation
    alone does not reproduce TotalSegmentator's external crop/resample policy.
    The input is never modified and native cm^-1 attenuation is not accepted.
    """
    source = np.asarray(resampled_hu)
    if source.dtype.kind not in 'iuf' or not np.isfinite(source).all():
        raise ValueError('HU input must be a finite numeric array')
    if source.size and (source.min() < np.iinfo(np.int32).min or
                        source.max() > np.iinfo(np.int32).max):
        raise ValueError('HU input exceeds int32 range')
    properties = CT_PROPERTIES if intensity_properties is None else intensity_properties
    if any(not np.isclose(float(properties.get(k, np.nan)), v, rtol=0, atol=1e-8)
           for k, v in CT_PROPERTIES.items()):
        raise ValueError('Require the frozen Dataset293 CT normalization')
    result = source.astype(np.int32).astype(np.float32)
    np.clip(result, CT_PROPERTIES['percentile_00_5'], CT_PROPERTIES['percentile_99_5'], out=result)
    result -= CT_PROPERTIES['mean']; result /= CT_PROPERTIES['std']
    return result


def _checkpoint_metadata(checkpoint):
    args = checkpoint.get('init_args', {})
    plans, dataset = args.get('plans'), args.get('dataset_json')
    if not isinstance(plans, dict) or not isinstance(dataset, dict):
        raise ValueError('Checkpoint requires original plans and dataset metadata')
    labels = dataset.get('labels', {})
    if labels.get('aorta') != AORTA_CHANNEL or set(labels.values()) != set(range(NUM_CLASSES)):
        raise ValueError('Require original Dataset293 labels with aorta channel 2 and 19 outputs')
    configuration = args.get('configuration', '3d_fullres')
    config = plans['configurations'][configuration]
    if list(config.get('spacing', [])) != [1.5, 1.5, 1.5]:
        raise ValueError('This controlled pilot requires original 1.5-mm spacing')
    if config.get('normalization_schemes') != ['CTNormalization']:
        raise ValueError('Require original CTNormalization scheme')
    properties = plans['foreground_intensity_properties_per_channel']['0']
    normalize_hu(np.zeros(1), properties)
    return {'plans': plans, 'configuration': configuration, 'dataset_json': dataset}


def build_model(checkpoint, device='cpu', deep_supervision=True):
    """Reconstruct the official graph and strictly load ALL pretrained weights.

    nnU-Net's ordinary -pretrained_weights loader intentionally skips output
    heads. This uses its plans/network API with an explicit complete state load.
    Old optimizer state, training epoch and scheduler are never restored.
    """
    torch = _torch()
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    path = Path(checkpoint)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    metadata = _checkpoint_metadata(saved)
    manager = PlansManager(metadata['plans'])
    config = manager.get_configuration(metadata['configuration'])
    model = get_network_from_plans(config.network_arch_class_name,
        config.network_arch_init_kwargs, config.network_arch_init_kwargs_req_import,
        1, NUM_CLASSES, allow_init=False, deep_supervision=deep_supervision)
    model.load_state_dict(saved['network_weights'], strict=True)
    model._fakect_pretrained_metadata = metadata
    loaded_sha256 = file_sha256(path)
    model._fakect_source_checkpoint_sha256 = saved.get('source_checkpoint_sha256', loaded_sha256)
    model._fakect_loaded_checkpoint_sha256 = loaded_sha256
    return model.to(torch.device(device))


def load_weights(model, checkpoint):
    """Load complete model weights only; leave any optimizer explicitly separate."""
    saved = _torch().load(checkpoint, map_location='cpu', weights_only=False)
    metadata = _checkpoint_metadata(saved)
    model.load_state_dict(saved['network_weights'], strict=True)
    digest = file_sha256(checkpoint)
    model._fakect_pretrained_metadata = metadata
    model._fakect_source_checkpoint_sha256 = saved.get('source_checkpoint_sha256', digest)
    model._fakect_loaded_checkpoint_sha256 = digest
    return {'checkpoint_sha256': digest, 'weights_sha256': weights_sha256(model)}


def save_checkpoint(path, model, metadata=None, *, foreground_channels=(AORTA_CHANNEL,)):
    """Publish a fresh immutable model archive; contains no restored optimizer."""
    torch = _torch()
    path = Path(path)
    if path.exists():
        raise FileExistsError(str(path))
    if not hasattr(model, '_fakect_pretrained_metadata'):
        raise ValueError('Model is missing verified pretrained metadata')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    # nnU-Net's decoder also references its encoder. Preserve aliases instead
    # of copying shared GPU storage repeatedly into the checkpoint archive.
    unique = {}
    state = {}
    for key, value in model.state_dict().items():
        identity = (value.data_ptr(), str(value.dtype), tuple(value.shape), tuple(value.stride()))
        if identity not in unique:
            unique[identity] = value.detach().cpu()
        state[key] = unique[identity]
    target = foreground_metadata(foreground_channels, model._fakect_pretrained_metadata['dataset_json']['labels'])
    provenance = {} if metadata is None else json.loads(json.dumps(metadata))
    if ('target_definition' in provenance and provenance['target_definition'] != target):
        raise ValueError('Checkpoint target_definition disagrees with foreground_channels')
    provenance['target_definition'] = target
    archive = {'schema_version': SCHEMA,
        'network_weights': state,
        'init_args': model._fakect_pretrained_metadata,
        'source_checkpoint_sha256': model._fakect_source_checkpoint_sha256,
        'inference_allowed_mirroring_axes': None,
        'metadata': provenance}
    try:
        torch.save(archive, temporary)
        # Hard-link publication refuses overwrite even if another writer won.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(path), 'sha256': file_sha256(path), 'weights_sha256': weights_sha256(model)}


def make_optimizer(model, learning_rate=1e-4, weight_decay=1e-4):
    torch = _torch()
    if not np.isfinite([learning_rate, weight_decay]).all() or learning_rate <= 0 or weight_decay < 0:
        raise ValueError('Require positive finite learning rate and nonnegative weight decay')
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def make_scaler(device='cuda'):
    torch = _torch()
    return torch.amp.GradScaler('cuda', enabled=torch.device(device).type == 'cuda', init_scale=1024.)


def _batch(value, device, name):
    torch = _torch()
    result = torch.as_tensor(value, device=device, dtype=torch.float32)
    if result.ndim == 3:
        result = result[None, None]
    elif result.ndim == 4:
        result = result[None]
    if result.ndim != 5 or result.shape[1] != 1:
        raise ValueError(name + ' must have shape [B,1,D,H,W], [1,D,H,W] or [D,H,W]')
    if not torch.isfinite(result).all():
        raise ValueError(name + ' contains nonfinite values')
    return result


def aorta_logit(logits, *, foreground_channels=(AORTA_CHANNEL,)):
    """Exact binary logit for a preserved output channel or foreground group."""
    torch = _torch()
    if logits.ndim != 5 or logits.shape[1] != NUM_CLASSES:
        raise ValueError('Require complete 19-channel 3D logits')
    channels = _foreground_channels(foreground_channels)
    z = logits.float()
    if channels == (AORTA_CHANNEL,):
        other = torch.cat((z[:, :AORTA_CHANNEL], z[:, AORTA_CHANNEL + 1:]), dim=1)
        return z[:, AORTA_CHANNEL:AORTA_CHANNEL + 1] - torch.logsumexp(other, dim=1, keepdim=True)
    other = tuple(channel for channel in range(NUM_CLASSES) if channel not in channels)
    return (torch.logsumexp(z[:, channels], dim=1, keepdim=True)
            - torch.logsumexp(z[:, other], dim=1, keepdim=True))


def aorta_loss(outputs, target, valid, deep_supervision_weights=None, *, foreground_channels=(AORTA_CHANNEL,)):
    """Masked BCE + foreground Dice; unknown tissue never becomes background.

    Coarser targets use nearest-neighbor sampling. A coarse voxel contributes
    only when its entire fine-grid pooling support is annotated. The default
    level weights follow nnU-Net: 1,1/2,... with the coarsest level excluded.
    """
    torch = _torch()
    import torch.nn.functional as functional
    channels = _foreground_channels(foreground_channels)
    levels = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    if not levels:
        raise ValueError('Require at least one segmentation output')
    y = _batch(target, levels[0].device, 'Target')
    mask = _batch(valid, levels[0].device, 'Validity')
    if y.shape != mask.shape or levels[0].shape[0] != y.shape[0]:
        raise ValueError('Target, validity and prediction batches must align')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('Validity must be binary')
    if not torch.all(((y == 0) | (y == 1)) | (mask == 0)):
        raise ValueError('Annotated targets must be binary aorta masks')
    if not torch.any(mask):
        raise ValueError('Require at least one annotated voxel')
    y = torch.where(mask.bool(), y, torch.zeros_like(y))
    weights = ([1.] if len(levels) == 1 else [2. ** -i for i in range(len(levels) - 1)] + [0.])
    if deep_supervision_weights is not None:
        weights = list(deep_supervision_weights)
    if len(weights) != len(levels) or not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError('Invalid deep-supervision weights')
    terms = []
    for logits, weight in zip(levels, weights):
        if weight == 0:
            continue
        shape = tuple(logits.shape[2:])
        if any(a > b for a, b in zip(shape, y.shape[2:])):
            raise ValueError('Deep supervision cannot enlarge the target grid')
        local_y = functional.interpolate(y, size=shape, mode='nearest')
        local_mask = (functional.adaptive_avg_pool3d(mask, shape) >= 1. - 1e-7).float()
        if not torch.any(local_mask):
            continue
        q = aorta_logit(logits, foreground_channels=channels)
        bce = (functional.binary_cross_entropy_with_logits(q, local_y, reduction='none') * local_mask).sum() / local_mask.sum()
        probability = torch.sigmoid(q) * local_mask
        truth = local_y * local_mask
        dice = 1. - (2. * (probability * truth).sum() + 1e-5) / (probability.sum() + truth.sum() + 1e-5)
        terms.append((float(weight), bce, dice))
    if not terms:
        raise ValueError('No annotated voxels survive active supervision levels')
    total = sum(x[0] for x in terms)
    bce = sum(w * b for w, b, _ in terms) / total
    dice = sum(w * d for w, _, d in terms) / total
    return {'loss': bce + dice, 'bce': bce, 'dice_loss': dice, 'active_levels': len(terms)}


def train_step(model, optimizer, image, target, valid, scaler=None, max_grad_norm=12., *,
               foreground_channels=(AORTA_CHANNEL,)):
    torch = _torch()
    device = next(model.parameters()).device
    x = _batch(image, device, 'Image')
    y, mask = _batch(target, device, 'Target'), _batch(valid, device, 'Validity')
    if x.shape != y.shape or x.shape != mask.shape or x.shape[0] != 1:
        raise ValueError('This controlled pilot requires aligned batch-one image/target/validity')
    if not np.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError('Require positive finite gradient limit')
    if scaler is None and device.type == 'cuda':
        raise ValueError('Pass a persistent CUDA GradScaler for mixed-precision training')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    context = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
    with context:
        losses = aorta_loss(model(x), y, mask, foreground_channels=foreground_channels)
    if not torch.isfinite(losses['loss']):
        raise FloatingPointError('Nonfinite pretrained fine-tuning loss')
    if scaler is not None and scaler.is_enabled():
        scaler.scale(losses['loss']).backward()
        scaler.unscale_(optimizer)
    else:
        losses['loss'].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer); scaler.update()
    else:
        optimizer.step()
    return {**{k: float(v.detach().cpu()) for k, v in losses.items() if k != 'active_levels'},
            'active_levels': losses['active_levels'], 'gradient_norm': float(norm.detach().cpu())}


def predict_tiled(model, image_normalized, patch_size=PATCH_SIZE, step_fraction=.5,
                  device=None, return_probabilities=False, accumulator_memory_mb=512,
                  temporary_directory=None, *, foreground_channels=(AORTA_CHANNEL,)):
    """Gaussian-overlap inference, no mirroring, with shared foreground grouping.

    Accumulate float32 logits on CPU or a bounded disk-backed array. This is a
    common untuned/tuned evaluator; it does not claim bitwise parity with the
    official float16 GPU accumulation path. Coordinates remain on the input
    model grid. Native-grid export belongs to the data adapter.
    """
    torch = _torch()
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
    channels = _foreground_channels(foreground_channels)
    patch = tuple(int(x) for x in patch_size)
    if len(patch) != 3 or min(patch) <= 0 or not 0 < step_fraction <= 1:
        raise ValueError('Invalid inference patch or step fraction')
    if accumulator_memory_mb <= 0:
        raise ValueError('Require positive accumulator memory budget')
    source = np.asarray(image_normalized, dtype=np.float32)
    if source.ndim == 4 and source.shape[0] == 1:
        source = source[0]
    if source.ndim != 3 or min(source.shape) <= 0 or not np.isfinite(source).all():
        raise ValueError('Inference requires one finite normalized CT volume')
    requested_device = next(model.parameters()).device if device is None else torch.device(device)
    if requested_device != next(model.parameters()).device:
        raise ValueError('Model and requested inference device differ')
    padded_shape = tuple(max(a, b) for a, b in zip(source.shape, patch))
    before = tuple((a - b) // 2 for a, b in zip(padded_shape, source.shape))
    padding = [(b, a - c - b) for a, c, b in zip(padded_shape, source.shape, before)]
    image = np.pad(source, padding, mode='constant', constant_values=0)
    steps = compute_steps_for_sliding_window(padded_shape, patch, step_fraction)
    gaussian = compute_gaussian(patch, sigma_scale=1 / 8, value_scaling_factor=1,
                                dtype=torch.float32, device=torch.device('cpu')).numpy()
    previous_training = model.training
    decoder = getattr(model, 'decoder', None)
    previous_ds = getattr(decoder, 'deep_supervision', None)
    if previous_ds is not None:
        decoder.deep_supervision = False
    model.eval()
    memory_bytes = (NUM_CLASSES + 1) * int(np.prod(padded_shape)) * 4
    with tempfile.TemporaryDirectory(prefix='fakect-logits-', dir=temporary_directory) as scratch:
        backed = memory_bytes > accumulator_memory_mb * 1024 ** 2
        def allocate(name, shape):
            if backed:
                value = np.memmap(Path(scratch) / name, mode='w+', shape=shape, dtype=np.float32)
                value[:] = 0
                return value
            return np.zeros(shape, dtype=np.float32)
        logits_sum = allocate('logits.dat', (NUM_CLASSES,) + padded_shape)
        counts = allocate('counts.dat', padded_shape)
        tiles = 0
        try:
            with torch.inference_mode():
                for start in itertools.product(*steps):
                    slices = tuple(slice(s, s + width) for s, width in zip(start, patch))
                    x = torch.from_numpy(np.ascontiguousarray(image[slices]))[None, None].to(requested_device)
                    context = torch.autocast('cuda', dtype=torch.float16) if requested_device.type == 'cuda' else nullcontext()
                    with context:
                        output = model(x)
                    output = output[0] if isinstance(output, (list, tuple)) else output
                    if tuple(output.shape) != (1, NUM_CLASSES) + patch or not torch.isfinite(output).all():
                        raise ValueError('Inference returned invalid 19-channel logits')
                    values = output[0].float().cpu().numpy()
                    logits_sum[(slice(None),) + slices] += values * gaussian[None]
                    counts[slices] += gaussian
                    tiles += 1
            mask = np.empty(source.shape, dtype=np.uint8)
            probability = np.empty(source.shape, dtype=np.float32) if return_probabilities else None
            # Work one axial plane at a time; never materialize a second 19-channel volume.
            for k in range(source.shape[0]):
                section = (k + before[0], slice(before[1], before[1] + source.shape[1]),
                           slice(before[2], before[2] + source.shape[2]))
                denominator = np.asarray(counts[section])
                if np.any(denominator <= 0):
                    raise RuntimeError('Sliding-window coverage contains an unvisited voxel')
                plane = np.asarray(logits_sum[(slice(None),) + section]) / denominator[None]
                winners = np.argmax(plane, axis=0)
                mask[k] = (winners == AORTA_CHANNEL if channels == (AORTA_CHANNEL,)
                           else np.isin(winners, channels))
                if probability is not None:
                    shifted = plane - plane.max(axis=0, keepdims=True)
                    exp = np.exp(shifted)
                    probability[k] = (exp[AORTA_CHANNEL] if channels == (AORTA_CHANNEL,)
                                      else exp[list(channels)].sum(axis=0)) / exp.sum(axis=0)
        finally:
            if previous_ds is not None:
                decoder.deep_supervision = previous_ds
            model.train(previous_training)
            for value in (logits_sum, counts):
                if isinstance(value, np.memmap):
                    value.flush(); value._mmap.close()
        return {'mask': mask, 'probability': probability,
                'metadata': {'tiles': tiles, 'patch_size': list(patch), 'step_fraction': step_fraction,
                    'mirroring': False, **foreground_metadata(channels),
                    'accumulator_dtype': 'float32', 'accumulator_disk_backed': backed,
                    'accumulator_bytes': memory_bytes, 'model_grid_shape': list(source.shape)}}


def production_smoke(checkpoint, image, target, valid, output_directory,
                     device='cuda', steps=2, learning_rate=1e-4, *, foreground_channels=(AORTA_CHANNEL,)):
    """Explicit caller-authorized TRAIN-patch gradient and saved-weight smoke.

    No cohort reader is called here. The orchestrator must supply a permitted
    training patch, bind its hashes, and enforce process host/GPU memory guards.
    """
    torch = _torch()
    if tuple(np.asarray(image).shape[-3:]) != PATCH_SIZE or steps < 1:
        raise ValueError('Production smoke requires a full 128-cubed training patch and positive steps')
    channels = _foreground_channels(foreground_channels)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(str(output))
    output.mkdir(parents=True)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model = build_model(checkpoint, device, True)
    initial = weights_sha256(model)
    optimizer = make_optimizer(model, learning_rate)
    scaler = make_scaler(device)
    updates = [train_step(model, optimizer, image, target, valid, scaler,
                          foreground_channels=channels) for _ in range(steps)]
    changed = weights_sha256(model) != initial
    first = predict_tiled(model, np.asarray(image).reshape(PATCH_SIZE), return_probabilities=True,
                          foreground_channels=channels)
    saved = save_checkpoint(output / 'checkpoint-smoke.pth', model, {'scope': 'TRAIN smoke only'},
                            foreground_channels=channels)
    del optimizer, scaler, model
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    restored = build_model(saved['path'], device, False)
    second = predict_tiled(restored, np.asarray(image).reshape(PATCH_SIZE), return_probabilities=True,
                           foreground_channels=channels)
    maximum_error = float(np.max(np.abs(first['probability'] - second['probability'])))
    parity = bool(np.array_equal(first['mask'], second['mask']) and
                  np.allclose(first['probability'], second['probability'], atol=1e-6, rtol=1e-5))
    final_fingerprint = weights_sha256(restored)
    result = {'passed': changed and parity and final_fingerprint == saved['weights_sha256'],
        'updates': updates, 'weights_changed': changed, 'initial_weights_sha256': initial,
        'final_weights_sha256': final_fingerprint, 'checkpoint': saved,
        'prediction_reload_parity': parity, 'maximum_probability_reload_error': maximum_error,
        'optimizer': 'fresh AdamW', 'learning_rate': learning_rate,
        'target_definition': foreground_metadata(channels, restored._fakect_pretrained_metadata['dataset_json']['labels']),
        'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.device(device).type == 'cuda' else 0.,
        'test_evaluated': False}
    if not result['passed']:
        raise RuntimeError('Pretrained production smoke failed: ' + json.dumps(result))
    return result
