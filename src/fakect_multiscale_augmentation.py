"""Bounded overview augmentation for anatomy-locator training and fixed QA.

Arrays use [1,K,J,I,2]: normalized attenuation and fractional acquisition
validity. Boxes use [center_kji, extent_kji], normalized to voxel *faces*;
voxel-center coordinate is (index+.5)/shape. The forward spatial map is

    u_output = .5 + scale * (u_input - .5) + translation_kji.

One scale multiplies all physical axes equally, even for anisotropic input
spacing. Translation fractions correspond to each axis's own physical FOV.
Scale>1 zooms in (effective source-space FOV and voxel spacing divide by scale).
The output shape and nominal network coordinate frame remain fixed. Axial
truncation invalidates cells without renumbering them or changing box location.

Random training transforms may use training boxes to ensure retention. Fixed
QA transforms never adapt to the supplied box: ineligible geometry raises a
ValueError. No function performs localization or accesses native image files.
The box is transformed supervision/QA, never an inference input channel.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import affine_transform

SCHEMA = 'fakect.overview-augmentation/1'
MAX_OVERVIEW_VOXELS = 2_000_000


def _triple(value, name, *, nonnegative=False):
    array = np.asarray(value)
    if array.shape != (3,) or array.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must contain three numeric values')
    array = array.astype(np.float64)
    if not np.isfinite(array).all() or (nonnegative and np.any(array < 0)):
        raise ValueError(f'Invalid {name}')
    return array


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be numeric')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'Invalid {name}')
    return value


def _box(value, *, in_fov):
    array = np.asarray(value)
    if array.shape not in ((6,), (1, 6)) or array.dtype.kind not in 'fiu':
        raise ValueError('box must have shape [1,6] or [6]')
    array = array.astype(np.float64)
    flat = array.reshape(6)
    if not np.isfinite(flat).all() or np.any(flat[3:] <= 0):
        raise ValueError('Box must be finite with positive extents')
    if in_fov and (np.any(flat[:3] - flat[3:] / 2 < -1e-7)
                   or np.any(flat[:3] + flat[3:] / 2 > 1 + 1e-7)):
        raise ValueError('Input target box extends beyond its FOV')
    return flat


def _inputs(x, box, safety_context_voxels):
    x = np.asarray(x)
    if x.ndim != 5 or x.shape[0] != 1 or x.shape[-1] != 2 or any(n <= 0 for n in x.shape[1:4]):
        raise ValueError('x must have shape [1,K,J,I,2]')
    if math.prod(x.shape[1:4]) > MAX_OVERVIEW_VOXELS:
        raise ValueError('Overview exceeds the 2,000,000-voxel augmentation budget')
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('x must contain finite float32 values')
    if np.any(x[..., 1] < 0) or np.any(x[..., 1] > 1):
        raise ValueError('Acquisition validity must lie in [0,1]')
    if type(safety_context_voxels) is not int or safety_context_voxels < 0:
        raise ValueError('safety_context_voxels must be a nonnegative integer')
    if np.asarray(box).shape != (1, 6):
        raise ValueError('Supervision box must have shape [1,6]')
    target = _box(box, in_fov=True)
    shape = np.asarray(x.shape[1:4], dtype=np.int64)
    margin = safety_context_voxels / shape
    if np.any(margin >= .5):
        raise ValueError('Safety context leaves no target space in the overview')
    return x, target, shape, margin


def _crop_faces(bounds, axial_size):
    bounds = np.asarray(bounds)
    if bounds.shape != (2,) or bounds.dtype.kind not in 'fiu':
        raise ValueError('crop_bounds_k must contain two normalized axial faces')
    bounds = bounds.astype(np.float64)
    if not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1] <= 1:
        raise ValueError('crop_bounds_k must satisfy 0 <= low < high <= 1')
    # Crop boundaries remove complete cells. Snapping inward never silently
    # retains cells outside the requested acquisition interval.
    low = int(np.ceil(bounds[0] * axial_size - 1e-9))
    high = int(np.floor(bounds[1] * axial_size + 1e-9))
    if high <= low:
        raise ValueError('Axial crop contains no complete output voxels')
    return low, high, bounds


def apply_transform(x, box, scale=1., translation_kji=(0., 0., 0.),
                    crop_bounds_k=(0., 1.), safety_context_voxels=2):
    """Apply one specified transform, rejecting any target/context clipping.

    Scale and translation are never clamped or retried here. Cropping uses
    normalized output k-face bounds and is snapped inward to complete voxels;
    the realized bounds are returned. Safety context is measured in output
    voxels around the exact affine target box on all three axes.

    Attenuation is resampled with validity-weighted trilinear interpolation.
    This preserves constant tissue intensity near invalid borders instead of
    blending padding into anatomy. Validity uses the identical spatial map.
    Unsupported samples and truncated slices have zero intensity and validity.
    """
    x, target, shape, margin = _inputs(x, box, safety_context_voxels)
    scale = _scalar(scale, 'scale', positive=True)
    translation = _triple(translation_kji, 'translation_kji')
    low, high, requested_crop = _crop_faces(crop_bounds_k, int(shape[0]))
    transformed = np.r_[.5 + scale * (target[:3] - .5) + translation, scale * target[3:]]
    target_low, target_high = transformed[:3] - transformed[3:] / 2, transformed[:3] + transformed[3:] / 2
    retained_low, retained_high = np.array([low / shape[0], 0., 0.]), np.array([high / shape[0], 1., 1.])
    if np.any(target_low - margin < retained_low - 1e-7) or np.any(target_high + margin > retained_high + 1e-7):
        raise ValueError('Transform is ineligible: complete target and safety context must remain inside FOV/crop')

    if scale == 1. and np.all(translation == 0):
        result = x.copy()
        result[..., 0][result[..., 1] == 0] = 0
    else:
        # In index-center coordinates q_in=(q_out-c-N*t)/scale+c.
        center = (shape.astype(np.float64) - 1) / 2
        offset = center - (center + shape * translation) / scale
        matrix = np.eye(3, dtype=np.float64) / scale
        valid = x[0, ..., 1]
        weight = affine_transform(valid, matrix, offset=offset, output_shape=tuple(shape),
                                  output=np.float32, order=1, mode='grid-constant', cval=0., prefilter=False)
        np.clip(weight, 0., 1., out=weight)
        weighted_image = np.multiply(x[0, ..., 0], valid, dtype=np.float32)
        numerator = affine_transform(weighted_image, matrix, offset=offset, output_shape=tuple(shape),
                                     output=np.float32, order=1, mode='grid-constant', cval=0., prefilter=False)
        result = np.zeros(x.shape, np.float32)
        np.divide(numerator, weight, out=result[0, ..., 0], where=weight > 0)
        result[0, ..., 1] = weight
    result[:, :low] = 0
    result[:, high:] = 0
    if not np.isfinite(result).all():
        raise ValueError('Augmentation produced nonfinite output')
    metadata = {'schema_version': SCHEMA, 'scale': scale, 'translation_kji': translation.tolist(),
                'crop_bounds_k': [low / int(shape[0]), high / int(shape[0])],
                'requested_crop_bounds_k': requested_crop.tolist(), 'crop_faces_k': [low, high],
                'shape_kji': shape.tolist(), 'safety_context_voxels': safety_context_voxels,
                'coordinate_rule': 'u_output=.5+scale*(u_input-.5)+translation; voxel center=(index+.5)/shape',
                'effective_source_spacing_scale': 1. / scale,
                'interpolation': 'validity-weighted trilinear; zero unsupported/truncated samples',
                'target_retained': True, 'parameter_selection': 'fixed; no target-adaptive resampling'}
    return result, transformed.astype(np.float32)[None], metadata


def augment_overview(x, box, rng, *, translate_fraction_kji, scale_range, crop_fraction_max,
                     safety_context_voxels=2):
    """Sample a feasible training transform without rejecting/distorting labels.

    Scale is sampled uniformly from the feasible subset of ``scale_range``.
    Translations are sampled uniformly from the intersection of the configured
    ranges and complete-target/context constraints. Axial truncation removes a
    random integer number of slices from each end, bounded independently by
    ``crop_fraction_max`` and target retention. Infeasible settings raise rather
    than silently introducing an identity or out-of-range scale.

    This conditional sampling uses *training* supervision. For a predetermined
    robustness probe call ``apply_transform``; report any ineligibility directly.
    """
    x, target, shape, margin = _inputs(x, box, safety_context_voxels)
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy.random.Generator')
    translations = _triple(translate_fraction_kji, 'translate_fraction_kji', nonnegative=True)
    if np.any(translations > 1):
        raise ValueError('translate_fraction_kji cannot exceed one FOV')
    limits = np.asarray(scale_range)
    if limits.shape != (2,) or limits.dtype.kind not in 'fiu':
        raise ValueError('scale_range must contain two positive ordered values')
    limits = limits.astype(np.float64)
    if not np.isfinite(limits).all() or not 0 < limits[0] <= limits[1]:
        raise ValueError('scale_range must contain two positive ordered values')
    crop_max = _scalar(crop_fraction_max, 'crop_fraction_max')
    if not 0 <= crop_max < .5:
        raise ValueError('crop_fraction_max must lie in [0,.5)')

    # Intersect the scale range with all linear target/FOV constraints. There
    # are no iterative retries and no sample-dependent execution-time budget.
    lo, hi = target[:3] - target[3:] / 2, target[:3] + target[3:] / 2
    feasible_max = float(np.min((1 - 2 * margin) / target[3:]))
    for coefficient in (.5 - lo, hi - .5):
        positive = coefficient > 0
        if np.any(positive):
            feasible_max = min(feasible_max, float(np.min((.5 + translations - margin)[positive] / coefficient[positive])))
    upper_scale = min(float(limits[1]), feasible_max)
    if upper_scale < limits[0] - 1e-12:
        raise ValueError('No feasible scale in the configured range retains target/context')
    upper_scale = max(float(limits[0]), upper_scale)
    scale = float(rng.uniform(limits[0], upper_scale)) if upper_scale > limits[0] else upper_scale
    center = .5 + scale * (target[:3] - .5)
    extent = scale * target[3:]
    minimum = np.maximum(-translations, margin + extent / 2 - center)
    maximum = np.minimum(translations, 1 - margin - extent / 2 - center)
    if np.any(minimum > maximum + 1e-12):
        raise ValueError('No feasible translation retains target/context')
    translation = rng.uniform(minimum, np.maximum(minimum, maximum))
    target_low, target_high = center + translation - extent / 2, center + translation + extent / 2
    max_end_slices = int(np.floor(crop_max * shape[0] + 1e-9))
    low_limit = min(max_end_slices, max(0, int(np.floor((target_low[0] - margin[0]) * shape[0] + 1e-9))))
    high_limit = min(max_end_slices, max(0, int(np.floor((1 - target_high[0] - margin[0]) * shape[0] + 1e-9))))
    low = int(rng.integers(low_limit + 1))
    high = int(shape[0]) - int(rng.integers(high_limit + 1))
    result, transformed, metadata = apply_transform(x, box, scale, translation,
        (low / shape[0], high / shape[0]), safety_context_voxels=safety_context_voxels)
    metadata.update(parameter_selection='training-box-conditioned feasible ranges',
                    requested_translate_fraction_kji=translations.tolist(), requested_scale_range=limits.tolist(),
                    feasible_scale_range=[float(limits[0]), upper_scale], crop_fraction_max=crop_max)
    return result, transformed, metadata


def inverse_box(box, metadata):
    """Map predicted/transformed boxes back to the original overview frame.

    Accepts [6] or [1,6] and preserves that shape. Coordinates may lie outside
    the original FOV so a poor prediction remains measurable instead of being
    clipped. Cropping changes validity only and needs no coordinate adjustment.
    """
    original_shape = np.asarray(box).shape
    target = _box(box, in_fov=False)
    if metadata.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported augmentation metadata schema')
    scale = _scalar(metadata['scale'], 'scale', positive=True)
    translation = _triple(metadata['translation_kji'], 'translation_kji')
    result = np.r_[(target[:3] - .5 - translation) / scale + .5, target[3:] / scale]
    return result.astype(np.float32).reshape(original_shape)
