"""Bounded, mask-only preflight for physically aligned target-grid sampling.

These fixed engineering criteria detect material resampling damage. They do not
establish anatomical validity, lumen severity, or clinical acceptability. No
attenuation data is loaded. Native and model arrays use k,j,i ordering.
"""
from collections import Counter
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from fakect_mask_pair_preview import sha256

MAX_NATIVE_VOXELS = 32_000_000
MAX_MODEL_VOXELS = 8_388_608
MAX_AXIS_SAMPLES = 2_000_000
OUTSIDE_FRACTION_INCREASE = .01
MATERIAL_COMPONENT_MM3 = 5.
BLOCKING_FLAGS = ('empty_model_target', 'material_new_disconnection',
                  'introduced_fragment_gt_5_mm3', 'native_component_lost_gt_5_mm3')


def _triple(value, name, integer=False, allow_zero=False):
    value = np.asarray(value)
    if (value.shape != (3,) or value.dtype.kind not in ('iu' if integer else 'iuf') or
            not np.isfinite(value).all() or np.any(value < 0 if allow_zero else value <= 0)):
        raise ValueError('Invalid ' + name)
    return value.astype(np.int64 if integer else np.float64)


def _foreground_box(mask, origin_ijk):
    """Trim unused padding without losing the native global grid origin."""
    indices = [np.flatnonzero(mask.any(axis=tuple(n for n in range(3) if n != axis))) for axis in range(3)]
    low = np.array([int(a[0]) for a in indices])
    high = np.array([int(a[-1]) + 1 for a in indices])
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    return mask[slices], np.asarray(origin_ijk) + low[::-1]


def _sampling_axes(shape_kji, origin_ijk, spacing_ijk_mm, source_shape_kji, target_spacing_ijk_mm):
    origin = np.asarray(origin_ijk, np.int64)[::-1]
    native_spacing = np.asarray(spacing_ijk_mm, float)[::-1]
    target_spacing = np.asarray(target_spacing_ijk_mm, float)[::-1]
    ratio = native_spacing / target_spacing
    full_shape = np.ceil(np.asarray(source_shape_kji) * native_spacing / target_spacing)
    low = np.maximum(0, np.floor(origin * ratio) - 1)
    high = np.minimum(full_shape, np.ceil((origin + shape_kji) * ratio) + 1)
    if (not np.isfinite(high).all() or np.any(high > np.iinfo(np.int64).max) or
            np.any(high <= low) or np.any(high - low > MAX_AXIS_SAMPLES)):
        raise ValueError('Target-grid axis exceeds bounded or representable sampling limits')
    low, high = low.astype(np.int64), high.astype(np.int64)
    if math.prod(map(int, high - low)) > np.iinfo(np.int64).max:
        raise ValueError('Target-grid count exceeds exact integer representation')
    axes = []
    valid = []
    # Match ContextVolume.read_block arithmetic, including subtract/add .5.
    for lo, hi, scale, native_origin, source_size, crop_size in zip(
            low, high, target_spacing / native_spacing, origin, source_shape_kji, shape_kji):
        coords = (np.arange(int(lo), int(hi), dtype=np.float64) + .5) * scale - .5
        nearest = np.floor(coords + .5).astype(np.int64) - native_origin
        axes.append(nearest)
        valid.append((coords >= -.5) & (coords < source_size - .5) & (nearest >= 0) & (nearest < crop_size))
    return axes, valid, low, high - low


def _sample_labels(array, origin_ijk, spacing_ijk_mm, source_shape_kji, target_spacing_ijk_mm,
                   max_model_voxels=MAX_MODEL_VOXELS):
    axes, valid, low, shape = _sampling_axes(array.shape, origin_ijk, spacing_ijk_mm,
                                           source_shape_kji, target_spacing_ijk_mm)
    if math.prod(map(int, shape)) > max_model_voxels:
        raise ValueError(f'Explicit target-mask crop exceeds {max_model_voxels:,} voxels; choose a bounded grid')
    sampled = array[np.ix_(*[np.clip(a, 0, n - 1) for a, n in zip(axes, array.shape)])].copy()
    sampled *= valid[0][:, None, None]
    sampled *= valid[1][None, :, None]
    sampled *= valid[2][None, None, :]
    return sampled, low


def _statistics(counts, voxel_mm3):
    counts = np.asarray(counts, np.int64).copy()
    counts[0] = 0
    foreground, largest = int(counts.sum()), int(counts.max(initial=0))
    return {'count': int(np.count_nonzero(counts)), 'foreground_voxels': foreground,
            'volume_mm3': float(foreground * voxel_mm3), 'largest_component_voxels': largest,
            'largest_component_volume_mm3': float(largest * voxel_mm3),
            'outside_largest_fraction': float(1 - largest / foreground) if foreground else None}


def _audit_array(mask, origin_ijk, spacing_ijk_mm, source_shape_kji, target_spacing_ijk_mm,
                 max_model_voxels=MAX_MODEL_VOXELS):
    mask = np.asarray(mask)
    origin = _triple(origin_ijk, 'native crop origin', integer=True, allow_zero=True)
    native_spacing = _triple(spacing_ijk_mm, 'native spacing')
    target_spacing = _triple(target_spacing_ijk_mm, 'target spacing')
    source_shape = _triple(source_shape_kji, 'source shape', integer=True)
    if (mask.ndim != 3 or not mask.size or mask.size > MAX_NATIVE_VOXELS or
            mask.dtype.kind not in 'bui' or not np.all((mask == 0) | (mask == 1)) or
            np.any(origin + np.asarray(mask.shape)[::-1] > source_shape[::-1])):
        raise ValueError('Invalid or oversized native binary mask/crop geometry')
    native_volume, model_volume = float(np.prod(native_spacing)), float(np.prod(target_spacing))
    if not mask.any():
        return {'ready': False, 'flags': ['empty_model_target'], 'warnings': ['empty_native_target'],
                'native': _statistics([0], native_volume), 'model': _statistics([0], model_volume),
                'method': 'empty native target', 'outside_largest_fraction_change': None,
                'relative_volume_change': None, 'native_components_lost_count': 0,
                'native_components_lost_gt_5_mm3': [], 'introduced_fragments_gt_5_mm3': []}
    original_shape = list(mask.shape)
    mask, origin = _foreground_box(mask.astype(bool, copy=False), origin)
    native_ids, count = label(mask, structure=generate_binary_structure(3, 1))
    native_counts = np.bincount(native_ids.ravel(), minlength=count + 1)
    native_counts[0] = 0
    native = _statistics(native_counts, native_volume)
    axes, valid, model_origin, model_shape = _sampling_axes(mask.shape, origin, native_spacing,
                                                           source_shape, target_spacing)
    fragments, lost_volumes = [], []
    if np.all(target_spacing <= native_spacing):
        # Every native voxel maps to a nonempty rectangular run along each axis.
        # These runs preserve adjacency and separation exactly. Component counts
        # can therefore be computed without constructing a fine 3D model box.
        weights = [np.bincount(a[ok], minlength=n) for a, ok, n in zip(axes, valid, mask.shape)]
        if any(np.any(w < 1) for w in weights):
            raise ValueError('Fine-grid sampling unexpectedly loses native voxel support')
        plane_weights = weights[1][:, None] * weights[2][None, :]
        counts = np.zeros(count + 1, np.int64)
        for k in range(mask.shape[0]):
            # np.bincount weighted sums are exact integers for these bounded
            # native planes; float64 exactly represents all relevant counts.
            weighted = plane_weights * weights[0][k]
            if int(weighted.max(initial=0)) * weighted.size > 2 ** 53:
                raise ValueError('Replication weights exceed exact float64 counting bounds')
            counts += np.bincount(native_ids[k].ravel(), weights=weighted.ravel(), minlength=count + 1).astype(np.int64)
        counts[0] = 0
        model = _statistics(counts, model_volume)
        method = 'exact native-component replication counts; no fine 3D allocation'
    else:
        mapped, model_origin = _sample_labels(native_ids, origin, native_spacing, source_shape,
                                              target_spacing, max_model_voxels)
        model_ids, model_count = label(mapped > 0, structure=generate_binary_structure(3, 1))
        counts = np.bincount(model_ids.ravel(), minlength=model_count + 1)
        counts[0] = 0
        model = _statistics(counts, model_volume)
        observed = np.bincount(mapped.ravel(), minlength=count + 1)
        observed[0] = 0
        lost_volumes = (native_counts[(native_counts > 0) & (observed == 0)] * native_volume).tolist()
        # Count (native component, model component) overlap once. This avoids a
        # full-volume scan for each original component and distinguishes a new
        # split from disconnected structures already present at native spacing.
        foreground = mapped > 0
        encoded = mapped[foreground].astype(np.int64) * (model_count + 1) + model_ids[foreground]
        pairs, overlaps = np.unique(encoded, return_counts=True)
        if len(pairs):
            parents = pairs // (model_count + 1)
            changes = np.r_[0, np.flatnonzero(np.diff(parents)) + 1, len(parents)]
            for lo, hi in zip(changes[:-1], changes[1:]):
                if hi - lo <= 1:
                    continue
                primary = int(lo + np.argmax(overlaps[lo:hi]))
                for index in range(int(lo), int(hi)):
                    volume = float(overlaps[index] * model_volume)
                    if index != primary and volume > MATERIAL_COMPONENT_MM3:
                        component = int(pairs[index] % (model_count + 1))
                        fragments.append({'native_component_id': int(parents[index]),
                                          'model_component_id': component, 'overlap_volume_mm3': volume,
                                          'model_component_volume_mm3': float(counts[component] * model_volume)})
        method = 'explicit physically aligned nearest mask; 6-connected component analysis'
    delta = (model['outside_largest_fraction'] - native['outside_largest_fraction']
             if model['foreground_voxels'] else None)
    ratio = model['volume_mm3'] / native['volume_mm3'] - 1
    material_lost = [float(v) for v in lost_volumes if v > MATERIAL_COMPONENT_MM3]
    flags, warnings = [], []
    if not model['foreground_voxels']:
        flags.append('empty_model_target')
    if delta is not None and delta > OUTSIDE_FRACTION_INCREASE:
        flags.append('material_new_disconnection')
    if fragments:
        flags.append('introduced_fragment_gt_5_mm3')
    if material_lost:
        flags.append('native_component_lost_gt_5_mm3')
    if model['outside_largest_fraction'] is not None and model['outside_largest_fraction'] > .01:
        warnings.append('model_material_disconnection_including_preexisting')
    if abs(ratio) > .1:
        warnings.append('total_volume_change_gt_10_percent')
    return {'ready': not flags, 'flags': flags, 'warnings': warnings, 'native': native, 'model': model,
            'method': method, 'relative_volume_change': ratio, 'outside_largest_fraction_change': delta,
            'native_mask_shape_kji': original_shape, 'audited_foreground_box_origin_ijk': origin.tolist(),
            'audited_foreground_box_shape_kji': list(mask.shape),
            'model_box_origin_kji': model_origin.tolist(), 'model_box_shape_kji': model_shape.tolist(),
            'native_components_lost_count': len(lost_volumes),
            'native_components_lost_gt_5_mm3_count': len(material_lost),
            'native_components_lost_gt_5_mm3': material_lost[:20],
            'introduced_fragments_gt_5_mm3_count': len(fragments),
            'introduced_fragments_gt_5_mm3': fragments[:20],
            'detail_lists_truncated': len(material_lost) > 20 or len(fragments) > 20}


def _load_mask(row):
    path = Path(row['mask_npz'])
    if sha256(path) != row['mask_npz_sha256']:
        raise ValueError('Compact mask checksum changed')
    with zipfile.ZipFile(path) as archive:
        with archive.open('after.npy') as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError('Unsupported mask NPY header version')
        if len(shape) != 3 or math.prod(shape) > MAX_NATIVE_VOXELS or dtype.kind not in 'bui':
            raise ValueError('Oversized or invalid native mask header')
    with np.load(path, allow_pickle=False) as data:
        mask = data['after']
        origin, spacing = data['crop_origin_ijk'], data['spacing_ijk_mm']
        if (data['sample_id'].shape != () or str(data['sample_id']) != row['sample_id'] or
                not np.array_equal(spacing, row['spacing_ijk_mm'])):
            raise ValueError('Compact mask identity or spacing disagrees with dataset')
    return mask, origin, spacing


def audit_target_grid(rows, spacing_ijk_mm):
    """Return plan-ready status and compact diagnostics for every supplied row.

    Four fixed failures block readiness: empty target, >1 percentage-point new
    foreground separation, entirely lost native component >5 mm³, or >5 mm³
    secondary fragment introduced within one formerly connected component.
    Existing disconnection and >10% volume drift remain explicit warnings.
    """
    target = _triple(spacing_ijk_mm, 'target spacing')
    rows = list(rows)
    ids = [row.get('sample_id') for row in rows]
    if not rows or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('Target-grid audit needs nonempty unique sample identities')
    samples, errors = [], []
    for row in rows:
        try:
            mask, origin, native_spacing = _load_mask(row)
            result = _audit_array(mask, origin, native_spacing, row['source_shape_kji'], target)
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
            result = {'ready': False, 'flags': ['invalid_sampling_input'], 'warnings': [], 'error': str(error)}
        result.update(sample_id=row['sample_id'], case_id=row.get('case_id'),
                      mask_npz_sha256=row.get('mask_npz_sha256'))
        samples.append(result)
        if not result['ready']:
            detail = result.get('error') or ', '.join(result['flags'])
            errors.append(f"Target-grid audit failed for {row['sample_id']}: {detail}")
    flag_counts = Counter(flag for result in samples for flag in result['flags'])
    warning_counts = Counter(flag for result in samples for flag in result['warnings'])
    deltas = [r['outside_largest_fraction_change'] for r in samples if r.get('outside_largest_fraction_change') is not None]
    ratios = [r['relative_volume_change'] for r in samples if r.get('relative_volume_change') is not None]
    summary = {'sample_count': len(samples), 'ready_sample_count': sum(r['ready'] for r in samples),
               'failed_sample_count': len(errors), 'flag_counts': dict(flag_counts),
               'warning_counts': dict(warning_counts), 'failed_sample_ids': [r['sample_id'] for r in samples if not r['ready']],
               'maximum_outside_largest_fraction_increase': max(deltas) if deltas else None,
               'relative_volume_change_range': [min(ratios), max(ratios)] if ratios else None}
    return {'schema_version': 'fakect.context-grid-audit/1', 'ready': not errors, 'errors': errors,
            'target_spacing_ijk_mm': target.tolist(), 'connectivity': 6,
            'thresholds': {'outside_largest_fraction_increase': OUTSIDE_FRACTION_INCREASE,
                           'material_component_mm3': MATERIAL_COMPONENT_MM3, 'comparison': 'strictly greater'},
            'limits': {'native_mask_voxels': MAX_NATIVE_VOXELS, 'explicit_model_mask_voxels': MAX_MODEL_VOXELS,
                       'fine_grid_method': 'exact nonempty native-voxel replication; no 3D model allocation'},
            'limitations': ['Engineering resampling criteria, not clinical or anatomical validation.',
                           'Audits the shared lower-face grid phase only; shifted sampling grids are not assessed.',
                           'Preexisting disconnection and large volume changes remain visible warnings.'],
            'summary': summary, 'samples': samples}
