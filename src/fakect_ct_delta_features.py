"""Baseline-relative CT features in a fixed physical aortic coordinate system.

These exploratory features require the paired baseline.  They are not inputs
available for an unpaired patient scan.  Positive/negative attenuation changes
are NOT equivalent to morphological addition/removal: identical tissue
attenuation can make an anatomical edit invisible in CT.

The frozen baseline centerline and RAW-derived reference radius define the
coordinates; no sibling labels, edit parameters, severity or source IDs enter
the descriptor.  The baseline target mask supplies only an attenuation scale.
Native samples contribute their physical voxel volume to cylindrical bins,
normalized by L*R**2 and the dimensionless bin volume.  Thus the representation
has neither a native-voxel count nor a phantom-size unit.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fakect_centerline_frame import build_centerline_frame


SCHEMA = 'fakect.baseline-relative-ct-features/1'
MAX_NATIVE_VOXELS = 32_000_000
CHUNK_VOXELS = 131_072


def _readonly(array):
    result = np.asarray(array)
    result.flags.writeable = False
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(row, key, verify):
    path = Path(row[key])
    if verify and _sha256(path) != row[key + '_sha256']:
        raise ValueError('Frozen CT input checksum changed: ' + key)
    return path


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be finite and positive')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return value


@dataclass(frozen=True)
class CaseGeometry:
    baseline_image_cm_inverse: np.ndarray
    bin_index: np.ndarray
    normalized_bin_volume: np.ndarray
    metadata: dict

    @property
    def feature_count(self):
        return 2 * len(self.normalized_bin_volume)


def prepare_array_geometry(baseline_image_cm_inverse, baseline_target_mask,
                           geometry, resolved_config, selection_manifest, *,
                           arc_bins=32, radial_bins=4, angular_bins=8,
                           radial_extent=4., delta_tolerance_cm_inverse=1e-6):
    """Prepare a reusable descriptor map from an already-decoded baseline.

    ``baseline_image_cm_inverse`` is already converted to physical attenuation
    units; use ``prepare_case_geometry`` for the native per-pixel source NPZ.
    This lower-level entry point supports unit/resolution fixtures without
    manufacturing cohort labels or loading whole-body data.
    """
    shape = tuple(int(v) for v in geometry['crop_shape_kji'])
    if len(shape) != 3 or min(shape) <= 0 or np.prod(shape) > MAX_NATIVE_VOXELS:
        raise ValueError('Invalid or excessive native crop shape')
    image = np.asarray(baseline_image_cm_inverse, dtype=np.float32)
    target = np.asarray(baseline_target_mask, dtype=bool)
    if image.shape != shape or target.shape != shape:
        raise ValueError('Baseline arrays do not match the native crop')
    if not np.all(np.isfinite(image)):
        raise ValueError('Baseline CT must be finite')
    spacing = np.asarray(geometry['spacing_ijk_mm'], dtype=float)
    origin = np.asarray(geometry['crop_origin_ijk'], dtype=float)
    if (spacing.shape != (3,) or origin.shape != (3,) or
            not np.all(np.isfinite(spacing)) or np.any(spacing <= 0) or
            not np.all(np.isfinite(origin))):
        raise ValueError('Native spacing and crop origin must be finite')
    reference_radius = _positive(selection_manifest['aorta_median_ring_radius_mm'],
                                 'aorta_median_ring_radius_mm')
    radial_extent = _positive(radial_extent, 'radial_extent')
    tolerance = _positive(delta_tolerance_cm_inverse, 'delta_tolerance_cm_inverse')
    bins = (arc_bins, radial_bins, angular_bins)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           or not 1 <= v <= 128 for v in bins) or np.prod(bins) > 65_536:
        raise ValueError('Descriptor bin counts must be bounded positive integers')
    if not np.any(target):
        raise ValueError('Baseline arterial reference mask is empty')
    attenuation_scale = float(np.median(image[target]))
    if not np.isfinite(attenuation_scale) or attenuation_scale <= tolerance:
        raise ValueError('Baseline arterial attenuation scale is not positive')
    nodes = np.asarray(resolved_config['roi']['center_ijk'], dtype=float)
    if not np.array_equal(nodes, np.asarray(geometry['roi_nodes_ijk'], dtype=float)):
        raise ValueError('Frozen main ROI controls disagree with source geometry')
    frame_config = resolved_config.get('centerline', {})
    frame = build_centerline_frame(nodes, spacing,
        smoothing_mm=frame_config.get('smoothing_mm', 1.),
        sample_step_mm=min(1., max(.25, reference_radius / 8)),
        min_curvature_per_mm=frame_config.get('min_curvature_per_mm', .002))
    xyz = frame['xyz_mm']
    path_distances = frame['spline_distance_mm']
    length = float(path_distances[-1])
    normal = frame['reference_normal']
    segments = np.diff(xyz, axis=0)
    segment_square = np.einsum('ij,ij->i', segments, segments)
    if np.any(segment_square <= 0):
        raise ValueError('Sampled reference centerline has a zero-length segment')
    tree = cKDTree(xyz)
    # The native frame uses index*spacing. RAW voxel centers use this offset.
    selection_geometry = selection_manifest.get('geometry', {})
    raw_boundary = np.asarray(selection_geometry.get('grid_boundary_origin_raw_mm',
        selection_manifest.get('coordinate_contract', {}).get(
            'grid_boundary_origin_raw_mm', [0., 0., 0.])), dtype=float)
    if raw_boundary.shape != (3,) or not np.all(np.isfinite(raw_boundary)):
        raise ValueError('Invalid RAW coordinate origin')
    slab = selection_manifest.get('z_slab_mm')
    if slab is not None:
        slab = np.asarray(slab, dtype=float)
        if slab.shape != (2,) or not np.all(np.isfinite(slab)) or slab[0] >= slab[1]:
            raise ValueError('Invalid frozen thoracic landmark slab')
    crop_low = selection_geometry.get('crop_low_ijk')
    crop_high = selection_geometry.get('crop_high_ijk_exclusive')
    if (crop_low is None) != (crop_high is None):
        raise ValueError('Thoracic anatomical crop requires both bounds')
    if crop_low is not None:
        crop_low, crop_high = np.asarray(crop_low, float), np.asarray(crop_high, float)
        if (crop_low.shape != (3,) or crop_high.shape != (3,) or
                not np.all(np.isfinite(crop_low)) or not np.all(np.isfinite(crop_high)) or
                np.any(crop_high <= crop_low)):
            raise ValueError('Invalid frozen thoracic anatomical crop')
    index = np.full(int(np.prod(shape)), -1, dtype=np.int32)
    for start in range(0, len(index), CHUNK_VOXELS):
        stop = min(start + CHUNK_VOXELS, len(index))
        flat = np.arange(start, stop)
        ijk = np.stack(np.unravel_index(flat, shape)[::-1], axis=1) + origin
        positions = ijk * spacing
        _, nearest = tree.query(positions, workers=1)
        candidates = np.stack((np.clip(nearest - 1, 0, len(segments) - 1),
                               np.clip(nearest, 0, len(segments) - 1)), axis=1)
        relative = positions[:, None, :] - xyz[candidates]
        fraction_raw = np.einsum('nci,nci->nc', relative, segments[candidates]) / segment_square[candidates]
        fraction = np.clip(fraction_raw, 0., 1.)
        offsets = relative - fraction[:, :, None] * segments[candidates]
        best = np.argmin(np.einsum('nci,nci->nc', offsets, offsets), axis=1)
        row = np.arange(len(flat))
        seg = candidates[row, best]
        u = fraction[row, best]
        u_raw = fraction_raw[row, best]
        offset = offsets[row, best]
        tangent = segments[seg] / np.sqrt(segment_square[seg, None])
        ref_normal = (1 - u[:, None]) * normal[seg] + u[:, None] * normal[seg + 1]
        ref_normal -= np.sum(ref_normal * tangent, axis=1)[:, None] * tangent
        ref_normal /= np.linalg.norm(ref_normal, axis=1)[:, None]
        ref_binormal = np.cross(tangent, ref_normal)
        radial = np.linalg.norm(offset, axis=1) / reference_radius
        angle = np.mod(np.arctan2(np.sum(offset * ref_binormal, axis=1),
                                np.sum(offset * ref_normal, axis=1)), 2 * np.pi)
        arc = (path_distances[seg] + u * np.diff(path_distances)[seg]) / length
        valid = (radial < radial_extent)
        valid &= ~((seg == 0) & (u_raw < 0))
        valid &= ~((seg == len(segments) - 1) & (u_raw > 1))
        if slab is not None:
            raw_z = positions[:, 2] + raw_boundary[2] + .5 * spacing[2]
            valid &= (raw_z >= slab[0]) & (raw_z <= slab[1])
        if crop_low is not None:
            valid &= np.all((ijk >= crop_low) & (ijk < crop_high), axis=1)
        a = np.minimum((arc * arc_bins).astype(int), arc_bins - 1)
        r = np.minimum((radial / radial_extent * radial_bins).astype(int), radial_bins - 1)
        t = np.minimum((angle / (2 * np.pi) * angular_bins).astype(int), angular_bins - 1)
        index[flat[valid]] = ((a * radial_bins + r) * angular_bins + t)[valid]
    radial_edges = np.linspace(0., radial_extent, radial_bins + 1)
    bin_volumes = np.broadcast_to(
        (np.diff(radial_edges ** 2) * np.pi / (arc_bins * angular_bins))[None, :, None],
        (arc_bins, radial_bins, angular_bins)).copy().ravel()
    metadata = {
        'schema_version': SCHEMA, 'array_order': 'kji',
        'shape_kji': list(shape), 'crop_origin_ijk': origin.tolist(),
        'spacing_ijk_mm': spacing.tolist(), 'voxel_volume_mm3': float(np.prod(spacing)),
        'arc_bins': int(arc_bins), 'radial_bins': int(radial_bins),
        'angular_bins': int(angular_bins), 'radial_extent': radial_extent,
        'attenuation_scale_cm_inverse': attenuation_scale,
        'delta_tolerance_cm_inverse': tolerance,
        'centerline_length_mm': length, 'reference_radius_mm': reference_radius,
        'normalization_volume_mm3': length * reference_radius ** 2,
        'focused_native_voxels': int(np.sum(index >= 0)),
        'focused_volume_mm3': float(np.sum(index >= 0) * np.prod(spacing)),
        'frame': frame['metadata'],
        'normal_direction': 'baseline transported reference, theta=0 along reference_normal',
        'feature_order': 'positive then negative attenuation; each channel arc,radial,angle C-order',
        'feature_inputs': ['paired attenuation', 'baseline centerline geometry',
                           'baseline reference radius', 'baseline arterial attenuation scale'],
        'target_mask_role': 'baseline median attenuation scale only; never selects descriptor voxels',
        'semantic_warning': 'CT delta sign is not anatomical addition/removal; equal attenuation edits can be invisible',
    }
    return CaseGeometry(_readonly(image.copy()), _readonly(index),
                        _readonly(bin_volumes), metadata)


def prepare_case_geometry(row, resolved_config, selection_manifest, *, verify=True, **settings):
    """Decode and verify a single baseline native crop, shared by all siblings."""
    if row.get('source_atn_units') != 'per_pixel' or row.get('image_units') != 'cm^-1':
        raise ValueError('Require native per_pixel baseline and cm^-1 sibling image units')
    source_path = _checked_file(row, 'source_npz', verify)
    with np.load(source_path, allow_pickle=False) as source:
        geometry = json.loads(str(source['geometry_json'].item()))
        attenuation = source['original_attenuation_per_pixel']
        target = source['target_mask']
    if (list(geometry['crop_origin_ijk']) != list(row['edit_crop_origin_ijk']) or
            list(geometry['crop_shape_kji']) != list(row['edit_crop_shape_kji']) or
            list(geometry['spacing_ijk_mm']) != list(row['spacing_ijk_mm'])):
        raise ValueError('Paired row does not match the baseline native geometry')
    # Keep the exact production exporter conversion and float32 arithmetic.
    image = np.asarray(attenuation / (geometry['spacing_ijk_mm'][0] / 10), dtype=np.float32)
    result = prepare_array_geometry(image, target, geometry, resolved_config,
                                    selection_manifest, **settings)
    result.metadata['source_npz'] = str(source_path)
    result.metadata['source_npz_sha256'] = row.get('source_npz_sha256')
    return result


def extract_array(case, sibling_image_cm_inverse):
    """Integrate the CT-only descriptor and return separate diagnostic counts."""
    image = np.asarray(sibling_image_cm_inverse)
    if image.shape != case.baseline_image_cm_inverse.shape or not np.all(np.isfinite(image)):
        raise ValueError('Sibling CT must be finite and match its paired baseline crop')
    delta = image.astype(np.float64).ravel() - case.baseline_image_cm_inverse.ravel()
    active = np.abs(delta) > case.metadata['delta_tolerance_cm_inverse']
    focused = case.bin_index >= 0
    selected = active & focused
    local_delta = delta[selected] / case.metadata['attenuation_scale_cm_inverse']
    bins = case.bin_index[selected]
    count = len(case.normalized_bin_volume)
    scale = case.metadata['voxel_volume_mm3'] / case.metadata['normalization_volume_mm3']
    positive = np.bincount(bins, weights=np.maximum(local_delta, 0), minlength=count)
    negative = np.bincount(bins, weights=np.maximum(-local_delta, 0), minlength=count)
    features = np.r_[positive, negative] * scale / np.tile(case.normalized_bin_volume, 2)
    all_energy = float(np.sum(np.abs(delta[active])))
    captured_energy = float(np.sum(np.abs(delta[selected])))
    total_voxels = int(np.sum(active))
    focused_voxels = int(np.sum(selected))
    summary = {
        'ct_near_baseline': focused_voxels == 0,
        'native_ct_changed_voxels': total_voxels,
        'focused_ct_changed_voxels': focused_voxels,
        'focused_positive_attenuation_voxels': int(np.sum(delta[selected] > 0)),
        'focused_negative_attenuation_voxels': int(np.sum(delta[selected] < 0)),
        'ct_changed_volume_mm3': total_voxels * case.metadata['voxel_volume_mm3'],
        'focused_ct_changed_volume_mm3': focused_voxels * case.metadata['voxel_volume_mm3'],
        'normalized_ct_changed_volume': focused_voxels * scale,
        'normalized_positive_ct_energy': float(np.sum(positive) * scale),
        'normalized_negative_ct_energy': float(np.sum(negative) * scale),
        'ct_delta_absolute_energy_cm_inverse_mm3': all_energy * case.metadata['voxel_volume_mm3'],
        'focused_ct_delta_absolute_energy_cm_inverse_mm3': captured_energy * case.metadata['voxel_volume_mm3'],
        'ct_delta_voxel_coverage_fraction': focused_voxels / total_voxels if total_voxels else 1.,
        'ct_delta_energy_coverage_fraction': captured_energy / all_energy if all_energy else 1.,
        'feature_l1': float(np.sum(features)),
        'feature_l2': float(np.linalg.norm(features)),
    }
    return np.asarray(features, dtype=np.float32), summary


def extract_variant(case, row, *, verify=True):
    """Read only the sibling image, never its labels/masks/edit parameters."""
    if (list(row['edit_crop_origin_ijk']) != case.metadata['crop_origin_ijk'] or
            list(row['edit_crop_shape_kji']) != case.metadata['shape_kji'] or
            list(row['spacing_ijk_mm']) != case.metadata['spacing_ijk_mm'] or
            row.get('image_units') != 'cm^-1'):
        raise ValueError('Sibling row changes the paired native crop or image units')
    if row.get('source_npz') != case.metadata.get('source_npz'):
        raise ValueError('Sibling row references a different original baseline')
    path = _checked_file(row, 'variant_npz', verify)
    with np.load(path, allow_pickle=False) as variant:
        image = variant['image']
    return extract_array(case, image)


def feature_names(case):
    """Names identify physical bins rather than any source/anatomical label ID."""
    m = case.metadata
    return [f'{channel}:s{s:02d}:r{r:02d}:theta{t:02d}'
            for channel in ('positive_attenuation', 'negative_attenuation')
            for s in range(m['arc_bins']) for r in range(m['radial_bins'])
            for t in range(m['angular_bins'])]
