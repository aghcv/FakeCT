"""Bounded native-grid ROI preview data; no morphology or source writes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fakect_tissues import coarse_labels
from fakect_released import uses_diagnostic_release, released_catalog, RELEASED_LABEL_ID

MAX_CROP_VOXELS = 16_777_216
MAX_DISPLAY_CELLS = 650_000
COLORS = {'background': '#000000', 'soft_tissue': '#bcaaa4', 'bone': '#fff0bc',
          'cartilage': '#66bb6a', 'muscle': '#b85c38', 'artery': '#f44336',
          'vein': '#367bf5', 'lung': '#4dd0c8', 'adipose': '#ffd600',
          'nervous_tissue': '#9575cd', 'fluid': '#b3e5fc', 'unknown': '#ff00cc', 'released': '#ff2ea6'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_bounds(center_ijk, half_width_mm, shape_kji, spacing_ijk_mm):
    """Inclusive native voxel centers within a conservative physical crop box."""
    center = np.asarray(center_ijk)
    shape = np.asarray(shape_kji)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if center.shape != (3,) or center.dtype.kind not in 'iu' or np.any(center < 0):
        raise ValueError('ROI center must contain three nonnegative integer i,j,k indices')
    if shape.shape != (3,) or shape.dtype.kind not in 'iu' or np.any(shape <= 0):
        raise ValueError('Source shape must contain three positive k,j,i dimensions')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Source spacing must contain three positive finite i,j,k millimetre values')
    if np.any(center >= shape[::-1]) or not np.isfinite(half_width_mm) or half_width_mm <= 0:
        raise ValueError('ROI center lies outside source dimensions, or crop half-width is invalid')
    # No allocation is performed until the clipped crop has passed its size bound.
    half = np.ceil(np.minimum(half_width_mm / spacing, shape[::-1])).astype(np.int64)
    low = np.maximum(0, center - half)
    high = np.minimum(shape[::-1], center + half + 1)
    if int(np.prod(high - low)) > MAX_CROP_VOXELS:
        raise ValueError(f'Preview crop exceeds {MAX_CROP_VOXELS:,} voxels; reduce crop_half_width_mm')
    return tuple(map(int, low)), tuple(map(int, high))


def sphere_mask(shape_kji, origin_ijk, center_ijk, spacing_ijk_mm, radius_mm):
    """Physical sphere evaluated at native voxel centers, array order k,j,i."""
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError('Sphere radius must be positive and finite')
    k, j, i = np.ogrid[:shape_kji[0], :shape_kji[1], :shape_kji[2]]
    squared = sum(((v + origin_ijk[a] - center_ijk[a]) * spacing_ijk_mm[a]) ** 2
                  for a, v in enumerate((i, j, k)))
    return squared <= radius_mm ** 2


def _tube_nodes(nodes_ijk, radii_mm=None):
    """Validate ordered controls without sorting or guessing connectivity."""
    nodes = np.asarray(nodes_ijk, dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1:] != (3,) or len(nodes) < 2
            or not np.all(np.isfinite(nodes)) or np.any(nodes < 0)):
        raise ValueError('Tube centers must contain at least two finite nonnegative i,j,k triples')
    if np.any(np.all(np.diff(nodes, axis=0) == 0, axis=1)):
        raise ValueError('Consecutive tube centers must be distinct; preserve their path order')
    if radii_mm is None:
        return nodes
    radii = np.asarray(radii_mm, dtype=float)
    if radii.shape != (len(nodes),) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError('Tube radii must contain one positive finite millimetre value per center')
    return nodes, radii


def tube_crop_bounds(nodes_ijk, half_width_mm, shape_kji, spacing_ijk_mm):
    """Bound every control and segment, expanded by a physical margin per axis."""
    nodes = _tube_nodes(nodes_ijk)
    return _node_crop_bounds(nodes, half_width_mm, shape_kji, spacing_ijk_mm)


def _node_crop_bounds(nodes, half_width_mm, shape_kji, spacing_ijk_mm):
    """Internal envelope also supports one fractional sphere center."""
    shape = np.asarray(shape_kji)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if shape.shape != (3,) or shape.dtype.kind not in 'iu' or np.any(shape <= 0):
        raise ValueError('Source shape must contain three positive k,j,i dimensions')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Source spacing must contain three positive finite i,j,k millimetre values')
    if np.any(nodes > shape[::-1] - 1) or not np.isfinite(half_width_mm) or half_width_mm <= 0:
        raise ValueError('ROI center lies outside source dimensions, or crop half-width is invalid')
    margin = np.minimum(half_width_mm / spacing, shape[::-1])
    low = np.maximum(0, np.floor(nodes.min(axis=0) - margin)).astype(np.int64)
    high = np.minimum(shape[::-1], np.ceil(nodes.max(axis=0) + margin) + 1).astype(np.int64)
    if int(np.prod(high - low)) > MAX_CROP_VOXELS:
        raise ValueError(f'Preview crop exceeds {MAX_CROP_VOXELS:,} voxels; shorten the tube or reduce crop_half_width_mm')
    return tuple(map(int, low)), tuple(map(int, high))


def tube_mask(shape_kji, origin_ijk, nodes_ijk, spacing_ijk_mm, radii_mm):
    """Exact union of linearly varying-radius balls along ordered line segments.

    Coordinates are native i,j,k voxel centers; distances and radii are in mm.
    Minimizing squared distance minus squared radius gives a quadratic on each
    segment. Its minimum is at the clipped stationary point when convex, and
    otherwise at an endpoint. Thus round caps and even steep tapers are included
    without discretizing the centerline or adding spline smoothing.
    """
    nodes, radii = _tube_nodes(nodes_ijk, radii_mm)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Source spacing must contain three positive finite i,j,k millimetre values')
    k, j, i = np.ogrid[:shape_kji[0], :shape_kji[1], :shape_kji[2]]
    positions = [(v + origin_ijk[a]) * spacing[a] for a, v in enumerate((i, j, k))]
    result = np.zeros(shape_kji, dtype=bool)
    for first, last, r0, r1 in zip(nodes[:-1], nodes[1:], radii[:-1], radii[1:]):
        start, delta = first * spacing, (last - first) * spacing
        displacement = [v - start[a] for a, v in enumerate(positions)]
        distance_squared = sum(v * v for v in displacement)
        dr = r1 - r0
        quadratic = float(np.dot(delta, delta) - dr * dr)
        projection = sum(displacement[a] * delta[a] for a in range(3)) + r0 * dr
        constant = distance_squared - r0 * r0
        endpoint = quadratic - 2 * projection + constant
        minimum = np.minimum(constant, endpoint)
        if quadratic > 0:
            t = np.clip(projection / quadratic, 0, 1)
            minimum = np.minimum(minimum, (quadratic * t - 2 * projection) * t + constant)
        # Allow only floating-point roundoff at the analytic surface.
        tolerance = 64 * np.finfo(float).eps * np.maximum(1, distance_squared + r0*r0 + r1*r1)
        result |= minimum <= tolerance
    return result


def _resolve_roi(roi, shape, spacing):
    kind = roi.get('shape', 'sphere')
    if kind == 'sphere':
        center = np.asarray(roi['center_ijk'], dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)) or np.any(center < 0):
            raise ValueError('Sphere center must contain three finite nonnegative i,j,k indices')
        radius = roi['radius_mm']
        if not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Sphere radius must be positive and finite')
        nodes, radii = np.asarray([center], dtype=float), np.asarray([radius], dtype=float)
        bounds = _node_crop_bounds(nodes, roi['crop_half_width_mm'], shape, spacing)
        focus = tuple(map(int, np.rint(center)))
    elif kind == 'tube':
        nodes, radii = _tube_nodes(roi['center_ijk'], roi['radius_mm'])
        bounds = tube_crop_bounds(nodes, roi['crop_half_width_mm'], shape, spacing)
        focus = tuple(map(int, np.rint(nodes[len(nodes) // 2])))
    else:
        raise ValueError('ROI shape must be sphere or tube')
    if roi['crop_half_width_mm'] < float(radii.max()):
        raise ValueError('crop_half_width_mm must be at least the largest ROI radius_mm')
    return {'roi_kind': kind, 'roi_nodes_ijk': tuple(tuple(map(float, n)) for n in nodes),
            'roi_radii_mm': tuple(map(float, radii)), 'focus_ijk': focus,
            'roi_node_low_ijk': tuple(map(float, nodes.min(axis=0))),
            'roi_node_high_ijk': tuple(map(float, nodes.max(axis=0))),
            'crop_low_ijk': bounds[0], 'crop_high_ijk_exclusive': bounds[1]}


def resolve_preview(config):
    """Validate metadata/selection without reading voxel payloads."""
    input_config = config['input']
    audit_bytes = input_config['audit'].read_bytes()
    catalog_bytes = input_config['catalog'].read_bytes()
    audit = json.loads(audit_bytes)
    catalog = json.loads(catalog_bytes)
    if uses_diagnostic_release(config):
        catalog = released_catalog(catalog)
    coarse_labels(np.array([0], dtype=np.int32), catalog)
    categories = {c['name']: c for c in catalog['categories']}
    tissue = config['selection']['tissue']
    if tissue not in categories or tissue in ('background', 'released'):
        raise ValueError(f'Select a catalog tissue other than background: {tissue}')
    for name in config['preview']['context_tissues']:
        if name not in categories or name == 'background':
            raise ValueError(f'Unknown/background context tissue: {name}')
    available = {r['original_id']: r for r in catalog['records']}
    source_ids = config['selection']['source_ids'] or tuple(
        key for key, r in available.items() if r['classification']['tissue_name'] == tissue)
    if not source_ids:
        raise ValueError(f'No original IDs are assigned to tissue {tissue}; revise the selection')
    for key in source_ids:
        if key not in available or available[key]['classification']['tissue_name'] != tissue:
            raise ValueError(f'Original ID {key} is not assigned to the selected tissue {tissue}')
    case = next((c for c in audit['cases'] if c['case_id'] == input_config['case_id']), None)
    if case is None:
        raise ValueError('Case is absent from the XCAT audit')
    root = input_config['root']
    source_dir = root / input_config['case_id']
    if source_dir.resolve() != Path(case['directory']).resolve():
        raise ValueError('Input root disagrees with the audited case directory; audit this source first')
    for name in ('par_path', 'log_path'):
        path = Path(case[name])
        expected = audit.get('source_metadata_sha256', {}).get(str(path))
        if not expected or digest(path) != expected:
            raise ValueError(f'Missing or changed audited metadata: {path}')
    shape, spacing = case['shape_kji'], case['spacing_ijk_mm']
    roi = _resolve_roi(config['roi'], shape, spacing)
    low, high = roi['crop_low_ijk'], roi['crop_high_ijk_exclusive']
    crop_shape = np.asarray(high) - low
    volume_stride = config['preview']['volume_stride']
    display_shape = (crop_shape + volume_stride - 1) // volume_stride
    if np.any(display_shape < 2):
        raise ValueError('volume_stride leaves fewer than two blocks per axis; reduce it')
    field_count = 1 + len(config['preview']['context_tissues'])
    if uses_diagnostic_release(config) and 'released' not in config['preview']['context_tissues']:
        field_count += 1  # The final 3D view automatically includes diagnostic releases.
    display_cells = int(np.prod(display_shape + 2)) * field_count
    if display_cells > MAX_DISPLAY_CELLS:
        recommendation = None
        # ceil(n / stride) must remain >= 2 along every crop axis. Search the
        # accepted integer strides in order, retaining the finest valid display.
        for stride in range(volume_stride + 1, int(crop_shape.min())):
            candidate_shape = (crop_shape + stride - 1) // stride
            candidate_cells = int(np.prod(candidate_shape + 2)) * field_count
            if candidate_cells <= MAX_DISPLAY_CELLS:
                recommendation = (stride, candidate_cells)
                break
        if recommendation is not None:
            stride, cells = recommendation
            advice = (f'Set [preview] volume_stride = {stride}, the smallest valid stride '
                      f'for this crop/context ({cells:,} estimated display cells). '
                      'This changes display sampling only; retain the crop context needed for edits')
        else:
            advice = ('No larger valid volume_stride fits the budget while retaining at least '
                      'two blocks per axis; reduce context_tissues or shorten the long crop axes '
                      'while retaining the context needed for edits')
        raise ValueError(f'3D preview exceeds the display budget: {display_cells:,} estimated '
                         f'cells at volume_stride = {volume_stride}, including padding for '
                         f'{field_count} fields; limit {MAX_DISPLAY_CELLS:,}. {advice}')
    slices = config['preview']['slice_ijk'] or roi['focus_ijk']
    if any(not lo <= v < hi for v, lo, hi in zip(slices, low, high)):
        raise ValueError('slice_ijk lies outside the preview crop; move the ROI or increase crop_half_width_mm')
    source_files = {}
    for channel in ('act', 'atn'):
        path = source_dir / f"{input_config['case_id']}_{channel}_{input_config['frame']}.bin"
        if path.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError(f'Unexpected binary source size: {path}')
        source_files[channel] = path
    if shape[1] * shape[2] * 4 > 256 * 1024 ** 2:
        raise ValueError('Source axial plane exceeds the bounded preview reader limit')
    return {'case': case, 'catalog': catalog, 'catalog_bytes': catalog_bytes,
            'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
            'audit_sha256': hashlib.sha256(audit_bytes).hexdigest(), 'source_ids': tuple(source_ids),
            'source_names': {str(key): available[key]['original_name'] for key in source_ids},
            'source_files': source_files, **roi,
            'slice_ijk': tuple(slices), 'shape_kji': tuple(shape), 'spacing_ijk_mm': tuple(spacing)}


def read_crop(path, shape_kji, low_ijk, high_ijk):
    """Read bounded whole axial planes, retaining only a crop; source is read-only."""
    path = Path(path)
    before = path.stat()
    if before.st_size != int(np.prod(shape_kji)) * 4:
        raise ValueError(f'Unexpected source size: {path}')
    low, high = low_ijk, high_ijk
    shape = tuple(hi - lo for lo, hi in zip(low[::-1], high[::-1]))
    if min(shape) < 1 or int(np.prod(shape)) > MAX_CROP_VOXELS:
        raise ValueError('Invalid or oversized crop')
    if any(lo < 0 or hi > n for lo, hi, n in zip(low, high, shape_kji[::-1])):
        raise ValueError('Crop bounds are outside the source')
    crop = np.empty(shape, dtype='<f4')
    plane_size = shape_kji[1] * shape_kji[2]
    with path.open('rb', buffering=0) as stream:
        stream.seek(low[2] * plane_size * 4)
        for n in range(shape[0]):
            plane = np.fromfile(stream, dtype='<f4', count=plane_size)
            if plane.size != plane_size:
                raise ValueError(f'Short source read: {path}')
            crop[n] = plane.reshape(shape_kji[1:])[low[1]:high[1], low[0]:high[0]]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed during preview: {path}')
    if not np.all(np.isfinite(crop)):
        raise ValueError(f'Nonfinite values in source crop: {path}')
    return crop, {'path': str(path), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
                  'source_bytes_read': shape[0] * plane_size * 4,
                  'crop_sha256': hashlib.sha256(crop.tobytes()).hexdigest(),
                  'integrity_scope': 'Crop content hash and source stat; full source volume not hashed'}


def prepare_crop(resolved, config):
    arrays, sources = {}, {}
    for channel, path in resolved['source_files'].items():
        arrays[channel], sources[channel] = read_crop(path, resolved['shape_kji'],
                                                     resolved['crop_low_ijk'], resolved['crop_high_ijk_exclusive'])
    if uses_diagnostic_release(config) and np.any(arrays['act'] == RELEASED_LABEL_ID):
        raise ValueError('Native source contains the reserved diagnostic released label; source-label collision')
    arrays['tissue'] = coarse_labels(arrays['act'], resolved['catalog'])
    arrays['candidates'] = np.isin(arrays['act'], resolved['source_ids'])
    if resolved.get('roi_kind', 'sphere') == 'tube':
        arrays['roi'] = tube_mask(arrays['act'].shape, resolved['crop_low_ijk'],
                                  resolved['roi_nodes_ijk'], resolved['spacing_ijk_mm'], resolved['roi_radii_mm'])
    else:
        arrays['roi'] = sphere_mask(arrays['act'].shape, resolved['crop_low_ijk'],
                                    config['roi']['center_ijk'], resolved['spacing_ijk_mm'], config['roi']['radius_mm'])
    arrays['selected'] = arrays['candidates'] & arrays['roi']
    arrays['attenuation_cm_inverse'] = arrays['atn'] / (resolved['spacing_ijk_mm'][0] / 10)
    return arrays, sources


def selection_diagnostics(arrays, resolved=None):
    """Describe spatial selection without equating organ IDs with vessels.

    Six-neighbor connectivity is a diagnostic, not automatic component picking:
    one anatomical vessel may contain several IDs or disconnected voxel pieces.
    """
    from scipy import ndimage

    selected = np.asarray(arrays['selected'], dtype=bool)
    components, count = ndimage.label(selected, structure=ndimage.generate_binary_structure(3, 1))
    sizes = sorted(map(int, np.bincount(components.ravel())[1:]), reverse=True)
    ids, counts = np.unique(arrays['act'][selected], return_counts=True)
    candidate_voxels = int(np.count_nonzero(arrays['candidates']))
    selected_voxels = int(np.count_nonzero(selected))
    warnings = []
    if selected_voxels == 0:
        warnings.append('The ROI contains no voxels from the selected tissue/source IDs; relocate or widen it.')
    elif count > 1:
        warnings.append(f'Selection contains {count} disconnected 6-connected components. Inspect nearby structures and narrow or relocate the ROI if needed; components do not necessarily represent distinct vessels.')
    result = {'candidate_voxels': candidate_voxels, 'roi_voxels': int(np.count_nonzero(arrays['roi'])),
              'selected_voxels': selected_voxels,
              'selected_fraction_of_candidates': selected_voxels / candidate_voxels if candidate_voxels else 0.0,
              'component_count_6': int(count), 'components_voxels_6': sizes,
              'selected_original_ids': list(map(int, ids)),
              'selected_original_id_counts': {str(int(key)): int(value) for key, value in zip(ids, counts)},
              'warnings': warnings}
    if resolved is not None:
        result['selected_original_names'] = {
            str(int(key)): resolved.get('source_names', {}).get(str(int(key))) for key in ids}
    return result
