"""Bounded native-grid ROI preview data; no morphology or source writes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fakect_tissues import coarse_labels

MAX_CROP_VOXELS = 16_777_216
COLORS = {'background': '#000000', 'soft_tissue': '#bcaaa4', 'bone': '#fff0bc',
          'cartilage': '#66bb6a', 'muscle': '#b85c38', 'artery': '#f44336',
          'vein': '#367bf5', 'lung': '#4dd0c8', 'adipose': '#ffd600',
          'nervous_tissue': '#9575cd', 'fluid': '#b3e5fc', 'unknown': '#ff00cc'}


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


def resolve_preview(config):
    """Validate metadata/selection without reading voxel payloads."""
    input_config = config['input']
    audit_bytes = input_config['audit'].read_bytes()
    catalog_bytes = input_config['catalog'].read_bytes()
    audit = json.loads(audit_bytes)
    catalog = json.loads(catalog_bytes)
    coarse_labels(np.array([0], dtype=np.int32), catalog)
    categories = {c['name']: c for c in catalog['categories']}
    tissue = config['selection']['tissue']
    if tissue not in categories or tissue == 'background':
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
    low, high = crop_bounds(config['roi']['center_ijk'], config['roi']['crop_half_width_mm'], shape, spacing)
    display_shape = np.ceil((np.asarray(high) - low) / config['preview']['volume_stride']).astype(int)
    if np.any(display_shape < 2):
        raise ValueError('volume_stride leaves fewer than two blocks per axis; reduce it')
    if int(np.prod(display_shape + 2)) * (1 + len(config['preview']['context_tissues'])) > 650_000:
        raise ValueError('3D preview exceeds the display budget; increase volume_stride or reduce crop/context')
    slices = config['preview']['slice_ijk'] or config['roi']['center_ijk']
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
            'source_files': source_files, 'crop_low_ijk': low, 'crop_high_ijk_exclusive': high,
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
    arrays['tissue'] = coarse_labels(arrays['act'], resolved['catalog'])
    arrays['selected'] = np.isin(arrays['act'], resolved['source_ids'])
    arrays['roi'] = sphere_mask(arrays['act'].shape, resolved['crop_low_ijk'],
                                config['roi']['center_ijk'], resolved['spacing_ijk_mm'], config['roi']['radius_mm'])
    arrays['attenuation_cm_inverse'] = arrays['atn'] / (resolved['spacing_ijk_mm'][0] / 10)
    return arrays, sources
