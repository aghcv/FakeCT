"""ROI-free, reviewable thoracic aortic masks from labeled XCAT phantoms.

The T12/T1 landmark slab is an explicit approximation, not an assertion that
the atlas splits the thoracic and abdominal portions of ``dias_aorta``.
"""
from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import ndimage

from fakect_anatomy import _role
from fakect_anatomy_surfaces import (_coordinate_contract, _load_evidence,
                                    inventory_surfaces)
from fakect_roi import read_crop
from fakect_tissues import coarse_labels

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.phantom-anatomy/1'
POLICY = 'atlas-aorta-t12-midpoint-t1-upper-proximal-branches/1'
MAX_VOXELS = 32_000_000


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    path.with_suffix(path.suffix + '.sha256').write_text(_sha(path) + '\n')


def load_config(path, repo_root=None):
    """Read a small INI whose schema intentionally has no ROI section."""
    root = Path(repo_root or ROOT).resolve()
    path = Path(path).resolve()
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    parser.read_string(path.read_text())
    fields = {'anatomy': {'schema_version', 'name', 'output_directory'},
              'input': {'root', 'case_id', 'frame', 'audit', 'catalog', 'surface_inventory'},
              'selection': {'boundary_policy', 'include_branches', 'branch_reach_mm',
                            'branch_reach_radii', 'connectivity', 'padding_mm'}}
    if parser.defaults() or set(parser.sections()) != set(fields):
        raise ValueError('Expected only [anatomy], [input], [selection]; no supplied ROI is used')
    for section, allowed in fields.items():
        if set(parser[section]) - allowed:
            raise ValueError(f'Unknown {section} fields: {sorted(set(parser[section]) - allowed)}')
    if parser.get('anatomy', 'schema_version') != SCHEMA:
        raise ValueError(f'Expected schema_version={SCHEMA}')

    def filepath(section, name, optional=False):
        raw = parser.get(section, name, fallback='').strip()
        if not raw:
            if optional:
                return None
            raise ValueError(f'{section}.{name} is required')
        p = Path(raw).expanduser()
        return (p if p.is_absolute() else root / p).resolve()

    case = parser.get('input', 'case_id')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', case):
        raise ValueError('case_id must be a safe directory identifier')
    frame = parser.getint('input', 'frame', fallback=1)
    connectivity = parser.getint('selection', 'connectivity', fallback=6)
    if frame < 1 or connectivity not in (6, 26):
        raise ValueError('Positive frame and connectivity=6 or26 required')
    boundary = parser.get('selection', 'boundary_policy', fallback='t12_midpoint_to_t1_upper')
    if boundary != 't12_midpoint_to_t1_upper':
        raise ValueError('Supported boundary_policy=t12_midpoint_to_t1_upper')
    reach = parser.get('selection', 'branch_reach_mm', fallback='').strip()
    settings = {'include_branches': parser.getboolean('selection', 'include_branches', fallback=True),
                'branch_reach_mm': float(reach) if reach else None,
                'branch_reach_radii': parser.getfloat('selection', 'branch_reach_radii', fallback=3.0),
                'padding_mm': parser.getfloat('selection', 'padding_mm', fallback=2.0),
                'connectivity': connectivity, 'boundary_policy': boundary}
    for key in ('branch_reach_mm', 'branch_reach_radii', 'padding_mm'):
        value = settings[key]
        if value is not None and (not np.isfinite(value) or value < 0 or value > 200):
            raise ValueError(f'{key} must be finite in [0,200]')
    if settings['branch_reach_mm'] is None and settings['branch_reach_radii'] <= 0:
        raise ValueError('A positive relative branch reach is required')
    return {'config_path': path, 'config_sha256': _sha(path),
            'name': parser.get('anatomy', 'name', fallback=case),
            'output_directory': filepath('anatomy', 'output_directory'),
            'input': {'root': filepath('input', 'root'), 'case_id': case, 'frame': frame,
                      'audit': filepath('input', 'audit'), 'catalog': filepath('input', 'catalog'),
                      'surface_inventory': filepath('input', 'surface_inventory', True)},
            'selection': settings}


def _source_metadata(config):
    inp = config['input']
    audit = json.loads(inp['audit'].read_text())
    catalog = json.loads(inp['catalog'].read_text())
    coarse_labels(np.zeros(1, np.int32), catalog, strict=True)
    cases = [c for c in audit['cases'] if str(c['case_id']) == inp['case_id']]
    if len(cases) != 1:
        raise ValueError('Expected one audited case')
    case = cases[0]
    directory = inp['root'] / inp['case_id']
    if directory.resolve() != Path(case['directory']).resolve():
        raise ValueError('Input root disagrees with audited case directory')
    if case.get('par_log_mismatches'):
        raise ValueError('Audited parameter/log geometry disagrees')
    for field in ('par_path', 'log_path'):
        p = Path(case[field])
        if audit.get('source_metadata_sha256', {}).get(str(p)) != _sha(p):
            raise ValueError(f'Audited metadata changed: {p}')
    shape = np.asarray(case['shape_kji'], dtype=int)
    spacing = np.asarray(case['spacing_ijk_mm'], dtype=float)
    if shape.shape != (3,) or spacing.shape != (3,) or np.any(shape <= 0) or np.any(spacing <= 0):
        raise ValueError('Invalid audited volume geometry')
    source = directory / f"{inp['case_id']}_act_{inp['frame']}.bin"
    if source.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError('ACT payload size disagrees with audited volume')
    # These envelope fields are solely the adapter's automatic full-grid broad
    # phase. They are never a user ROI and never restrict the selected anatomy.
    center = (shape[::-1] - 1) / 2
    resolved = {'case': case, 'catalog': catalog, 'shape_kji': shape.tolist(),
                'spacing_ijk_mm': spacing.tolist(), 'roi_nodes_ijk': [center.tolist()],
                'roi_radii_mm': [float(np.max(shape[::-1] * spacing))],
                'catalog_sha256': _sha(inp['catalog']), 'audit_sha256': _sha(inp['audit'])}
    evidence, provenance = _load_evidence()
    contract = _coordinate_contract(config, resolved, evidence, provenance)
    if contract['status'] != 'bounded_contract_validated':
        raise ValueError('RAW/BIN registration contract unavailable: ' + ', '.join(contract['reasons']))
    return resolved, source, contract


def _surfaces(config, resolved, contract):
    cache = config['input']['surface_inventory']
    if cache is None:
        result = inventory_surfaces(config, resolved)
        return result, {'method': 'fresh_RAW_inventory', 'sha256': None}
    if cache.with_suffix(cache.suffix + '.sha256').read_text().strip() != _sha(cache):
        raise ValueError('Surface inventory cache checksum mismatch')
    raw = json.loads(cache.read_text())
    if 'surface_evidence' in raw:
        if raw.get('complete') is not True or raw.get('request', {}).get('catalog_sha256') != resolved['catalog_sha256']:
            raise ValueError('Surface cache is incomplete or uses another catalog')
        result = raw['surface_evidence']
    else:
        result = raw
    if (result.get('schema_version') != 'fakect.anatomy-surface-inventory/1' or
            str(result['case_id']) != config['input']['case_id'] or result['frame'] != config['input']['frame']):
        raise ValueError('Surface inventory case/frame/schema mismatch')
    if result['coordinate_contract']['status'] != 'bounded_contract_validated':
        raise ValueError('Cached surface registration was unavailable')
    for entry in result['files'] + result.get('coordinate_dependencies', []):
        stat = Path(entry['path']).stat()
        if stat.st_size != entry['bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise ValueError(f'Cached surface source changed: {entry["path"]}')
    for field in ('grid_boundary_origin_raw_mm', 'spacing_ijk_mm'):
        if not np.allclose(result['coordinate_contract'][field], contract[field], atol=1e-9, rtol=0):
            raise ValueError('Cached RAW/BIN geometry disagrees with current contract')
    return result, {'method': 'verified_full_surface_inventory_cache', 'path': str(cache),
                    'sha256': _sha(cache), 'ignored_fields': ['old_ROI', 'old_selected_source_ids',
                                                          'old_crop', 'old_connectivity_results'],
                    'source_integrity': 'Cached file hashes bound to unchanged size/mtime; current coordinate contract revalidated'}


def anatomical_roles(catalog, include_branches=True):
    settings = {'include_branches': include_branches}
    roles = {}
    for row in catalog['records']:
        if row['classification'].get('tissue_name') == 'artery':
            roles[int(row['original_id'])] = _role(row, settings)[0]
    return roles


def landmark_domain(records, catalog, settings):
    """Derive physical bounds from all copies of the T12 and T1 surfaces."""
    by_id = {int(r['original_id']): r for r in catalog['records']}
    anchor_ids = {i for i, role in anatomical_roles(catalog).items() if role == 'anchor'}
    anchors = [r for r in records if r.get('id_join_status') == 'unique_name_match' and
               set(r.get('candidate_original_ids', [])) <= anchor_ids and r.get('candidate_original_ids')]
    if not anchors:
        raise ValueError('No uniquely mapped atlas aortic RAW anchor')
    landmarks = {}
    for name in ('T12_vertebrae', 'T1_vertebrae'):
        rows = [r for r in records if r['original_name'] == name]
        if not rows:
            raise ValueError(f'Missing automatic thoracic landmark {name}')
        landmarks[name] = {'minimum_z_mm': min(r['bounds_raw_mm']['min_xyz'][2] for r in rows),
                           'maximum_z_mm': max(r['bounds_raw_mm']['max_xyz'][2] for r in rows),
                           'surface_count': len(rows),
                           'candidate_original_ids': sorted({i for r in rows for i in r['candidate_original_ids']}),
                           'surfaces': [{k:r[k] for k in ('source_file', 'local_surface_index', 'original_name',
                                                        'bounds_raw_mm', 'candidate_original_ids')} for r in rows]}
    lo = (landmarks['T12_vertebrae']['minimum_z_mm'] + landmarks['T12_vertebrae']['maximum_z_mm']) / 2
    hi = landmarks['T1_vertebrae']['maximum_z_mm']
    low = np.min(np.asarray([r['bounds_raw_mm']['min_xyz'] for r in anchors], dtype=float), axis=0)
    high = np.max(np.asarray([r['bounds_raw_mm']['max_xyz'] for r in anchors], dtype=float), axis=0)
    if lo >= hi or high[2] <= lo or low[2] >= hi:
        raise ValueError('Thoracic landmarks are inconsistent with the aorta')
    radii = [v for r in anchors for v in r.get('centerline', {}).get('ring_mean_radii_mm', [])
             if r.get('centerline', {}).get('status') == 'verified_two_pole_rings']
    reach = settings['branch_reach_mm']
    if reach is None:
        if not radii:
            raise ValueError('Aortic ring radii unavailable; provide branch_reach_mm explicitly')
        reach = settings['branch_reach_radii'] * float(np.median(radii))
    if not settings['include_branches']:
        reach = 0.0
    pad = settings['padding_mm']
    low[:2] -= reach + pad
    high[:2] += reach + pad
    low[2], high[2] = lo - pad, hi + pad
    return {'landmarks': landmarks, 'z_slab_mm': [float(lo), float(hi)],
            'branch_reach_mm': float(reach), 'branch_distance': 'Euclidean physical distance from selected core',
            'aorta_median_ring_radius_mm': float(np.median(radii)) if radii else None,
            'anchor_source_ids': sorted({i for r in anchors for i in r['candidate_original_ids']}),
            'anchor_names': [by_id[i]['original_name'] for i in sorted(anchor_ids)],
            'crop_bounds_raw_mm': {'min_xyz': low.tolist(), 'max_xyz': high.tolist()}}


def validate(config):
    """Read metadata/RAW evidence without decoding the ACT volume."""
    resolved, source, contract = _source_metadata(config)
    surfaces, cache = _surfaces(config, resolved, contract)
    domain = landmark_domain(surfaces['records'], resolved['catalog'], config['selection'])
    spacing = np.asarray(resolved['spacing_ijk_mm'])
    boundary = np.asarray(contract['grid_boundary_origin_raw_mm'])
    limits = np.asarray(resolved['shape_kji'][::-1])
    bounds = domain['crop_bounds_raw_mm']
    raw_low = np.floor((np.asarray(bounds['min_xyz']) - boundary) / spacing - 0.5).astype(int)
    raw_high = np.ceil((np.asarray(bounds['max_xyz']) - boundary) / spacing - 0.5).astype(int) + 1
    low, high = np.maximum(raw_low, 0), np.minimum(raw_high, limits)
    if np.any(low != raw_low) or np.any(high != raw_high):
        raise ValueError('Automatic thoracic domain leaves available field of view')
    if np.any(high <= low) or np.prod(high - low) > MAX_VOXELS:
        raise ValueError(f'Automatic native crop exceeds {MAX_VOXELS:,} voxels; reduce branch reach')
    report = {'schema_version': SCHEMA, 'policy': POLICY, 'case_id': config['input']['case_id'],
              'frame': config['input']['frame'], 'status': 'metadata_validated',
              'roi_required': False, 'coordinate_reviewed': False, 'provisional': True,
              'scope': 'T12-midpoint to T1-upper landmark slab, atlas aorta and proximal connected arterial branches',
              'limitations': ['Landmark slab is an explicit anatomical approximation; review its inferior cutoff.',
                             'Branch reach limits proximal extent and does not establish complete named branch coverage.',
                             'Native contact is geometric connectivity, not proof of physiological connectivity.',
                             'Atlas aorta label also contains abdominal aorta; thorax category alone does not delimit it.',
                             'This requires XCAT anatomical labels; attenuation-only clinical CT is unsupported.'],
              'config_sha256': config['config_sha256'], 'catalog_sha256': resolved['catalog_sha256'],
              'audit_sha256': resolved['audit_sha256'], 'catalog_path': str(config['input']['catalog']),
              'audit_path': str(config['input']['audit']), 'surface_inventory': cache,
              'source_surface_files': surfaces['files'], 'coordinate_contract': contract,
              'settings': config['selection'], **domain,
              'geometry': {'shape_kji': resolved['shape_kji'], 'spacing_ijk_mm': spacing.tolist(),
                           'grid_boundary_origin_raw_mm': boundary.tolist(),
                           'crop_low_ijk': low.tolist(), 'crop_high_ijk_exclusive': high.tolist(),
                           'crop_shape_kji': (high-low)[::-1].tolist(),
                           'crop_first_voxel_center_raw_mm': (boundary + (low + .5) * spacing).tolist()},
              'act_source': {'path': str(source), 'bytes': source.stat().st_size,
                             'mtime_ns': source.stat().st_mtime_ns},
              'selector_code_sha256': {name: _sha(ROOT / 'src' / name) for name in
                                      ('fakect_phantom_anatomy.py', 'fakect_anatomy.py',
                                       'fakect_anatomy_surfaces.py', 'fakect_anatomy_centerlines.py',
                                       'fakect_tissues.py', 'fakect_roi.py')}}
    return report


def select_masks(labels, catalog, z_slab_mm, origin_raw_mm, spacing_ijk_mm,
                 branch_reach_mm, connectivity=6, include_branches=True):
    """Select connected voxel components, never all voxels sharing a branch ID.

    ``origin_raw_mm`` is the center of voxel (0,0,0), not its lower boundary.
    """
    labels = np.asarray(labels)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    origin = np.asarray(origin_raw_mm, dtype=float)
    if (labels.ndim != 3 or not np.all(np.isfinite(labels)) or
            not np.all(labels == np.rint(labels)) or connectivity not in (6, 26) or
            spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)) or
            origin.shape != (3,) or not np.all(np.isfinite(origin)) or
            not np.isfinite(branch_reach_mm) or branch_reach_mm < 0):
        raise ValueError('Invalid native labels, physical geometry, branch reach, or connectivity')
    lo, hi = map(float, z_slab_mm)
    if not np.isfinite(lo + hi) or lo >= hi:
        raise ValueError('Invalid thoracic landmark slab')
    roles = anatomical_roles(catalog, include_branches)
    anchor_ids = [i for i, role in roles.items() if role == 'anchor']
    branch_ids = [i for i, role in roles.items() if role == 'branch']
    if not anchor_ids:
        raise ValueError('Catalog has no classified aortic anchor')
    z = origin[2] + np.arange(labels.shape[0]) * spacing[2]
    slab = ((z >= lo) & (z <= hi))[:, None, None]
    anchors = np.isin(labels, anchor_ids) & slab
    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    component, count = ndimage.label(anchors, structure)
    sizes = np.bincount(component.ravel())
    if count == 0:
        raise ValueError('No native aortic anchor within the automatic thoracic domain')
    sizes[0] = 0
    chosen = int(np.argmax(sizes))
    core = component == chosen
    del component
    candidates = np.isin(labels, branch_ids) & slab
    candidate_count = int(np.count_nonzero(candidates))
    if include_branches and branch_reach_mm > 0:
        distances = ndimage.distance_transform_edt(~core, sampling=spacing[::-1])
        in_reach = candidates & (distances <= branch_reach_mm)
        del distances
        selected = ndimage.binary_propagation(core, structure=structure, mask=core | in_reach)
    else:
        in_reach = np.zeros(labels.shape, bool)
        selected = core.copy()
    branches = selected & ~core
    selected_ids, selected_counts = np.unique(labels[selected].astype(np.int32), return_counts=True)
    arterial_ids = list(roles)
    excluded = np.isin(labels, [i for i, role in roles.items() if role == 'excluded']) & slab
    excluded_ids, excluded_counts = np.unique(labels[excluded].astype(np.int32), return_counts=True)
    counts = {'core_voxels': int(core.sum()), 'branch_voxels': int(branches.sum()),
              'selected_voxels': int(selected.sum()), 'aorta_components_in_slab': int(count),
              'disconnected_aorta_voxels': int(anchors.sum() - core.sum()),
              'eligible_branch_voxels_in_slab': candidate_count,
              'branch_voxels_beyond_reach': int(candidates.sum() - in_reach.sum()),
              'disconnected_branch_voxels_in_reach': int(in_reach.sum() - branches.sum()),
              'excluded_arterial_voxels_in_slab': int(excluded.sum()),
              'aortic_voxels_outside_slab_in_crop': int(np.isin(labels, anchor_ids).sum() - anchors.sum()),
              'all_arterial_voxels_in_crop': int(np.isin(labels, arterial_ids).sum()),
              'selected_original_id_voxels': {str(int(i)): int(n) for i,n in zip(selected_ids,selected_counts)},
              'excluded_original_id_voxels': {str(int(i)): int(n) for i,n in zip(excluded_ids,excluded_counts)}}
    return {'mask': selected, 'core_mask': core, 'branch_mask': branches}, counts


def run(config):
    """Write a new immutable selection revision without editing source inputs."""
    output = config['output_directory']
    if output.exists():
        raise ValueError('Output directory already exists; choose a new selection revision')
    report = validate(config)
    if _sha(config['config_path']) != config['config_sha256']:
        raise ValueError('Input config changed during validation')
    output.mkdir(parents=True)
    (output / 'INCOMPLETE').write_text('Selection generation in progress\n')
    (output / 'input.ini').write_bytes(config['config_path'].read_bytes())
    geometry = report['geometry']
    labels, evidence = read_crop(report['act_source']['path'], geometry['shape_kji'],
                                 geometry['crop_low_ijk'], geometry['crop_high_ijk_exclusive'])
    if evidence['mtime_ns'] != report['act_source']['mtime_ns']:
        raise ValueError('ACT source changed after metadata validation')
    catalog = json.loads(config['input']['catalog'].read_text())
    arrays, counts = select_masks(labels, catalog, report['z_slab_mm'],
                                  geometry['crop_first_voxel_center_raw_mm'], geometry['spacing_ijk_mm'],
                                  report['branch_reach_mm'], config['selection']['connectivity'],
                                  config['selection']['include_branches'])
    report.update(status='complete_provisional_review', complete=True, counts=counts,
                  native_read=evidence, selected_source_ids=sorted(int(i) for i in counts['selected_original_id_voxels']))
    names = {r['original_id']: r for r in catalog['records']}
    report['selected_labels'] = [{'original_id': i, 'original_name': names[i]['original_name'],
                                  'hierarchy_path': names[i]['classification']['hierarchy_path'],
                                  'selected_voxels': counts['selected_original_id_voxels'][str(i)]}
                                 for i in report['selected_source_ids']]
    report['mask_semantics'] = 'Selected voxel components, not whole admitted original IDs; proximal connected branches inside landmark slab and physical reach'
    report['array_sha256'] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
    np.savez_compressed(output / 'mask.npz', **arrays, crop_low_ijk=np.array(geometry['crop_low_ijk']),
                        crop_origin_ijk=np.array(geometry['crop_low_ijk']),
                        source_shape_kji=np.array(geometry['shape_kji']),
                        case_id=np.array(config['input']['case_id']), frame=np.array(config['input']['frame']),
                        spacing_ijk_mm=np.array(geometry['spacing_ijk_mm']),
                        crop_first_voxel_center_raw_mm=np.array(geometry['crop_first_voxel_center_raw_mm']))
    report['mask_npz_sha256'] = _sha(output / 'mask.npz')
    report['mask_npz'] = 'mask.npz'
    if _sha(config['config_path']) != config['config_sha256']:
        raise ValueError('Input config changed during selection')
    _json(output / 'selection.json', report)
    (output / 'INCOMPLETE').unlink()
    return report
