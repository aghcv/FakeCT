"""Optional, audited anatomical ID selection on original XCAT labels.

An explicit preprocessing step resolves anatomy on a bounded native crop. Its
immutable artifact is consumed by metadata-only preview/freeze validation.
Neither triangle ordinals nor attenuation values are used as anatomical IDs.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import ndimage

SCHEMA = 'fakect.anatomy-selection/1'
POLICY = 'thoracic-aorta-and-connected-systemic-branches/1'
CORE = '01_Cardiovascular/Heart/Aorta'
BRANCHES = ('01_Cardiovascular/Systemic_Arteries',
            '01_Cardiovascular/Coronary_Circulation/Coronary_Arteries',
            '01_Cardiovascular/Cerebrovascular/Arteries')
PULMONARY = ('01_Cardiovascular/Heart/Pulmonary_Artery',
             '01_Cardiovascular/Pulmonary_Circulation')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _name(value):
    return re.sub(r'[\s-]+', '_', str(value).strip().lower())


def _within(path, parent):
    return path == parent or path.startswith(parent + '/')


def _settings(config):
    selection = config['selection']
    anatomy = _name(selection.get('anatomy', ''))
    if anatomy and anatomy != 'thoracic_aorta':
        raise ValueError('Supported automatic anatomy: thoracic_aorta')
    if anatomy and selection['tissue'] != 'artery':
        raise ValueError('thoracic_aorta requires selection.tissue=artery')
    labels = [_name(x) for x in selection.get('anatomical_labels', ())]
    if len(set(labels)) != len(labels) or any(not x for x in labels):
        raise ValueError('anatomical_labels must contain distinct nonempty names')
    connectivity = selection.get('connectivity', 6)
    if connectivity not in (6, 26) or isinstance(connectivity, bool):
        raise ValueError('selection.connectivity must be 6 or 26')
    return {'anatomy': anatomy, 'anatomical_labels': labels,
            'include_branches': selection.get('include_branches', True),
            'connectivity': connectivity,
            'protect_other_arteries': selection.get('protect_other_arteries', True)}


def _manual_ids(records, requested):
    if not requested:
        return {r['original_id'] for r in records}
    found, ids = set(), set()
    for row in records:
        classification = row['classification']
        aliases = {_name(row['original_name']), _name(classification.get('canonical_name', '')),
                   _name(classification.get('normalized_name', ''))}
        matches = set(requested) & aliases
        if matches:
            found.update(matches)
            ids.add(row['original_id'])
    if found != set(requested):
        raise ValueError('Unknown anatomical_labels for the selected tissue: ' +
                         ', '.join(sorted(set(requested) - found)))
    return ids


def _role(row, settings):
    classification = row['classification']
    path = classification.get('hierarchy_path', '')
    original = _name(row['original_name'])
    # A known old atlas rule matches "lad" inside "bladder". Do not promote
    # this lexical contradiction to vascular evidence or change that catalog.
    if re.search(r'(?:^|_)(?:gall_)?bladder(?:_|$)', original):
        return 'excluded', 'Nonvascular original name contradicts arterial atlas classification'
    if any(_within(path, p) for p in PULMONARY):
        return 'excluded', 'Pulmonary circulation excluded before connectivity analysis'
    if classification.get('status') != 'classified':
        return 'excluded', 'Anatomical classification is unresolved'
    if _within(path, CORE):
        return 'anchor', 'Atlas aortic anchor'
    if settings['include_branches'] and any(_within(path, p) for p in BRANCHES):
        return 'branch', 'Atlas systemic/coronary/cerebrovascular arterial candidate'
    return 'excluded', 'Outside requested anatomical hierarchy or branch inclusion disabled'


def _stat(path):
    path = Path(path).resolve()
    value = path.stat()
    return {'path': str(path), 'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns}


def _request(config, resolved):
    root = Path(__file__).resolve().parents[1]
    modules = ('fakect_anatomy.py', 'fakect_anatomy_surfaces.py', 'fakect_anatomy_centerlines.py',
               'fakect_roi.py', 'fakect_tissues.py')
    evidence = root/'references/atlas/dpi/xcat_v2_generator_layout_evidence_v1.json'
    return {'policy': POLICY, 'settings': _settings(config),
            'case_id': config['input']['case_id'], 'frame': config['input']['frame'],
            'tissue': config['selection']['tissue'],
            'source_ids': list(config['selection'].get('source_ids', ())),
            'catalog_sha256': resolved['catalog_sha256'],
            'audit_sha256': resolved['audit_sha256'],
            'geometry': {key: resolved[key] for key in
                         ('shape_kji', 'spacing_ijk_mm', 'crop_low_ijk', 'crop_high_ijk_exclusive',
                          'roi_kind', 'roi_nodes_ijk', 'roi_radii_mm')},
            'source_files': {key: _stat(path) for key, path in resolved['source_files'].items()},
            'selector_code_sha256': {name: _hash_file(root/'src'/name) for name in modules},
            'layout_evidence_sha256': _hash_file(evidence)}


def _check_stats(records):
    for record in records:
        current = _stat(record['path'])
        if any(current[key] != record[key] for key in ('path', 'bytes', 'mtime_ns')):
            raise ValueError(f'Anatomical selection source changed: {record["path"]}; resolve anatomy again')


def _dependencies(audit):
    yield from audit.get('request', {}).get('source_files', {}).values()
    yield from audit.get('surface_evidence', {}).get('files', [])
    yield from audit.get('surface_evidence', {}).get('coordinate_dependencies', [])


def verify_selection_sources(resolved):
    """Read metadata and the small saved selector, never raw voxel payloads."""
    audit = resolved.get('anatomical_selection')
    if not audit:
        return
    path = resolved.get('anatomy_selection_path')
    if path:
        path = Path(path)
        expected = resolved['anatomy_selection_sha256']
        if (_hash_file(path) != expected or
                path.with_suffix(path.suffix + '.sha256').read_text().strip() != expected):
            raise ValueError('Saved anatomical selection changed; resolve a new selection revision')
    _check_stats(_dependencies(audit))


def verify_selection_crop(resolved, arrays):
    audit = resolved.get('anatomical_selection') or {}
    expected = audit.get('source_crop_sha256', {})
    for channel, checksum in expected.items():
        if channel not in ('act', 'atn') or _hash_bytes(np.ascontiguousarray(arrays[channel]).tobytes()) != checksum:
            raise ValueError('Native source crop differs from anatomical selection; resolve anatomy again')


def resolve_selection_filter(config, resolved):
    """Return stable ID candidates; native discovery is an explicit earlier step."""
    settings = _settings(config)
    if not settings['anatomy'] and not settings['anatomical_labels']:
        if config['selection'].get('anatomy_file'):
            raise ValueError('anatomy_file requires selection.anatomy')
        return {'source_ids': resolved['source_ids']}
    records = [r for r in resolved['catalog']['records']
               if r['classification']['tissue_name'] == config['selection']['tissue']]
    artery_ids = {r['original_id'] for r in resolved['catalog']['records']
                  if r['classification']['tissue_name'] == 'artery'}
    allowed = set(resolved['source_ids']) & _manual_ids(records, settings['anatomical_labels'])
    if not allowed:
        raise ValueError('Anatomical names and explicit source_ids have an empty intersection')
    raw, checksum, artifact = None, None, None
    if settings['anatomy']:
        artifact = config['selection'].get('anatomy_file')
        if not artifact or not Path(artifact).is_file():
            raise ValueError('Set selection.anatomy_file to a fresh JSON path, then run '
                             'python3 scripts/resolve_anatomy.py --config YOUR.ini before preview/preflight')
        artifact = Path(artifact).resolve()
        if (artifact.parent/'INCOMPLETE').exists():
            raise ValueError('Anatomical selection is INCOMPLETE')
        raw = artifact.read_bytes()
        checksum = _hash_bytes(raw)
        if artifact.with_suffix(artifact.suffix + '.sha256').read_text().strip() != checksum:
            raise ValueError('Anatomical selection checksum mismatch')
        audit = json.loads(raw)
        if audit.get('schema_version') != SCHEMA or audit.get('complete') is not True:
            raise ValueError('Expected a complete fakect.anatomy-selection/1 artifact')
        if _canonical(audit.get('request')) != _canonical(_request(config, resolved)):
            raise ValueError('Anatomical selection no longer matches ROI, source, catalog or selector; '
                             'choose a new anatomy_file and resolve anatomy again')
        selected = audit.get('selected_source_ids', [])
        if (not selected or any(type(x) is not int for x in selected) or
                len(set(selected)) != len(selected) or not set(selected) <= allowed):
            raise ValueError('Invalid resolved anatomical IDs')
        by_id = {r['original_id']: r for r in records}
        if any(_role(by_id[key], settings)[0] == 'excluded' for key in selected):
            raise ValueError('Resolved IDs violate anatomical eligibility')
    else:
        selected = sorted(allowed)
        audit = {'schema_version': SCHEMA, 'complete': True, 'anatomy': None,
                 'method': 'explicit_original_or_canonical_names',
                 'selected_source_ids': selected, 'records': [
                     {'original_id': r['original_id'], 'original_name': r['original_name'],
                      'canonical_name': r['classification'].get('canonical_name'),
                      'hierarchy_path': r['classification'].get('hierarchy_path'),
                      'decision': 'included' if r['original_id'] in allowed else 'excluded',
                      'reason': 'Explicit name/ID intersection' if r['original_id'] in allowed else 'Not requested',
                      'surface_matches': []} for r in records if r['original_id'] in resolved['source_ids']],
                 'warnings': ['Names filter original IDs; native connectivity was not requested.'],
                 'connectivity_scope': 'not_requested', 'counts': {}}
    protected = sorted(artery_ids - set(selected)) if settings['protect_other_arteries'] else []
    result = {'source_ids': tuple(selected), 'anatomical_selection': audit,
              'anatomy_selection_bytes': raw, 'anatomy_selection_sha256': checksum,
              'anatomy_selection_path': artifact, 'anatomical_protected_ids': tuple(protected)}
    verify_selection_sources(result)
    return result


def _surface_matches(surface_evidence):
    result = defaultdict(list)
    for row in surface_evidence['records']:
        result[row['original_name']].append(row)
    return result


def select_anatomy_labels(labels, roi, catalog, source_ids, settings, surface_evidence,
                         attenuation_cm_inverse=None):
    """Resolve whole original IDs from aortic-anchor connectivity in this crop.

    Disconnected pieces sharing an admitted ID cannot be distinguished by an ID
    list; their counts are explicit in the audit, not silently described as a
    component-level mask. The ROI still clips the downstream selected target.
    """
    labels, roi = np.asarray(labels), np.asarray(roi)
    if labels.shape != roi.shape or labels.ndim != 3 or roi.dtype != np.bool_:
        raise ValueError('Anatomical selection needs a matching native 3D Boolean ROI')
    if (labels.dtype.kind not in 'iuf' or not np.isfinite(labels).all() or
            np.any(labels <= -(2**31)) or np.any(labels >= 2**31) or
            np.any(labels != np.trunc(labels))):
        raise ValueError('Anatomy requires original integral ACT IDs, not attenuation or averaged labels')
    rows = [r for r in catalog['records'] if r['classification']['tissue_name'] == 'artery']
    manual = _manual_ids(rows, settings['anatomical_labels'])
    available = set(source_ids) & manual
    surfaces = _surface_matches(surface_evidence)
    surface_refs = defaultdict(list)
    for index, surface in enumerate(surface_evidence['records']):
        surface_refs[surface['original_name']].append(index)
    roi_ids, roi_counts = np.unique(labels[roi], return_counts=True)
    roi_count = dict(zip(map(int, roi_ids), map(int, roi_counts)))
    crop_ids, crop_counts = np.unique(labels, return_counts=True)
    crop_count = dict(zip(map(int, crop_ids), map(int, crop_counts)))
    roles = {}
    for row in rows:
        key = row['original_id']
        role, reason = _role(row, settings)
        if key not in available:
            role, reason = 'excluded', 'Outside explicit source-ID/anatomical-name intersection'
        elif not surfaces.get(row['original_name']):
            role, reason = 'excluded', 'No exact-name RAW surface evidence for this case/frame'
        roles[key] = (role, reason)
    eligible = np.isin(labels, [key for key, (role, _) in roles.items() if role != 'excluded'])
    anchors = roi & np.isin(labels, [key for key, (role, _) in roles.items() if role == 'anchor'])
    if not anchors.any():
        raise ValueError('No atlas-qualified, RAW-supported aortic anchor intersects this ROI; '
                         'review ROI, optional labels and source correspondence')
    structure = ndimage.generate_binary_structure(3, 1 if settings['connectivity'] == 6 else 3)
    components, _ = ndimage.label(eligible, structure=structure)
    seeded = np.unique(components[anchors])
    seeded = seeded[seeded != 0]
    connected = np.isin(components, seeded)
    selected_ids = sorted(map(int, np.unique(labels[connected & roi])))
    if not selected_ids:
        raise ValueError('Anatomical filtering left no target')
    selected_mask = roi & np.isin(labels, selected_ids)
    extra = selected_mask & ~connected
    warnings = []
    if surface_evidence.get('coordinate_contract', {}).get('status') != 'bounded_contract_validated':
        warnings.append('RAW/BIN spatial correspondence is unavailable under the pinned coordinate contract. '
                        'RAW names and native ACT connectivity remain available; centerline spatial screening was not used.')
    if extra.any():
        warnings.append(f'{int(extra.sum())} additional ROI voxels share admitted IDs but are not '
                        'aorta-connected in this crop. This is an ID filter; review these pieces or refine the ROI.')
    records = []
    for row in rows:
        key = row['original_id']
        if key not in crop_count:
            continue
        role, reason = roles[key]
        keep = key in selected_ids
        if keep:
            reason = 'Aortic anchor in ROI' if role == 'anchor' else 'Qualified branch label connected to an aortic anchor and intersecting ROI'
        elif role != 'excluded':
            reason = 'No ROI voxels' if not roi_count.get(key) else 'No eligible native connection to the ROI aortic anchor within this crop'
        matches = surfaces.get(row['original_name'], [])
        record = {'original_id': key, 'original_name': row['original_name'],
                  'canonical_name': row['classification'].get('canonical_name'),
                  'hierarchy_path': row['classification'].get('hierarchy_path'),
                  'role': role, 'roi_voxels': roi_count.get(key, 0), 'crop_voxels': crop_count[key],
                  'connected_roi_voxels': int(np.count_nonzero(roi & connected & (labels == key))),
                  'decision': 'included' if keep else 'excluded', 'reason': reason,
                  # Repeated RAW names can have hundreds of IDs and surfaces.
                  # Store geometry once, rather than duplicating their Cartesian join.
                  'surface_refs': surface_refs[row['original_name']]}
        if attenuation_cm_inverse is not None and roi_count.get(key):
            values = np.asarray(attenuation_cm_inverse)[roi & (labels == key)]
            record['attenuation_cm_inverse'] = {'min': float(values.min()), 'max': float(values.max()),
                                                'mean': float(values.mean())}
        if keep and any(len(s.get('candidate_original_ids', [])) > 1 for s in matches):
            warnings.append(f'ID {key} ({row["original_name"]}) has an ambiguous RAW-name join; '
                            'local surface ordinals are not voxel IDs. Generic systemic branches remain generically named.')
        records.append(record)
    return {'schema_version': SCHEMA, 'complete': True, 'anatomy': settings['anatomy'],
            'policy': POLICY, 'method': 'atlas_eligibility_then_native_aortic_anchor_connectivity',
            'selected_source_ids': selected_ids, 'records': records, 'warnings': warnings,
            'connectivity_scope': f'{settings["connectivity"]}-neighbor paths through eligible original arterial labels '
                                  'inside the bounded crop; anchors must overlap the ROI. Pulmonary labels cannot carry paths. '
                                  'This is voxel contact, not proof of physiological connectivity.',
            'mask_semantics': 'ROI intersected with the resolved WHOLE signed original IDs. Growth retains existing lineage semantics.',
            'counts': {'artery_roi_voxels_before': sum(roi_count.get(r['original_id'], 0) for r in rows),
                       'selected_roi_voxels': int(selected_mask.sum()),
                       'connected_roi_voxels': int((connected & roi).sum()),
                       'unconnected_retained_roi_voxels': int(extra.sum()),
                       'anchor_roi_voxels': int(anchors.sum()),
                       'selected_id_count': len(selected_ids)},
            'surface_evidence': surface_evidence}


def sample_surface_centerlines(surface_evidence, arrays, resolved):
    """Corroborate RAW-name joins by sampling existing ACT; never voxelize meshes."""
    contract = surface_evidence.get('coordinate_contract', {})
    if contract.get('status') != 'bounded_contract_validated':
        return
    origin = contract.get('grid_boundary_origin_mm', contract.get('grid_boundary_origin_raw_mm'))
    if origin is None:
        return
    spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0) or
            not np.allclose(spacing, contract.get('spacing_ijk_mm', []), rtol=1e-10, atol=1e-10)):
        raise ValueError('RAW/BIN spacing disagrees with resolved native volume')
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Invalid declared RAW/BIN boundary origin')
    low = np.asarray(resolved['crop_low_ijk'])
    shape = np.asarray(arrays['act'].shape)[::-1]
    for surface in surface_evidence['records']:
        centerline = surface.get('centerline', {})
        if centerline.get('status') != 'verified_two_pole_rings':
            continue
        points = np.asarray(centerline['centerline_raw_mm'], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Invalid verified surface centerline')
        chunks, total_samples = [], 0
        for first, last in zip(points[:-1], points[1:]):
            # Crop clipping before resampling bounds both time and allocations.
            a, b = (first - origin)/spacing, (last - origin)/spacing
            delta = b - a
            start, stop = 0., 1.
            for axis in range(3):
                if delta[axis] == 0:
                    if not low[axis] <= a[axis] <= low[axis] + shape[axis]:
                        stop = -1.; break
                else:
                    bounds = sorted(((low[axis]-a[axis])/delta[axis],
                                     (low[axis]+shape[axis]-a[axis])/delta[axis]))
                    start, stop = max(start, bounds[0]), min(stop, bounds[1])
            if stop < start:
                continue
            count = max(2, int(np.ceil(np.linalg.norm(delta)*(stop-start)*2)) + 1)
            if count > 100_000:
                raise ValueError('Surface centerline segment exceeds bounded sampling budget')
            total_samples += count
            if total_samples > 200_000:
                raise ValueError('Surface centerline exceeds bounded aggregate sampling budget')
            chunks.append(np.floor(a + np.linspace(start, stop, count)[:, None]*delta).astype(np.int64)-low)
        if not chunks:
            continue
        indices = np.unique(np.concatenate(chunks), axis=0)
        indices = indices[np.all((indices >= 0) & (indices < shape), axis=1)]
        if not len(indices):
            continue
        values = arrays['act'][tuple(indices[:, ::-1].T)].astype(np.int32)
        counts = Counter(map(int, values))
        allowed = set(surface.get('candidate_original_ids', []))
        surface['native_centerline_samples'] = {
            'sample_voxels': len(indices), 'roi_sample_voxels': int(arrays['roi'][tuple(indices[:, ::-1].T)].sum()),
            'sampled_id_counts': dict(sorted(counts.items())),
            'matching_name_id_counts': {key: count for key, count in sorted(counts.items()) if key in allowed},
            'semantics': 'Existing native ACT cells sampled along verified RAW ring-centroid segments; '
                         'name agreement corroborates a local ID mapping, not a unique whole-surface assignment.'}


def create_anatomy_selection(config, input_bytes=None):
    from fakect_roi import resolve_preview, prepare_crop
    from fakect_anatomy_surfaces import inventory_surfaces
    settings = _settings(config)
    if not settings['anatomy']:
        raise ValueError('Set selection.anatomy=thoracic_aorta for automatic resolution')
    output = config['selection'].get('anatomy_file')
    if not output or Path(output).suffix.lower() != '.json':
        raise ValueError('Set selection.anatomy_file to a fresh .json path')
    output = Path(output).resolve()
    if output.parent.exists():
        raise FileExistsError('Anatomy output folder already exists; choose a fresh anatomy_file revision')
    resolved = resolve_preview(config, apply_anatomy=False)
    request = _request(config, resolved)
    arrays, sources = prepare_crop(resolved, config)
    surfaces = inventory_surfaces(config, resolved)
    sample_surface_centerlines(surfaces, arrays, resolved)
    audit = select_anatomy_labels(arrays['act'], arrays['roi'], resolved['catalog'],
                                 resolved['source_ids'], settings, surfaces, arrays['attenuation_cm_inverse'])
    audit['request'] = request
    audit['source_crop_sha256'] = {channel: _hash_bytes(np.ascontiguousarray(arrays[channel]).tobytes())
                                   for channel in ('act', 'atn')}
    audit['native_sources'] = sources
    if _canonical(request) != _canonical(_request(config, resolved)):
        raise ValueError('Selection dependencies changed during anatomy resolution')
    _check_stats(_dependencies(audit))
    # Claim the revision exclusively after successful computation. A competing
    # job must never change another run's INCOMPLETE marker or input snapshot.
    output.parent.mkdir(parents=True, exist_ok=False)
    (output.parent/'INCOMPLETE').write_text('Anatomical selection is incomplete\n')
    raw = (_canonical(audit) + '\n').encode()
    with output.open('xb') as stream:
        stream.write(raw)
    output.with_suffix(output.suffix + '.sha256').write_text(_hash_bytes(raw) + '\n')
    if input_bytes is not None:
        (output.parent/'input.ini').write_bytes(input_bytes)
    (output.parent/'INCOMPLETE').unlink()
    return audit
