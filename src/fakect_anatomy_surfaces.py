"""Bounded RAW identity inventory and explicitly scoped envelope diagnostics.

Surface block ordinals never identify BIN voxels. Repeated dictionary names
produce candidate ID sets, not invented one-to-one surface correspondence.
Only a pinned DPI coordinate contract permits a broad-phase bounding-box test;
that test does not establish triangle intersection or anatomical connectivity.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import warnings

import numpy as np

from fakect_anatomy_centerlines import MAX_TRIANGLES, extract_centerline, centerline_envelope_overlap

ROOT = Path(__file__).resolve().parents[1]
LAYOUT_EVIDENCE = ROOT / 'references/atlas/dpi/xcat_v2_generator_layout_evidence_v1.json'
LAYOUT_EVIDENCE_SHA256 = 'a8207559fa6b39aa3ddc945f1c7e824ce98184f57b80943b95042b3282dd0d9e'
RAW_GROUPS = ('body', 'bones', 'heart', 'muscle', 'organs', 'vessels')
TRIANGLE_CHUNK_ROWS = 4096
MAX_LINE_BYTES = 4096
_HEADER = re.compile(r'[A-Za-z_][A-Za-z0-9_ .()/+\-]*\Z')


def _fingerprint(path):
    """Hash a surface dependency while checking for concurrent file changes.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.

    Returns
    -------
    dict
        Resolved path, byte size, nanosecond modification time and SHA256.

    Raises
    ------
    ValueError
        The file size or modification time changes while hashing.
    """
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed while reading: {path}')
    return {'path': str(path.resolve()), 'bytes': before.st_size,
            'mtime_ns': before.st_mtime_ns, 'sha256': digest.hexdigest()}


def _read_small_source(path):
    """Read bounded coordinate metadata and bind the exact interpreted bytes.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.

    Returns
    -------
    tuple of bytes and dict
        Content and stable path/size/mtime/SHA256 provenance.

    Raises
    ------
    ValueError
        Metadata exceeds one MiB or the source changes while reading.

    Notes
    -----
    Bind the bytes actually interpreted to stable source metadata.
    """
    path = Path(path)
    before = path.stat()
    if before.st_size > 1024 * 1024:
        raise ValueError(f'Coordinate metadata exceeds the one-MiB reader limit: {path}')
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Coordinate metadata changed while reading: {path}')
    return content, {'path': str(path.resolve()), 'bytes': before.st_size,
                     'mtime_ns': before.st_mtime_ns, 'sha256': hashlib.sha256(content).hexdigest()}


def _load_evidence():
    """Load and verify the pinned RAW/BIN coordinate evidence.

    Parameters
    ----------
    None
        This function takes no explicit arguments.

    Returns
    -------
    tuple of dict and dict
        Decoded evidence and provenance of the evidence file.

    Raises
    ------
    ValueError
        The evidence content differs from the pinned SHA256.
    """
    content, source = _read_small_source(LAYOUT_EVIDENCE)
    if source['sha256'] != LAYOUT_EVIDENCE_SHA256:
        raise ValueError('Pinned RAW/BIN coordinate evidence changed; review its provenance')
    return json.loads(content), source


def _coordinate_contract(config, resolved, evidence, provenance):
    """Validate a narrowly scoped RAW/BIN coordinate registration contract.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    resolved : dict
        Native source, catalog and crop geometry from ``resolve_preview``; indices use
        i,j,k and array shapes use k,j,i.
    evidence : dict
        Pinned generator/layout evidence and supported case/frame scope.
    provenance : dict
        Stable metadata and checksum of the pinned evidence source.

    Returns
    -------
    dict
        Validated spacing, raw-mm ROI envelope and grid origin, or unavailable status
        with explicit reasons.

    Raises
    ------
    ValueError
        Resolved ROI nodes or radii are malformed after coordinate checks pass.

    Notes
    -----
    Reads parameter metadata and generator bytes. Unsupported registration returns unavailable evidence; bounds overlap alone does not prove geometry intersection.

    Return unavailable spatial evidence unless every bounded check passes.
    """
    inp = config['input']
    result = {'status': 'unavailable', 'evidence': provenance,
              'evidence_id': evidence['evidence_id'],
              'authority': evidence['authoritative_field_grid_contract']['registration_authority'],
              'spatial_claim': 'Axis-aligned bounds overlap only; not triangle intersection or connectivity.',
              'independent_registration': False, 'reasons': []}
    reasons = result['reasons']
    if str(inp['case_id']) not in evidence['scope']['phantom_ids']:
        reasons.append('case_outside_pinned_scope')
    if inp['frame'] != evidence['scope']['selected_frame_ordinal']:
        reasons.append('frame_outside_pinned_scope')
    par_path = Path(resolved.get('case', {}).get('par_path') or
                    Path(inp['root']) / (str(inp['case_id']) + '.par'))
    required = ('startslice', 'endslice', 'x_array_size', 'y_array_size',
                'pixel_width', 'slice_width', 'activ_output_format', 'color_code', 'activity_unit')
    try:
        content, parameter_source = _read_small_source(par_path)
        rows = re.findall(r'^\s*(\w+)\s*=\s*([^#\r\n]+)', content.decode(), re.M)
        if any(sum(k == key for k, _ in rows) != 1 for key in required):
            raise ValueError('missing or duplicate required field')
        params = dict(rows)
        values = {key: float(params[key]) for key in required}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError('nonfinite parameter')
        result['parameter_source'] = parameter_source
        if values['startslice'] != 1:
            reasons.append('startslice_other_than_1')
        if (values['activ_output_format'], values['color_code'], values['activity_unit']) != (0, 1, 0):
            reasons.append('unsupported_generator_output_profile')
        expected_shape = [values['endslice'] - values['startslice'] + 1,
                          values['y_array_size'], values['x_array_size']]
        expected_spacing = [10 * values['pixel_width'], 10 * values['pixel_width'],
                            10 * values['slice_width']]
        shape = np.asarray(resolved['shape_kji'], dtype=float)
        spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
        if (shape.shape != (3,) or spacing.shape != (3,) or
                not np.all(np.isfinite(shape)) or not np.all(np.isfinite(spacing)) or
                np.any(shape <= 0) or np.any(spacing <= 0) or
                np.any(shape != np.floor(shape)) or
                not np.array_equal(shape, expected_shape) or
                not np.allclose(spacing, expected_spacing, rtol=1e-10, atol=1e-10)):
            reasons.append('parameter_geometry_disagrees_with_resolved_volume')
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        reasons.append('parameter_contract_unavailable')
    generator = Path(inp['root']) / Path(evidence['external_generator']['observed_path']).name
    try:
        observed = _fingerprint(generator)
        result['generator_observation'] = observed
        expected = evidence['external_generator']
        if observed['sha256'] != expected['sha256'] or observed['bytes'] != expected['size_bytes']:
            reasons.append('generator_content_mismatch')
    except OSError:
        reasons.append('generator_content_unavailable')
    if reasons:
        return result
    # ROI index positions address native voxel centers, while DPI's origin is
    # the lower grid boundary. A half-voxel offset is therefore mandatory.
    origin = np.array([-0.5 * shape[2] * spacing[0], -0.5 * shape[1] * spacing[1], 0.])
    nodes = np.asarray(resolved['roi_nodes_ijk'], dtype=float)
    radii = np.asarray(resolved['roi_radii_mm'], dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1:] != (3,) or len(nodes) == 0 or
            radii.shape != (len(nodes),) or not np.all(np.isfinite(nodes)) or
            not np.all(np.isfinite(radii)) or np.any(radii <= 0)):
        raise ValueError('Resolved ROI requires finite nodes and one positive radius per node')
    world_nodes = origin + (nodes + 0.5) * spacing
    result.update(status='bounded_contract_validated',
                  grid_boundary_origin_mm=origin.tolist(),
                  grid_boundary_origin_raw_mm=origin.tolist(),
                  voxel_center_offset_ijk=[0.5, 0.5, 0.5], spacing_ijk_mm=spacing.tolist(),
                  roi_envelope_raw_mm={'min_xyz': (world_nodes - radii[:, None]).min(axis=0).tolist(),
                                       'max_xyz': (world_nodes + radii[:, None]).max(axis=0).tolist()},
                  limitations=evidence['known_limits_and_fail_closed_rules'])
    return result


def _scan_file(path, group, name_ids, coordinate_contract, artery_names=()):
    """Inventory RAW blocks in bounded chunks and recover eligible centerlines.

    Parameters
    ----------
    path : pathlib.Path
        Input file path.
    group : str
        RAW anatomical group name recorded in provenance.
    name_ids : mapping of str to iterable of int
        Exact original-name joins to signed voxel IDs; repeated names retain all
        candidates.
    coordinate_contract : dict
        Validated or unavailable raw-grid registration contract.
    artery_names : iterable of str
        Names eligible for bounded triangle retention and centerline recovery.

    Returns
    -------
    tuple of dict and list of dict
        Stable file fingerprint and block records containing names, candidate IDs,
        bounds and optional ring evidence.

    Raises
    ------
    ValueError
        A RAW header/triangle is invalid, a block is empty, a line exceeds its budget,
        or the source changes.

    Notes
    -----
    Only one eligible surface retains up to ``MAX_TRIANGLES``; triangles are not returned.

    Stream chunks; retain at most MAX_TRIANGLES for one arterial block.
    """
    before = path.stat()
    hasher = hashlib.sha256()
    records, pending = [], []
    current = None

    def flush():
        if not pending:
            return
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            try:
                array = np.fromstring(b'\n'.join(pending).decode('ascii'), sep=' ', dtype=np.float64)
            except (ValueError, UnicodeError, Warning) as exc:
                raise ValueError(f'Malformed triangle geometry in {path} after line {current["line"]}') from exc
        if array.size != len(pending) * 9 or not np.all(np.isfinite(array)):
            raise ValueError(f'Nonfinite or malformed triangle geometry in {path} after line {current["line"]}')
        vertices = array.reshape(-1, 3)
        current['_low'] = np.minimum(current['_low'], vertices.min(axis=0))
        current['_high'] = np.maximum(current['_high'], vertices.max(axis=0))
        current['triangle_count'] += len(pending)
        if current['_retain_triangles']:
            if current['triangle_count'] <= MAX_TRIANGLES:
                current['_triangles'].append(array.reshape(-1, 3, 3))
            else:
                current['_triangles'].clear()
                current['_retain_triangles'] = False
                current['_centerline_reason'] = 'surface_exceeds_bounded_centerline_budget'
        pending.clear()

    def finish():
        if current is None:
            return
        flush()
        if current['triangle_count'] == 0:
            raise ValueError(f'RAW surface has no triangle geometry: {path}:{current["line"]}')
        low, high = current.pop('_low'), current.pop('_high')
        current['bounds_raw_mm'] = {'min_xyz': low.tolist(), 'max_xyz': high.tolist()}
        retained = current.pop('_retain_triangles')
        chunks = current.pop('_triangles')
        reason = current.pop('_centerline_reason')
        centerline = (extract_centerline(np.concatenate(chunks)) if retained else
                      {'status': 'unavailable', 'reason': reason})
        current['centerline'] = centerline
        overlap = centerline_envelope_overlap(centerline, coordinate_contract)
        if overlap is not None:
            current['centerline_envelope_overlap_roi_envelope'] = overlap
        if coordinate_contract['status'] == 'bounded_contract_validated':
            envelope = coordinate_contract['roi_envelope_raw_mm']
            current['bounds_overlap_roi_envelope'] = bool(
                np.all(high >= envelope['min_xyz']) and np.all(low <= envelope['max_xyz']))
            current['spatial_status'] = 'bounds_overlap_only'
        else:
            current['spatial_status'] = 'unavailable'
        records.append(current)

    with path.open('rb') as stream:
        line_number = 0
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError(f'RAW line exceeds {MAX_LINE_BYTES} bytes: {path}:{line_number}')
            hasher.update(raw)
            stripped = raw.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            try:
                float(tokens[0])
                numeric = True
            except ValueError:
                numeric = False
            if numeric:
                if current is None or len(tokens) != 9:
                    raise ValueError(f'RAW triangle requires a named block and nine numbers: {path}:{line_number}')
                pending.append(stripped)
                if len(pending) >= TRIANGLE_CHUNK_ROWS:
                    flush()
            else:
                try:
                    name = stripped.decode('ascii')
                except UnicodeError as exc:
                    raise ValueError(f'RAW surface header is not ASCII: {path}:{line_number}') from exc
                if not _HEADER.fullmatch(name):
                    raise ValueError(f'Malformed RAW surface header: {path}:{line_number}')
                finish()
                ids = sorted(name_ids.get(name, []))
                current = {'source_file': str(path.resolve()), 'group': group,
                           'local_surface_index': len(records), 'line': line_number,
                           'original_name': name, 'triangle_count': 0,
                           '_low': np.full(3, np.inf), '_high': np.full(3, -np.inf),
                           '_retain_triangles': name in artery_names, '_triangles': [],
                           '_centerline_reason': 'not_a_catalog_artery_candidate',
                           'candidate_original_ids': ids,
                           'id_join_status': 'unique_name_match' if len(ids) == 1 else
                                             'ambiguous_name' if ids else 'unmapped_name'}
        finish()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'RAW source changed during inventory: {path}')
    return ({'path': str(path.resolve()), 'group': group, 'bytes': before.st_size,
             'mtime_ns': before.st_mtime_ns, 'sha256': hasher.hexdigest(),
             'surface_count': len(records), 'triangle_count': sum(r['triangle_count'] for r in records)}, records)


def inventory_surfaces(config, resolved):
    """Inventory six RAW groups without reading or voxelizing BIN images.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    resolved : dict
        Native source, catalog and crop geometry from ``resolve_preview``; indices use
        i,j,k and array shapes use k,j,i.

    Returns
    -------
    dict
        Surface records, file hashes, coordinate dependencies and identity/spatial
        limitations.

    Raises
    ------
    ValueError
        Case/frame identity, RAW geometry, coordinate evidence or source stability is
        invalid.
    OSError
        Required surface files cannot be read.

    Notes
    -----
    Inventory all six RAW groups; never read BIN data or return triangles.

    ``resolved`` is the result of ``fakect_roi.resolve_preview``. Missing RAW
    files or malformed geometry fail. An unsupported coordinate contract still
    returns identity evidence, with spatial status explicitly unavailable.
    Only one bounded arterial surface at a time is retained for ring recovery.
    """
    inp = config['input']
    case_id, frame = str(inp['case_id']), inp['frame']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', case_id) or isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
        raise ValueError('Surface inventory requires a safe case ID and positive integer frame')
    name_ids, artery_names = defaultdict(list), set()
    for row in resolved['catalog']['records']:
        name_ids[row['original_name']].append(int(row['original_id']))
        if row.get('classification', {}).get('tissue_name') == 'artery':
            artery_names.add(row['original_name'])
    evidence, provenance = _load_evidence()
    contract = _coordinate_contract(config, resolved, evidence, provenance)
    files, records = [], []
    for group in RAW_GROUPS:
        path = Path(inp['root']) / case_id / f'{case_id}_{frame}_{group}.raw'
        file_record, surface_records = _scan_file(path, group, name_ids, contract, artery_names)
        files.append(file_record)
        records.extend(surface_records)
    return {'schema_version': 'fakect.anatomy-surface-inventory/1', 'case_id': case_id, 'frame': frame,
            'files': files, 'records': records, 'coordinate_contract': contract,
            'coordinate_dependencies': [contract[key] for key in
                                        ('evidence', 'parameter_source', 'generator_observation') if key in contract],
            'index_semantics': 'Local RAW block occurrence only; never a signed BIN ID or DPI global block index.',
            'identity_semantics': 'Exact dictionary original-name match. Repeated names retain all candidate IDs without occurrence-to-ID assignment.',
            'spatial_semantics': 'Bounds overlap is broad-phase evidence only. Native BIN label contact is authoritative; no triangle intersection is claimed.'}
