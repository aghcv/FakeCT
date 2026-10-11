"""Read-only intensity and anatomical-axis evidence for XCAT CT adapters.

The generated values are water-normalized attenuation, not a newly simulated
clinical acquisition. RAS signs require both the bounded generator contract and
separate named-surface landmarks; no identity affine or RAW ordinal is assumed.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from fakect_anatomy_surfaces import _load_evidence, _coordinate_contract

SCHEMA = 'fakect.pretrained-input-evidence/1'
_AXIS_NOTES = ('x-axis runs from the right side of the body to the left',
               'y-axis runs from the front of the body to the back',
               'z-axis runs from the feet to the head')


def _read(path, checksum=None, maximum_bytes=64 * 1024 ** 2):
    path = Path(path).resolve()
    before = path.stat()
    if before.st_size > maximum_bytes:
        raise ValueError(f'Evidence metadata exceeds its bounded reader budget: {path}')
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Evidence source changed during read: {path}')
    digest = hashlib.sha256(content).hexdigest()
    if checksum is not None and digest != checksum:
        raise ValueError(f'Evidence checksum mismatch: {path}')
    return content, {'path': str(path), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
                     'sha256': digest, 'integrity_scope': 'content_hash'}


def _check_stat(record):
    path = Path(record['path']).resolve()
    stat = path.stat()
    if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
        raise ValueError(f'Frozen evidence source changed: {path}')
    result = dict(record)
    result['path'] = str(path)
    result['integrity_scope'] = 'recorded_hash_bound_to_current_stat' if 'sha256' in result else 'size_and_mtime_only'
    return result


def _parameter_values(text):
    rows = re.findall(r'^\s*(\w+)\s*=\s*([^#\r\n]+)', text, re.M)
    required = ('pixel_width', 'slice_width', 'energy', 'startslice', 'phan_rotx', 'phan_roty', 'phan_rotz')
    values = {}
    for key in required:
        matches = [value.strip() for name, value in rows if name == key]
        if len(matches) != 1:
            raise ValueError(f'Expected one XCAT parameter {key}')
        try:
            value = float(matches[0])
        except ValueError as error:
            raise ValueError(f'Invalid XCAT parameter {key}') from error
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite XCAT parameter {key}')
        values[key] = value
    return values


def calibration_evidence(parameter_text, log_text, spacing_ijk_mm):
    """Read the same-energy water reference and verify per-pixel scaling."""
    par = _parameter_values(parameter_text)
    if par['pixel_width'] <= 0 or par['slice_width'] <= 0 or par['energy'] <= 0:
        raise ValueError('Positive XCAT widths and photon energy are required')
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    expected = np.array([par['pixel_width'], par['pixel_width'], par['slice_width']]) * 10
    if spacing.shape != (3,) or not np.allclose(spacing, expected, rtol=1e-10, atol=1e-10):
        raise ValueError('Calibration spacing disagrees with XCAT parameter file')
    energies = re.findall(r'radionuclide energy\s*\(keV\)\s*=\s*([0-9.eE+\-]+)', log_text, re.I)
    if len(energies) != 1 or not np.isclose(float(energies[0]), par['energy'], rtol=0, atol=1e-6):
        raise ValueError('XCAT photon energy disagrees between parameters and generation log')
    sections = re.split(r'Linear Attenuation Coefficients\s*\(1/(cm|pixel)\)\s*:', log_text)
    water = {}
    for index in range(1, len(sections), 2):
        unit, body = sections[index], sections[index + 1]
        if unit in water:
            raise ValueError('Duplicate attenuation coefficient table')
        matches = re.findall(r'^\s*Body\s*\(water\)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$', body, re.M)
        if len(matches) != 1:
            raise ValueError('Expected one water attenuation entry per coefficient table')
        token = matches[0]
        digits = len(token.partition('.')[2])
        water[unit] = {'value': float(token), 'printed_value': token,
                       'rounding_half_unit': .5 * 10 ** (-digits)}
    if set(water) != {'cm', 'pixel'} or water['cm']['value'] <= 0 or water['pixel']['value'] <= 0:
        raise ValueError('Positive water attenuation in both cm and pixel units is required')
    error = abs(water['pixel']['value'] - water['cm']['value'] * par['pixel_width'])
    tolerance = water['pixel']['rounding_half_unit'] + water['cm']['rounding_half_unit'] * par['pixel_width'] + 1e-12
    if error > tolerance:
        raise ValueError('Logged water coefficients violate XCAT per-pixel scaling')
    return {'status': 'generation_log_water_reference', 'mu_water_cm_inverse': water['cm']['value'],
            'mu_water_printed_value': water['cm']['printed_value'],
            'mu_water_rounding_half_unit_cm_inverse': water['cm']['rounding_half_unit'],
            'mu_water_per_pixel_printed': water['pixel']['value'],
            'photon_energy_keV': par['energy'], 'energy_semantics': 'XCAT photon energy in keV; not scanner tube potential in kVp.',
            'source_atn_units': 'per_pixel', 'source_atn_to_cm_inverse_scale': 1 / par['pixel_width'],
            'prepared_image_units': 'cm^-1', 'output_units': 'water_normalized_HU',
            'formula': 'HU = 1000 * (mu_cm_inverse / mu_water_cm_inverse - 1)',
            'zero_attenuation_HU': -1000., 'clinical_reconstruction': False,
            'calibration_precision': 'Water coefficient is rounded in the generation log; no additional precision is claimed.'}


def landmark_orientation(records, parameter_text, shape_kji, spacing_ijk_mm, boundary_origin_raw_mm):
    """Confirm generator axis signs with named anatomy, then construct RAS."""
    par = _parameter_values(parameter_text)
    if par['startslice'] != 1 or any(par[key] != 0 for key in ('phan_rotx', 'phan_roty', 'phan_rotz')):
        raise ValueError('Axis-aligned pilot requires startslice=1 and zero global phantom rotations')
    folded = ' '.join(parameter_text.lower().split())
    if not all(note in folded for note in _AXIS_NOTES):
        raise ValueError('XCAT parameter axis-direction notes are missing or changed')
    by_name = defaultdict(list)
    for row in records:
        by_name[row['original_name']].append(row)
    landmarks = {}
    for name, group in (('llung', 'organs'), ('rlung', 'organs'), ('lkidney', 'organs'),
                        ('rkidney', 'organs'), ('sternum', 'bones'), ('T6_vertebrae', 'bones'),
                        ('head', 'body'), ('foot_left', 'body'), ('foot_right', 'body')):
        matches = [row for row in by_name[name] if row['group'] == group]
        if not matches:
            raise ValueError(f'Missing independent anatomical orientation landmark: {name}')
        lows = np.asarray([row['bounds_raw_mm']['min_xyz'] for row in matches], dtype=float)
        highs = np.asarray([row['bounds_raw_mm']['max_xyz'] for row in matches], dtype=float)
        if lows.shape != (len(matches), 3) or highs.shape != lows.shape or not np.isfinite(lows).all() or not np.isfinite(highs).all() or np.any(highs < lows):
            raise ValueError(f'Invalid anatomical landmark bounds: {name}')
        low, high = lows.min(axis=0), highs.max(axis=0)
        landmarks[name] = {'group': group, 'bounds_raw_mm': {'min_xyz': low.tolist(), 'max_xyz': high.tolist()},
                           'bounds_midpoint_raw_mm': ((low + high) / 2).tolist(),
                           'surface_occurrences': [{'source_file': row['source_file'],
                                                    'local_surface_index': row['local_surface_index']}
                                                   for row in matches]}
    checks = []
    for low_name, high_name, axis, meaning in (
            ('rlung', 'llung', 0, '+i points toward anatomical left'),
            ('rkidney', 'lkidney', 0, '+i points toward anatomical left'),
            ('sternum', 'T6_vertebrae', 1, '+j points toward anatomical posterior'),
            ('foot_left', 'head', 2, '+k points toward anatomical superior'),
            ('foot_right', 'head', 2, '+k points toward anatomical superior')):
        delta = np.array(landmarks[high_name]['bounds_midpoint_raw_mm']) - landmarks[low_name]['bounds_midpoint_raw_mm']
        if delta[axis] <= 0 or delta[axis] <= max(abs(delta[n]) for n in range(3) if n != axis):
            raise ValueError(f'Anatomical orientation landmark contradicts generator axis: {low_name} -> {high_name}')
        checks.append({'from_landmark': low_name, 'to_landmark': high_name, 'delta_raw_mm': delta.tolist(),
                       'dominant_axis': axis, 'interpretation': meaning, 'passed': True})
    shape = np.asarray(shape_kji, dtype=float)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    boundary = np.asarray(boundary_origin_raw_mm, dtype=float)
    if (shape.shape != (3,) or spacing.shape != (3,) or boundary.shape != (3,) or
            not np.isfinite(shape).all() or not np.isfinite(spacing).all() or not np.isfinite(boundary).all() or
            np.any(shape <= 0) or np.any(spacing <= 0) or np.any(shape != np.floor(shape))):
        raise ValueError('Invalid native grid for anatomical affine')
    expected_boundary = np.array([-.5 * shape[2] * spacing[0], -.5 * shape[1] * spacing[1], 0])
    if not np.allclose(boundary, expected_boundary, rtol=0, atol=1e-8):
        raise ValueError('Boundary origin contradicts bounded startslice=1 generator contract')
    signs = np.array([-1., -1., 1.])
    affine = np.eye(4)
    affine[:3, :3] = np.diag(signs * spacing)
    affine[:3, 3] = signs * (boundary + .5 * spacing)
    return {'status': 'supported_by_generator_notes_and_named_surface_landmarks',
            'native_axis_directions': ['L', 'P', 'S'], 'nifti_world_coordinates': 'RAS',
            'affine_ras_ijk': affine.tolist(), 'shape_kji': shape.astype(int).tolist(),
            'spacing_ijk_mm': spacing.tolist(), 'grid_boundary_origin_raw_mm': boundary.tolist(),
            'full_volume_first_voxel_center_raw_mm': (boundary + .5 * spacing).tolist(),
            'array_contract': 'Affine addresses full native voxel-center ijk. Transpose stored kji arrays to ijk; crop affine adds its global ijk origin.',
            'landmarks': landmarks, 'landmark_checks': checks,
            'independent_surface_to_volume_registration': False,
            'scope': 'Anatomical axis-sign checks independent of parameter text; RAW-to-BIN mapping still relies on the pinned generator contract.'}


def case_evidence(sample_row):
    """Read small checked evidence/cache files; do not read native voxel arrays.

    The row is from a frozen context/sibling dataset and supplies the selection
    manifest, native geometry and original ACT/ATN records. All returned source
    dependencies remain external and immutable; this function writes nothing.
    """
    row = sample_row
    case_id, frame = str(row['case_id']), row['frame']
    if frame != 1 or row.get('source_atn_units') != 'per_pixel' or row.get('image_units') != 'cm^-1' or row.get('source_atn_dtype') != '<f4':
        raise ValueError('Pilot evidence requires frame-1 per-pixel ATN and cm^-1 prepared images')
    if any(row[key]['bytes'] != 4 * int(np.prod(row['source_shape_kji'])) for key in ('source_act', 'source_atn')):
        raise ValueError('Source byte lengths disagree with the native float32 grid')
    dependencies = [_check_stat(row[key]) for key in ('source_act', 'source_atn')]
    content, source = _read(row['selection_manifest'], row['selection_manifest_sha256'])
    dependencies.append(source)
    selection = json.loads(content)
    if selection.get('case_id') != case_id or selection.get('frame') != frame or not selection.get('complete'):
        raise ValueError('Anatomical selection metadata does not match this complete case/frame')
    content, source = _read(selection['audit_path'], selection['audit_sha256'])
    audit = json.loads(content)
    dependencies.append(source)
    inventory_source = selection['surface_inventory']
    content, source = _read(inventory_source['path'], inventory_source['sha256'])
    dependencies.append(source)
    cached = json.loads(content)
    surfaces = cached.get('surface_evidence', cached)
    if surfaces.get('case_id') != case_id or surfaces.get('frame') != frame:
        raise ValueError('Surface inventory case/frame mismatch')
    dependencies.extend(_check_stat(item) for item in surfaces['files'])
    old_contract = selection['coordinate_contract']
    if old_contract.get('status') != 'bounded_contract_validated':
        raise ValueError('Selection lacks a bounded RAW/BIN coordinate contract')
    par_path = Path(old_contract['parameter_source']['path'])
    parameter_bytes, source = _read(par_path, old_contract['parameter_source']['sha256'], 1024 ** 2)
    dependencies.append(source)
    source_root = par_path.parent
    if par_path.name != case_id + '.par' or any(Path(row[key]['path']).resolve().parent != source_root / case_id for key in ('source_act', 'source_atn')):
        raise ValueError('Parameter and source volume locations disagree')
    log_path = source_root / case_id / (case_id + '_log')
    log_checksum = audit.get('source_metadata_sha256', {}).get(str(log_path.resolve()))
    par_checksum = audit.get('source_metadata_sha256', {}).get(str(par_path.resolve()))
    if not log_checksum or par_checksum != old_contract['parameter_source']['sha256']:
        raise ValueError('Frozen source audit does not bind both calibration metadata files')
    log_bytes, source = _read(log_path, log_checksum, maximum_bytes=1024 ** 2)
    dependencies.append(source)
    evidence, provenance = _load_evidence()
    geometry = selection['geometry']
    if (list(geometry['shape_kji']) != list(row['source_shape_kji']) or
            not np.allclose(geometry['spacing_ijk_mm'], row['spacing_ijk_mm'], rtol=0, atol=1e-9)):
        raise ValueError('Selection and dataset native geometry disagree')
    current = _coordinate_contract({'input': {'root': source_root, 'case_id': case_id, 'frame': frame}},
                                   {'case': {'par_path': str(par_path)}, 'shape_kji': row['source_shape_kji'],
                                    'spacing_ijk_mm': row['spacing_ijk_mm'], 'roi_nodes_ijk': [[0., 0., 0.]],
                                    'roi_radii_mm': [1.]}, evidence, provenance)
    if current['status'] != 'bounded_contract_validated':
        raise ValueError('Current RAW/BIN contract failed: ' + ', '.join(current['reasons']))
    dependencies.extend(current[key] for key in ('evidence', 'parameter_source', 'generator_observation'))
    parameter_text, log_text = parameter_bytes.decode(), log_bytes.decode()
    calibration = calibration_evidence(parameter_text, log_text, row['spacing_ijk_mm'])
    orientation = landmark_orientation(surfaces['records'], parameter_text, row['source_shape_kji'],
                                       row['spacing_ijk_mm'], current['grid_boundary_origin_raw_mm'])
    for item in dependencies:
        _check_stat(item)
    unique = {}
    for item in dependencies:
        previous = unique.setdefault(item['path'], item)
        if any(previous.get(key) != item.get(key) for key in ('bytes', 'mtime_ns', 'sha256')):
            raise ValueError('Conflicting evidence observations for ' + item['path'])
    return {'schema_version': SCHEMA, 'case_id': case_id, 'frame': frame,
            'calibration': calibration, 'orientation': orientation,
            'coordinate_evidence_id': current['evidence_id'],
            'dependencies': list(unique.values()),
            'warnings': ['Water-normalized XCAT attenuation is not a reconstructed clinical CT acquisition.',
                         'The water calibration uses the rounded value printed in each generation log.',
                         'Anatomical sign checks do not independently register RAW surfaces to BIN voxels.',
                         'Whole-volume ACT/ATN payloads are bound by size/mtime, not full content hashes.']}
