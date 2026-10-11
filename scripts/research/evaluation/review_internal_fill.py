#!/usr/bin/env python3
"""Measure conservative internal clearance of a label in all possible edit regions.

This evidence does not change tissue classification, stiffness, or source data.
The main selection tube is expanded by the largest cumulative dilation budget
across the recipe sweep, bounding every reachable target descendant. Background
padding deliberately underestimates clearance where source context is cropped.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_study_config import load_study_config
from fakect_roi import resolve_preview, tube_mask, sphere_mask

SCHEMA = 'fakect.internal-fill-spatial-review/2'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _stat(path):
    stat = Path(path).stat()
    return {'path': str(Path(path).resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def cumulative_dilation_budget(config):
    """Sum maximum distance times maximum passes, including Cartesian sweeps."""
    total = 0.
    details = []
    for name in config['recipe']['steps']:
        edit = config['edits'][name]
        if edit['operation'] != 'dilation':
            continue
        sweep = config.get('sweeps', {}).get(name, {})
        distance = max(sweep.get('distance_mm', (edit['distance_mm'],)))
        iterations = max(sweep.get('iterations', (edit['iterations'],)))
        total += distance * iterations
        details.append({'step_name': name, 'max_distance_mm': distance, 'max_iterations': iterations})
    return float(total), details


def conservative_clearance(act, spacing_ijk_mm):
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if act.ndim != 3 or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Clearance needs a 3D label volume and three positive finite spacings')
    center = ndimage.distance_transform_edt(np.pad(act != 0, 1), sampling=spacing[::-1])[1:-1, 1:-1, 1:-1]
    # Subtract both half diagonals: this bounds separation between the entire
    # potential donor voxel and any background voxel, not just their centers.
    return center, np.maximum(center - np.linalg.norm(spacing), 0.)


def growth_envelope(main_roi, spacing_ijk_mm, budget_mm):
    """Exact native Euclidean expansion, including varying-radius transitions."""
    if not main_roi.any():
        raise ValueError('Main ROI has no native voxels')
    distance = ndimage.distance_transform_edt(~main_roi, sampling=np.asarray(spacing_ijk_mm)[::-1])
    return distance <= budget_mm + 1e-9


def _measure(mask, centers, clearance, attenuation, margin_mm):
    values = centers[mask]
    lower = clearance[mask]
    signal = attenuation[mask]
    return {'voxels': int(mask.sum()),
        'center_clearance_mm_min': float(values.min()) if len(values) else None,
        'center_clearance_mm_quantiles_0_25_50_75_100': np.percentile(values, [0, 25, 50, 75, 100]).tolist() if len(values) else [],
        'conservative_surface_clearance_lower_bound_mm': float(lower.min()) if len(lower) else None,
        'voxels_within_margin': int((lower <= margin_mm).sum()),
        'attenuation_cm_inverse_min_max': [float(signal.min()), float(signal.max())] if len(signal) else []}


def inspect_case(config_path, *, label_id=4, margin_mm=3., repo_root=ROOT):
    config_path = Path(config_path).resolve()
    input_bytes = config_path.read_bytes()
    config = load_study_config(config_path, repo_root=repo_root)
    # An anatomy artifact is not required to inspect a conservative spatial
    # superset. No tissue/label selection IDs are inferred from this read.
    resolved = resolve_preview(config, apply_anatomy=False)
    low, high = np.asarray(resolved['crop_low_ijk']), np.asarray(resolved['crop_high_ijk_exclusive'])
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low[::-1], high[::-1]))
    spacing = np.asarray(resolved['spacing_ijk_mm'])
    sources = {channel: _stat(path) for channel, path in resolved['source_files'].items()}
    arrays = {channel: np.array(np.memmap(path, mode='r', dtype='<f4', shape=resolved['shape_kji'])[slices])
              for channel, path in resolved['source_files'].items()}
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError('Source crop contains nonfinite values')
    act, atn = arrays['act'], arrays['atn']
    centers, clearance = conservative_clearance(act, spacing)
    budget, budget_steps = cumulative_dilation_budget(config)
    if resolved['roi_kind'] == 'tube':
        main = tube_mask(act.shape, low, resolved['roi_nodes_ijk'], spacing, resolved['roi_radii_mm'])
        ideal_low = (np.asarray(resolved['roi_nodes_ijk']) * spacing -
                     (np.asarray(resolved['roi_radii_mm']) + budget)[:, None]).min(axis=0)
        ideal_high = (np.asarray(resolved['roi_nodes_ijk']) * spacing +
                      (np.asarray(resolved['roi_radii_mm']) + budget)[:, None]).max(axis=0)
    else:
        center, radius = config['roi']['center_ijk'], config['roi']['radius_mm']
        main = sphere_mask(act.shape, low, center, spacing, radius)
        ideal_low = np.asarray(center) * spacing - radius - budget
        ideal_high = np.asarray(center) * spacing + radius + budget
    # The proposed override must not rely on an unobserved part of the envelope.
    if np.any(ideal_low < low * spacing) or np.any(ideal_high > (high - 1) * spacing):
        raise ValueError('Conservative cumulative-growth envelope leaves the loaded crop; enlarge crop before evaluating an override')
    envelope = growth_envelope(main, spacing, budget)
    potential = (act == label_id) & envelope
    aorta_ids = [row['original_id'] for row in resolved['catalog']['records']
                if row['classification'].get('hierarchy_path') == '01_Cardiovascular/Heart/Aorta']
    aorta = np.isin(act, aorta_ids) & main
    adjacent = (act == label_id) & ndimage.binary_dilation(aorta, structure=ndimage.generate_binary_structure(3, 1))
    attenuation = atn / (spacing[0] / 10)
    borrowing = _measure(potential, centers, clearance, attenuation, margin_mm)
    face = _measure(adjacent, centers, clearance, attenuation, margin_mm)
    catalog_record = next((row for row in resolved['catalog']['records'] if row['original_id'] == label_id), None)
    if catalog_record is None:
        raise ValueError('Requested label is absent from source catalog')
    if config_path.read_bytes() != input_bytes:
        raise ValueError('INI changed during spatial evidence collection')
    if any(_stat(path) != sources[channel] for channel, path in resolved['source_files'].items()):
        raise ValueError('Native source changed during spatial evidence collection')
    return {'case_id': config['input']['case_id'], 'status': 'ok', 'config': str(config_path),
        'config_sha256': _sha(input_bytes), 'catalog_sha256': resolved['catalog_sha256'],
        'audit_sha256': resolved['audit_sha256'], 'source_files': sources,
        'act_crop_sha256': _sha(act.tobytes()), 'atn_crop_sha256': _sha(atn.tobytes()),
        'crop_low_ijk': low.tolist(), 'crop_high_ijk_exclusive': high.tolist(),
        'spacing_ijk_mm': spacing.tolist(), 'roi_nodes_ijk': resolved['roi_nodes_ijk'],
        'roi_radii_mm': resolved['roi_radii_mm'], 'conservative_dilation_budget_mm': budget,
        'budget_steps': budget_steps, 'label_id': label_id, 'margin_mm': margin_mm,
        'potential_borrowing_envelope': borrowing, 'original_aorta_face_adjacent_label': face,
        'catalog_record': catalog_record,
        'interior_margin_supported': bool(borrowing['voxels'] and not borrowing['voxels_within_margin']),
        'automatic_override_approved': False}


def collect_project(project_path, output, *, label_id=4, margin_mm=3., repo_root=ROOT):
    root = Path(repo_root).resolve()
    project_path = Path(project_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Spatial evidence output exists; choose a fresh revision')
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError('margin_mm must be finite and nonnegative')
    project = json.loads(project_path.read_bytes())
    if project.get('schema_version') != 'fakect.cohort-project/1':
        raise ValueError('Expected fakect.cohort-project/1')
    document = {'schema_version': SCHEMA, 'project': str(project_path),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'label_id': label_id,
        'margin_mm': margin_mm, 'complete': False, 'cases': [], 'source_volumes_modified': False,
        'script_sha256': _sha(Path(__file__).read_bytes()),
        'method': 'Native ACT=0 is background. Pad the finite crop with background, compute exact Euclidean distance from voxel centers to any background center, and subtract the full native voxel diagonal (two half diagonals) to bound separation between entire donor and background voxel cells. Crop padding makes this clearance a lower bound. Form the exact native Euclidean distance expansion of the complete main ROI voxel mask by the maximum cumulative dilation budget (all swept distances and pass counts), a conservative superset of original selection and target growth including varying-radius tube transitions. Reject an envelope whose continuous coordinate bounds extend beyond the loaded crop. Internal ACT=0 cavities also count as background conservatively.',
        'interpretation': 'This supports a manually reviewed interior-fill exception only for the recorded source/config envelope; it does not prove tissue histology or reclassify the entire original label. A per-label override changes stiffness throughout this config, but mutations remain limited to edit regions. Any enlarged/repositioned ROI or edit budget requires new evidence. Erosion recipient seed voxels remain unchanged; dilation can borrow donor voxels. No override or freeze is automatically approved.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    for entry in project['cases']:
        config = Path(entry['config'])
        config = config if config.is_absolute() else root / config
        try:
            row = inspect_case(config, label_id=label_id, margin_mm=margin_mm, repo_root=root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row = {'case_id': str(entry['case_id']), 'config': str(config), 'status': 'failed',
                   'error': str(error), 'interior_margin_supported': False, 'automatic_override_approved': False}
        document['cases'].append(row)
        output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'case_id': row['case_id'], 'status': row['status'],
            'interior_margin_supported': row['interior_margin_supported'],
            'clearance_lower_bound_mm': row.get('potential_borrowing_envelope', {}).get('conservative_surface_clearance_lower_bound_mm'),
            'error': row.get('error')}, allow_nan=False), flush=True)
    document['complete'] = True
    document['failed_case_count'] = sum(row['status'] != 'ok' for row in document['cases'])
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label-id', type=int, default=4)
    parser.add_argument('--margin-mm', type=float, default=3.)
    args = parser.parse_args()
    try:
        result = collect_project(args.project, args.output, label_id=args.label_id, margin_mm=args.margin_mm)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, 'error: ' + str(error) + '\n')
    if result['failed_case_count']:
        parser.exit(2, 'error: some case evidence could not be collected; inspect the recorded failures\n')


if __name__ == '__main__':
    main()
