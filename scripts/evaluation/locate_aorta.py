#!/usr/bin/env python3
"""Read-only localization of an explicit XCAT aorta ID and provisional arch ROI.

The first pass reads spaced, complete axial planes; the second reads every plane
within their detected interval plus one stride. The result is an observed extent,
not proof that small disconnected pieces are absent in unsampled outer planes.
Surface names corroborate the dictionary; surface ordinals are never voxel IDs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage
from skimage.graph import route_through_array

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_roi import read_crop, tube_crop_bounds, tube_mask
from fakect_tissues import read_organ_table


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_planes(path, shape, levels, target_id):
    points, rows = [], []
    sample_hash = hashlib.sha256()
    plane_size = shape[1] * shape[2]
    with path.open('rb', buffering=0) as stream:
        for number, k in enumerate(levels):
            stream.seek(int(k) * plane_size * 4)
            plane = np.fromfile(stream, dtype='<f4', count=plane_size)
            if len(plane) != plane_size:
                raise ValueError(f'Short source plane at k={k}')
            sample_hash.update(plane.tobytes())
            plane = plane.reshape(shape[1:])
            j, i = np.nonzero(plane == target_id)
            if len(i):
                points.append(np.column_stack((np.full(len(i), k), j, i)).astype(np.int32))
                rows.append({'k': int(k), 'count': int(len(i)),
                             'bbox_low_ij': [int(i.min()), int(j.min())],
                             'bbox_high_ij_exclusive': [int(i.max()+1), int(j.max()+1)],
                             'mean_ij': [float(i.mean()), float(j.mean())]})
            if number % 64 == 0 or number == len(levels)-1:
                print(f'Read {number+1}/{len(levels)} planes; k={k}', flush=True)
    return np.concatenate(points) if points else np.empty((0, 3), dtype=np.int32), rows, sample_hash.hexdigest()


def surface_evidence(path, name):
    vertices, first, content_hash = [], None, hashlib.sha256()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip() == name:
                if first is not None:
                    raise ValueError(f'Multiple named surfaces {name}; disambiguate the source explicitly')
                first = line_number
                content_hash.update(line.encode())
                continue
            if first is None:
                continue
            fields = line.split()
            if len(fields) != 9:
                break
            try:
                triangle = np.asarray([float(value) for value in fields]).reshape(3, 3)
            except ValueError:
                break
            if not np.all(np.isfinite(triangle)):
                raise ValueError('Nonfinite surface vertices')
            content_hash.update(line.encode())
            vertices.append(triangle)
    if not vertices:
        raise ValueError(f'No triangle surface named {name} in {path}')
    values = np.concatenate(vertices)
    stat = path.stat()
    return {'path': str(path), 'name': name, 'header_line': first,
            'triangle_count': len(vertices), 'surface_xyz_min': values.min(axis=0).tolist(),
            'surface_xyz_max': values.max(axis=0).tolist(), 'named_surface_text_sha256': content_hash.hexdigest(),
            'file_bytes': stat.st_size, 'file_mtime_ns': stat.st_mtime_ns,
            'coordinate_status': 'Native-volume affine not inferred from raw coordinates; only named surface evidence retained.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/aghorban/slurm/xcat/260602/260602_act_1.bin'))
    parser.add_argument('--shape-kji', type=int, nargs=3, default=(1860, 750, 750))
    parser.add_argument('--spacing-ijk-mm', type=float, nargs=3, default=(1, 1, 1))
    parser.add_argument('--label-id', type=int, default=2922)
    parser.add_argument('--coarse-stride', type=int, default=32)
    parser.add_argument('--dictionary', type=Path, default=ROOT/'references/atlas/xcat/organ_ids.primary.txt')
    parser.add_argument('--catalog', type=Path, default=ROOT/'outputs/tissues/atlas-v1/label-catalog.json')
    parser.add_argument('--surface', type=Path, default=Path('/home/aghorban/slurm/xcat/260602/260602_1_heart.raw'))
    parser.add_argument('--arch-start-ijk', type=int, nargs=3, default=(387, 360, 1352))
    parser.add_argument('--arch-end-ijk', type=int, nargs=3, default=(389, 419, 1336))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; choose a new localization report path')
    shape, spacing = tuple(args.shape_kji), tuple(args.spacing_ijk_mm)
    if min(shape) < 1 or min(spacing) <= 0 or not np.all(np.isfinite(spacing)) or args.coarse_stride < 1:
        parser.error('Shape, spacing, and coarse stride must be positive and finite')
    if shape[1]*shape[2]*4 > 256*1024**2:
        parser.error('Axial plane exceeds bounded reader limit')
    before, started = args.source.stat(), time.monotonic()
    if before.st_size != int(np.prod(shape))*4:
        parser.error('Source size disagrees with float32 shape')
    dictionary = read_organ_table(args.dictionary)
    name = dictionary[args.label_id]
    catalog = json.loads(args.catalog.read_bytes())
    records = {r['original_id']: r for r in catalog['records']}
    record = records[args.label_id]
    if record['original_name'] != name or record['classification']['tissue_name'] != 'artery':
        parser.error('Dictionary/catalog do not support the requested aorta artery label')
    coarse_levels = list(range(0, shape[0], args.coarse_stride))
    coarse_points, coarse_rows, coarse_hash = scan_planes(args.source, shape, coarse_levels, args.label_id)
    if not len(coarse_points):
        parser.error('Target absent from coarse planes; reduce the stride before selecting an ROI')
    first = max(0, int(coarse_points[:, 0].min())-args.coarse_stride)
    stop = min(shape[0], int(coarse_points[:, 0].max())+args.coarse_stride+1)
    levels = list(range(first, stop))
    points, fine_rows, fine_hash = scan_planes(args.source, shape, levels, args.label_id)
    low_kji, high_kji = points.min(axis=0), points.max(axis=0)+1
    low = np.maximum(0, low_kji[::-1]-1)
    high = np.minimum(shape[::-1], high_kji[::-1]+1)
    if int(np.prod(high-low)) > 8_000_000:
        parser.error('Observed target bounding volume exceeds localization memory limit')
    # Python integers avoid int32 overflow when seeking beyond the 2-GiB offset.
    labels, source_record = read_crop(args.source, shape, tuple(map(int, low)), tuple(map(int, high)))
    target = labels == args.label_id
    components, count = ndimage.label(target, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel())[1:]
    component_rows = []
    for number in np.argsort(sizes)[::-1]:
        locations = np.argwhere(components == number+1)
        component_rows.append({'voxels': int(sizes[number]),
                               'bbox_low_ijk': (locations.min(axis=0)[::-1]+low).tolist(),
                               'bbox_high_ijk_exclusive': (locations.max(axis=0)[::-1]+low+1).tolist()})
    # Report direct neighboring arterial IDs as contacts, never as assumed aorta.
    contact = ndimage.binary_dilation(target, structure=ndimage.generate_binary_structure(3, 1)) & ~target
    contacts = []
    for key, number in zip(*np.unique(labels[contact], return_counts=True)):
        other = records.get(int(key), {})
        if other.get('classification', {}).get('tissue_name') != 'artery':
            continue
        locations = np.argwhere(contact & (labels == key))
        contacts.append({'original_id': int(key), 'name': other['original_name'], 'contact_voxels': int(number),
                         'bbox_low_ijk': (locations.min(axis=0)[::-1]+low).tolist(),
                         'bbox_high_ijk_exclusive': (locations.max(axis=0)[::-1]+low+1).tolist()})
    # In-plane connected limbs document why arbitrary k-sorting cannot trace an arch.
    plane_components = []
    for k in range(max(first, args.arch_end_ijk[2]), stop, 8):
        local_k = k-low[2]
        if not 0 <= local_k < target.shape[0]:
            continue
        plane_labels, n = ndimage.label(target[local_k])
        rows = []
        for group in range(1, n+1):
            j, i = np.nonzero(plane_labels == group)
            if len(i) >= 5:
                rows.append({'voxels': int(len(i)), 'mean_ijk': [float(i.mean()+low[0]), float(j.mean()+low[1]), k]})
        if rows:
            plane_components.append({'k': k, 'components': rows})
    start = tuple((np.asarray(args.arch_start_ijk)-low)[::-1])
    end = tuple((np.asarray(args.arch_end_ijk)-low)[::-1])
    if (any(v < 0 or v >= n for v, n in zip(start, target.shape))
            or any(v < 0 or v >= n for v, n in zip(end, target.shape)) or not target[start] or not target[end]):
        parser.error('Provisional arch endpoints must lie in the observed target; edit endpoint coordinates')
    edt = ndimage.distance_transform_edt(target, sampling=spacing[::-1])
    costs = np.where(target, 1/np.maximum(edt, .1)**2, np.inf)
    route, cost = route_through_array(costs, start, end, fully_connected=True, geometric=True)
    route = np.asarray(route, dtype=int)
    native = route[:, ::-1]+low
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(native, axis=0)*spacing, axis=1))]
    sample = np.unique(np.r_[0, np.searchsorted(arc, np.arange(15, arc[-1], 15)), len(route)-1])
    nodes = native[sample]
    # Visible selection envelope, not measured wall radius or a reviewed edit.
    radii = np.maximum(10., np.ceil(edt[tuple(route[sample].T)])+3)
    padding = float(radii.max()+9)
    roi_low, roi_high = tube_crop_bounds(nodes, padding, shape, spacing)
    crop_shape = np.asarray(roi_high)-roi_low
    roi_mask = tube_mask(tuple(crop_shape[::-1]), roi_low, nodes, spacing, radii)
    native_target = points[:, ::-1]
    inside_crop = np.all((native_target >= roi_low) & (native_target < roi_high), axis=1)
    target_locations = native_target[inside_crop]-roi_low
    target_in_roi = int(np.count_nonzero(roi_mask[tuple(target_locations[:, ::-1].T)]))
    volume_stride, context_groups = 2, 3
    display_cells = int(np.prod(np.ceil(crop_shape/volume_stride).astype(int)+2)*(1+context_groups))
    after = args.source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during localization')
    report = {'schema_version': 'fakect.aorta-localization/1',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'provisional_unreviewed',
              'source': {'path': str(args.source), 'dtype': '<f4', 'shape_kji': list(shape),
                         'spacing_ijk_mm': list(spacing), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns},
              'identity': {'original_id': args.label_id, 'original_name': name,
                           'hierarchy_path': record['classification']['hierarchy_path'],
                           'tissue': record['classification']['tissue_name'],
                           'dictionary_path': str(args.dictionary), 'dictionary_sha256': digest(args.dictionary),
                           'catalog_path': str(args.catalog), 'catalog_sha256': digest(args.catalog)},
              'surface': surface_evidence(args.surface, name),
              'coarse_scan': {'k_stride': args.coarse_stride, 'planes_read': len(coarse_levels),
                              'plane_payload_sha256': coarse_hash, 'planes_with_target': coarse_rows},
              'refined_scan': {'k_range_half_open': [first, stop], 'planes_read': len(levels),
                               'plane_payload_sha256': fine_hash, 'planes_with_target': fine_rows},
              'observed_target': {'bbox_low_ijk': low_kji[::-1].tolist(),
                                  'bbox_high_ijk_exclusive': high_kji[::-1].tolist(),
                                  'voxels': int(target.sum()), 'components_6': component_rows},
              'arterial_contacts': contacts,
              'contact_interpretation': 'Six-neighbor spatial adjacency only; generic arteries and named branches are not automatically aortic continuations or included in target.',
              'arch_plane_components': plane_components,
              'provisional_arch_roi': {'shape': 'tube', 'source_ids': [args.label_id],
                                       'center_ijk': nodes.tolist(), 'radius_mm': radii.tolist(),
                                       'crop_half_width_mm': padding, 'crop_low_ijk': list(roi_low),
                                       'crop_high_ijk_exclusive': list(roi_high),
                                       'crop_voxels': int(np.prod(crop_shape)), 'volume_stride': volume_stride,
                                       'target_voxels_in_crop': int(inside_crop.sum()),
                                       'target_voxels_in_roi': target_in_roi,
                                       'target_fraction_in_roi': target_in_roi / int(inside_crop.sum()),
                                       'display_budget_cells_for_3_context_groups': display_cells,
                                       'edit_2m_voxel_cap_passes': int(np.prod(crop_shape)) <= 2_000_000,
                                       'display_650k_cap_passes': display_cells <= 650_000,
                                       'coordinate_reviewed': False, 'operation': 'none',
                                       'path_length_mm': float(arc[-1]),
                                       'path_method': 'Provisional 26-neighbor minimum inverse-square interior-distance path sampled every approximately15mm; supplied path order retained.',
                                       'radius_method': 'ceil(interior EDT at each control)+3mm, minimum10mm; selection envelope only, not a calibrated vessel radius.',
                                       'target_display_contract': 'Display every original-ID2922 voxel in the crop as the target context; separately overlay ROI and target∩ROI.'},
              'source_crop_provenance': source_record,
              'limitations': ['Unreviewed ROI: do not generate a cohort or train from these coordinates before user review.',
                              'The explicit dias_aorta label is the defined target; complete anatomic aorta may continue under generic artery IDs.',
                              'Fine scan covers the detected interval only; small disconnected target pieces between coarse outer planes are not ruled out.',
                              'Raw surface ordinal is not a voxel label and raw surface coordinates are not assumed registered.'],
              'script_sha256': digest(__file__), 'elapsed_seconds': time.monotonic()-started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.out), 'observed_target': report['observed_target'],
                      'provisional_arch_roi': report['provisional_arch_roi']}, indent=2))


if __name__ == '__main__':
    main()
