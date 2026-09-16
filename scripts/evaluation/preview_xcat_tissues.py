#!/usr/bin/env python3
"""Render coordinate-matched attenuation, original-ID and coarse-group previews.

Source reads are whole contiguous axial planes. This avoids expensive sagittal
memmap page faults on NFS while retaining bounded memory and explicit sampling.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_tissues import coarse_labels

COLORS = {
    'background': '#000000', 'soft_tissue': '#bcaaa4', 'bone': '#fff0bc',
    'cartilage': '#66bb6a', 'muscle': '#b85c38', 'artery': '#f44336',
    'vein': '#367bf5', 'lung': '#4dd0c8', 'adipose': '#ffd600',
    'nervous_tissue': '#9575cd', 'fluid': '#b3e5fc', 'unknown': '#ff00cc',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_samples(case, frame, crosshair, context_step, inplane_step):
    shape = tuple(case['shape_kji'])
    k, j, i = crosshair
    context_k = np.arange(0, shape[0], context_step)
    histogram_k = case['sample_planes_k']
    stride_j, stride_i = case['sample_stride_ji']
    needed = sorted(set(context_k.tolist() + histogram_k + [k]))
    context_index = {int(value): n for n, value in enumerate(context_k)}
    selected_histogram = set(histogram_k)
    channels, source_records = {}, []
    for channel in ('atn', 'act'):
        path = Path(case['directory']) / f"{case['case_id']}_{channel}_{frame}.bin"
        before = path.stat()
        expected = int(np.prod(shape)) * 4
        if before.st_size != expected:
            raise ValueError(f'Unexpected source size: {path}')
        coronal = np.empty((len(context_k), len(range(0, shape[2], inplane_step))), dtype='<f4')
        sagittal = np.empty((len(context_k), len(range(0, shape[1], inplane_step))), dtype='<f4')
        histogram, axial = [], None
        with path.open('rb', buffering=0) as stream:
            for n, z in enumerate(needed):
                stream.seek(z * shape[1] * shape[2] * 4)
                plane = np.fromfile(stream, dtype='<f4', count=shape[1] * shape[2])
                if plane.size != shape[1] * shape[2]:
                    raise ValueError(f'Short source read: {path}, plane {z}')
                plane = plane.reshape(shape[1:])
                if z == k:
                    axial = plane.copy()
                if z in context_index:
                    row = context_index[z]
                    coronal[row] = plane[j, ::inplane_step]
                    sagittal[row] = plane[::inplane_step, i]
                if z in selected_histogram:
                    histogram.append(plane[::stride_j, ::stride_i].ravel().copy())
                if n % 64 == 0 or n == len(needed) - 1:
                    print(f'{channel}: read {n + 1}/{len(needed)} selected axial planes', flush=True)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed during preview: {path}')
        arrays = [axial, coronal, sagittal]
        histogram = np.concatenate(histogram)
        if any(not np.all(np.isfinite(a)) for a in arrays + [histogram]):
            raise ValueError(f'Nonfinite sampled source values: {path}')
        channels[channel] = {'views': arrays, 'histogram': histogram}
        source_records.append({'path': str(path), 'bytes': before.st_size,
                               'mtime_ns': before.st_mtime_ns, 'dtype': '<f4',
                               'contiguous_planes_read': needed,
                               'requested_bytes': len(needed) * shape[1] * shape[2] * 4,
                               'sample_sha256': hashlib.sha256(b''.join(a.tobytes() for a in arrays + [histogram])).hexdigest()})
    return channels, source_records


def draw(case, catalog, channels, crosshair, context_step, inplane_step, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba
    from matplotlib.patches import Patch

    colors = {c['name']: COLORS[c['name']] for c in catalog['categories']}
    lut = np.tile(to_rgba(colors['unknown']), (256, 1))
    for category in catalog['categories']:
        lut[category['id']] = to_rgba(colors[category['name']])
    tissue_cmap = ListedColormap(lut)
    original_cmap = plt.get_cmap('turbo').copy()
    original_cmap.set_bad(colors['background'])
    original = channels['act']['views']
    grouped = [coarse_labels(a, catalog) for a in original]
    group_sample = coarse_labels(channels['act']['histogram'], catalog)
    counts = Counter(map(int, group_sample))
    by_id = {r['original_id']: r for r in catalog['records']}
    unknown_code = next(c['id'] for c in catalog['categories'] if c['name'] == 'unknown')
    background_code = next(c['id'] for c in catalog['categories'] if c['name'] == 'background')
    spacing = case['spacing_ijk_mm']
    width_cm = spacing[0] / 10
    attenuation = [a / width_cm for a in channels['atn']['views']]
    hist_atn = channels['atn']['histogram'] / width_cm
    k, j, i = crosshair
    shape = case['shape_kji']
    geometry = [(shape[2], shape[1], spacing[0], spacing[1], 'Axial', 'i', 'j', k, 'k', spacing[2], i, j, 1, 1),
                (shape[2], shape[0], spacing[0], spacing[2], 'Coronal', 'i', 'k', j, 'j', spacing[1], i, k, inplane_step, context_step),
                (shape[1], shape[0], spacing[1], spacing[2], 'Sagittal', 'j', 'k', i, 'i', spacing[0], j, k, inplane_step, context_step)]
    fig = plt.figure(figsize=(21, 17), facecolor='white')
    grid = fig.add_gridspec(3, 4, left=.12, right=.96, bottom=.10, top=.89,
                           width_ratios=[1.2, 1.3, 1, 1], hspace=.54, wspace=.35)
    axes = [[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(3)]
    values, frequencies = np.unique(hist_atn, return_counts=True)
    nonzero = values != 0
    axes[0][0].bar(values[nonzero], frequencies[nonzero], width=.001, color='#546e7a')
    axes[0][0].set(yscale='log', xlabel='Attenuation (1/cm)', ylabel='Sampled voxel count',
                   title=f"Attenuation samples\n{len(case['sample_planes_k'])} axial planes; ji stride {case['sample_stride_ji']}; zero omitted")
    labels, frequencies = np.unique(channels['act']['histogram'], return_counts=True)
    selected = sorted((n for n, label in enumerate(labels) if label != 0), key=lambda n: frequencies[n])[-12:]
    axes[1][0].barh(range(len(selected)), frequencies[selected], color='#546e7a')
    axes[1][0].set_yticks(range(len(selected)), [f"{int(labels[n])}: {by_id.get(int(labels[n]), {}).get('original_name', 'unnamed')}" for n in selected], fontsize=8)
    axes[1][0].set(xlabel='Sampled voxel count', title='Original signed organ IDs\nMost frequent IDs; zero omitted')
    categories = [c for c in catalog['categories'] if c['name'] != 'background']
    axes[2][0].barh(range(len(categories)), [counts[c['id']] for c in categories],
                   color=[colors[c['name']] for c in categories], edgecolor='#888888', linewidth=.4)
    axes[2][0].set_yticks(range(len(categories)), [c['name'].replace('_', ' ') for c in categories], fontsize=9)
    axes[2][0].invert_yaxis()
    axes[2][0].set(xlabel='Sampled voxel count', title='Proposed tissue groups\nSame sample grid; background omitted')
    max_count = max(counts[c['id']] for c in categories)
    axes[2][0].set_xlim(0, max(max_count * 1.25, 1))
    for n, category in enumerate(categories):
        axes[2][0].text(counts[category['id']] + max_count * .025, n, str(counts[category['id']]), va='center', fontsize=8)
    row_titles = ('Attenuation', 'Original IDs (illustrative hashed colors)', 'Proposed tissue groups')
    view_stats = []
    for col, (nx, ny, sx, sy, name, xlabel, ylabel, position, coordinate, sp, cross_x, cross_y, step_x, step_y) in enumerate(geometry, 1):
        for row in range(3):
            ax = axes[row][col]
            arr = attenuation[col - 1] if row == 0 else original[col - 1] if row == 1 else grouped[col - 1]
            if row == 1:
                arr = np.ma.masked_where(arr == 0, np.mod(arr.astype(np.int32) * 137, 257))
            extent = (-step_x / 2, (arr.shape[1] - .5) * step_x,
                      -step_y / 2, (arr.shape[0] - .5) * step_y)
            cmap = 'gray' if row == 0 else original_cmap if row == 1 else tissue_cmap
            vmax = max(float(a.max()) for a in attenuation) if row == 0 else 256 if row == 1 else 255
            im = ax.imshow(arr, origin='lower', extent=extent, aspect=sy / sx,
                           cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            ax.set(xlim=(-.5, nx - .5), ylim=(-.5, ny - .5),
                   xlabel=f'{xlabel} (zero-based voxel index)', ylabel=f'{ylabel} (zero-based voxel index)',
                   title=f'{name}: {coordinate}={position}; relative {position * sp:.1f} mm\n{row_titles[row]}')
            ax.secondary_xaxis('top', functions=(lambda v, s=sx: v * s, lambda v, s=sx: v / s)).set_xlabel('relative mm')
            ax.axvline(cross_x, color='#00eeee', lw=.5, alpha=.75)
            ax.axhline(cross_y, color='#00eeee', lw=.5, alpha=.75)
            if row == 0:
                fig.colorbar(im, ax=ax, shrink=.48, pad=.025, label='1/cm')
        tissue = grouped[col - 1]
        view_stats.append({'view': name, 'shape': list(tissue.shape),
                           'original_ids': len(np.unique(original[col - 1])),
                           'group_counts': {str(key): value for key, value in sorted(Counter(map(int, tissue.ravel())).items())},
                           'unknown_voxels': int(np.count_nonzero(tissue == unknown_code))})
    fig.suptitle(f"XCAT {case['case_id']}, frame {channels['frame']} | {case['organ_file']} | atlas policy {catalog['policy_version']}\n"
                 f"Crosshair: i={i}, j={j}, k={k}. Axial: full resolution. Whole-body context: k stride {context_step}, in-plane stride {inplane_step}.\n"
                 'Native array [k,j,i]; anatomical orientation and physical origin unverified. Coarse groups preserve original labels.',
                 fontsize=15, y=.98)
    fig.legend(handles=[Patch(facecolor=colors[c['name']], edgecolor='#777777', label=c['name'].replace('_', ' ')) for c in catalog['categories']],
               loc='lower center', bbox_to_anchor=(.53, .040), ncol=6, frameon=False, fontsize=11)
    fig.text(.53, .020, 'Unknown/review is magenta. Adipose has no assigned anatomical labels in this policy; fat material is still present.\n'
             'Counts describe the sampled grid, not the entire volume. Decimated context can omit thin vessels.', ha='center', fontsize=10)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return {'views': view_stats, 'histogram_sampled_voxels': int(group_sample.size),
            'histogram_background_voxels': counts[background_code],
            'histogram_unknown_voxels': counts[unknown_code],
            'histogram_group_counts': {c['name']: counts[c['id']] for c in catalog['categories']},
            'histogram_missing_dictionary_ids': [int(value) for value in np.unique(channels['act']['histogram']) if int(value) not in by_id],
            'colors_by_group': colors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT / 'docs/integration/2026-09-16/xcat-results.json')
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--case', default='260602')
    parser.add_argument('--frame', type=int, default=1)
    parser.add_argument('--crosshair-kji', type=int, nargs=3)
    parser.add_argument('--context-step', type=int, default=8)
    parser.add_argument('--inplane-step', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True, help='New PNG path; JSON provenance is saved beside it')
    args = parser.parse_args()
    if args.context_step < 1 or args.inplane_step < 1 or args.frame < 1:
        parser.error('Frame and sampling strides must be positive integers')
    if args.out.suffix.lower() != '.png' or args.out.exists() or args.out.with_suffix('.json').exists():
        parser.error('--out must be a new PNG path with no existing JSON companion')
    audit = json.loads(args.audit.read_text())
    case = next((c for c in audit['cases'] if c['case_id'] == args.case), None)
    if case is None:
        parser.error('Case is absent from the audit')
    catalog = json.loads(args.catalog.read_text())
    # Validate the category map before reading source volumes.
    coarse_labels(np.array([0], dtype=np.int32), catalog)
    shape = case['shape_kji']
    crosshair = args.crosshair_kji or [int(shape[0] * .68), shape[1] // 2, shape[2] // 2]
    if any(v < 0 or v >= n for v, n in zip(crosshair, shape)):
        parser.error('Crosshair lies outside source dimensions')
    for path in (case['par_path'], case['log_path']):
        expected = audit.get('source_metadata_sha256', {}).get(path)
        if expected and sha256(path) != expected:
            parser.error(f'Source metadata changed since audit: {path}')
    channels, sources = read_samples(case, args.frame, crosshair, args.context_step, args.inplane_step)
    channels['frame'] = args.frame
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = draw(case, catalog, channels, crosshair, args.context_step, args.inplane_step, args.out)
    report = {'schema_version': 'fakect.tissue-preview.v1', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'case_id': args.case, 'frame': args.frame, 'crosshair_kji': crosshair,
              'shape_kji': shape, 'spacing_ijk_mm': case['spacing_ijk_mm'],
              'coordinate_status': 'Native array indices and relative mm; anatomical orientation/origin unverified',
              'context_step_k': args.context_step, 'context_inplane_step': args.inplane_step,
              'histogram_planes_k': case['sample_planes_k'], 'histogram_stride_ji': case['sample_stride_ji'],
              'attenuation_conversion': 'atn per-pixel divided by pixel_width_cm; not HU',
              'catalog_sha256': sha256(args.catalog), 'audit_sha256': sha256(args.audit),
              'script_sha256': sha256(__file__), 'figure_sha256': sha256(args.out),
              'sources': sources, 'source_integrity_scope': 'Source stat checked before/after; sampled values hashed, full volumes not hashed',
              **result}
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'figure': str(args.out), 'source_bytes_read': sum(s['requested_bytes'] for s in sources),
                      'histogram_sampled_voxels': result['histogram_sampled_voxels'],
                      'histogram_group_counts': result['histogram_group_counts']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
