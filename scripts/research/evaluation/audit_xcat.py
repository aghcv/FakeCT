#!/usr/bin/env python3
"""Read-only XCAT metadata/file-size audit and bounded, deterministic float32 samples.

No full-volume arrays are loaded. Every .bin is stat'ed; selected frame pairs and
averages are memory-mapped and sampled on 17 axial planes with in-plane stride 8.
The provisional C-order [k,j,i] interpretation is not anatomical orientation proof.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import numpy as np

DEFAULT_CASES = ['260602'] + [str(v) for v in range(260611, 260622)]
GEOM_KEYS = ['x_array_size', 'y_array_size', 'startslice', 'endslice', 'pixel_width', 'slice_width']


def parse_par(path):
    return {k: v.strip() for k, v in re.findall(r'^\s*(\w+)\s*=\s*([^#\n]+)', path.read_text(), re.M)}


def parse_log(text):
    patterns = {'x_array_size': r'x_array_size\s*=\s*(\d+)', 'y_array_size': r'y_array_size\s*=\s*(\d+)',
                'pixel_width': r'pixel width\s*=\s*([\d.]+)', 'slice_width': r'slice width\s*=\s*([\d.]+)',
                'startslice': r'starting slice number\s*=\s*(\d+)', 'endslice': r'ending slice number\s*=\s*(\d+)'}
    result = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            result[key] = float(m[1])
    h = re.search(r'Total Body\s+Height\s+([\d.]+)\s+mm', text)
    if h:
        result['body_height_mm'] = float(h[1])
    return result


def load_organ_ids(path):
    records = re.findall(r'^\s*(.*?)\s*=\s*(-?\d+)\s*$', path.read_text(), re.M)
    return {int(value): name for name, value in records}


def samples(volume, planes, stride=8):
    return np.concatenate([np.array(volume[int(k), ::stride, ::stride]).ravel() for k in planes])


def describe(sample, known_ids=None):
    finite = np.isfinite(sample)
    values = sample[finite]
    u, counts = np.unique(values, return_counts=True)
    top = np.argsort(counts)[-15:][::-1]
    result = {'sampled_voxels': int(sample.size), 'finite_fraction': float(finite.mean()),
              'nonzero_fraction': float(np.count_nonzero(values) / max(values.size, 1)),
              'unique_values': int(u.size), 'min': float(values.min()), 'max': float(values.max()),
              'fraction_not_exact_integer': float(np.mean(values != np.rint(values))),
              'fraction_not_integer_at_1e-3': float(np.mean(np.abs(values-np.rint(values)) > 1e-3)),
              'top_values': [{'value': float(u[i]), 'count': int(counts[i])} for i in top]}
    if known_ids is not None:
        result['negative_voxels'] = int(np.count_nonzero(values < 0))
        result['unknown_ids'] = [float(v) for v in u if v != 0 and (v != round(float(v)) or int(v) not in known_ids)]
    else:
        result['nonzero_levels'] = [float(v) for v in u if v != 0] if len(u) < 100 else None
    return result


def audit_case(base, case, known_ids):
    par = parse_par(base / (case + '.par'))
    directory = base / case
    log_path = directory / (case + '_log')
    log = parse_log(log_path.read_text())
    shape = (int(par['endslice']) - int(par['startslice']) + 1, int(par['y_array_size']), int(par['x_array_size']))
    voxel_bytes = int(np.prod(shape)) * 4
    files = [p for p in directory.iterdir() if p.is_file()]
    binary_files = [p for p in files if p.suffix == '.bin']
    mismatch = [{'path': str(p), 'bytes': p.stat().st_size} for p in binary_files if p.stat().st_size != voxel_bytes]
    channels = defaultdict(list)
    for path in binary_files:
        m = re.fullmatch(re.escape(case) + r'_(act|atn)_(\d+|av)\.bin', path.name)
        channels[m[1] if m else 'unrecognized'].append(m[2] if m else path.name)
    nframes = int(par['out_frames'])
    selected_frames = sorted({1, (nframes + 1) // 2, nframes})
    planes = np.linspace(0, shape[0] - 1, 17, dtype=int)
    summaries = {}
    mapping = defaultdict(set)
    for frame in [str(x) for x in selected_frames] + ['av']:
        pair_samples = {}
        for channel in ['act', 'atn']:
            path = directory / f'{case}_{channel}_{frame}.bin'
            if not path.exists() or path.stat().st_size != voxel_bytes:
                continue
            volume = np.memmap(path, dtype='<f4', mode='r', shape=shape, order='C')
            sample = samples(volume, planes)
            pair_samples[channel] = sample
            stats = describe(sample, known_ids if channel == 'act' and frame != 'av' else None)
            if channel == 'atn':
                stats['max_normalized_cm_inverse'] = stats['max'] / float(par['pixel_width'])
            if frame == '1' and channel == 'act':
                nonzero = sample[sample != 0]
                reverse = nonzero.view('>f4')
                stats['big_endian_alternative_fraction_tiny_nonzero'] = float(np.mean((np.abs(reverse) < 1e-20) & (reverse != 0)))
            summaries[f'{channel}_{frame}'] = stats
            del volume
        if frame != 'av' and set(pair_samples) == {'act', 'atn'}:
            for scalar in np.unique(pair_samples['atn']):
                if scalar:
                    ids = np.unique(pair_samples['act'][pair_samples['atn'] == scalar])
                    mapping[float(scalar)].update(int(v) for v in ids)
    raw = sorted(directory.glob('*.raw'))
    raw_prefix = raw[0].open('rb').read(200).decode('ascii', errors='replace') if raw else ''
    result = {'case_id': case, 'directory': str(directory), 'par_path': str(base/(case+'.par')), 'log_path': str(log_path),
              'organ_file': par.get('organ_file'), 'gender_parameter': par.get('gender'), 'shape_kji': list(shape),
              'spacing_ijk_mm': [float(par['pixel_width'])*10]*2+[float(par['slice_width'])*10],
              'field_of_view_ijk_mm': [shape[2]*float(par['pixel_width'])*10, shape[1]*float(par['pixel_width'])*10, shape[0]*float(par['slice_width'])*10],
              'expected_binary_bytes_float32': voxel_bytes, 'log_metadata': log,
              'par_log_mismatches': {k: {'par': float(par[k]), 'log': log.get(k)} for k in GEOM_KEYS if float(par[k]) != log.get(k)},
              'binary_count': len(binary_files), 'raw_count': len(raw), 'file_count': len(files),
              'logical_bytes': sum(p.stat().st_size for p in files), 'allocated_bytes': sum(p.stat().st_blocks * 512 for p in files),
              'binary_size_mismatches': mismatch, 'channels': {k: sorted(v, key=lambda x: (x == 'av', int(x) if x.isdigit() else 0)) for k,v in channels.items()},
              'missing_expected_channels': [f'{c}_{f}' for c in ['act','atn'] for f in [str(i) for i in range(1,nframes+1)]+['av'] if f not in channels[c]],
              'raw_first_file_prefix': raw_prefix, 'color_code': par.get('color_code'), 'activ_output_format': par.get('activ_output_format'),
              'energy_kev': float(par['energy']), 'sample_planes_k': planes.tolist(), 'sample_stride_ji': [8,8],
              'samples': summaries,
              'sample_value_bytes_total': sum(v['sampled_voxels'] * 4 for v in summaries.values()),
              'sampling_axial_plane_span_bytes_upper_bound': len(summaries) * len(planes) * shape[1] * shape[2] * 4,
              'sampled_carotid_ids_present': sorted({i for ids in mapping.values() for i in ids if i in [1185,1186]}),
              'scalar_to_organ_ids_sample': [{'attenuation_per_pixel': val, 'attenuation_cm_inverse': val/float(par['pixel_width']), 'organ_id_count': len(ids), 'organ_ids': sorted(ids)} for val,ids in sorted(mapping.items())]}
    return result


def plot_case(base, case, result, known_ids, path):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shape = tuple(result['shape_kji']); spacing = result['spacing_ijk_mm']
    k, j, i = int(shape[0] * .68), shape[1]//2, shape[2]//2
    slices = []; hist_samples = []
    for channel in ['atn','act']:
        volume = np.memmap(base/case/f'{case}_{channel}_1.bin', dtype='<f4', mode='r', shape=shape)
        hist_samples.append(samples(volume, result['sample_planes_k']))
        arrays = [np.array(volume[k,:,:]), np.array(volume[::2,j,::2]), np.array(volume[::2,::2,i])]
        if channel == 'atn':
            arrays = [a / (spacing[0]/10) for a in arrays]
        slices.append(arrays)
        del volume
    fig, axes = plt.subplots(2,4,figsize=(20,10),constrained_layout=True)
    values, counts = np.unique(hist_samples[0],return_counts=True)
    nz = values != 0
    axes[0,0].bar(values[nz]/(spacing[0]/10), counts[nz], width=.001)
    axes[0,0].set(xlabel='Attenuation (1/cm; per-pixel / pixel width cm)', ylabel='Sampled voxel count (log)', title='17 axial planes, stride 8; zero omitted', yscale='log')
    vals, cnt = np.unique(hist_samples[1][hist_samples[1] != 0],return_counts=True)
    selected = np.argsort(cnt)[-13:]
    axes[1,0].barh(np.arange(len(selected)),cnt[selected])
    axes[1,0].set_yticks(np.arange(len(selected)),[f'{int(vals[n])}: {known_ids.get(int(vals[n]),"unknown")}' for n in selected],fontsize=7)
    axes[1,0].set(xlabel='Sampled voxel count',title='Most frequent signed organ IDs; zero omitted')
    geometries = [(shape[2],shape[1],spacing[0],spacing[1],f'Axial: k={k}; relative z={k*spacing[2]:.1f} mm','i','j'),
                  (shape[2],shape[0],spacing[0],spacing[2],f'Coronal: j={j}; relative y={j*spacing[1]:.1f} mm','i','k'),
                  (shape[1],shape[0],spacing[1],spacing[2],f'Sagittal: i={i}; relative x={i*spacing[0]:.1f} mm','j','k')]
    for row in [0,1]:
        for col,(nx,ny,sx,sy,title,xlabel,ylabel) in enumerate(geometries,1):
            ax=axes[row,col]; arr=slices[row][col-1]
            if row == 1:
                arr=np.where(arr == 0,np.nan,np.mod(arr*137,257))
            step = 1 if col == 1 else 2
            extent = (-step/2, (arr.shape[1]-.5)*step, -step/2, (arr.shape[0]-.5)*step)
            im=ax.imshow(arr,origin='lower',extent=extent,aspect=sy/sx,cmap='gray' if row==0 else 'turbo',interpolation='nearest',vmin=0)
            ax.set(xlabel=f'{xlabel} (zero-based voxel index)',ylabel=f'{ylabel} (zero-based voxel index)',title=title)
            ax.secondary_xaxis('top',functions=(lambda v,s=sx:v*s,lambda v,s=sx:v/s)).set_xlabel('relative mm')
            ax.axvline(i if col != 3 else j,color='cyan',lw=.6,alpha=.6)
            ax.axhline(j if col == 1 else k,color='cyan',lw=.6,alpha=.6)
            if row == 0:
                fig.colorbar(im,ax=ax,shrink=.6,label='1/cm')
    fig.suptitle(f'XCAT {case}, frame 1 | {result["organ_file"]}\nProvisional array [k,j,i]; anatomical orientation/origin unverified. Top: normalized attenuation. Bottom: signed organ IDs (hashed colors).',fontsize=12)
    fig.savefig(path,dpi=150)
    plt.close(fig)
    return {'path': str(path), 'case_id': case, 'frame':1, 'crosshair_kji':[k,j,i], 'anatomical_orientation':'unverified',
            'max_source_file_span_bytes': 2 * result['expected_binary_bytes_float32'],
            'io_note': 'Orthogonal strided views may touch many mapped pages, up to both source files; only sampled 2D arrays are materialized.'}


def finalize_metadata(report):
    for c in report['cases']:
        c['spacing_ijk_mm'] = [round(v, 9) for v in c['spacing_ijk_mm']]
        c['field_of_view_ijk_mm'] = [round(v, 9) for v in c['field_of_view_ijk_mm']]
        c['sample_value_bytes_total'] = sum(v['sampled_voxels'] * 4 for v in c['samples'].values())
        c['sampling_axial_plane_span_bytes_upper_bound'] = len(c['samples']) * len(c['sample_planes_k']) * c['shape_kji'][1] * c['shape_kji'][2] * 4
        c['sampled_carotid_ids_present'] = sorted({i for rec in c['scalar_to_organ_ids_sample'] for i in rec['organ_ids'] if i in [1185,1186]})
    for key in ['sample_value_bytes_total','sampling_axial_plane_span_bytes_upper_bound']:
        report['totals'][key] = sum(c[key] for c in report['cases'])
    for c in report['cases']:
        c['sampled_frame_ids_without_names'] = sorted({int(v) for k,rec in c['samples'].items() if k.startswith('act_') and k != 'act_av' for v in rec['unknown_ids']})
    report['sampled_frame_ids_without_names'] = sorted({v for c in report['cases'] for v in c['sampled_frame_ids_without_names']})
    frame_labels = [v for c in report['cases'] for k,v in c['samples'].items() if k.startswith('act_') and k != 'act_av']
    report['validation'] = {
        'all_binary_sizes_match': all(not c['binary_size_mismatches'] for c in report['cases']),
        'all_par_log_geometry_agrees': all(not c['par_log_mismatches'] for c in report['cases']),
        'all_expected_frames_present': all(not c['missing_expected_channels'] for c in report['cases']),
        'all_sampled_values_finite': all(v['finite_fraction'] == 1 for c in report['cases'] for v in c['samples'].values()),
        'all_sampled_frame_labels_integer': all(v['fraction_not_exact_integer'] == 0 for v in frame_labels),
        'all_sampled_frame_labels_in_organ_ids': all(not v['unknown_ids'] for v in frame_labels),
        'scope': 'Size/completeness checks cover all named volumes; value checks cover only the declared samples.'}
    report['sampling_io_note'] = 'sample_value_bytes_total counts float32 values materialized, not OS I/O. sampling_axial_plane_span_bytes_upper_bound is the sum of full axial plane spans touched by sampling. Actual physical I/O depends on sparse extents, page size, cache and read-ahead. Preview accesses additional cross-sectional planes. No full-volume copy is created.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('/home/aghorban/slurm/xcat'))
    parser.add_argument('--cases',nargs='+',default=DEFAULT_CASES)
    parser.add_argument('--output-prefix',type=Path,default=Path('outputs/evaluation/xcat'))
    parser.add_argument('--plot-case',default='260602')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    known_ids=load_organ_ids(args.base/'organ_ids.txt')
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'base':str(args.base),
            'method':'Read-only stat of each direct case file. NumPy <f4 C-order memmaps; 17 evenly spaced axial planes, ji stride 8, frames 1/middle/last and averages. Estimates are samples, not exhaustive voxel statistics.',
            'sampling_io_note':'sample_value_bytes_total counts float32 values materialized, not OS I/O. sampling_axial_plane_span_bytes_upper_bound is the sum of full axial plane spans touched by sampling. Actual physical I/O depends on sparse extents, page size, cache and read-ahead. Preview accesses additional cross-sectional planes. No full-volume copy is created.',
            'axis_order_status':'Provisional [k,j,i] in C order. File sizes and anatomical-looking slices support this; anatomical orientation and origin require separate landmark validation.',
            'organ_ids_path':str(args.base/'organ_ids.txt'),'known_organ_ids':len(known_ids),
            'negative_known_organ_ids':sum(k<0 for k in known_ids),'cases':[]}
    for case in args.cases:
        result=audit_case(args.base,case,known_ids)
        report['cases'].append(result)
        print(f'{case}: {result["shape_kji"]}, {result["spacing_ijk_mm"]} mm, {result["binary_count"]} bins, act1 {result["samples"]["act_1"]["unique_values"]} sampled IDs',flush=True)
    report['totals']={key:sum(c[key] for c in report['cases']) for key in ['file_count','binary_count','raw_count','logical_bytes','allocated_bytes','sample_value_bytes_total','sampling_axial_plane_span_bytes_upper_bound']}
    report['totals']['instantaneous_frame_pairs']=sum(len(c['channels']['act'])-('av' in c['channels']['act']) for c in report['cases'])
    report['totals']['baseline_anatomies']=len(report['cases'])
    finalize_metadata(report)
    Path(str(args.output_prefix)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    if not args.no_plots:
        record=next((c for c in report['cases'] if c['case_id']==args.plot_case),None)
        if record:
            report['preview']=plot_case(args.base,args.plot_case,record,known_ids,Path(str(args.output_prefix)+'-preview.png'))
    Path(str(args.output_prefix)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# XCAT source audit','',report['method'],'',
           '| Case | Organ model | Shape (k,j,i) | Spacing (mm) | Float32 volume GiB | Sample IDs frame 1 | Negative samples |',
           '|---|---|---|---|---:|---:|---:|']
    for c in report['cases']:
        a=c['samples']['act_1']
        lines.append(f'| {c["case_id"]} | {c["organ_file"]} | {c["shape_kji"]} | {c["spacing_ijk_mm"]} | {c["expected_binary_bytes_float32"]/2**30:.3f} | {a["unique_values"]} | {a["negative_voxels"]} |')
    t=report['totals']
    lines += ['',f'{t["baseline_anatomies"]} source anatomies; {t["instantaneous_frame_pairs"]} paired motion frames; {t["binary_count"]} binary volumes including averages; {t["raw_count"]} ASCII mesh files.',
              f'Total apparent file size {t["logical_bytes"]/2**40:.3f} TiB; allocated size {t["allocated_bytes"]/2**30:.3f} GiB (sparse volumes).',
              f'Sampling materialized {t["sample_value_bytes_total"]/2**20:.2f} MiB total float32 values sequentially; sampled axial-plane spans sum to {t["sampling_axial_plane_span_bytes_upper_bound"]/2**30:.3f} GiB. Actual disk I/O depends on sparse allocation, paging, cache and read-ahead. Preview accesses additional planes.',
              '', '## Input-contract implications','',
              '- Every binary file size agrees with x*y*(endslice-startslice+1)*4 bytes. Parameter/log geometry agrees for the audited cases. Float32 little endian gives meaningful sampled IDs; anatomical orientation/origin remain unverified.',
              '- `.raw` files begin with organ names and ASCII triangle vertices. They are meshes, not voxel arrays.',
              f'- Dictionary coverage check: {len(report["sampled_frame_ids_without_names"])} sampled signed IDs are absent from the supplied `organ_ids.txt`: {report["sampled_frame_ids_without_names"]}. Preserve those IDs and resolve their names/material grouping before promoting a cohort; do not silently map them to zero.',
              '- With `color_code=1`, per-frame `act` volumes provide signed organ IDs. Preserve negative IDs, background 0, and an ID-to-name mapping. Organ names are not unique.',
              '- `act_av` blends IDs across motion and has fractional values; it must not be accepted as a categorical segmentation.',
              '- `atn` samples agree with the output log\'s attenuation-per-pixel table. Normalize by `pixel_width` in cm to obtain inverse-cm attenuation before comparing across resolutions. These values are not HU.',
              '- Multiple organ IDs share one attenuation value. Use paired act IDs for anatomical selection and maintain a separate explicit tissue grouping. Intensity bins alone cannot isolate many organs.',
              '- Split train/validation/test by source anatomy before emitting slices, frames, or geometry variants; 612 motion frames are not 612 independent subjects.',
              '- A float32 volume is 1.60–4.41 GiB. Full-volume copies, displacement fields, distance maps, and TensorFlow training need separate memory budgets; use bounded ROI/chunk processing.',
              '- Sampling validates representative data only. Full-volume finite/integer checks, signed-ID coverage, orientation landmarks, correspondence, and unchanged-exterior checks remain import acceptance gates.',
              '- Preview coordinates are zero-based voxel indices and relative millimetres; no unverified anatomical direction labels are assigned.', '']
    if 'preview' in report:
        lines += [f'![Sample preview]({Path(report["preview"]["path"]).name})','']
    lines += ['## Validation scope', '', 'All reported input sizes and frame-presence checks cover the full case inventory. Finite/integer/dictionary checks cover only the declared deterministic samples. No full-volume checksum or anatomy-orientation validation was performed.', '', '```json', json.dumps(report['validation'], indent=2), '```', '']
    Path(str(args.output_prefix)+'.md').write_text('\n'.join(lines))
    print(json.dumps(report['totals'],indent=2),flush=True)


if __name__ == '__main__':
    main()
