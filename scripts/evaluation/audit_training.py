#!/usr/bin/env python3
"""Read-only FakeCT audit: headers/CSV/logs/Keras metadata and tiny morphology arrays.

No TensorFlow import, volume pixel reads, training, Slurm submission, or source edits.
Only --out-json is written. Existing scientific Python dependencies are required.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
import pandas as pd
import nrrd
import scipy
from scipy.ndimage import binary_dilation
import skimage
from skimage import measure


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--dataset-root', default='/home/aghorban/repo/FakeCT/data/dataset')
    p.add_argument('--slurm-dir', default='/home/aghorban/slurm')
    p.add_argument('--morphology-ref', default='c6a5c2c9c47ebcbee116c8ade18269eef4ac09ff')
    p.add_argument('--out-json', default='outputs/evaluation/training-results.json')
    args = p.parse_args()
    dataset = Path(args.dataset_root)
    base = dataset / 'paired_datasets'
    manifest = base / 'pairs.csv'
    df = pd.read_csv(manifest)
    result = {'scope': 'Read-only files/metadata; synthetic arrays only; no TensorFlow, training, or job submission.',
              'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'pynrrd': nrrd.__version__,
                           'scipy': scipy.__version__, 'skimage': skimage.__version__,
                           'nrrd_read_signature': str(inspect.signature(nrrd.read))},
              'manifest': {'path': str(manifest), 'sha256': sha256(manifest), 'rows': len(df),
                           'cases': df.patient_id.nunique(), 'images': df.image.nunique(),
                           'masks': df['mask'].nunique(),
                           'duplicate_image_slice': int(df.duplicated(['image', 'x_index']).sum())}}
    paths = set(df.image) | set(df['mask'])
    missing = sorted(x for x in paths if not Path(x).exists())
    result['manifest']['missing_paths'] = missing
    if missing:
        raise FileNotFoundError(f'Manifest paths missing ({len(missing)}); no hidden path rewriting performed.')
    headers = {path: nrrd.read_header(path) for path in sorted(paths)}
    image_shapes = collections.Counter(tuple(map(int, headers[x]['sizes'][:2])) for x in df.image.unique())
    result['manifest']['image_shape_case_counts'] = {str(k): v for k, v in image_shapes.items()}
    type_bytes = {'short': 2, 'int16': 2, 'unsigned short': 2, 'uint16': 2,
                  'int': 4, 'int32': 4, 'uint': 4, 'unsigned int': 4, 'uint32': 4,
                  'unsigned char': 1, 'uchar': 1, 'uint8': 1, 'float': 4, 'double': 8}
    total_bytes = sum(math.prod(map(int, h['sizes'])) * type_bytes[h['type']] for h in headers.values())
    result['memory_estimates'] = {'all_image_mask_arrays_GiB': total_bytes / 2**30,
                                'note': 'Header-derived uncompressed estimate; not measured RSS.',
                                'expanded_shuffle': []}
    segment_headers = collections.Counter()
    for x in df['mask'].unique():
        h = headers[x]
        fields = tuple(sorted((k, str(v)) for k, v in h.items() if k.startswith('Segment') and k.endswith(('_Name', '_LabelValue', '_Layer'))))
        segment_headers[fields] += 1
    result['manifest']['segment_header_groups'] = [{'count': v, 'fields': dict(k)} for k, v in segment_headers.items()]
    ids = df.patient_id.dropna().unique().tolist()
    np.random.default_rng(42).shuffle(ids)
    nt = max(1, int(len(ids) * .8))
    train_ids, test_ids = set(ids[:nt]), set(ids[nt:])
    combined = df[df.patient_id.isin(train_ids)].sample(frac=1.0, random_state=42)
    nv = max(1, int(len(combined) * .1))
    splits = {'train': combined.iloc[nv:], 'validation': combined.iloc[:nv], 'test': df[df.patient_id.isin(test_ids)]}
    first = splits['train'].iloc[0]
    shape = tuple(map(int, headers[first.image]['sizes'][:2]))
    split_out = {'seed': 42, 'first_training_case': first.patient_id, 'selected_shape': shape, 'splits': {}}
    for name, part in splits.items():
        kept = part[part.image.map(lambda x: tuple(map(int, headers[x]['sizes'][:2])) == shape)]
        split_out['splits'][name] = {'nominal_rows': len(part), 'nominal_cases': part.patient_id.nunique(),
                                    'yielded_rows': len(kept), 'yielded_cases': kept.patient_id.nunique(),
                                    'retained_ids': sorted(kept.patient_id.unique())}
    split_out['train_validation_case_overlap'] = sorted(set(splits['train'].patient_id) & set(splits['validation'].patient_id))
    split_out['test_train_or_validation_case_overlap'] = sorted((set(splits['train'].patient_id) | set(splits['validation'].patient_id)) & set(splits['test'].patient_id))
    split_out['limitation'] = 'Reproduces path-derived case IDs; true subject linkage is unverified.'
    result['historical_split_reproduction'] = split_out
    for ctx in (0, 3, 15, 30, 60):
        ch = 2*ctx+1
        result['memory_estimates']['expanded_shuffle'].append({'shape': shape, 'context': ctx, 'channels': ch,
            'shuffle256_GiB': 256*math.prod(shape)*(ch+1)*4/2**30,
            'batch8_inputs_GiB': 8*math.prod(shape)*ch*4/2**30})
    result['metrics'] = []
    for path in sorted(base.glob('**/*metrics*.csv')):
        with path.open() as f:
            rows = [{k: float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        result['metrics'].append({'path': str(path), 'sha256': sha256(path), 'rows': rows})
    result['models'] = []
    for path in sorted(base.glob('**/*.keras')):
        item = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                item['metadata'] = json.loads(z.read('metadata.json'))
                config = json.loads(z.read('config.json'))
                item['input_layers'] = [n['config'] for n in config.get('config', {}).get('layers', []) if n.get('class_name') == 'InputLayer']
        result['models'].append(item)
    result['launcher_snapshots'] = []
    for path in sorted(Path(args.slurm_dir).glob('fakenoise*.sh')):
        result['launcher_snapshots'].append({'path': str(path), 'sha256': sha256(path), 'contents': path.read_text()})
    logs = []
    for path in sorted((Path(args.slurm_dir)/'logs').glob('fakenoise*.out')):
        if path.stat().st_size > 25_000_000:
            logs.append({'path': str(path), 'skipped': 'above bounded 25MB limit'})
            continue
        out = path.read_text(errors='replace')
        errp = path.with_suffix('.err')
        err = errp.read_text(errors='replace') if errp.exists() and errp.stat().st_size <= 25_000_000 else ''
        completed = re.findall(r'\[INFO\] Model saved to ([^\r\n]+)', out)
        signals = [line[:500] for line in err.splitlines() if re.search(r'GPU will not be used|CUDA_ERROR|ValueError:|ImportError:|ModuleNotFoundError:|CANCELLED|DUE TO TIME LIMIT', line)]
        logs.append({'stdout_path': str(path), 'stderr_path': str(errp), 'saved_models': completed,
                     'epoch10_present': 'Epoch 10/10' in out, 'diagnostic_lines': signals})
    result['logs'] = logs
    result['gpu_evidence_caveat'] = ('Historical completion stderr records CUDA failures. Launcher snapshots describe current files, '
        'and logs do not persist exact submitted script/version; this does not prove current TF2.17 launcher failed.')
    s = subprocess.check_output(['git', '-C', args.repo, 'show', f'{args.morphology_ref}:src/fakect.py'], text=True)
    revision = subprocess.check_output(['git', '-C', args.repo, 'rev-parse', args.morphology_ref], text=True).strip()
    needed = {'_gaussian_shape', '_roi_mask_vox', '_bitwise_dilate6', '_bitwise_erode6', 'proximity_bridge', 'apply_roi_scale'}
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name in needed]
    if {n.name for n in nodes} != needed:
        raise RuntimeError('Required morphology functions missing; refusing partial audit')
    ns = {'np': np, 'binary_dilation': binary_dilation, 'measure': measure}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'audited-phantom-functions', 'exec'), ns)
    a = np.zeros((49,49,49), dtype=np.uint8)
    a[10:39,21:28,21:28] = 1
    cases = []
    for radius in (8,12,16):
        roi = ns['_roi_mask_vox'](a.shape, (24,24,24), radius)
        for scale in (.5,.8,1.,1.2,1.5,2.):
            b = ns['apply_roi_scale'](a, (24,24,24), radius, scale, bridge_dist=0)
            cases.append({'radius': radius, 'scale': scale, 'bridge': 0, 'before': int(a.sum()), 'after': int(b.sum()),
                          'changed_voxels': int(np.sum(a != b)), 'changed_outside_initial_roi': int(np.sum((a != b) & ~roi))})
    c = np.zeros((33,33,33), dtype=np.uint8)
    c[10:23,14:19,14:19] = 1
    c[2,2,2] = 1
    c[2,2,4] = 1
    roi = ns['_roi_mask_vox'](c.shape, (16,16,16), 8)
    bridge_cases = []
    for bridge in (0,2):
        d = ns['apply_roi_scale'](c, (16,16,16), 8, 1.5, bridge_dist=bridge)
        bridge_cases.append({'bridge': bridge, 'remote_gap_before': int(c[2,2,3]), 'remote_gap_after': int(d[2,2,3]),
                             'changed_outside_initial_roi': int(np.sum((c != d) & ~roi))})
    result['morphology'] = {'git_ref': args.morphology_ref, 'git_sha': revision,
        'source_sha256': hashlib.sha256(s.encode()).hexdigest(), 'method': 'AST extraction of unmodified functions; no GUI imports',
        'fixture': 'zeros((49,49,49)); mask[10:39,21:28,21:28]=1; center(24,24,24)',
        'cases': cases, 'bridge_fixture': 'zeros((33,33,33)); mask[10:23,14:19,14:19]=1; mask[2,2,2]=mask[2,2,4]=1; center(16,16,16),radius8,scale1.5',
        'bridge_cases': bridge_cases}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out_json': args.out_json, 'manifest_rows': len(df), 'cases': df.patient_id.nunique(),
                      'metrics_files': len(result['metrics']), 'model_files': len(result['models']),
                      'morphology_cases': len(cases), 'historical_completed_logs': sum(bool(x.get('saved_models')) for x in logs)}))


if __name__ == '__main__':
    main()
