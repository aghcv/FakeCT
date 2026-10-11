"""Immutable engineering pilot contracts for overview/detail CoA learning.

Annotations are separate from image data. Explicit pilot acceptance does not
change a draft annotation into a clinically reviewed diagnosis. Test families
are assigned but never loaded, predicted or used by this pilot.
"""
from __future__ import annotations

import configparser
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import time

import numpy as np

from fakect_evaluation_design import build_evaluation_design

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.multiscale-pilot/1'
SCORE = 'coa_local_target_volume_loss_fraction'
CODE_FILES = ('src/fakect_multiscale_experiment.py', 'src/fakect_multiscale_learning.py',
              'src/fakect_multiscale_volume.py', 'src/fakect_multiscale_network.py',
              'src/fakect_context_volume.py', 'src/fakect_evaluation_design.py',
              'scripts/research/train_multiscale_coa.py', 'scripts/research/slurm/fakect_multiscale_gpu.sh')
FIELDS = {
    'experiment': {'schema_version', 'output_directory'},
    'data': {'manifest', 'annotations', 'annotation_use', 'allow_provisional_anatomy'},
    'split': {'case_counts', 'seed'},
    'input.overview': {'shape_kji', 'cache_directory', 'max_chunk_voxels'},
    'input.detail': {'shape_kji', 'spacing_ijk_mm', 'proposals'},
    'model': {'base_filters', 'learning_rate', 'seed', 'clip_min', 'clip_max'},
    'pilot': {'locator_epochs', 'classifier_epochs', 'minimum_changed_coverage',
              'gpu_memory_limit_mb', 'require_gpu'},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
    Path(str(path) + '.sha256').write_text(sha256(path) + '\n')


def checked_json(path, expected=None):
    path = Path(path)
    digest = sha256(path)
    if digest != Path(str(path) + '.sha256').read_text().strip():
        raise ValueError('Checksum mismatch: ' + str(path))
    if expected is not None and expected != digest:
        raise ValueError('Pinned manifest changed: ' + str(path))
    if (path.parent / 'INCOMPLETE').exists():
        raise ValueError('Incomplete artifact: ' + str(path))
    return json.loads(path.read_text())


def load_config(path):
    path = Path(path).resolve()
    p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                  empty_lines_in_values=False)
    p.read_string(path.read_text())
    if p.defaults() or set(p.sections()) != set(FIELDS):
        raise ValueError('Expected sections: ' + ', '.join(FIELDS))
    c = {s: dict(p[s]) for s in FIELDS}
    for section, fields in FIELDS.items():
        if set(c[section]) != fields:
            raise ValueError(f'[{section}] requires exactly {sorted(fields)}')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported multiscale pilot schema')
    if c['data']['annotation_use'] != 'accepted_engineering_proxy':
        raise ValueError('Explicit annotation_use=accepted_engineering_proxy is required')
    for section, key in [('experiment', 'output_directory'), ('data', 'manifest'),
                         ('data', 'annotations'), ('input.overview', 'cache_directory')]:
        v = Path(c[section][key]).expanduser()
        c[section][key] = str((v if v.is_absolute() else ROOT / v).resolve())
    for section, key in [('data', 'allow_provisional_anatomy'), ('pilot', 'require_gpu')]:
        c[section][key] = p.getboolean(section, key)
    for section, key in [('split', 'case_counts'), ('input.overview', 'shape_kji'),
                         ('input.detail', 'shape_kji')]:
        c[section][key] = [int(v.strip()) for v in c[section][key].split(',')]
        if len(c[section][key]) != 3 or min(c[section][key]) < 1:
            raise ValueError(f'{section}.{key} needs three positive integers')
    for section, key, lo, hi in [('split', 'seed', 0, 2**31-1), ('model', 'seed', 0, 2**31-1),
                               ('model', 'base_filters', 2, 16), ('input.detail', 'proposals', 2, 2),
                               ('input.overview', 'max_chunk_voxels', 1, 4_000_000),
                               ('pilot', 'locator_epochs', 1, 500), ('pilot', 'classifier_epochs', 1, 100),
                               ('pilot', 'gpu_memory_limit_mb', 512, 65536)]:
        v = int(c[section][key])
        if not lo <= v <= hi:
            raise ValueError(f'{section}.{key} outside [{lo}, {hi}]')
        c[section][key] = v
    for section, key in [('model', 'learning_rate'), ('model', 'clip_min'), ('model', 'clip_max'),
                         ('pilot', 'minimum_changed_coverage')]:
        c[section][key] = float(c[section][key])
        if not math.isfinite(c[section][key]):
            raise ValueError(f'Nonfinite {section}.{key}')
    c['input.detail']['spacing_ijk_mm'] = [float(v.strip()) for v in c['input.detail']['spacing_ijk_mm'].split(',')]
    spacing = c['input.detail']['spacing_ijk_mm']
    if len(spacing) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing):
        raise ValueError('Detail spacing needs three positive finite numbers')
    if c['model']['learning_rate'] <= 0 or c['model']['clip_min'] >= c['model']['clip_max']:
        raise ValueError('Invalid optimizer or intensity range')
    if not 0 < c['pilot']['minimum_changed_coverage'] <= 1:
        raise ValueError('Coverage must be in (0,1]')
    for section, budget in [('input.overview', 2_000_000), ('input.detail', 8_388_608)]:
        shape = c[section]['shape_kji']
        if min(shape) < 16 or any(v % 8 for v in shape) or math.prod(shape) > budget:
            raise ValueError(f'{section} needs dimensions >=16 divisible by 8, within {budget:,} voxels')
    output = Path(c['experiment']['output_directory'])
    if any(output == Path(c['data'][key]).parent for key in ('manifest', 'annotations')):
        raise ValueError('Output must differ from source artifact directories')
    return c


def join_annotations(rows, labels):
    """Exact sample join, preserving source case and family identity."""
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate dataset sample')
    by_id = {r['sample_id']: r for r in labels}
    if len(by_id) != len(labels) or set(by_id) != {r['sample_id'] for r in rows}:
        raise ValueError('Annotation IDs must match the cohort exactly, without duplicates')
    result = []
    for row in rows:
        label = by_id[row['sample_id']]
        if any(label[key] != row[key] for key in ('case_id', 'anatomy_family')):
            raise ValueError('Annotation case/family identity mismatch')
        score = label[SCORE]
        if (type(label['coa_benchmark_positive']) is not bool or label.get('errors') or
                isinstance(score, bool) or not isinstance(score, (int, float)) or
                not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Unknown or invalid annotation: ' + row['sample_id'])
        result.append({**row, 'annotation': label})
    return result


def select_pilot(rows):
    """Four predetermined, balanced scenarios per development case; no test data."""
    chosen = []
    for case in sorted({r['case_id'] for r in rows if r['split'] != 'test'}):
        group = [r for r in rows if r['case_id'] == case]
        baseline = [r for r in group if r['baseline']]
        edited = [r for r in group if not r['baseline']]
        def companion(r):
            return {k:v for k,v in r['parameters'].items() if k != 'coa_narrow'}
        # Keep companion edits identical across the negative/positive pilot siblings.
        lowest = min(edited, key=lambda r: (r['parameters']['arch_outer_expand']['distance_mm'], r['sample_id']))
        edited = [r for r in edited if companion(r) == companion(lowest)]
        negatives = sorted((r for r in edited if
                            not r['annotation']['coa_benchmark_positive']),
                           key=lambda r: (r['annotation'][SCORE], r['sample_id']))
        positives = sorted((r for r in edited if r['annotation']['coa_benchmark_positive']),
                           key=lambda r: (r['annotation'][SCORE], r['sample_id']))
        if (len(baseline) != 1 or not negatives or len(positives) < 2 or
                positives[0]['annotation'][SCORE] == positives[-1]['annotation'][SCORE]):
            raise ValueError('Pilot needs one baseline, one edited negative and two positives per case')
        chosen.extend({**r, 'pilot_role': role, 'pilot_companion_parameters': {} if r['baseline'] else companion(lowest)}
                      for r,role in zip([baseline[0], negatives[0], positives[0], positives[-1]],
                                        ['original', 'edited_negative', 'lower_positive', 'higher_positive']))
    return chosen


def plan(c):
    path = Path(c['data']['manifest'])
    dataset = checked_json(path)
    annotation_path = Path(c['data']['annotations'])
    annotation = checked_json(annotation_path)
    if dataset.get('schema_version') != 'fakect.context-dataset/1' or dataset.get('complete') is not True:
        raise ValueError('Expected a complete context cohort')
    if dataset.get('anatomical_extent_reviewed') is not True and not c['data']['allow_provisional_anatomy']:
        raise ValueError('Provisional anatomy requires explicit engineering acceptance')
    if annotation.get('schema_version') != 'fakect.coa-label-review/1' or annotation.get('score_name') != SCORE:
        raise ValueError('Unsupported annotation contract')
    if annotation['definition']['dataset_manifest'] != str(path):
        raise ValueError('Annotations reference a different cohort')
    provenance = {r['path']: r['sha256'] for r in annotation['source_provenance']}
    if provenance.get(str(path)) != sha256(path):
        raise ValueError('Annotation/cohort hash mismatch')
    for filename, digest in provenance.items():
        if sha256(filename) != digest:
            raise ValueError('Annotation source provenance changed: ' + filename)
    rows = join_annotations(dataset['samples'], annotation['samples'])
    if len(rows) != dataset['sample_count']:
        raise ValueError('Source sample count disagrees')
    cutoff = annotation['definition']['positive_loss_fraction']
    for row in rows:
        a = row['annotation']
        if (a['original_baseline'] != row['baseline'] or
                a['coa_benchmark_positive'] != (a[SCORE] >= cutoff)):
            raise ValueError('Annotation policy disagrees with row')
        if row['baseline']:
            if a[SCORE] != 0 or a['coa_edit_applied']:
                raise ValueError('Original reference must have zero induced CoA score')
        else:
            steps = [s for s in row['step_outcomes'] if s['step_name'] == 'coa_narrow']
            if (len(steps) != 1 or steps[0] != row['step_outcomes'][-1] or
                    steps[0]['iteration'] != 1 or steps[0]['operation'] != 'erosion' or
                    steps[0]['status'] != 'applied'):
                raise ValueError('Proxy label requires a single final accepted CoA erosion')
            step = steps[0]
            guard = step.get('erosion_safeguard') or {}
            before, after = guard.get('baseline_target_voxels'), guard.get('retained_target_voxels')
            if (guard.get('accepted') is not True or guard.get('connectivity_ok') is not True or
                    type(before) is not int or type(after) is not int or not 0 <= after <= before or
                    before <= 0 or before-after != step['removed_voxels'] or
                    not math.isclose(a[SCORE], (before-after)/before, rel_tol=0, abs_tol=1e-12)):
                raise ValueError('Annotation score disagrees with achieved CoA edit')
    design = build_evaluation_design(rows, {'strategy': 'holdout',
        'case_counts': c['split']['case_counts'], 'seeds': [c['model']['seed']], 'threshold': .5}, c['split']['seed'])
    assignments = design['folds'][0]['family_assignment']
    rows = [{**r, 'split': assignments[r['anatomy_family']]} for r in rows]
    originals = {}
    sources = {}
    for row in rows:
        key = (row['original_reference_mask_sha256'], tuple(row['spacing_ijk_mm']))
        if originals.setdefault(key, row['split']) != row['split']:
            raise ValueError('Identical reference anatomy crosses partitions')
        source = row['source_atn']
        if sources.setdefault(source['path'], row['split']) != row['split']:
            raise ValueError('Shared attenuation source crosses partitions')
    selected = select_pilot(rows)
    # Bound native reads before any image allocation.
    for row in selected:
        ns = row['spacing_ijk_mm'][::-1]
        ts = c['input.detail']['spacing_ijk_mm'][::-1]
        bound = [min(n, math.ceil((p-1)*b/a)+2) for n,p,a,b in zip(
            row['source_shape_kji'], c['input.detail']['shape_kji'], ns, ts)]
        if math.prod(bound) > 64_000_000:
            raise ValueError('Detail read exceeds native slab budget for ' + row['case_id'])
        stat = Path(row['source_atn']['path']).stat()
        source = row['source_atn']
        if (stat.st_size, stat.st_mtime_ns) != (source['bytes'], source['mtime_ns']):
            raise ValueError('Pinned source stat changed: ' + source['path'])
    return {'schema_version': SCHEMA, 'ready': True, 'scope': 'engineering_learning_check',
            'config': deepcopy(c), 'dataset_sha256': sha256(path), 'annotation_sha256': sha256(annotation_path),
            'annotation_acceptance': {'use': c['data']['annotation_use'], 'source_status': annotation['status'],
                                      'threshold': cutoff, 'clinical_review': False,
                                      'near_threshold_policy': 'retained_with_flag'},
            'design': design, 'samples': selected,
            'reserved_test_sample_ids': [r['sample_id'] for r in rows if r['split'] == 'test'],
            'pilot_counts': {split: {'samples': sum(r['split'] == split for r in selected),
                                    'positive': sum(r['split'] == split and r['annotation']['coa_benchmark_positive'] for r in selected)}
                             for split in ('train', 'validation')},
            'sampling': 'Per development case: original, lowest edited negative, lowest positive, highest positive; all edited samples share the lowest dilation setting and other companion parameters.',
            'inference_contract': 'Attenuation and acquisition geometry only; masks provide supervision and coverage QA, never proposals.'}


def freeze(c, input_bytes):
    document = plan(c)
    sources = {name: (ROOT / name).read_bytes() for name in CODE_FILES}
    out = Path(c['experiment']['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE').touch()
    (out / 'input.ini').write_bytes(input_bytes)
    hashes = {}
    for name, content in sources.items():
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    document.update(code_sha256=hashes, input_sha256=hashlib.sha256(input_bytes).hexdigest())
    write_json(out / 'experiment-lock.json', document)
    (out / 'INCOMPLETE').unlink()
    return document


def inspect_lock(c, input_bytes):
    out = Path(c['experiment']['output_directory'])
    lock = checked_json(out / 'experiment-lock.json')
    if lock['config'] != c or lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest():
        raise ValueError('Configuration changed; choose a new experiment revision')
    if sha256(out / 'input.ini') != lock['input_sha256']:
        raise ValueError('Frozen INI changed')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if sha256(ROOT / name) != digest or sha256(out / 'source' / name) != digest:
            raise ValueError('Pinned implementation changed: ' + name)
    for key, name in [('manifest', 'dataset_sha256'), ('annotations', 'annotation_sha256')]:
        checked_json(c['data'][key], lock[name])
    return lock


def bbox_target(volume, overview):
    """Supervision only: positive mask voxel-edge bounds in padded overview FOV."""
    mask = volume.mask_after
    axes = [np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    if any(not len(axis) for axis in axes):
        raise ValueError('Empty anatomy target')
    lo = np.array([a[0] for a in axes]) + volume.mask_origin_ijk[::-1]
    hi = np.array([a[-1]+1 for a in axes]) + volume.mask_origin_ijk[::-1]
    native = np.array(volume.spacing_ijk_mm[::-1])
    extent = np.array(overview['image'].shape) * np.array(overview['spacing_ijk_mm'][::-1])
    return np.r_[((lo+hi)/2)*native/extent, (hi-lo)*native/extent].astype(np.float32)


def proposal_origins(box, overview, c):
    """Image-predicted box -> two physical patches; no mask/ROI argument."""
    box = np.asarray(box, dtype=float)
    if box.shape != (6,) or not np.isfinite(box).all() or np.any(box[3:] <= 0):
        raise ValueError('Invalid predicted anatomy box')
    extent = np.array(overview['image'].shape) * np.array(overview['spacing_ijk_mm'][::-1])
    patch = np.array(c['input.detail']['shape_kji'])
    spacing = np.array(c['input.detail']['spacing_ijk_mm'][::-1])
    origins, boxes = [], []
    for fraction in (.25, .75):
        center = box[:3].copy()
        center[0] += (fraction-.5)*box[3]
        origin = np.floor(center*extent/spacing - patch/2).astype(int)
        origins.append(origin.tolist())
        boxes.append(np.r_[(origin+patch/2)*spacing/extent, patch*spacing/extent].tolist())
    return origins, np.asarray(boxes, dtype=np.float32)


def coverage(volume, origins, c):
    """Native-center coverage QA; does not choose or modify image proposals."""
    shape = volume.mask_after.shape
    union = np.zeros(shape, dtype=bool)
    native = np.array(volume.spacing_ijk_mm[::-1])
    spacing = np.array(c['input.detail']['spacing_ijk_mm'][::-1])
    patch = np.array(c['input.detail']['shape_kji'])
    offset = volume.mask_origin_ijk[::-1]
    centers = [(np.arange(n)+o+.5)*s for n,o,s in zip(shape,offset,native)]
    for origin in origins:
        # Select native centers inside model patch voxel faces, including padded edges.
        lo = np.array([np.searchsorted(axis, face, side='left')
                       for axis,face in zip(centers,np.asarray(origin)*spacing)])
        hi = np.array([np.searchsorted(axis, face, side='left')
                       for axis,face in zip(centers,(np.asarray(origin)+patch)*spacing)])
        if np.all(hi > lo):
            union[tuple(slice(int(l), int(h)) for l,h in zip(lo,hi))] = True
    changed = volume.mask_before != volume.mask_after
    def ratio(mask):
        total = int(np.count_nonzero(mask))
        return float(np.count_nonzero(mask & union) / total) if total else None
    before, after = volume.mask_before.astype(bool), volume.mask_after.astype(bool)
    return {'target_coverage': ratio(after), 'before_coverage': ratio(before),
            'changed_coverage': ratio(changed), 'added_coverage': ratio(after & ~before),
            'removed_coverage': ratio(before & ~after),
            'changed_voxels': int(np.count_nonzero(changed))}


def verify_no_cancellation(volume, row):
    """Conservative CoA coverage proof for this one-dilation/two-erosion release."""
    steps = [s for s in row['step_outcomes'] if s['status'] == 'applied']
    if not steps and row['baseline']:
        return
    if ([s['step_name'] for s in steps] != ['arch_outer_expand', 'arch_inner_narrow', 'coa_narrow'] or
            [s['operation'] for s in steps] != ['dilation', 'erosion', 'erosion'] or
            any(s['iteration'] != 1 for s in steps)):
        raise ValueError('Unsupported per-condition coverage proof: recipe order or passes changed')
    before, after = volume.mask_before.astype(bool), volume.mask_after.astype(bool)
    for mask, key in [(after & ~before, 'added_voxels'), (before & ~after, 'removed_voxels')]:
        if int(mask.sum()) != sum(s[key] for s in steps):
            raise ValueError('Net changes conceal cancellation; per-step masks are required')


def normalized_input(image, valid, c):
    low, high = c['model']['clip_min'], c['model']['clip_max']
    intensity = np.clip((image-low)/(high-low), 0, 1).astype(np.float32)
    return np.stack((intensity * (valid > 0), valid.astype(np.float32)), axis=-1)


def report(c):
    """Self-contained progress report; optional PNG previews use local references."""
    out = Path(c['experiment']['output_directory'])
    lock = checked_json(out / 'experiment-lock.json')
    sections = []
    for name in ('preparation', 'locator-history', 'proposal-audit', 'classifier-history', 'results', 'failure'):
        path = out / (name + '.json')
        if path.exists():
            value = checked_json(path)
            summary = ''
            if name == 'proposal-audit':
                def percent(v):
                    return 'N/A' if v is None else f'{100*v:.1f}%'
                body = ''.join('<tr><td>'+html.escape(r['sample_id'])+'</td><td>'+html.escape(r['split'])+
                    f'</td><td>{r["center_error_mm"]:.2f}</td><td>'+percent(r['target_coverage'])+
                    '</td><td>'+percent(r['removed_coverage'])+'</td><td>'+percent(r['changed_coverage'])+
                    '</td><td>'+('PASS' if r['coverage_passed'] else 'STOP')+'</td></tr>' for r in value['samples'])
                summary = '<table><tr><th>Sample</th><th>Split</th><th>Center error mm</th><th>Aorta covered</th><th>Erosion covered</th><th>Changes covered</th><th>Gate</th></tr>'+body+'</table>'
            if name == 'results':
                summary = '<p><strong>'+html.escape(value['status'])+'</strong></p>'
            sections.append(f'<h2>{html.escape(name)}</h2>{summary}<details><summary>Metadata</summary><pre>{html.escape(json.dumps(value, indent=2))}</pre></details>')
    previews = ''.join(f'<figure><img loading="lazy" src="previews/{html.escape(p.name)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>'
                       for p in sorted((out / 'previews').glob('*.png')))
    document = '<!doctype html><meta charset="utf-8"><title>CoA multiresolution pilot</title><style>body{font:16px system-ui;margin:2rem auto;max-width:1100px;padding:1rem}pre{white-space:pre-wrap;background:#f3f5f7;padding:1rem}img{max-width:100%}figure{margin:1rem 0}table{border-collapse:collapse;font-size:14px}td,th{padding:.4rem;border:1px solid #ccc;text-align:left}</style>'
    document += '<h1>CoA multiresolution learning pilot</h1><p>Engineering study of a provisional local-volume-loss label. Test families remain reserved. This is not clinical severity validation.</p>'
    document += '<h2>Frozen design</h2><pre>' + html.escape(json.dumps({k:lock[k] for k in ('pilot_counts','annotation_acceptance','sampling','inference_contract')},indent=2)) + '</pre>'
    document += previews + ''.join(sections)
    (out / 'report.html').write_text(document)
    return str(out / 'report.html')
