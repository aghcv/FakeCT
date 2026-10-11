#!/usr/bin/env python3
"""Audit/export a fixed XCAT CT panel and evaluate a pinned pretrained aorta model."""
from __future__ import annotations

import argparse
import configparser
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_multiscale_experiment import checked_json, sha256, write_json
from fakect_pretrained_evidence import case_evidence
from fakect_pretrained_adapter import export_thorax_nifti, plan_thorax_crop

SCHEMA = 'fakect.pretrained-aorta-benchmark/1'
FIELDS = {
    'benchmark': {'schema_version', 'source_manifest', 'output_directory', 'cases', 'panel', 'target_review_status'},
    'input': {'context_mm', 'slab_slices'},
    'model': {'package_version', 'environment_directory', 'cache_directory', 'weight_tasks', 'task', 'roi_subset', 'fast'},
    'evaluation': {'surface_tolerance_mm', 'boundary_margin_mm', 'local_context_mm'},
}
CODE_FILES = ('scripts/benchmark_pretrained_aorta.py', 'requirements-pretrained.txt',
    'scripts/slurm/fakect_pretrained_prepare.sh', 'scripts/slurm/fakect_pretrained_gpu.sh', 'scripts/slurm/fakect_pretrained_report.sh',
    'src/fakect_pretrained_evidence.py', 'src/fakect_pretrained_adapter.py', 'src/fakect_pretrained_report.py',
    'src/fakect_context_volume.py', 'src/fakect_multiscale_experiment.py',
    'src/fakect_anatomy_surfaces.py', 'src/fakect_anatomy_centerlines.py',
    'src/fakect_two_stage_3d.py', 'src/fakect_surface_overlay.py', 'src/fakect_volume_preview.py')


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(FIELDS):
        raise ValueError('Unexpected benchmark sections')
    config = {section: dict(parser[section]) for section in FIELDS}
    for section, fields in FIELDS.items():
        if set(config[section]) != fields:
            raise ValueError('Wrong fields in ' + section)
    if config['benchmark']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported pretrained benchmark schema')
    for section, keys in [('benchmark', ('source_manifest', 'output_directory')),
                          ('model', ('environment_directory', 'cache_directory'))]:
        for key in keys:
            path = Path(config[section][key]).expanduser()
            config[section][key] = str((path if path.is_absolute() else ROOT / path).resolve())
    for key in ('cases', 'panel'):
        config['benchmark'][key] = [v.strip() for v in config['benchmark'][key].split(',')]
        if len(set(config['benchmark'][key])) != len(config['benchmark'][key]):
            raise ValueError('Repeated benchmark ' + key)
    if config['benchmark']['target_review_status'] != 'engineering_proxy':
        raise ValueError('This pilot requires explicit engineering_proxy target acknowledgement')
    if config['benchmark']['panel'] != ['baseline', 'lower_narrowing', 'higher_narrowing', 'enlargement']:
        raise ValueError('This frozen pilot supports the four prespecified panel roles')
    config['input']['context_mm'] = float(config['input']['context_mm'])
    config['input']['slab_slices'] = int(config['input']['slab_slices'])
    config['model']['weight_tasks'] = [int(v) for v in config['model']['weight_tasks'].split(',')]
    config['model']['fast'] = parser.getboolean('model', 'fast')
    if config['model']['task'] != 'total' or config['model']['roi_subset'] != 'aorta' or config['model']['fast']:
        raise ValueError('This pilot requires default total/aorta 1.5 mm inference')
    config['evaluation'] = {k: float(v) for k, v in config['evaluation'].items()}
    if (not 0 <= config['input']['context_mm'] <= 100 or not 1 <= config['input']['slab_slices'] <= 64
            or any(not np.isfinite(v) or v < 0 for v in config['evaluation'].values())
            or config['evaluation']['local_context_mm'] <= config['evaluation']['boundary_margin_mm']):
        raise ValueError('Invalid export or evaluation settings')
    return config


def select_panel(manifest, cases):
    if manifest.get('complete') is not True:
        raise ValueError('A complete frozen sibling population is required')
    if set(cases) & {'260612', '260617'}:
        raise ValueError('Recent reserved infant cases are outside this development pilot')
    result = []
    for case in cases:
        rows = [r for r in manifest['samples'] if r['case_id'] == case]
        base = [r for r in rows if r['baseline']]
        narrow = sorted((r for r in rows if not r['baseline'] and r['mask_counts']['removed'] > 0
                         and r['mask_counts']['added'] == 0), key=lambda r: (r['mask_counts']['removed'], r['sample_id']))
        expand = sorted((r for r in rows if not r['baseline'] and r['mask_counts']['added'] > 0
                         and r['mask_counts']['removed'] == 0), key=lambda r: (r['mask_counts']['added'], r['sample_id']))
        if len(base) != 1 or len(narrow) < 2 or not expand:
            raise ValueError('Insufficient original/removed-only/added-only samples for ' + case)
        chosen = [base[0], narrow[(len(narrow) - 1) // 4], narrow[-1], expand[-1]]
        if len({r['sample_id'] for r in chosen}) != 4:
            raise ValueError('Benchmark roles must have distinct samples')
        result.extend(dict(row, benchmark_role=role) for row, role in zip(chosen,
                      ('baseline', 'lower_narrowing', 'higher_narrowing', 'enlargement')))
    return result


def model_cache(config):
    model = config['model']
    if importlib.metadata.version('TotalSegmentator') != model['package_version']:
        raise ValueError('Pinned TotalSegmentator version required')
    cache = Path(model['cache_directory'])
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['TOTALSEG_HOME_DIR'] = str(cache)
    from totalsegmentator.config import setup_totalseg, set_config_key, get_weights_dir
    from totalsegmentator.libs import download_pretrained_weights
    from totalsegmentator.map_tasks_config import TASK_CONFIGS
    standard = TASK_CONFIGS['total']['sub_modes']['default']['task_id']
    if set(model['weight_tasks']) != set(standard) | {298}:
        raise ValueError('Pinned model task IDs do not match the official default-task registry')
    setup_totalseg('fakect-local-benchmark')
    set_config_key('send_usage_stats', False)
    for task in model['weight_tasks']:
        download_pretrained_weights(task)
    weights = Path(get_weights_dir())
    files = {str(p): sha256(p) for p in sorted(weights.rglob('*')) if p.is_file()}
    if not any(p.endswith('.pth') for p in files):
        raise ValueError('No downloaded pretrained checkpoint')
    versions = {name: importlib.metadata.version(name) for name in
                ('TotalSegmentator', 'torch', 'nnunetv2', 'nibabel', 'numpy', 'scipy', 'plotly', 'scikit-image', 'matplotlib')}
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (cache / 'environment-freeze.txt').write_text(freeze)
    write_json(cache / 'model-lock.json', {'versions': versions, 'weights': files,
               'environment_sha256': sha256(cache / 'environment-freeze.txt'), 'weight_tasks': model['weight_tasks'],
               'model': model['task'], 'roi_subset': model['roi_subset'], 'telemetry_enabled': False})
    return {'model_lock': str(cache / 'model-lock.json'), 'versions': versions, 'weights': len(files)}


def plan(config, content):
    source = checked_json(config['benchmark']['source_manifest'])
    rows = select_panel(source, config['benchmark']['cases'])
    model_lock = Path(config['model']['cache_directory']) / 'model-lock.json'
    pinned_model = checked_json(model_lock)
    if pinned_model['versions']['TotalSegmentator'] != config['model']['package_version']:
        raise ValueError('Pretrained model lock version mismatch')
    dependencies = {config['benchmark']['source_manifest']: sha256(config['benchmark']['source_manifest']),
                    str(model_lock): sha256(model_lock)}
    evidence = {}
    for case in config['benchmark']['cases']:
        base = next(r for r in rows if r['case_id'] == case and r['baseline'])
        evidence[case] = case_evidence(base)
        selection = json.loads(Path(base['selection_manifest']).read_text())
        for row in (r for r in rows if r['case_id'] == case):
            plan_thorax_crop(row, selection, evidence[case], context_mm=config['input']['context_mm'])
        mask_path = str((Path(base['selection_manifest']).parent / selection['mask_npz']).resolve())
        for name, digest in [(mask_path, selection['mask_npz_sha256']),
                             (base['selection_manifest'], base['selection_manifest_sha256'])]:
            if sha256(name) != digest:
                raise ValueError('Frozen anatomical reference changed')
            dependencies[name] = digest
    out = Path(config['benchmark']['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'input.ini').write_bytes(content)
    code = {}
    for name in CODE_FILES:
        source_path = ROOT / name
        code[name] = sha256(source_path)
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    lock = {'schema_version': SCHEMA, 'config': config, 'samples': rows, 'evidence': evidence,
            'code_sha256': code, 'dependencies': dependencies, 'input_sha256': sha256(out / 'input.ini'),
            'model': pinned_model, 'scope': 'Region-assisted development benchmark, native atlas thoracic core',
            'models_trained': False, 'reserved_infant_images_evaluated': False}
    write_json(out / 'benchmark-lock.json', lock)
    return {'samples': len(rows), 'cases': config['benchmark']['cases'], 'output': str(out)}


def inspect(config, content, *, check_model=False):
    out = Path(config['benchmark']['output_directory'])
    lock = checked_json(out / 'benchmark-lock.json')
    if lock['config'] != config or hashlib.sha256(content).hexdigest() != lock['input_sha256']:
        raise ValueError('Frozen benchmark input changed')
    for name, digest in lock['code_sha256'].items():
        if sha256(ROOT / name) != digest or sha256(out / 'source' / name) != digest:
            raise ValueError('Frozen benchmark source changed: ' + name)
    for name, digest in lock['dependencies'].items():
        if sha256(name) != digest:
            raise ValueError('Frozen benchmark dependency changed: ' + name)
    for evidence in lock['evidence'].values():
        for item in evidence['dependencies']:
            stat = Path(item['path']).stat()
            if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
                raise ValueError('Frozen XCAT evidence source changed')
    if check_model:
        for name, version in lock['model']['versions'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError('Pretrained software version changed: ' + name)
        for name, digest in lock['model']['weights'].items():
            if sha256(name) != digest:
                raise ValueError('Pretrained checkpoint changed')
        freeze = Path(config['model']['cache_directory']) / 'environment-freeze.txt'
        if sha256(freeze) != lock['model']['environment_sha256']:
            raise ValueError('Pinned pretrained environment freeze changed')
    return lock


def prepare(config, lock, case):
    if case not in config['benchmark']['cases']:
        raise ValueError('Unknown benchmark case')
    out = Path(config['benchmark']['output_directory']) / 'cases' / case
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE').touch()
    samples = []
    for row in (r for r in lock['samples'] if r['case_id'] == case):
        folder = out / row['benchmark_role']
        folder.mkdir()
        paths = {key: folder / (key + '.nii.gz') for key in ('target', 'domain', 'changed')}
        image = folder / 'ct-hu.nii.gz'
        selection = json.loads(Path(row['selection_manifest']).read_text())
        exported = export_thorax_nifti(row, selection, lock['evidence'][case], image,
                    **config['input'], mask_output_paths=paths)
        record = {k: row[k] for k in ('sample_id', 'case_id', 'baseline', 'benchmark_role', 'anatomy_family', 'mask_counts')}
        for name, path in [('input_nifti', image), ('truth_nifti', paths['target']),
                           ('domain_nifti', paths['domain']), ('changed_nifti', paths['changed'])]:
            record[name] = str(path)
            record[name + '_sha256'] = exported['artifacts'][str(path)]['sha256']
        record['export'] = exported
        samples.append(record)
        print('Prepared', row['sample_id'], row['benchmark_role'], flush=True)
    write_json(out / 'prepared.json', {'complete': True, 'samples': samples,
                'benchmark_lock_sha256': sha256(out.parents[1] / 'benchmark-lock.json')})
    (out / 'INCOMPLETE').unlink()
    return {'case': case, 'samples': len(samples)}


def prediction_case(config, lock, case):
    import nibabel as nib
    import torch
    if not torch.cuda.is_available():
        raise ValueError('GPU inference is required for this authorized pilot')
    out = Path(config['benchmark']['output_directory'])
    metadata = checked_json(out / 'cases' / case / 'prepared.json')
    if metadata['benchmark_lock_sha256'] != sha256(out / 'benchmark-lock.json'):
        raise ValueError('Prepared images belong to another benchmark')
    predicted = out / 'predictions' / case
    predicted.mkdir(parents=True, exist_ok=False)
    (predicted / 'INCOMPLETE').touch()
    os.environ['TOTALSEG_HOME_DIR'] = config['model']['cache_directory']
    from totalsegmentator.config import get_config
    if get_config()['send_usage_stats'] is not False:
        raise ValueError('Local-only predictor configuration changed')
    rows = []
    for row in metadata['samples']:
        for name in ('input_nifti', 'truth_nifti', 'domain_nifti', 'changed_nifti'):
            if sha256(row[name]) != row[name + '_sha256']:
                raise ValueError('Prepared benchmark artifact changed')
        folder = predicted / row['benchmark_role']
        command = [str(Path(config['model']['environment_directory']) / 'bin' / 'TotalSegmentator'),
                   '-i', row['input_nifti'], '-o', str(folder), '--task', 'total', '--roi_subset', 'aorta',
                   '--device', 'gpu', '--nr_thr_resamp', '2', '--nr_thr_saving', '1']
        print('Predicting', row['sample_id'], row['benchmark_role'], flush=True)
        subprocess.run(command, check=True)
        path = folder / 'aorta.nii.gz'
        source, inferred = nib.load(row['input_nifti']), nib.load(str(path))
        if source.shape != inferred.shape or not np.allclose(source.affine, inferred.affine, rtol=0, atol=1e-5):
            raise ValueError('Pretrained output failed native shape/affine roundtrip')
        rows.append(dict(row, status='completed', prediction_nifti=str(path), prediction_nifti_sha256=sha256(path)))
    write_json(predicted / 'prediction.json', {'complete': True, 'samples': rows,
        'benchmark_lock_sha256': sha256(out / 'benchmark-lock.json'), 'model_lock': lock['model']})
    (predicted / 'INCOMPLETE').unlink()
    return {'case': case, 'samples': len(rows), 'gpu': torch.cuda.get_device_name(0)}


def report(config, lock):
    from fakect_pretrained_report import write_report
    out = Path(config['benchmark']['output_directory'])
    rows = []
    for case in config['benchmark']['cases']:
        result = checked_json(out / 'predictions' / case / 'prediction.json')
        if result['benchmark_lock_sha256'] != sha256(out / 'benchmark-lock.json'):
            raise ValueError('Predictions belong to a different benchmark')
        rows.extend(result['samples'])
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in lock['samples']]:
        raise ValueError('Benchmark prediction inventory is incomplete or reordered')
    manifest = {'schema_version': SCHEMA, 'samples': rows, 'predictor': lock['model'],
                'benchmark_lock_sha256': sha256(out / 'benchmark-lock.json'), 'models_trained': False}
    write_json(out / 'evaluation-manifest.json', manifest)
    result = write_report(out / 'evaluation-manifest.json', out / 'review-r1', settings=config['evaluation'])
    write_json(out / 'benchmark-receipt.json', dict(result, complete=True, models_trained=False,
               benchmark_lock_sha256=sha256(out / 'benchmark-lock.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('model-cache', 'plan', 'list', 'prepare', 'predict', 'report'), default='plan')
    parser.add_argument('--case-id')
    parser.add_argument('--ids-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    path = Path(args.config)
    content, config = path.read_bytes(), load_config(path)
    if args.validate_only:
        source = checked_json(config['benchmark']['source_manifest'])
        rows = select_panel(source, config['benchmark']['cases'])
        audit = {}
        for case in config['benchmark']['cases']:
            base = next(r for r in rows if r['case_id'] == case and r['baseline'])
            evidence = case_evidence(base)
            selection = json.loads(Path(base['selection_manifest']).read_text())
            crop = plan_thorax_crop(base, selection, evidence, context_mm=config['input']['context_mm'])
            audit[case] = {'water_mu_cm_inverse': evidence['calibration']['mu_water_cm_inverse'],
                          'orientation': evidence['orientation'], 'native_shape_kji': crop['shape_kji']}
        result = {'validated': True, 'samples': len(rows), 'case_input_audit': audit}
    elif args.stage == 'model-cache':
        result = model_cache(config)
    elif args.stage == 'plan':
        result = plan(config, content)
    else:
        lock = inspect(config, content, check_model=args.stage == 'predict')
        if args.stage == 'list':
            if args.ids_only:
                print('\n'.join(config['benchmark']['cases']))
                return
            result = {'cases': config['benchmark']['cases']}
        elif args.stage == 'prepare':
            result = prepare(config, lock, args.case_id)
        elif args.stage == 'predict':
            result = prediction_case(config, lock, args.case_id)
        else:
            result = report(config, lock)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
