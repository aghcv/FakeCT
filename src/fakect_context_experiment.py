"""Immutable experiments for streamed, larger-context aorta segmentation."""
import configparser
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from fakect_evaluation_design import build_evaluation_design
from fakect_mask_pair_preview import sha256
from fakect_context_volume import MAX_SOURCE_SLAB_VOXELS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.context-experiment/1'
CODE_FILES = ('fakect_context_experiment.py', 'fakect_context_learning.py',
              'fakect_context_volume.py', 'fakect_context_data.py', 'fakect_context_grid.py',
              'fakect_evaluation_design.py', 'fakect_mask_pair_preview.py',
              'fakect_two_stage_data.py', 'fakect_cohort_registry.py',
              'fakect_segmentation.py', 'fakect_model_resampling.py')


def _write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    path.with_suffix(path.suffix+'.sha256').write_text(sha256(path)+'\n')


def load_config(path, *, repo_root=None):
    from fakect_context_learning import settings_for
    root = Path(repo_root or ROOT).resolve()
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                        strict=True, empty_lines_in_values=False)
    parser.read_string(Path(path).read_text(encoding='utf-8-sig'))
    fields = {'experiment': {'schema_version', 'scope', 'mode', 'output_directory'},
              'data': {'manifest', 'allow_provisional_anatomy'},
              'split': {'strategy', 'case_counts', 'seed'}, 'model': None}
    if parser.defaults() or set(parser.sections()) != set(fields):
        raise ValueError('Require exactly [experiment], [data], [split], [model] without defaults')
    result = {key: dict(parser[key]) for key in fields}
    for section, keys in fields.items():
        if keys is not None and set(result[section]) != keys:
            raise ValueError(f'[{section}] requires exactly {sorted(keys)}')
    exp, data, split = (result[name] for name in ('experiment', 'data', 'split'))
    if exp['schema_version'] != SCHEMA or exp['scope'] != 'engineering':
        raise ValueError(f'Expected {SCHEMA}, scope=engineering')
    if exp['mode'] not in ('diagnostic', 'cohort'):
        raise ValueError('experiment.mode must be diagnostic or cohort')
    for values, key in ((exp, 'output_directory'), (data, 'manifest')):
        path_value = Path(values[key]).expanduser()
        values[key] = str((path_value if path_value.is_absolute() else root/path_value).resolve())
    if Path(exp['output_directory']) == Path(data['manifest']).parent:
        raise ValueError('Dataset and experiment outputs must differ')
    data['allow_provisional_anatomy'] = parser.getboolean('data', 'allow_provisional_anatomy')
    if split['strategy'] != 'holdout':
        raise ValueError('Current larger-context experiment implements family-disjoint holdout')
    split['case_counts'] = [int(x.strip()) for x in split['case_counts'].split(',')]
    split['seed'] = int(split['seed'])
    if len(split['case_counts']) != 3 or min(split['case_counts']) < 1 or split['seed'] < 0:
        raise ValueError('Require positive train,validation,test case counts and nonnegative seed')
    result['model'] = settings_for(result)
    return result


def plan(config, *, verify_payloads=True):
    from fakect_context_data import inspect_dataset
    from fakect_context_learning import settings_for
    path = Path(config['data']['manifest'])
    if not path.is_file():
        return {'ready': False, 'errors': [f'Prepare context dataset first: {path}']}
    dataset = inspect_dataset(path, verify_payloads=verify_payloads)
    samples = dataset['samples']
    settings = settings_for(config)
    design = build_evaluation_design(samples,
        {'strategy': 'holdout', 'case_counts': config['split']['case_counts'],
         'seeds': [settings['seed']], 'threshold': settings['threshold']}, config['split']['seed'])
    assignment = design['folds'][0]['family_assignment']
    rows = [{**row, 'split': assignment[row['anatomy_family']]} for row in samples]
    identities = {}
    for row in rows:
        digest = row.get('original_reference_mask_sha256')
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Every sample requires a valid original_reference_mask_sha256 for split checks')
        key = (digest, tuple(row['spacing_ijk_mm']))
        if identities.setdefault(key, row['split']) != row['split']:
            raise ValueError('Identical original reference anatomy crosses splits; group its families together')
    grids = {}
    errors = []
    for row in rows:
        if row['case_id'] in grids:
            continue
        native = row['spacing_ijk_mm'][::-1]
        target = settings['target_spacing_mm'][::-1]
        # Match ContextVolume.model_shape exactly, including floating-point ceil.
        shape = [math.ceil(n*a/b) for n,a,b in zip(row['source_shape_kji'],native,target)]
        tiles = math.prod(math.ceil(n/p) for n,p in zip(shape,settings['patch_size_kji']))
        # A patch spans (p-1)*target/native native-center intervals, plus both
        # interpolation neighbors. Clipping cannot require more than the source.
        slab_shape = [min(n, math.ceil((p-1)*b/a)+2) for n,p,a,b in
                      zip(row['source_shape_kji'],settings['patch_size_kji'],native,target)]
        slab_voxels = math.prod(slab_shape)
        if slab_voxels > MAX_SOURCE_SLAB_VOXELS:
            errors.append(f"Case {row['case_id']} native interpolation slab bound is {slab_voxels:,} voxels, "
                          f'exceeding the {MAX_SOURCE_SLAB_VOXELS:,}-voxel budget; reduce patch size or target spacing')
        grids[row['case_id']] = {'model_shape_kji': shape, 'native_spacing_ijk_mm': row['spacing_ijk_mm'],
                                 'full_scan_tiles_per_sample': tiles,
                                 'native_slab_upper_bound_shape_kji': slab_shape,
                                 'native_slab_upper_bound_voxels': slab_voxels,
                                 'native_slab_upper_bound_bytes': 4*slab_voxels}
    # A diagnostic is explicitly resubstitution on one TRAIN sample, not a test.
    diagnostic = config['experiment']['mode'] == 'diagnostic'
    requested = settings.get('diagnostic_sample_id', '')
    training = sorted((r for r in rows if r['split'] == 'train'), key=lambda row: row['sample_id'])
    selected = next((r for r in training if r['sample_id'] == requested), None) if requested else training[0]
    if diagnostic and selected is None:
        raise ValueError('diagnostic_sample_id must belong to the training partition')
    provisional = dataset.get('anatomical_extent_reviewed') is not True
    if provisional and not config['data']['allow_provisional_anatomy']:
        errors.append('Anatomical extent is provisional; review it or explicitly allow an engineering experiment')
    grid_audit = None
    if not diagnostic:
        from fakect_context_grid import audit_target_grid
        grid_audit = audit_target_grid(rows, settings['target_spacing_mm'])
        if not grid_audit['ready']:
            errors.extend(grid_audit['errors'])
    warnings = ['Source images are attenuation-copy proxies, not reconstructed clinical CT.',
                'Anatomical boundaries/proximal branches and family identities remain provisional.',
                'Whole-volume validation and testing can be expensive; tile counts cover the full source field of view.']
    if diagnostic:
        warnings.append('Diagnostic fits/evaluates one training patch only; its Dice is not held-out performance.')
    return {'schema_version': 'fakect.context-plan/1', 'ready': not errors, 'errors': errors,
            'dataset_manifest': str(path), 'dataset_manifest_sha256': sha256(path),
            'sample_count': len(rows), 'samples': rows, 'design': design, 'model_grids': grids,
            'model_settings': settings, 'diagnostic': diagnostic,
            'diagnostic_sample_id': selected['sample_id'] if diagnostic else None,
            'split_counts': {part: sum(r['split']==part for r in rows) for part in ('train','validation','test')},
            'evaluation_tiles': {part: sum(grids[r['case_id']]['full_scan_tiles_per_sample']
                                          for r in rows if r['split']==part) for part in ('validation','test')},
            'memory_bounds': {'native_slab_budget_voxels': MAX_SOURCE_SLAB_VOXELS,
                              'native_slab_budget_bytes': 4*MAX_SOURCE_SLAB_VOXELS,
                              'maximum_native_slab_bytes': max(g['native_slab_upper_bound_bytes'] for g in grids.values()),
                              'scope': 'Native float32 slab only; excludes crop caches, output arrays, and neural-network memory.'},
            'patch_extent_kji_mm': [n*s for n,s in zip(settings['patch_size_kji'],settings['target_spacing_mm'][::-1])],
            'scope': 'engineering', 'anatomical_extent_reviewed': not provisional,
            'target_grid_audit': grid_audit, 'warnings': warnings}


def freeze(config, input_bytes):
    planned = plan(config)
    if not planned['ready']:
        raise ValueError('; '.join(planned['errors']))
    output = Path(config['experiment']['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'input.ini').write_bytes(input_bytes)
    (output/'source').mkdir()
    hashes = {}
    # The diagnostic does not execute the cohort-wide grid audit.
    code_files = [name for name in CODE_FILES if name != 'fakect_context_grid.py' or not planned['diagnostic']]
    for name in code_files:
        (output/'source'/name).write_bytes((ROOT/'src'/name).read_bytes())
        hashes[name] = sha256(ROOT/'src'/name)
    lock = {**planned, 'schema_version': 'fakect.context-lock/1', 'config': deepcopy(config),
            'input_sha256': hashlib.sha256(input_bytes).hexdigest(), 'code_sha256': hashes}
    _write(output/'experiment-lock.json', lock)
    return lock


def inspect_lock(config, input_bytes):
    output = Path(config['experiment']['output_directory'])
    path = output/'experiment-lock.json'
    if sha256(path) != path.with_suffix('.json.sha256').read_text().strip():
        raise ValueError('Experiment lock checksum changed')
    lock = json.loads(path.read_text())
    if (lock.get('schema_version') != 'fakect.context-lock/1' or lock['config'] != config or
            lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest() or
            sha256(output/'input.ini') != lock['input_sha256']):
        raise ValueError('Experiment INI changed; use a new revision')
    for name, digest in lock['code_sha256'].items():
        if sha256(ROOT/'src'/name) != digest or sha256(output/'source'/name) != digest:
            raise ValueError('Experiment implementation changed; use a new revision')
    if sha256(lock['dataset_manifest']) != lock['dataset_manifest_sha256']:
        raise ValueError('Context dataset changed after planning')
    return lock


def fit(config, input_bytes):
    from fakect_context_learning import fit as fit_model
    lock = inspect_lock(config, input_bytes)
    return fit_model(config, lock, Path(config['experiment']['output_directory'])/'model',
                     diagnostic=lock['diagnostic'])


def evaluate(config, input_bytes):
    from fakect_context_learning import evaluate as evaluate_model
    lock = inspect_lock(config, input_bytes)
    output = Path(config['experiment']['output_directory'])
    return evaluate_model(config, lock, output/'model', output/'evaluation',
                          diagnostic=lock['diagnostic'])
