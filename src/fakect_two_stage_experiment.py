"""INI orchestration for the paired segmentation/reference-shape pilot.

Generation is separate. This explicit engineering pilot preserves provisional
family provenance; it does not relax the verified registry experiment contract.
"""
import configparser
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from fakect_evaluation_design import build_evaluation_design
from fakect_model_resampling import output_shape
from fakect_segmentation import validate_clip_bounds

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.two-stage-experiment/1'
MODEL_DEFAULTS = {
    'architecture': 'unet3d', 'patch_size_kji': [64, 64, 64],
    'target_spacing_mm': [0.6, 0.6, 0.6], 'base_filters': 4, 'depth': 2,
    'batch_size': 1, 'epochs': 5, 'learning_rate': 0.001, 'seed': 42,
    'patches_per_volume': 4, 'change_weight': 5.0, 'threshold': 0.5,
    'clip_min': 0.0, 'clip_max': 0.5, 'max_train_batches_per_epoch': 0,
    'save_predictions': True,
}
CODE_FILES = ('fakect_two_stage_experiment.py', 'fakect_two_stage_data.py',
              'fakect_two_stage_learning.py', 'fakect_model_resampling.py',
              'fakect_evaluation_design.py', 'fakect_segmentation.py',
              'fakect_cohort_registry.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _boolean(value, name):
    if value not in ('true', 'false'):
        raise ValueError(f'{name} must be true or false')
    return value == 'true'


def _integer(value, name, minimum=0, maximum=2**31-1):
    if not value.isascii() or not value.isdigit():
        raise ValueError(f'{name} must be an integer >= {minimum}')
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')
    return result


def _model(values, base):
    result = deepcopy(base)
    if set(values) - set(MODEL_DEFAULTS):
        raise ValueError(f'Unknown model settings: {sorted(set(values)-set(MODEL_DEFAULTS))}')
    for key, value in values.items():
        if key == 'architecture':
            result[key] = value
        elif key == 'save_predictions':
            result[key] = _boolean(value, key)
        elif key in ('patch_size_kji', 'target_spacing_mm'):
            parts = value.split(',')
            if len(parts) != 3:
                raise ValueError(f'{key} requires three comma-separated values')
            result[key] = ([_integer(x.strip(), key, 4, 128) for x in parts]
                           if key == 'patch_size_kji' else [float(x) for x in parts])
        elif isinstance(MODEL_DEFAULTS[key], int):
            result[key] = _integer(value, key, 0 if key in ('seed', 'max_train_batches_per_epoch') else 1)
        else:
            result[key] = float(value)
    if result['architecture'] != 'unet3d':
        raise ValueError('architecture must be unet3d')
    if not 1 <= result['depth'] <= 3:
        raise ValueError('depth must be 1, 2 or 3')
    if any(n % (2**result['depth']) for n in result['patch_size_kji']):
        raise ValueError('patch_size_kji must be divisible by 2**depth')
    if not 2 <= result['base_filters'] <= 32 or result['batch_size'] > 16:
        raise ValueError('base_filters must be 2..32; batch_size must be 1..16')
    if any(not math.isfinite(x) or x <= 0 for x in result['target_spacing_mm']):
        raise ValueError('target_spacing_mm must contain positive finite spacing')
    for key in ('clip_min', 'clip_max', 'learning_rate', 'change_weight', 'threshold'):
        if not math.isfinite(result[key]):
            raise ValueError(f'{key} must be finite')
    if (result['clip_min'] >= result['clip_max'] or result['learning_rate'] <= 0
            or result['change_weight'] < 1 or not 0 < result['threshold'] < 1):
        raise ValueError('Invalid clipping, learning rate, change weight or threshold')
    result['clip_min'], result['clip_max'] = validate_clip_bounds(result['clip_min'], result['clip_max'])
    return result


def load_config(path, *, repo_root=None):
    root = Path(repo_root or ROOT).resolve()
    path = Path(path).expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       strict=True, empty_lines_in_values=False)
    parser.optionxform = str
    try:
        parser.read_string(path.read_text(encoding='utf-8-sig'))
    except configparser.Error as exc:
        raise ValueError(f'Invalid two-stage INI: {exc}') from exc
    required = {'experiment', 'data', 'split', 'model'}
    if parser.defaults() or not required <= set(parser.sections()) or set(parser.sections()) - required - {
            'model.segmentation', 'model.reconstruction'}:
        raise ValueError('Require [experiment], [data], [split], [model]; optional model task sections only')
    fields = {'experiment': {'schema_version', 'output_directory', 'scope'},
              'data': {'project', 'directory', 'write_change_masks'},
              'split': {'strategy', 'case_counts', 'seed'}}
    sections = {name: dict(parser[name]) for name in parser.sections()}
    for name, values in sections.items():
        if any('\n' in value or not value.strip() for value in values.values()):
            raise ValueError(f'[{name}] requires nonblank single-line values')
        if name in fields and set(values) != fields[name]:
            raise ValueError(f'[{name}] requires exactly {sorted(fields[name])}')
    exp, data, split = (sections[name] for name in ('experiment', 'data', 'split'))
    if exp['schema_version'] != SCHEMA or exp['scope'] != 'engineering':
        raise ValueError(f'Use schema_version={SCHEMA} and scope=engineering for this pilot')
    if split['strategy'] != 'holdout':
        raise ValueError('The two-stage pilot currently supports holdout; family folds remain a later extension')
    split['case_counts'] = [_integer(v.strip(), 'case_counts', 1) for v in split['case_counts'].split(',')]
    if len(split['case_counts']) != 3:
        raise ValueError('case_counts requires train,validation,test counts')
    split['seed'] = _integer(split['seed'], 'split.seed')
    data['write_change_masks'] = _boolean(data['write_change_masks'], 'write_change_masks')
    for values, key in ((exp, 'output_directory'), (data, 'directory'), (data, 'project')):
        location = Path(values[key]).expanduser()
        values[key] = str((location if location.is_absolute() else root/location).resolve())
    if Path(exp['output_directory']) == Path(data['directory']):
        raise ValueError('Experiment and data directories must differ')
    model = _model(sections['model'], MODEL_DEFAULTS)
    stage1 = _model(sections.get('model.segmentation', {}), model)
    stage2 = _model(sections.get('model.reconstruction', {}), model)
    if stage1['target_spacing_mm'] != stage2['target_spacing_mm']:
        raise ValueError('Both stages must use the same physical grid for chained evaluation')
    return {'experiment': exp, 'data': data, 'split': split, 'model': model,
            'stage1': stage1, 'stage2': stage2}


def prepare(config):
    from fakect_two_stage_data import prepare_dataset, inspect_dataset
    manifest = Path(config['data']['directory'])/'dataset-manifest.json'
    if manifest.exists():
        result = inspect_dataset(manifest, verify_payloads=True)
        _check_dataset_config(config, result)
        return result
    return prepare_dataset(config['data']['project'], config['data']['directory'],
                           write_changes=config['data']['write_change_masks'])


def _check_dataset_config(config, manifest):
    if manifest.get('project_sha256') != sha256(config['data']['project']):
        raise ValueError('Requested project differs from prepared dataset; use a new data revision')
    flags = ['changes_npz' in sample for sample in manifest['samples']]
    if not flags or any(flag != config['data']['write_change_masks'] for flag in flags):
        raise ValueError('write_change_masks differs from prepared dataset; use a new data revision')


def plan(config, *, verify_payloads=True):
    from fakect_two_stage_data import inspect_dataset
    manifest_path = Path(config['data']['directory'])/'dataset-manifest.json'
    if not manifest_path.is_file():
        return {'ready': False, 'errors': [f'Missing paired data manifest: {manifest_path}'],
                'next_stage': 'prepare', 'scope': 'engineering'}
    manifest = inspect_dataset(manifest_path, verify_payloads=verify_payloads)
    _check_dataset_config(config, manifest)
    rows = manifest['samples']
    evaluation = {'strategy': 'holdout', 'case_counts': config['split']['case_counts'],
                  'seeds': [config['model']['seed']], 'threshold': config['model']['threshold']}
    design = build_evaluation_design(rows, evaluation, config['split']['seed'])
    assignment = design['folds'][0]['family_assignment']
    samples = [{**row, 'split': assignment[row['anatomy_family']]} for row in rows]
    grids = {}
    for row in samples:
        geometry = row['geometry']
        grids[row['case_id']] = {
            'native_shape_kji': geometry['crop_shape_kji'],
            'native_spacing_ijk_mm': geometry['spacing_ijk_mm'],
            'model_shape_kji': list(output_shape(geometry['crop_shape_kji'],
                geometry['spacing_ijk_mm'], config['stage1']['target_spacing_mm'])),
            'model_spacing_ijk_mm': config['stage1']['target_spacing_mm'],
        }
    counts = {name: sum(s['split'] == name for s in samples) for name in ('train', 'validation', 'test')}
    # A byte-identical baseline cannot appear on both sides of a held-out split.
    known = {}
    for row in samples:
        digest = row.get('before_sha256') or row.get('before_mask_sha256')
        if digest:
            key = (digest, tuple(row.get('geometry', {}).get('spacing_ijk_mm', [])))
            previous = known.setdefault(key, row['split'])
            if previous != row['split']:
                raise ValueError('Identical baseline anatomy crosses a split; merge its families first')
    return {'schema_version': 'fakect.two-stage-plan/1', 'ready': True, 'scope': 'engineering',
            'family_status': manifest.get('family_status', 'provisional'),
            'dataset_manifest': str(manifest_path), 'dataset_manifest_sha256': sha256(manifest_path),
            'sample_count': len(samples), 'split_counts': counts, 'design': design,
            'stage_settings': {'model1': config['stage1'], 'model2': config['stage2']},
            'model_grids': grids,
            'samples': samples, 'dataset_metadata': {k: v for k, v in manifest.items() if k != 'samples'},
            'warnings': ['Anatomy families are provisional engineering groups, not verified independent patients.',
                         'Targets cover the reviewed selected-aorta crop, not validated whole-scan anatomy.',
                         'Images are attenuation-copy proxies; clinical generalization is not established.']}


def _code_hashes():
    return {name: sha256(ROOT/'src'/name) for name in CODE_FILES}


def freeze(config, input_bytes):
    result = plan(config)
    if not result['ready']:
        raise ValueError('; '.join(result['errors']))
    directory = Path(config['experiment']['output_directory'])
    lock = {**result, 'schema_version': 'fakect.two-stage-lock/1', 'config': config,
            'input_sha256': hashlib.sha256(input_bytes).hexdigest(), 'code_sha256': _code_hashes()}
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'input.ini').write_bytes(input_bytes)
    _write(directory/'experiment-lock.json', lock)
    (directory/'experiment-lock.sha256').write_text(sha256(directory/'experiment-lock.json')+'\n')
    return lock


def inspect_lock(config, input_bytes):
    directory = Path(config['experiment']['output_directory'])
    path = directory/'experiment-lock.json'
    if sha256(path) != (directory/'experiment-lock.sha256').read_text().strip():
        raise ValueError('Experiment lock checksum changed')
    lock = json.loads(path.read_text())
    if (lock['config'] != config or lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest()
            or sha256(directory/'input.ini') != lock['input_sha256']):
        raise ValueError('INI differs from frozen experiment; use a new output_directory')
    if lock['code_sha256'] != _code_hashes():
        raise ValueError('Model implementation changed; use a new experiment revision')
    if sha256(lock['dataset_manifest']) != lock['dataset_manifest_sha256']:
        raise ValueError('Paired dataset manifest changed after planning')
    return lock


def fit(config, input_bytes, stage):
    from fakect_two_stage_learning import fit_stage, settings_for, inspect_manifest
    lock = inspect_lock(config, input_bytes)
    if stage not in (1, 2):
        raise ValueError('Model stage must be 1 or 2')
    directory = Path(config['experiment']['output_directory'])/f'model{stage}'
    metadata_path = directory/'model-metadata.json'
    if metadata_path.exists():
        if sha256(metadata_path) != metadata_path.with_name('model-metadata.json.sha256').read_text().strip():
            raise ValueError('Completed model metadata checksum mismatch')
        metadata = json.loads(metadata_path.read_text())
        _, fingerprint = inspect_manifest(lock)
        if (metadata.get('stage') != stage or metadata.get('settings') != settings_for(config, stage)
                or metadata.get('manifest_fingerprint') != fingerprint
                or metadata.get('model_sha256') != sha256(directory/'model.keras')
                or metadata.get('source_sha256') != sha256(ROOT/'src'/'fakect_two_stage_learning.py')
                or metadata.get('history_sha256') != sha256(directory/'history.json')
                or metadata.get('test_payloads_loaded') is not False
                or metadata.get('epochs_completed') != settings_for(config, stage)['epochs']):
            raise ValueError('Completed model does not match this experiment')
        return {**metadata, 'reused': True}
    return fit_stage(config, lock, stage, directory)


def _evaluate_one(config, lock, stage, directory):
    from fakect_two_stage_learning import evaluate_stage, inspect_manifest, settings_for
    output = directory/f'test-model{stage}'
    path = output/'evaluation.json'
    if path.exists():
        if sha256(path) != path.with_name('evaluation.json.sha256').read_text().strip():
            raise ValueError('Saved evaluation checksum mismatch')
        result = json.loads(path.read_text())
        _, fingerprint = inspect_manifest(lock)
        expected = {row['sample_id'] for row in lock['samples'] if row['split'] == 'test'}
        records = result.get('records', [])
        if (result.get('stage') != stage or result.get('manifest_fingerprint') != fingerprint
                or result.get('model_sha256') != sha256(directory/f'model{stage}'/'model.keras')
                or (stage == 2 and result.get('stage1_model_sha256') != sha256(directory/'model1'/'model.keras'))
                or result.get('threshold') != settings_for(config, stage)['threshold']
                or result.get('target_spacing_ijk_mm') != settings_for(config, stage)['target_spacing_mm']
                or result.get('test_samples') != len(expected)
                or len(records) != len(expected) or {r['sample_id'] for r in records} != expected):
            raise ValueError('Saved test results differ from frozen model/partition')
        for row in records:
            if 'prediction_file' in row:
                prediction = output/row['prediction_file']
                if prediction.parent.resolve() != output.resolve() or row.get('prediction_sha256') != sha256(prediction):
                    raise ValueError('Saved prediction checksum mismatch')
        return result
    return evaluate_stage(config, lock, stage, directory/f'model{stage}', output,
                          **({'stage1_model_path': directory/'model1'} if stage == 2 else {}))


def evaluate(config, input_bytes):
    lock = inspect_lock(config, input_bytes)
    directory = Path(config['experiment']['output_directory'])
    result1 = _evaluate_one(config, lock, 1, directory)
    result2 = _evaluate_one(config, lock, 2, directory)
    result = {'schema_version': 'fakect.two-stage-results/1', 'scope': 'engineering',
              'dataset_manifest_sha256': lock['dataset_manifest_sha256'],
              'split_counts': lock['split_counts'], 'design': lock['design'],
              'stage1': result1, 'stage2': result2, 'warnings': lock['warnings']}
    _write(directory/'results.json', result)
    return result
