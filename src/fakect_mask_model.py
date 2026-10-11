"""Reference-shape learning from supplied binary masks, with no CT model.

This runner consumes frozen paired masks. The current cohort is a reviewed
ROI-selected aorta lineage; using whole-thoracic masks requires a new dataset
and model revision. Atlas extraction belongs upstream of this experiment.
"""
import configparser
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from fakect_evaluation_design import build_evaluation_design
from fakect_model_resampling import output_shape
from fakect_two_stage_experiment import MODEL_DEFAULTS, _integer, _model, sha256

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.mask-model-experiment/1'
CODE_FILES = ('fakect_mask_model.py', 'fakect_two_stage_experiment.py',
              'fakect_two_stage_data.py', 'fakect_two_stage_learning.py',
              'fakect_model_resampling.py', 'fakect_evaluation_design.py',
              'fakect_segmentation.py', 'fakect_cohort_registry.py')


def _write(path, document):
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')


def load_config(path, *, repo_root=None):
    root = Path(repo_root or ROOT).resolve()
    parser = configparser.ConfigParser(interpolation=None, strict=True,
        inline_comment_prefixes=('#',), empty_lines_in_values=False)
    parser.optionxform = str
    try:
        parser.read_string(Path(path).read_text(encoding='utf-8-sig'))
    except configparser.Error as exc:
        raise ValueError(f'Invalid mask-model INI: {exc}') from exc
    fields = {'experiment': {'schema_version', 'scope', 'output_directory'},
              'data': {'manifest', 'target_scope'},
              'split': {'strategy', 'case_counts', 'seed'}, 'model': None}
    if parser.defaults() or set(parser.sections()) != set(fields):
        raise ValueError('Require only [experiment], [data], [split], [model]')
    sections = {name: dict(parser[name]) for name in fields}
    for name, values in sections.items():
        if any('\n' in value or not value.strip() for value in values.values()):
            raise ValueError(f'[{name}] requires nonblank single-line values')
        if fields[name] is not None and set(values) != fields[name]:
            raise ValueError(f'[{name}] requires exactly {sorted(fields[name])}')
    exp, data, split = (sections[name] for name in ('experiment', 'data', 'split'))
    if exp['schema_version'] != SCHEMA or exp['scope'] != 'engineering':
        raise ValueError(f'Use schema_version={SCHEMA} and scope=engineering')
    if data['target_scope'] != 'selected_lineage':
        raise ValueError('Current paired adapter supports target_scope=selected_lineage only; '
                         'whole-thoracic masks need a new prepared dataset adapter')
    if split['strategy'] != 'holdout':
        raise ValueError('This mask-model runner currently supports holdout')
    split['case_counts'] = [_integer(v.strip(), 'case_counts', 1)
                           for v in split['case_counts'].split(',')]
    if len(split['case_counts']) != 3:
        raise ValueError('case_counts requires train,validation,test counts')
    split['seed'] = _integer(split['seed'], 'split.seed')
    for values, key in ((exp, 'output_directory'), (data, 'manifest')):
        value = Path(values[key]).expanduser()
        values[key] = str((value if value.is_absolute() else root/value).resolve())
    if Path(exp['output_directory']) == Path(data['manifest']).parent:
        raise ValueError('Experiment output and dataset directories must differ')
    model = _model(sections['model'], MODEL_DEFAULTS)
    return {'experiment': exp, 'data': data, 'split': split, 'model': model}


def plan(config, *, verify_payloads=True):
    from fakect_two_stage_data import inspect_dataset
    from fakect_two_stage_learning import inspect_manifest, settings_for
    path = Path(config['data']['manifest'])
    if not path.is_file():
        return {'ready': False, 'errors': [f'Missing prepared paired manifest: {path}'],
                'scope': 'engineering'}
    dataset = inspect_dataset(path, verify_payloads=verify_payloads)
    settings = settings_for(config, 2)
    design = build_evaluation_design(dataset['samples'],
        {'strategy': 'holdout', 'case_counts': config['split']['case_counts'],
         'seeds': [settings['seed']], 'threshold': settings['threshold']},
        config['split']['seed'])
    assignment = design['folds'][0]['family_assignment']
    samples = [{**row, 'split': assignment[row['anatomy_family']]}
               for row in dataset['samples']]
    # Uses the same family, case and shared-reference leakage checks as fitting.
    samples, fingerprint = inspect_manifest({'samples': samples})
    grids = {}
    for row in samples:
        geometry = row['geometry']
        grid = {'native_shape_kji': geometry['crop_shape_kji'],
                'native_spacing_ijk_mm': geometry['spacing_ijk_mm'],
                'model_shape_kji': list(output_shape(geometry['crop_shape_kji'],
                    geometry['spacing_ijk_mm'], settings['target_spacing_mm'])),
                'model_spacing_ijk_mm': settings['target_spacing_mm']}
        if row['case_id'] in grids and grids[row['case_id']] != grid:
            raise ValueError('A case has inconsistent sample geometries')
        grids[row['case_id']] = grid
    return {'schema_version': 'fakect.mask-model-plan/1', 'ready': True,
            'scope': 'engineering', 'target_scope': config['data']['target_scope'],
            'input': 'supplied observed binary mask; no learned segmentation model',
            'family_status': dataset.get('family_status', 'provisional'),
            'dataset_manifest': str(path), 'dataset_manifest_sha256': sha256(path),
            'manifest_fingerprint': fingerprint, 'sample_count': len(samples),
            'split_counts': {part: sum(row['split'] == part for row in samples)
                             for part in ('train', 'validation', 'test')},
            'design': design, 'model_settings': settings, 'model_grids': grids,
            'samples': samples,
            'warnings': ['Prepared masks cover reviewed ROI-selected aorta lineage, not the whole thoracic aorta.',
                         'Family groups are provisional; every related anatomy and synthetic sibling stays together.',
                         'A reconstructed reference is a learned shape estimate, not a clinical diagnosis.',
                         'Whole-thoracic coverage requires a new paired dataset and retraining.']}


def _code_hashes():
    return {name: sha256(ROOT/'src'/name) for name in CODE_FILES}


def freeze(config, input_bytes):
    result = plan(config)
    if not result['ready']:
        raise ValueError('; '.join(result['errors']))
    directory = Path(config['experiment']['output_directory'])
    lock = {**result, 'schema_version': 'fakect.mask-model-lock/1',
            'config': deepcopy(config), 'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
            'code_sha256': _code_hashes()}
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'input.ini').write_bytes(input_bytes)
    (directory/'source').mkdir()
    for name in CODE_FILES:
        (directory/'source'/name).write_bytes((ROOT/'src'/name).read_bytes())
    _write(directory/'experiment-lock.json', lock)
    (directory/'experiment-lock.sha256').write_text(sha256(directory/'experiment-lock.json')+'\n')
    return lock


def inspect_lock(config, input_bytes):
    directory = Path(config['experiment']['output_directory'])
    path = directory/'experiment-lock.json'
    if sha256(path) != (directory/'experiment-lock.sha256').read_text().strip():
        raise ValueError('Experiment lock checksum changed')
    lock = json.loads(path.read_text())
    if lock.get('schema_version') != 'fakect.mask-model-lock/1':
        raise ValueError('Expected a mask-model experiment lock')
    if (lock['config'] != config or lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest()
            or sha256(directory/'input.ini') != lock['input_sha256']):
        raise ValueError('INI differs from frozen experiment; use a new output_directory')
    if lock['code_sha256'] != _code_hashes():
        raise ValueError('Model implementation changed; use a new experiment revision')
    if any(sha256(directory/'source'/name) != digest for name, digest in lock['code_sha256'].items()):
        raise ValueError('Frozen source snapshot checksum changed')
    if sha256(lock['dataset_manifest']) != lock['dataset_manifest_sha256']:
        raise ValueError('Paired dataset manifest changed after planning')
    return lock


def _completed_model(config, lock):
    """Check completed model provenance without loading TensorFlow or payloads."""
    from fakect_two_stage_learning import settings_for, inspect_manifest
    directory = Path(config['experiment']['output_directory'])/'model2'
    path = directory/'model-metadata.json'
    if sha256(path) != path.with_name(path.name+'.sha256').read_text().strip():
        raise ValueError('Completed model metadata checksum mismatch')
    metadata = json.loads(path.read_text())
    _, fingerprint = inspect_manifest(lock)
    settings = settings_for(config, 2)
    if (metadata.get('stage') != 2 or metadata.get('settings') != settings
            or metadata.get('manifest_fingerprint') != fingerprint
            or metadata.get('model_sha256') != sha256(directory/'model.keras')
            or metadata.get('source_sha256') != sha256(ROOT/'src'/'fakect_two_stage_learning.py')
            or metadata.get('history_sha256') != sha256(directory/'history.json')
            or metadata.get('test_payloads_loaded') is not False
            or metadata.get('epochs_completed') != settings['epochs']):
        raise ValueError('Completed model does not match this experiment')
    return metadata


def fit(config, input_bytes):
    from fakect_two_stage_learning import fit_stage
    lock = inspect_lock(config, input_bytes)
    directory = Path(config['experiment']['output_directory'])/'model2'
    if (directory/'model-metadata.json').exists():
        return {**_completed_model(config, lock), 'reused': True}
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Partial model output exists; inspect it and use a new experiment revision')
    return fit_stage(config, lock, 2, directory)


def evaluate(config, input_bytes):
    from fakect_two_stage_learning import evaluate_stage, inspect_manifest, settings_for
    lock = inspect_lock(config, input_bytes)
    metadata = _completed_model(config, lock)
    directory = Path(config['experiment']['output_directory'])
    output = directory/'test-model2'
    path = output/'evaluation.json'
    if path.exists():
        if sha256(path) != path.with_name(path.name+'.sha256').read_text().strip():
            raise ValueError('Saved evaluation checksum mismatch')
        result = json.loads(path.read_text())
        _, fingerprint = inspect_manifest(lock)
        settings = settings_for(config, 2)
        expected = {row['sample_id'] for row in lock['samples'] if row['split'] == 'test'}
        records = result.get('records', [])
        if (result.get('stage') != 2 or result.get('manifest_fingerprint') != fingerprint
                or result.get('model_sha256') != metadata['model_sha256']
                or result.get('stage1_model_sha256') is not None
                or result.get('threshold') != settings['threshold']
                or result.get('target_spacing_ijk_mm') != settings['target_spacing_mm']
                or result.get('test_samples') != len(expected) or len(records) != len(expected)
                or {row['sample_id'] for row in records} != expected):
            raise ValueError('Saved test results differ from frozen mask model/partition')
        for row in records:
            if settings['save_predictions'] and 'prediction_file' not in row:
                raise ValueError('Saved evaluation is missing required predictions')
            if 'prediction_file' in row:
                prediction = output/row['prediction_file']
                if prediction.parent.resolve() != output.resolve() or row.get('prediction_sha256') != sha256(prediction):
                    raise ValueError('Saved prediction checksum mismatch')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Partial evaluation exists; inspect it before a new revision')
        result = evaluate_stage(config, lock, 2, directory/'model2', output,
                                stage1_model_path=None)
    document = {'schema_version': 'fakect.mask-model-results/1', 'scope': 'engineering',
                'input': lock['input'], 'target_scope': lock['target_scope'],
                'dataset_manifest_sha256': lock['dataset_manifest_sha256'],
                'split_counts': lock['split_counts'], 'design': lock['design'],
                'stage2': result, 'warnings': lock['warnings']}
    _write(directory/'results.json', document)
    return document
