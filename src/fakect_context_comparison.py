"""Freeze a matched baseline-only versus synthetic-sibling experiment."""
import configparser
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import fakect_context_experiment as original
from fakect_context_comparison_learning import ARMS, settings_for, training_cases
from fakect_context_learning import settings_for as model_settings, _signed_write
from fakect_mask_pair_preview import sha256

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.context-comparison/1'
CODE_FILES = tuple('src/'+name for name in dict.fromkeys(original.CODE_FILES + (
    'fakect_context_comparison.py', 'fakect_context_comparison_learning.py',
    'fakect_context_cached_evaluation.py'))) + (
    'scripts/compare_context_aorta.py', 'scripts/report_context_comparison.py',
    'scripts/slurm/fakect_context_comparison_gpu.sh')


def load_config(path, *, repo_root=None):
    root = Path(repo_root or ROOT).resolve()
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
        strict=True, empty_lines_in_values=False)
    parser.read_string(Path(path).read_text(encoding='utf-8-sig'))
    fields = {'experiment': {'schema_version', 'scope', 'output_directory'},
              'data': {'manifest', 'allow_provisional_anatomy'},
              'split': {'strategy', 'case_counts', 'seed'}, 'comparison': None, 'model': None}
    if parser.defaults() or set(parser.sections()) != set(fields):
        raise ValueError('Require exactly [experiment], [data], [split], [comparison], [model]')
    config = {key: dict(parser[key]) for key in fields}
    for section, keys in fields.items():
        if keys is not None and set(config[section]) != keys:
            raise ValueError(f'[{section}] requires exactly {sorted(keys)}')
    experiment, data, split = (config[key] for key in ('experiment', 'data', 'split'))
    if experiment['schema_version'] != SCHEMA or experiment['scope'] != 'engineering':
        raise ValueError(f'Expected {SCHEMA} with scope=engineering')
    for section, key in ((experiment, 'output_directory'), (data, 'manifest')):
        target = Path(section[key]).expanduser()
        section[key] = str((target if target.is_absolute() else root/target).resolve())
    if Path(experiment['output_directory']) == Path(data['manifest']).parent:
        raise ValueError('Dataset and experiment outputs must differ')
    data['allow_provisional_anatomy'] = parser.getboolean('data', 'allow_provisional_anatomy')
    if split['strategy'] != 'holdout':
        raise ValueError('This controlled comparison requires a family-disjoint holdout')
    split['case_counts'] = [int(value.strip()) for value in split['case_counts'].split(',')]
    split['seed'] = int(split['seed'])
    if len(split['case_counts']) != 3 or min(split['case_counts']) < 1 or split['seed'] < 0:
        raise ValueError('Require positive train,validation,test case counts and nonnegative seed')
    ignored = {'patches_per_volume', 'diagnostic_steps_per_epoch', 'diagnostic_sample_id'} & set(config['model'])
    if ignored:
        raise ValueError(f'Comparison does not use model settings {sorted(ignored)}; use [comparison] controls')
    config['model'] = model_settings(config)
    config['comparison'] = settings_for(config)
    if config['model']['max_train_batches_per_epoch']:
        raise ValueError('Use comparison.updates_per_epoch for both arms')
    if config['model']['save_predictions']:
        raise ValueError('Comparison currently streams metrics; save_predictions must be false')
    return config


def plan(config, *, verify_payloads=True):
    adapter = deepcopy(config)
    adapter['experiment']['mode'] = 'cohort'
    base = original.plan(adapter, verify_payloads=verify_payloads)
    if not base.get('ready'):
        return base
    comparison = settings_for(config)
    cases = training_cases(base, comparison)
    arms = {arm: {'training_samples': [row['sample_id'] for row in base['samples']
        if row['split'] == 'train' and (arm != 'baseline_only' or row['baseline'])],
        'updates_per_epoch': comparison['updates_per_epoch'],
        'total_updates': comparison['updates_per_epoch']*config['model']['epochs']}
        for arm in ARMS}
    return {**base, 'schema_version': 'fakect.context-comparison-plan/1', 'arms': arms,
        'comparison': comparison, 'training_case_count': len(cases),
        'updates_per_case_per_epoch': comparison['updates_per_epoch']//len(cases),
        'matched_controls': ['family split', 'initial weights', 'optimizer update count',
            'case order', 'baseline-derived physical patch origins', 'augmentation seeds',
            'architecture and physical spacing', 'whole-FOV validation/test protocol'],
        'validation_epochs': [epoch for epoch in range(1, config['model']['epochs']+1)
            if epoch % comparison['validation_every_epochs'] == 0 or epoch == config['model']['epochs']],
        'interpretation': 'Comparison of synthetic unseen-family segmentation. Family identities and anatomical extent remain provisional; no clinical-generalization claim.'}


def freeze(config, input_bytes):
    planned = plan(config)
    if not planned['ready']:
        raise ValueError('; '.join(planned['errors']))
    # Read sources before creating an immutable output, so missing code leaves no partial lock.
    sources = {name: (ROOT/name).read_bytes() for name in CODE_FILES}
    output = Path(config['experiment']['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'input.ini').write_bytes(input_bytes)
    hashes = {}
    for name, content in sources.items():
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    lock = {**planned, 'schema_version': 'fakect.context-comparison-lock/1',
        'config': deepcopy(config), 'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
        'code_sha256': hashes}
    _signed_write(output/'experiment-lock.json', lock)
    return lock


def inspect_lock(config, input_bytes):
    output = Path(config['experiment']['output_directory'])
    path = output/'experiment-lock.json'
    if sha256(path) != Path(str(path)+'.sha256').read_text().strip():
        raise ValueError('Comparison lock checksum changed')
    lock = json.loads(path.read_text())
    if (lock.get('schema_version') != 'fakect.context-comparison-lock/1' or
        lock['config'] != config or lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest()
        or sha256(output/'input.ini') != lock['input_sha256']):
        raise ValueError('Comparison INI changed; use a fresh revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Comparison source inventory changed; use a fresh revision')
    for name, digest in lock['code_sha256'].items():
        if sha256(ROOT/name) != digest or sha256(output/'source'/name) != digest:
            raise ValueError(f'Comparison implementation changed: {name}; use a fresh revision')
    if sha256(lock['dataset_manifest']) != lock['dataset_manifest_sha256']:
        raise ValueError('Comparison dataset changed after planning')
    return lock


def fit(config, input_bytes, arm):
    from fakect_context_comparison_learning import fit_arm
    lock = inspect_lock(config, input_bytes)
    if arm not in ARMS:
        raise ValueError(f'Choose an arm from {ARMS}')
    return fit_arm(config, lock, arm, Path(config['experiment']['output_directory'])/arm/'model')


def evaluate(config, input_bytes, arm):
    from fakect_context_comparison_learning import evaluate_arm
    lock = inspect_lock(config, input_bytes)
    if arm not in ARMS:
        raise ValueError(f'Choose an arm from {ARMS}')
    output = Path(config['experiment']['output_directory'])/arm
    return evaluate_arm(config, lock, arm, output/'model', output/'evaluation')
