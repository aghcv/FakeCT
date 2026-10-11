"""Immutable focused CoA classifier and training-only memorization diagnostic.

The reviewed image-only locator defines the high-resolution crops. This module
inherits its dataset, normalization and family assignment without changing any
earlier frozen source. The classifier consumes attenuation and validity only.
"""
from copy import deepcopy
import configparser
import hashlib
import math
from pathlib import Path

import numpy as np

import fakect_coa_classifier_experiment as classifier
from fakect_multiscale_experiment import ROOT, SCORE, sha256, checked_json, write_json

SCHEMA = 'fakect.coa-focused/1'
FIELDS = {
    'experiment': {'schema_version', 'output_directory'},
    'locator': {'experiment_directory', 'review_directory'},
    'diagnostic': {'case_ids', 'epochs', 'learning_rate', 'cache_limit_mb',
                   'minimum_probability_gap', 'maximum_bce'},
    'training': {'epochs', 'learning_rate', 'seed', 'host_memory_limit_mb',
                 'gpu_memory_limit_mb', 'require_gpu'},
    'model': {'base_filters', 'regression_weight'},
}
CODE_FILES = classifier.CODE_FILES + (
    'src/fakect_coa_focused_experiment.py', 'src/fakect_coa_focused_network.py',
    'src/fakect_coa_focused_learning.py', 'scripts/research/train_coa_focused.py',
    'scripts/research/slurm/fakect_coa_focused_gpu.sh',
)


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(FIELDS):
        raise ValueError('Expected exactly ' + ', '.join('['+s+']' for s in FIELDS))
    c = {s: dict(parser[s]) for s in FIELDS}
    if any(set(c[s]) != fields for s, fields in FIELDS.items()):
        raise ValueError('Unsupported focused classifier input fields')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported focused classifier schema')
    for section, key in [('experiment', 'output_directory'), ('locator', 'experiment_directory'),
                         ('locator', 'review_directory')]:
        value = Path(c[section][key]).expanduser()
        c[section][key] = str((value if value.is_absolute() else ROOT/value).resolve())
    for section, key, low, high in [
            ('diagnostic', 'epochs', 1, 100), ('diagnostic', 'cache_limit_mb', 256, 4096),
            ('training', 'epochs', 1, 100), ('training', 'seed', 0, 2**31-1),
            ('training', 'host_memory_limit_mb', 512, 12288),
            ('training', 'gpu_memory_limit_mb', 512, 12288), ('model', 'base_filters', 2, 8)]:
        c[section][key] = int(c[section][key])
        if not low <= c[section][key] <= high:
            raise ValueError(f'{section}.{key} outside [{low},{high}]')
    for section, key in [('diagnostic', 'learning_rate'), ('training', 'learning_rate'),
                         ('diagnostic', 'minimum_probability_gap'), ('diagnostic', 'maximum_bce'),
                         ('model', 'regression_weight')]:
        value = float(c[section][key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{section}.{key} must be positive and finite')
        c[section][key] = value
    if not 0 < c['diagnostic']['minimum_probability_gap'] < 1 or not 0 < c['diagnostic']['maximum_bce'] < math.log(2):
        raise ValueError('Diagnostic gates must require meaningful probability separation and BCE below chance')
    c['diagnostic']['case_ids'] = [s.strip() for s in c['diagnostic']['case_ids'].split(',')]
    ids = c['diagnostic']['case_ids']
    if not 1 <= len(ids) <= 4 or any(not s for s in ids) or len(ids) != len(set(ids)):
        raise ValueError('diagnostic.case_ids needs one to four unique case IDs')
    c['training']['require_gpu'] = parser.getboolean('training', 'require_gpu')
    if c['diagnostic']['cache_limit_mb'] > c['training']['host_memory_limit_mb']//2:
        raise ValueError('Diagnostic cache must leave at least half the host budget for TensorFlow and readers')
    out = Path(c['experiment']['output_directory'])
    if any(out == Path(value) or Path(value) in out.parents for value in c['locator'].values()):
        raise ValueError('Focused output must be separate from the locator and review')
    return c


def select_diagnostic(c, contract):
    """Select matched edited-negative/strong-positive pairs from training only."""
    rows, proposals = contract['samples'], contract['proposals']
    ids = c['diagnostic']['case_ids']
    if not ids or any(not s for s in ids) or len(ids) != len(set(ids)):
        raise ValueError('Diagnostic case IDs must be nonempty and unique')
    selected = []
    for case_id in c['diagnostic']['case_ids']:
        case = [r for r in rows if r['case_id'] == case_id]
        if not case or any(r['split'] != 'train' for r in case):
            raise ValueError('Diagnostic case must belong exclusively to training: '+case_id)
        pair = []
        for role in ('edited_negative', 'higher_positive'):
            matches = [r for r in case if r['pilot_role'] == role]
            if len(matches) != 1:
                raise ValueError('Diagnostic needs exactly one '+role+' for '+case_id)
            pair.append(matches[0])
        negative, positive = pair
        if (negative['baseline'] or positive['baseline'] or
                negative['annotation']['coa_benchmark_positive'] is not False or
                positive['annotation']['coa_benchmark_positive'] is not True or
                not 0 <= negative['annotation'][SCORE] < .2 <= positive['annotation'][SCORE] <= 1):
            raise ValueError('Diagnostic labels must straddle the fixed 20% threshold')
        if negative['anatomy_family'] != positive['anatomy_family']:
            raise ValueError('Diagnostic siblings must belong to the same anatomy family')
        if ('pilot_companion_parameters' not in negative or
                negative['pilot_companion_parameters'] != positive.get('pilot_companion_parameters')):
            raise ValueError('Diagnostic companion edits must be identical within each case')
        without_coa = lambda row: {key: value for key, value in row['parameters'].items() if key != 'coa_narrow'}
        if without_coa(negative) != without_coa(positive):
            raise ValueError('Diagnostic may vary only CoA parameters within each case')
        a, b = (proposals[r['sample_id']] for r in pair)
        if not np.array_equal(a['origins_kji'], b['origins_kji']):
            raise ValueError('Diagnostic siblings require identical reviewed native crop origins')
        selected.extend(pair)
    return selected


def plan(c):
    contract = classifier.source_contract(c)
    rows = select_diagnostic(c, contract)
    detail = contract['parent_config']['input.detail']
    bytes_needed = len(rows)*detail['proposals']*math.prod(detail['shape_kji'])*2*4
    if bytes_needed > c['diagnostic']['cache_limit_mb']*1024**2:
        raise ValueError('Diagnostic input panel exceeds the bounded RAM cache; select fewer cases')
    return {'schema_version': SCHEMA, 'ready': True, 'config': deepcopy(c), **contract,
            'scope': 'focused_classifier_training_diagnostic_then_development_pilot',
            'diagnostic_sample_ids': [r['sample_id'] for r in rows],
            'diagnostic_counts': {'samples': len(rows), 'positive': len(rows)//2,
                                  'cases': len(rows)//2, 'input_cache_bytes': bytes_needed},
            'training_contract': {
                'locator_frozen': True, 'predictor_inputs': 'detail attenuation and acquisition validity only',
                'feature_stride': 4, 'feature_spacing_ijk_mm': [4*v for v in detail['spacing_ijk_mm']],
                'diagnostic_selection': 'matched edited negative and higher positive from training cases; companion edits and image-derived crop origins equal',
                'diagnostic_checkpoint_selection': 'minimum training BCE; per-case learning gate; no validation or test prediction',
                'pilot_checkpoint_selection': 'minimum validation BCE, fresh initialization after diagnostic passes',
                'reserved_test_evaluated': False,
                'storage': 'shared native attenuation plus immutable edit crops; bounded transient RAM inputs',
            }}


def freeze(c, input_bytes):
    document = plan(c)
    sources = {name: (ROOT/name).read_bytes() for name in CODE_FILES}
    out = Path(c['experiment']['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'INCOMPLETE').touch()
    (out/'input.ini').write_bytes(input_bytes)
    hashes = {}
    for name, content in sources.items():
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    document.update(code_sha256=hashes, input_sha256=hashlib.sha256(input_bytes).hexdigest())
    write_json(out/'experiment-lock.json', document)
    (out/'INCOMPLETE').unlink()
    return document


def inspect_lock(c, input_bytes):
    out = Path(c['experiment']['output_directory'])
    lock = checked_json(out/'experiment-lock.json')
    if (lock['schema_version'] != SCHEMA or lock['config'] != c or
            lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest() or
            sha256(out/'input.ini') != lock['input_sha256']):
        raise ValueError('Focused input changed; use a new experiment revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Focused source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if sha256(ROOT/name) != digest or sha256(out/'source'/name) != digest:
            raise ValueError('Pinned focused source changed: '+name)
    for name, digest in lock['dependency_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Pinned focused dependency changed: '+name)
    current = plan(c)
    if any(lock.get(key) != value for key, value in current.items()):
        raise ValueError('Focused locator, sample or geometry contract changed')
    return lock


def runtime_config(c, lock):
    value = deepcopy(lock['parent_config'])
    value['experiment']['output_directory'] = c['experiment']['output_directory']
    t = c['training']
    value['model'].update(learning_rate=t['learning_rate'], seed=t['seed'], base_filters=c['model']['base_filters'])
    value['pilot'].update(classifier_epochs=t['epochs'], require_gpu=t['require_gpu'],
                          gpu_memory_limit_mb=t['gpu_memory_limit_mb'])
    value['training'].update(host_memory_limit_mb=t['host_memory_limit_mb'], maximum_traces=1)
    return value


def require_smoke(c, lock):
    """The revised architecture requires its own production-size resource check."""
    root = Path(c['experiment']['output_directory'])
    result = checked_json(root/'runtime-smoke.json')
    detail = lock['parent_config']['input.detail']
    if (result.get('source_lock_sha256') != sha256(root/'experiment-lock.json') or
            result.get('status') != 'passed' or result.get('checkpoint_parity') is not True or
            result.get('detail_shape_kji') != detail['shape_kji'] or result.get('proposals') != detail['proposals'] or
            result.get('traces') != {'train': 1, 'predict': 1} or
            (c['training']['require_gpu'] and result.get('gpu_visible') is not True)):
        raise ValueError('Focused classifier requires its own passing full-size GPU smoke check')
    return result


def require_diagnostic(c, lock):
    """Recalculate learning gates on the recorded matched training panel."""
    from fakect_coa_classifier_report import summarize
    root = Path(c['experiment']['output_directory'])
    folder = root/'diagnostic'
    if (folder/'failure.json').exists() or (folder/'INCOMPLETE').exists():
        raise ValueError('Failed or incomplete focused diagnostic cannot authorize pilot training')
    result = checked_json(folder/'results.json')
    predictions = checked_json(folder/'classifier-predictions.json')
    rows = result['predictions']
    selected = {r['sample_id']: r for r in select_diagnostic(c, lock)}
    if (result.get('stage') != 'diagnostic' or result.get('status') != 'diagnostic_passed' or
            result.get('source_lock_sha256') != sha256(root/'experiment-lock.json') or
            result.get('test_evaluated') is not False or result.get('locator_unchanged') is not True or
            result.get('learning_gate', {}).get('passed') is not True or
            result.get('selected_checkpoint_prediction_parity', {}).get('passed') is not True or
            len(rows) != len(selected) or {r['sample_id'] for r in rows} != set(selected) or
            predictions.get('predictions') != rows):
        raise ValueError('The focused training diagnostic must pass before the development pilot')
    for row in rows:
        expected = selected[row['sample_id']]
        if (row.get('split') != 'train' or row.get('case_id') != expected['case_id'] or
                row.get('family') != expected['anatomy_family'] or row.get('role') != expected['pilot_role'] or
                row['label'] != int(expected['annotation']['coa_benchmark_positive']) or
                row['score_label'] != expected['annotation'][SCORE]):
            raise ValueError('Diagnostic predictions do not match their frozen training labels')
    for case_id in c['diagnostic']['case_ids']:
        pair = [r for r in rows if r['case_id'] == case_id]
        metrics = summarize(pair)
        if (metrics['sensitivity'] != 1 or metrics['specificity'] != 1 or
                metrics['bce'] > c['diagnostic']['maximum_bce'] or
                metrics['positive_probability_mean']-metrics['negative_probability_mean'] < c['diagnostic']['minimum_probability_gap']):
            raise ValueError('Diagnostic learning gate failed for '+case_id)
    model = folder/'model/classifier.keras'
    digest = sha256(model)
    if (predictions.get('checkpoint_sha256') != digest or
            result.get('checkpoint_sha256', {}).get('classifier.keras', digest) != digest or
            predictions.get('selected_epoch') != result.get('selected_epoch') or
            predictions.get('test_evaluated') is not False):
        raise ValueError('Diagnostic checkpoint does not match its predictions')
    paths = [folder/'results.json', folder/'classifier-predictions.json', model]
    return {'source_lock_sha256': sha256(root/'experiment-lock.json'),
            'dependency_sha256': {str(path): sha256(path) for path in paths},
            'scope': 'training-only feasibility gate; no generalization claim'}
