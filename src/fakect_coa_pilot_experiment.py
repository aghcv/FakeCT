"""Immutable 32/8 development pilot after a passed focused CoA diagnostic.

The clean input inherits the reviewed locator, native image geometry, model,
annotations and family assignment from the successful 200-epoch reference.
Only pilot duration and bounded whole-panel cache size are new training choices.
"""
from copy import deepcopy
import configparser
import hashlib
import math
from pathlib import Path

import numpy as np

import fakect_coa_classifier_experiment as classifier
import fakect_coa_paired_experiment_v2 as parent
from fakect_multiscale_experiment import ROOT, SCORE, sha256, checked_json, write_json

SCHEMA = 'fakect.coa-focused-pilot/1'
FIELDS = {
    'experiment': {'schema_version', 'output_directory'},
    'reference': {'experiment_directory'},
    'training': {'epochs', 'learning_rate', 'seed', 'cache_limit_mb',
                 'host_memory_limit_mb', 'gpu_memory_limit_mb', 'require_gpu'},
}
CODE_FILES = parent.CODE_FILES + (
    'src/fakect_coa_pilot_experiment.py', 'src/fakect_coa_pilot_learning.py',
    'scripts/research/train_coa_pilot.py', 'scripts/research/slurm/fakect_coa_pilot_gpu.sh',
)
REFERENCE_FILES = {
    'results': 'diagnostic/results.json', 'history': 'diagnostic/classifier-history.json',
    'initialization': 'diagnostic/initialization.json', 'audit': 'diagnostic/continuation-audit.json',
    'predictions': 'diagnostic/classifier-predictions.json', 'smoke': 'runtime-smoke.json',
}
INHERITED_FIELDS = ('parent_config', 'samples', 'proposals', 'locator_checkpoint',
                    'reserved_test_sample_ids', 'pilot_counts', 'design', 'annotation_acceptance')


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(FIELDS):
        raise ValueError('Expected exactly [experiment], [reference], [training]')
    c = {section: dict(parser[section]) for section in FIELDS}
    if any(set(c[section]) != fields for section, fields in FIELDS.items()):
        raise ValueError('Unsupported focused pilot input fields')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported focused pilot schema')
    for section, key in (('experiment', 'output_directory'), ('reference', 'experiment_directory')):
        value = Path(c[section][key]).expanduser()
        c[section][key] = str((value if value.is_absolute() else ROOT/value).resolve())
    t = c['training']
    for key, low, high in (('epochs', 1, 200), ('seed', 0, 2**31-1),
                           ('cache_limit_mb', 256, 6144), ('host_memory_limit_mb', 512, 12288),
                           ('gpu_memory_limit_mb', 512, 12288)):
        t[key] = int(t[key])
        if not low <= t[key] <= high:
            raise ValueError(f'training.{key} outside [{low},{high}]')
    t['learning_rate'] = float(t['learning_rate'])
    if not math.isfinite(t['learning_rate']) or t['learning_rate'] <= 0:
        raise ValueError('training.learning_rate must be positive and finite')
    t['require_gpu'] = parser.getboolean('training', 'require_gpu')
    if t['cache_limit_mb'] > t['host_memory_limit_mb']//2:
        raise ValueError('The input cache must leave at least half the host budget for runtime allocations')
    out = Path(c['experiment']['output_directory'])
    reference = Path(c['reference']['experiment_directory'])
    if out == reference or reference in out.parents:
        raise ValueError('Pilot output must be separate from its frozen reference')
    return c


def validate_panel(lock):
    """Require the exact balanced family-separated 32/8 development panel."""
    rows = classifier.validate_samples(lock)
    if lock['pilot_counts'] != {'train': {'samples': 32, 'positive': 16},
                                'validation': {'samples': 8, 'positive': 4}}:
        raise ValueError('Focused pilot requires the unchanged balanced 32/8 panel')
    roles = {'original', 'edited_negative', 'lower_positive', 'higher_positive'}
    for split, case_count in (('train', 8), ('validation', 2)):
        selected = [row for row in rows if row['split'] == split]
        cases = {row['case_id'] for row in selected}
        if len(cases) != case_count:
            raise ValueError('Focused pilot case counts changed: '+split)
        for case in cases:
            group = [row for row in selected if row['case_id'] == case]
            if len(group) != 4 or {row['pilot_role'] for row in group} != roles:
                raise ValueError('Every pilot case needs the four prespecified sibling roles')
            for row in group:
                original = row['pilot_role'] == 'original'
                positive = row['pilot_role'] in ('lower_positive', 'higher_positive')
                label = row['annotation']
                if (row['baseline'] is not original or label['coa_benchmark_positive'] is not positive or
                        (label[SCORE] >= .2) != positive or (original and label[SCORE] != 0)):
                    raise ValueError('Pilot role, baseline or fixed engineering label is inconsistent')
    return rows


def validate_reference(c, reference_config, reference_lock, documents, *,
                       lock_digest, checkpoint_digest):
    """Bind the passed 200-epoch arm and unchanged initialization/settings."""
    if reference_config['experiment']['schema_version'] != parent.SCHEMA or reference_config['diagnostic']['epochs'] != 200:
        raise ValueError('Pilot requires the completed 200-epoch paired reference')
    for key in ('learning_rate', 'seed', 'host_memory_limit_mb', 'gpu_memory_limit_mb', 'require_gpu'):
        if c['training'][key] != reference_config['training'][key]:
            raise ValueError('Pilot must preserve the successful reference training.'+key)
    validate_panel(reference_lock)
    result, history, initial, audit, predictions, smoke = (
        documents[key] for key in ('results', 'history', 'initialization', 'audit', 'predictions', 'smoke'))
    selected = parent.select_diagnostic(reference_config, reference_lock)
    pair_count = len(selected)//2
    if (result.get('stage') != 'diagnostic' or result.get('status') != 'diagnostic_passed' or
            result.get('epochs_completed') != 200 or result.get('classifier_trained') is not True or
            result.get('learning_gate', {}).get('passed') is not True or result.get('batch_size') != 2 or
            result.get('optimizer_steps') != 200*pair_count or result.get('training_pairs_per_epoch') != pair_count or
            result.get('sample_presentations') != 400*pair_count or result.get('validation') is not None):
        raise ValueError('Pilot requires a passed, complete training-only 200-epoch diagnostic')
    for document in (result, audit, predictions, smoke):
        if document.get('source_lock_sha256') != lock_digest:
            raise ValueError('Reference artifact belongs to another frozen lock')
        if document is not smoke and document.get('test_evaluated') is not False:
            raise ValueError('Reference must exclude reserved test evaluation')
    for document in (result, audit):
        if (document.get('locator_unchanged') is not True or document.get('locator_retrained') is not False or
                document.get('selected_checkpoint_prediction_parity', {}).get('passed') is not True):
            raise ValueError('Reference locator or classifier checkpoint parity failed')
    initial_hash = initial.get('weights_sha256')
    if (not isinstance(initial_hash, str) or len(initial_hash) != 64 or
            reference_lock.get('reference_initial_classifier_weights_sha256') != initial_hash or
            audit.get('initial_classifier_weights_sha256') != initial_hash or
            audit.get('reference_initial_classifier_weights_sha256') != initial_hash or
            audit.get('same_classifier_initialization_as_reference') is not True or
            initial.get('seed') != c['training']['seed'] or audit.get('initialization_seed') != c['training']['seed'] or
            audit.get('shuffle_seed') != c['training']['seed'] or initial.get('validation') is not None):
        raise ValueError('Reference classifier initialization changed')
    epochs = history.get('epochs', [])
    if len(epochs) != 200 or [row.get('epoch') for row in epochs] != list(range(1, 201)):
        raise ValueError('Reference history must contain all 200 sequential epochs')
    for row in epochs:
        value = row.get('checkpoint_selection_bce')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Reference history has invalid training BCE')
    best = min(epochs, key=lambda row: row['checkpoint_selection_bce'])
    if any(document.get('selected_epoch') != best['epoch'] for document in (result, audit, predictions, history)):
        raise ValueError('Reference selected checkpoint is not its minimum training BCE')
    if (result.get('checkpoint_sha256', {}).get('classifier.keras') != checkpoint_digest or
            result.get('checkpoint_sha256', {}).get('locator.keras') != reference_lock['locator_checkpoint']['sha256'] or
            predictions.get('checkpoint_sha256') != checkpoint_digest or
            audit.get('selected_classifier_checkpoint_sha256') != checkpoint_digest or
            result.get('predictions') != predictions.get('predictions')):
        raise ValueError('Reference checkpoint or prediction artifacts disagree')
    for rows in (result.get('predictions', []), initial.get('predictions', []), best.get('predictions', [])):
        parent._prediction_join(rows, selected)
    by_id = {row['sample_id']: row for row in best['predictions']}
    for row in result['predictions']:
        previous = by_id[row['sample_id']]
        if not np.allclose([row['probability'], row['score_prediction']],
                           [previous['probability'], previous['score_prediction']], atol=1e-6, rtol=1e-5):
            raise ValueError('Reference loaded predictions differ from its selected checkpoint history')
    detail = reference_lock['parent_config']['input.detail']
    if (smoke.get('status') != 'passed' or smoke.get('checkpoint_parity') is not True or
            smoke.get('batch_size') != 2 or smoke.get('detail_shape_kji') != detail['shape_kji'] or
            smoke.get('proposals') != detail['proposals'] or smoke.get('traces') != {'train': 1, 'predict': 1} or
            (c['training']['require_gpu'] and smoke.get('gpu_visible') is not True)):
        raise ValueError('Reference requires its passing production-size batch-two GPU smoke')
    return {'reference_initial_classifier_weights_sha256': initial_hash,
            'reference_summary': {'epochs_completed': 200, 'selected_epoch': best['epoch'],
                                  'status': result['status'], 'checkpoint_sha256': checkpoint_digest,
                                  'initial_weights_sha256': initial_hash,
                                  'scope': 'Passed training-only feasibility diagnostic; fresh pilot initialization.'}}


def source_contract(c):
    reference = Path(c['reference']['experiment_directory'])
    for marker in (reference/'INCOMPLETE', reference/'failure.json',
                   reference/'diagnostic/INCOMPLETE', reference/'diagnostic/failure.json'):
        if marker.exists():
            raise ValueError('Failed or incomplete reference cannot authorize the pilot')
    rc = parent.load_config(reference/'input.ini')
    if Path(rc['experiment']['output_directory']).resolve() != reference.resolve():
        raise ValueError('Reference input points to another output directory')
    lock = parent.inspect_lock(rc, (reference/'input.ini').read_bytes())
    diagnostic = parent.require_diagnostic(rc, lock)
    documents = {key: checked_json(reference/name) for key, name in REFERENCE_FILES.items()}
    checkpoint = reference/'diagnostic/model/classifier.keras'
    verified = validate_reference(c, rc, lock, documents,
                                  lock_digest=sha256(reference/'experiment-lock.json'),
                                  checkpoint_digest=sha256(checkpoint))
    contract = {key: deepcopy(lock[key]) for key in INHERITED_FIELDS}
    contract.update(verified, reference_config=deepcopy(rc), model_config=deepcopy(rc['model']),
                    diagnostic_dependency=diagnostic)
    dependencies = deepcopy(lock['dependency_sha256'])
    paths = [reference/'input.ini', reference/'experiment-lock.json', checkpoint]
    paths.extend(reference/name for name in REFERENCE_FILES.values())
    dependencies.update({str(path): sha256(path) for path in paths})
    contract['dependency_sha256'] = dependencies
    return contract


def plan(c):
    contract = source_contract(c)
    rows = validate_panel(contract)
    detail = contract['parent_config']['input.detail']
    needed = len(rows)*detail['proposals']*math.prod(detail['shape_kji'])*2*4
    if needed > c['training']['cache_limit_mb']*1024**2:
        raise ValueError('Pilot cache must hold all 40 normalized samples; increase cache_limit_mb')
    if c['training']['cache_limit_mb'] > c['training']['host_memory_limit_mb']//2:
        raise ValueError('Pilot cache exceeds half the host budget')
    return {'schema_version': SCHEMA, 'ready': True, 'config': deepcopy(c), **contract,
            'scope': 'focused_classifier_family_separated_development_pilot',
            'cache_plan': {'samples': len(rows), 'input_cache_bytes': needed,
                           'input_cache_mib': needed/1024**2,
                           'cache_limit_mib': c['training']['cache_limit_mb']},
            'training_contract': {
                'locator_frozen': True, 'optimizer_batch_size': 2,
                'predictor_inputs': 'detail attenuation and acquisition validity only',
                'initialization': 'Fresh weights and seeded shuffle; initial weight fingerprint must match the passed reference.',
                'checkpoint_selection': 'Minimum validation BCE; validation is development data, reserved test families remain unused.',
                'input_cache': 'All 40 normalized native crop samples resident in a bounded transient RAM cache; no disk duplication.',
                'smoke': 'Own batch-two GPU train/reload smoke with full planned cache live and guarded.',
                'reserved_test_evaluated': False,
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
        raise ValueError('Pilot input changed; use a new experiment revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Pilot source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if sha256(ROOT/name) != digest or sha256(out/'source'/name) != digest:
            raise ValueError('Pinned pilot source changed: '+name)
    for name, digest in lock['dependency_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Pinned pilot dependency changed: '+name)
    current = plan(c)
    if any(lock.get(key) != value for key, value in current.items()):
        raise ValueError('Pilot reference, data or model contract changed')
    return lock


def runtime_config(c, lock):
    value = deepcopy(lock['parent_config'])
    value['experiment']['output_directory'] = c['experiment']['output_directory']
    t = c['training']
    value['model'].update(learning_rate=t['learning_rate'], seed=t['seed'],
                          base_filters=lock['model_config']['base_filters'])
    value['pilot'].update(classifier_epochs=t['epochs'], require_gpu=t['require_gpu'],
                          gpu_memory_limit_mb=t['gpu_memory_limit_mb'])
    value['training'].update(host_memory_limit_mb=t['host_memory_limit_mb'], maximum_traces=1)
    return value


def require_diagnostic(c, lock):
    """Recheck the passed parent diagnostic without creating a local diagnostic."""
    current = source_contract(c)
    for key in INHERITED_FIELDS + ('reference_config', 'model_config',
                                  'reference_initial_classifier_weights_sha256', 'dependency_sha256'):
        if current[key] != lock[key]:
            raise ValueError('Passed parent diagnostic or inherited pilot contract changed')
    return current['diagnostic_dependency']


def require_smoke(c, lock):
    """Require a pilot-specific full-size smoke with the whole input cache live."""
    root = Path(c['experiment']['output_directory'])
    result = checked_json(root/'runtime-smoke.json')
    source = sha256(root/'experiment-lock.json')
    detail = lock['parent_config']['input.detail']
    if (result.get('source_lock_sha256') != source or result.get('status') != 'passed' or
            result.get('synthetic_data_only') is not True or result.get('checkpoint_parity') is not True or
            result.get('batch_size') != 2 or result.get('detail_shape_kji') != detail['shape_kji'] or
            result.get('proposals') != detail['proposals'] or result.get('traces') != {'train': 1, 'predict': 1} or
            (c['training']['require_gpu'] and result.get('gpu_visible') is not True)):
        raise ValueError('Pilot requires its own passing full-size batch-two GPU smoke')
    samples = result.get('samples', [])
    if not samples:
        raise ValueError('Pilot smoke needs recorded classifier gradient updates')
    for row in samples:
        for key in ('loss', 'gradient_global_norm'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Pilot smoke gradient/loss evidence is invalid')
    if not any(row['gradient_global_norm'] > 0 for row in samples):
        raise ValueError('Pilot smoke must demonstrate a nonzero classifier gradient')
    if not any(row.get('full_cache_resident') is True and row['gradient_global_norm'] > 0 for row in samples):
        raise ValueError('Pilot smoke must record a nonzero model update while the complete cache is resident')
    probe = result.get('full_cache_memory_probe', {})
    cache = lock['cache_plan']
    if (probe.get('status') != 'passed' or probe.get('passed') is not True or
            probe.get('source_lock_sha256') != source or probe.get('sample_count') != cache['samples'] or
            probe.get('resident_bytes') != cache['input_cache_bytes'] or
            probe.get('resident_mib') != cache['input_cache_mib'] or
            probe.get('cache_limit_mib') != c['training']['cache_limit_mb'] or
            probe.get('host_guard_passed') is not True or
            probe.get('physical_pages_touched') is not True or
            probe.get('model_update_while_resident') is not True or
            probe.get('resident_training_traces') != {'train': 1, 'predict': 1} or
            probe.get('host_limit_mib') != c['training']['host_memory_limit_mb']):
        raise ValueError('Pilot smoke must verify the entire planned input cache at runtime')
    observed = probe.get('host_peak_rss_mib')
    if (isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed) or
            not cache['input_cache_mib'] <= observed <= c['training']['host_memory_limit_mb']):
        raise ValueError('Full-cache memory probe exceeds the host budget or lacks physical allocation evidence')
    return result
