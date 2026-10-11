"""Fresh learning-rate control against the completed focused CoA pilot.

The completed 50-epoch development arm supplies the exact image panel, model,
normalization, locator, initialization and partition. Only Adam's learning rate
changes, from 0.001 to 0.0003; validation need not have passed any accuracy gate.
"""
from copy import deepcopy
import configparser
import hashlib
import math
from pathlib import Path

import numpy as np

import fakect_coa_pilot_experiment as parent
from fakect_multiscale_experiment import ROOT, SCORE, sha256, checked_json, write_json

SCHEMA = 'fakect.coa-focused-pilot-control/1'
FIELDS = deepcopy(parent.FIELDS)
CODE_FILES = parent.CODE_FILES + (
    'src/fakect_coa_pilot_control_experiment.py', 'src/fakect_coa_pilot_control_learning.py',
    'scripts/train_coa_pilot_control.py', 'scripts/slurm/fakect_coa_pilot_control_gpu.sh',
)
REFERENCE_FILES = {
    'results': 'pilot/results.json', 'history': 'pilot/classifier-history.json',
    'initialization': 'pilot/initialization.json', 'audit': 'pilot/continuation-audit.json',
    'predictions': 'pilot/classifier-predictions.json', 'smoke': 'runtime-smoke.json',
    'replay': 'pilot/proposal-replay.json', 'cache': 'pilot/cache-loaded.json',
}
INHERITED_FIELDS = parent.INHERITED_FIELDS
validate_panel = parent.validate_panel
runtime_config = parent.runtime_config
require_smoke = parent.require_smoke


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(FIELDS):
        raise ValueError('Expected exactly [experiment], [reference], [training]')
    c = {section: dict(parser[section]) for section in FIELDS}
    if any(set(c[section]) != fields for section, fields in FIELDS.items()):
        raise ValueError('Unsupported focused pilot control input fields')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported focused pilot control schema')
    for section, key in (('experiment', 'output_directory'), ('reference', 'experiment_directory')):
        value = Path(c[section][key]).expanduser()
        c[section][key] = str((value if value.is_absolute() else ROOT/value).resolve())
    t = c['training']
    for key in ('epochs', 'seed', 'cache_limit_mb', 'host_memory_limit_mb', 'gpu_memory_limit_mb'):
        t[key] = int(t[key])
    t['learning_rate'] = float(t['learning_rate'])
    t['require_gpu'] = parser.getboolean('training', 'require_gpu')
    if t['epochs'] != 50 or t['learning_rate'] != .0003:
        raise ValueError('The single-factor control fixes 50 epochs and learning_rate=0.0003')
    if not 0 <= t['seed'] <= 2**31-1:
        raise ValueError('training.seed outside supported range')
    for key in ('cache_limit_mb', 'host_memory_limit_mb', 'gpu_memory_limit_mb'):
        if not 256 <= t[key] <= 12288:
            raise ValueError('training.'+key+' outside supported range')
    if t['cache_limit_mb'] > t['host_memory_limit_mb']//2:
        raise ValueError('The input cache must leave at least half the host budget for runtime allocations')
    out, reference = (Path(c[s][k]) for s, k in
                      (('experiment', 'output_directory'), ('reference', 'experiment_directory')))
    if out == reference or reference in out.parents or out in reference.parents:
        raise ValueError('Control output must be separate from its frozen reference')
    return c


def _prediction_join(records, rows):
    by_id = {row['sample_id']: row for row in rows}
    if len(records) != len(by_id) or {row.get('sample_id') for row in records} != set(by_id):
        raise ValueError('Completed pilot predictions must join all 40 development samples')
    for record in records:
        row = by_id[record['sample_id']]
        expected = {'case_id': row['case_id'], 'family': row['anatomy_family'],
                    'split': row['split'], 'role': row['pilot_role'],
                    'label': int(row['annotation']['coa_benchmark_positive']),
                    'score_label': row['annotation'][SCORE]}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError('Completed pilot prediction identity or annotation changed')
        for key in ('probability', 'score_label', 'score_prediction'):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Completed pilot predictions must be finite fractions')


def validate_reference(c, rc, lock, documents, *, lock_digest, checkpoint_digest):
    """Require completed development training, without a validation performance gate."""
    if rc['experiment']['schema_version'] != parent.SCHEMA or rc['training']['epochs'] != 50 or rc['training']['learning_rate'] != .001:
        raise ValueError('Control requires the completed 50-epoch 0.001 pilot reference')
    if c['training']['epochs'] != 50 or c['training']['learning_rate'] != .0003:
        raise ValueError('Control changes only learning_rate to 0.0003 for 50 epochs')
    for key, value in rc['training'].items():
        if key != 'learning_rate' and c['training'].get(key) != value:
            raise ValueError('Control must preserve reference training.'+key)
    rows = validate_panel(lock)
    result, history, initial, audit, predictions, replay, cache = (
        documents[key] for key in ('results', 'history', 'initialization', 'audit', 'predictions', 'replay', 'cache'))
    if (result.get('stage') != 'pilot' or result.get('status') not in ('learning_check_passed', 'learning_check_failed') or
            result.get('epochs_completed') != 50 or result.get('classifier_trained') is not True or
            result.get('batch_size') != 2 or result.get('optimizer_steps') != 800 or
            result.get('training_pairs_per_epoch') != 16 or result.get('sample_presentations') != 1600 or
            result.get('training', {}).get('samples') != 32 or result.get('validation', {}).get('samples') != 8):
        raise ValueError('Control requires a complete 50-epoch 32/8 development pilot')
    for document in (result, audit, predictions, cache, documents['smoke']):
        if document.get('source_lock_sha256') != lock_digest:
            raise ValueError('Completed pilot artifact belongs to another frozen lock')
    for document in (result, audit, predictions, cache):
        if document.get('test_evaluated') is not False:
            raise ValueError('Completed pilot must exclude reserved test evaluation')
    for document in (result, audit):
        if (document.get('locator_unchanged') is not True or document.get('locator_retrained') is not False or
                document.get('selected_checkpoint_prediction_parity', {}).get('passed') is not True or
                document.get('full_cache_loaded') is not True):
            raise ValueError('Completed pilot locator, full cache or checkpoint parity failed')
    if (replay.get('passed') is not True or audit.get('proposal_replay_passed') is not True or
            cache.get('all_development_samples_resident') is not True or
            cache.get('preloaded_before_first_gradient') is not True or
            cache.get('sample_count') != 40 or cache.get('resident_bytes') != lock['cache_plan']['input_cache_bytes']):
        raise ValueError('Completed pilot crop replay or whole-panel cache failed')
    initial_hash = initial.get('weights_sha256')
    if (not isinstance(initial_hash, str) or len(initial_hash) != 64 or
            initial_hash != lock['reference_initial_classifier_weights_sha256'] or
            audit.get('initial_classifier_weights_sha256') != initial_hash or
            audit.get('reference_initial_classifier_weights_sha256') != initial_hash or
            audit.get('same_classifier_initialization_as_reference') is not True or
            initial.get('seed') != c['training']['seed'] or audit.get('initialization_seed') != c['training']['seed'] or
            audit.get('shuffle_seed') != c['training']['seed']):
        raise ValueError('Completed pilot classifier initialization changed')
    epochs = history.get('epochs', [])
    if len(epochs) != 50 or [row.get('epoch') for row in epochs] != list(range(1, 51)):
        raise ValueError('Completed pilot history must contain all 50 sequential epochs')
    train_ids = {row['sample_id'] for row in rows if row['split'] == 'train'}
    for epoch in epochs:
        bce = epoch.get('checkpoint_selection_bce')
        if isinstance(bce, bool) or not isinstance(bce, (int, float)) or not math.isfinite(bce) or bce < 0 or bce != epoch.get('validation', {}).get('bce'):
            raise ValueError('Completed pilot history must select minimum validation BCE')
        _prediction_join(epoch.get('predictions', []), rows)
        pairs = epoch.get('training_pair_order', [])
        if (epoch.get('batch_size') != 2 or epoch.get('optimizer_updates_this_epoch') != 16 or len(pairs) != 16 or
                any(len(pair) != 2 for pair in pairs) or
                len(epoch.get('training_sample_order', [])) != 32 or
                [sample for pair in pairs for sample in pair] != epoch['training_sample_order'] or
                set(epoch['training_sample_order']) != train_ids):
            raise ValueError('Completed pilot balanced update history changed')
    best = min(epochs, key=lambda epoch: epoch['checkpoint_selection_bce'])
    if any(document.get('selected_epoch') != best['epoch'] for document in (result, history, audit, predictions)):
        raise ValueError('Completed pilot selected checkpoint is not minimum validation BCE')
    if (result.get('checkpoint_sha256', {}).get('classifier.keras') != checkpoint_digest or
            result.get('checkpoint_sha256', {}).get('locator.keras') != lock['locator_checkpoint']['sha256'] or
            predictions.get('checkpoint_sha256') != checkpoint_digest or
            audit.get('selected_classifier_checkpoint_sha256') != checkpoint_digest or
            result.get('predictions') != predictions.get('predictions')):
        raise ValueError('Completed pilot checkpoint or prediction artifacts disagree')
    for records in (result.get('predictions', []), initial.get('predictions', [])):
        _prediction_join(records, rows)
    selected_by_id = {record['sample_id']: record for record in best['predictions']}
    for record in result['predictions']:
        previous = selected_by_id[record['sample_id']]
        if not np.allclose([record['probability'], record['score_prediction']],
                           [previous['probability'], previous['score_prediction']], atol=1e-6, rtol=1e-5):
            raise ValueError('Completed pilot loaded predictions differ from selected history')
    return {'reference_initial_classifier_weights_sha256': initial_hash,
            'reference_summary': {'epochs_completed': 50, 'selected_epoch': best['epoch'],
                                  'status': result['status'], 'checkpoint_sha256': checkpoint_digest,
                                  'initial_weights_sha256': initial_hash,
                                  'scope': 'Completed development reference; fresh control weights and unchanged panel.'}}


def source_contract(c):
    reference = Path(c['reference']['experiment_directory'])
    for marker in (reference/'INCOMPLETE', reference/'failure.json', reference/'pilot/INCOMPLETE', reference/'pilot/failure.json'):
        if marker.exists():
            raise ValueError('Failed or incomplete reference cannot authorize the control')
    rc = parent.load_config(reference/'input.ini')
    if Path(rc['experiment']['output_directory']).resolve() != reference.resolve():
        raise ValueError('Reference input points to another output directory')
    lock = parent.inspect_lock(rc, (reference/'input.ini').read_bytes())
    passed_diagnostic = parent.require_diagnostic(rc, lock)
    parent.require_smoke(rc, lock)
    documents = {key: checked_json(reference/name) for key, name in REFERENCE_FILES.items()}
    checkpoint = reference/'pilot/model/classifier.keras'
    lock_digest = sha256(reference/'experiment-lock.json')
    verified = validate_reference(c, rc, lock, documents, lock_digest=lock_digest,
                                  checkpoint_digest=sha256(checkpoint))
    contract = {key: deepcopy(lock[key]) for key in INHERITED_FIELDS}
    contract.update(verified, reference_config=deepcopy(rc), model_config=deepcopy(lock['model_config']))
    paths = [reference/'input.ini', reference/'experiment-lock.json', checkpoint]
    paths.extend(reference/name for name in REFERENCE_FILES.values())
    dependencies = deepcopy(lock['dependency_sha256'])
    dependencies.update({str(path): sha256(path) for path in paths})
    contract['dependency_sha256'] = dependencies
    contract['diagnostic_dependency'] = {'completed_pilot_directory': str(reference),
        'completed_pilot_source_lock_sha256': lock_digest, 'completed_pilot': verified['reference_summary'],
        'completed_pilot_results_sha256': sha256(reference/REFERENCE_FILES['results']),
        'passed_diagnostic': passed_diagnostic}
    return contract


def plan(c):
    contract = source_contract(c)
    rows = validate_panel(contract)
    detail = contract['parent_config']['input.detail']
    needed = len(rows)*detail['proposals']*math.prod(detail['shape_kji'])*2*4
    if needed > c['training']['cache_limit_mb']*1024**2 or c['training']['cache_limit_mb'] > c['training']['host_memory_limit_mb']//2:
        raise ValueError('Control needs the complete 40-sample cache within half the host budget')
    return {'schema_version': SCHEMA, 'ready': True, 'config': deepcopy(c), **contract,
            'scope': 'focused_classifier_learning_rate_control',
            'cache_plan': {'samples': len(rows), 'input_cache_bytes': needed,
                           'input_cache_mib': needed/1024**2, 'cache_limit_mib': c['training']['cache_limit_mb']},
            'training_contract': {'changed_factor': {'learning_rate': {'reference': .001, 'control': .0003}},
                'unchanged_epochs': 50, 'optimizer_updates': 800, 'sample_presentations': 1600,
                'locator_frozen': True, 'optimizer_batch_size': 2,
                'predictor_inputs': 'detail attenuation and acquisition validity only',
                'initialization': 'Fresh weights and seeded shuffle; same reference initial weight fingerprint.',
                'checkpoint_selection': 'Minimum validation BCE; reserved test families remain unused.',
                'input_cache': 'All 40 normalized native crop samples resident in bounded transient RAM; no disk duplication.',
                'smoke': 'Own batch-two GPU train/reload smoke with full planned cache live and guarded.',
                'reserved_test_evaluated': False}}


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
            lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest() or sha256(out/'input.ini') != lock['input_sha256']):
        raise ValueError('Control input changed; use a new experiment revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Control source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if sha256(ROOT/name) != digest or sha256(out/'source'/name) != digest:
            raise ValueError('Pinned control source changed: '+name)
    for name, digest in lock['dependency_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Pinned control dependency changed: '+name)
    current = plan(c)
    if any(lock.get(key) != value for key, value in current.items()):
        raise ValueError('Control reference, data or model contract changed')
    return lock


def require_diagnostic(c, lock):
    """Keep both the completed development control and passed diagnostic pinned."""
    current = source_contract(c)
    for key in INHERITED_FIELDS + ('reference_config', 'model_config',
                                  'reference_initial_classifier_weights_sha256', 'dependency_sha256'):
        if current[key] != lock[key]:
            raise ValueError('Completed parent or inherited pilot contract changed')
    return current['diagnostic_dependency']
