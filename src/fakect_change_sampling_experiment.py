"""Immutable TRAIN-only diagnostic for present-class, burden-balanced sampling.

The source population and locator proposals are reused without alteration. A
small deterministic panel exercises enlargement, narrowing and unchanged
controls on two source TRAIN anatomies. This diagnostic cannot establish
generalization or authorize downstream classifier training.
"""
from copy import deepcopy
import configparser
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import fakect_multiscale_experiment as base
import fakect_change_population_experiment as reference

ROOT = base.ROOT
SCHEMA = 'fakect.change-sampling-pilot/1'
NEW_CODE = ('src/fakect_change_sampling_experiment.py',
            'src/fakect_change_sampling_learning.py',
            'src/fakect_change_sampling_report.py',
            'src/fakect_change_sampling.py',
            'scripts/research/train_change_sampling.py',
            'scripts/research/slurm/fakect_change_sampling_gpu.sh',
            'scripts/research/slurm/fakect_change_sampling_report.sh')
CODE_FILES = tuple(dict.fromkeys(reference.CODE_FILES + NEW_CODE))
FIELDS = {
    'experiment': {'schema_version', 'output_directory'},
    'reference': {'experiment_directory', 'fold'},
    'diagnostic': {'case_ids', 'burden_quantiles'},
    'training': {'epochs', 'steps_per_epoch', 'patch_shape_kji', 'base_filters',
                 'learning_rate', 'change_weight', 'dice_weight', 'changed_fraction',
                 'inference_overlap', 'seed'},
    'validation': {'minimum_dice', 'minimum_precision', 'minimum_recall',
                   'maximum_control_change_fraction'},
}


def _path(value):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else ROOT/path).resolve())


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(FIELDS):
        raise ValueError('Expected experiment, reference, diagnostic, training, validation')
    config = {section: dict(parser[section]) for section in FIELDS}
    if any(set(config[section]) != fields for section, fields in FIELDS.items()):
        raise ValueError('Unsupported or missing sampling diagnostic fields')
    if any('\n' in value or '\r' in value for section in config.values() for value in section.values()):
        raise ValueError('Every input value must occupy one line')
    if config['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported change-sampling diagnostic schema')
    for section in ('experiment', 'reference'):
        key = 'output_directory' if section == 'experiment' else 'experiment_directory'
        config[section][key] = _path(config[section][key])
    output, parent = (Path(config[section][key]) for section, key in
                      [('experiment', 'output_directory'), ('reference', 'experiment_directory')])
    if output == parent or output in parent.parents or parent in output.parents:
        raise ValueError('Diagnostic output must be separate from the frozen reference')
    cases = [value.strip() for value in config['diagnostic']['case_ids'].split(',')]
    if len(cases) != 2 or len(set(cases)) != 2 or any(not value for value in cases):
        raise ValueError('The bounded diagnostic requires exactly two unique source TRAIN cases')
    config['diagnostic']['case_ids'] = cases
    config['diagnostic']['burden_quantiles'] = int(config['diagnostic']['burden_quantiles'])
    if config['diagnostic']['burden_quantiles'] != 3:
        raise ValueError('Use three predeclared burden strata per present class and case')
    training = config['training']
    training['patch_shape_kji'] = [int(value.strip()) for value in training['patch_shape_kji'].split(',')]
    if (len(training['patch_shape_kji']) != 3 or any(value < 16 or value % 8 for value in training['patch_shape_kji'])
            or math.prod(training['patch_shape_kji']) > 1_048_576):
        raise ValueError('Patch dimensions must be >=16, divisible by8 and within the voxel budget')
    for key, low, high in [('epochs', 1, 200), ('steps_per_epoch', 14, 112),
                          ('base_filters', 2, 8), ('inference_overlap', 0, 64), ('seed', 0, 2**31-1)]:
        training[key] = int(training[key])
        if not low <= training[key] <= high:
            raise ValueError('Invalid training.'+key)
    if training['steps_per_epoch'] % 14:
        raise ValueError('steps_per_epoch must be a multiple of14 for balanced case/control exposure')
    if training['inference_overlap'] > min(training['patch_shape_kji'])//2:
        raise ValueError('Inference overlap exceeds half the smallest patch')
    for key in ('learning_rate', 'change_weight', 'dice_weight', 'changed_fraction'):
        training[key] = float(training[key])
        if not math.isfinite(training[key]):
            raise ValueError('Nonfinite training.'+key)
    if (not 0 < training['learning_rate'] <= .1 or not 1 <= training['change_weight'] <= 64
            or not 0 <= training['dice_weight'] <= 10 or not 0 <= training['changed_fraction'] <= 1):
        raise ValueError('Invalid optimizer, loss or changed-patch sampling settings')
    for key in config['validation']:
        config['validation'][key] = float(config['validation'][key])
        if not math.isfinite(config['validation'][key]) or not 0 <= config['validation'][key] <= 1:
            raise ValueError('Diagnostic thresholds must be in [0,1]')
    return config


def _read_json(path, expected=None):
    """All inputs are hashed; older unsigned metadata remains compatible."""
    path = Path(path)
    if Path(str(path)+'.sha256').exists():
        return base.checked_json(path, expected)
    if (path.parent/'INCOMPLETE').exists():
        raise ValueError('Incomplete artifact: '+str(path))
    if expected is not None and base.sha256(path) != expected:
        raise ValueError('Pinned metadata changed: '+str(path))
    return json.loads(path.read_text())


def _dependency(dependencies, path, expected=None):
    path = Path(path).resolve()
    digest = base.sha256(path)
    if expected is not None and digest != expected:
        raise ValueError('Pinned source payload changed: '+str(path))
    if str(path) in dependencies and dependencies[str(path)] != digest:
        raise ValueError('Conflicting source payload: '+str(path))
    dependencies[str(path)] = digest
    if path.suffix == '.json':
        _read_json(path, digest)
    return digest


def _counts(row):
    counts = row.get('mask_counts', {})
    result = {}
    for name in ('added', 'removed'):
        value = counts.get(name)
        if type(value) is not int or value < 0:
            raise ValueError('Require actual nonnegative integer net mask_counts: '+row['sample_id'])
        result[name] = value
    return result


def select_panel(rows, case_ids):
    """Six unique edited siblings per case: three burden ranks per net class.

    Candidates are stratified by actual net-mask voxel counts, not requested
    edit strength. Each class's low/middle/high rank band prefers respectively
    a single-class/mixed/single-class sibling; ties use stable sample IDs.
    Selection consults source TRAIN targets only and is not an outcome search.
    """
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate source TRAIN sample ID')
    selected, slots = {}, []
    for case in sorted(case_ids):
        candidates = [row for row in rows if row['case_id'] == case and not row['baseline']]
        for row in candidates:
            _counts(row)
        used = set()
        for name, other in [('added', 'removed'), ('removed', 'added')]:
            ordered = sorted((row for row in candidates if _counts(row)[name] > 0),
                             key=lambda row: (_counts(row)[name], row['sample_id']))
            if len(ordered) < 3:
                raise ValueError('Need three positive '+name+' siblings for '+case)
            for tier in range(3):
                start, stop = len(ordered)*tier//3, len(ordered)*(tier+1)//3
                available = [(rank, row) for rank, row in enumerate(ordered[start:stop], start)
                             if row['sample_id'] not in used]
                if not available:
                    raise ValueError('Cannot choose unique siblings in every burden stratum')
                middle = (start+stop-1)/2
                prefer_mixed = tier == 1
                rank, row = min(available, key=lambda item:
                    ((_counts(item[1])[other] > 0) != prefer_mixed,
                     abs(item[0]-middle), item[1]['sample_id']))
                sid = row['sample_id']
                used.add(sid)
                selected[sid] = deepcopy(row)
                slots.append({'sample_id': sid, 'case_id': case, 'class': name,
                              'burden_stratum': ('low', 'middle', 'high')[tier],
                              'mode': name, 'burden_bin': tier,
                              'rank': rank, 'positive_candidates': len(ordered),
                              'stratum_rank_interval': [start, stop],
                              'target_voxels': _counts(row)[name],
                              'actual_mask_counts': _counts(row),
                              'mixed_changes': _counts(row)[other] > 0})
    return [selected[sid] for sid in sorted(selected)], slots


def _verify_masks(row, dependencies):
    _dependency(dependencies, row['mask_npz'], row['mask_npz_sha256'])
    with np.load(row['mask_npz'], allow_pickle=False) as arrays:
        before, after, added, removed = (arrays[key] for key in ('before', 'after', 'added', 'removed'))
        if before.ndim != 3 or any(value.shape != before.shape for value in (after, added, removed)):
            raise ValueError('Aligned three-dimensional net masks are required')
        if any(not np.all((value == 0) | (value == 1)) for value in (before, after, added, removed)):
            raise ValueError('Net masks must be binary')
        if (not np.array_equal(added, (after == 1) & (before == 0))
                or not np.array_equal(removed, (before == 1) & (after == 0))):
            raise ValueError('Net mask identities disagree: '+row['sample_id'])
        counts = {name: int(np.count_nonzero(value)) for name, value in [('added', added), ('removed', removed)]}
        if counts != _counts(row) or (row['baseline'] and any(counts.values())):
            raise ValueError('Actual mask burden disagrees with the source descriptor')


def plan(config):
    parent = Path(config['reference']['experiment_directory'])
    old = _read_json(parent/'experiment-lock.json')
    reference.inspect_lock(old['config'], (parent/'input.ini').read_bytes())
    if old.get('schema_version') != reference.SCHEMA or old.get('reserved_test_evaluated') is not False:
        raise ValueError('Require the frozen completed expanded-population experiment')
    matches = [stage for stage in old['stages'] if stage['outer_fold_id'] == config['reference']['fold']]
    if len(matches) != 1:
        raise ValueError('Require one exact frozen source fold')
    source_stage = matches[0]
    cases = config['diagnostic']['case_ids']
    if (any(case not in source_stage['train_case_ids'] for case in cases)
            or set(cases) & set(source_stage['validation_case_ids']+source_stage['final_test_case_ids']+old['reserved_case_ids'])):
        raise ValueError('Diagnostic cases must belong exclusively to the source TRAIN partition')
    for key in reference.MATCHED_TRAINING:
        if config['training'][key] != old['config']['training'][key]:
            raise ValueError('Diagnostic must preserve source network/loss/sampling settings: '+key)
    if config['validation'] != old['config']['validation']:
        raise ValueError('Preserve the existing declared localization thresholds')
    dependencies = {}
    parent_hash = _dependency(dependencies, parent/'experiment-lock.json')
    _dependency(dependencies, parent/'input.ini')
    folder = parent/'stages'/source_stage['id']
    result = _read_json(folder/'results.json')
    if result.get('status') != 'completed' or result.get('reserved_test_evaluated') is not False:
        raise ValueError('Source fold must be completed with reserved anatomy unopened')
    _dependency(dependencies, folder/'results.json')
    audit = _read_json(folder/'all-training-proposals.json')
    _dependency(dependencies, folder/'all-training-proposals.json')
    source_ids = source_stage['train_sample_ids']
    source_rows = {row['sample_id']: row for row in old['samples']}
    if len(source_rows) != len(old['samples']) or len(set(source_ids)) != len(source_ids):
        raise ValueError('Source sample IDs must be unique')
    proposals = {row['sample_id']: row for row in audit['samples']}
    if (audit.get('selection_allowed') is not False or len(proposals) != len(audit['samples'])
            or set(proposals) != set(source_ids) or any(row.get('split') != 'train' for row in proposals.values())):
        raise ValueError('Frozen proposals must cover exactly the source TRAIN IDs without selection')
    rows = [source_rows[sid] for sid in source_ids if source_rows[sid]['case_id'] in cases]
    if any(row['case_id'] not in source_stage['train_case_ids'] for row in rows):
        raise ValueError('Non-TRAIN anatomy entered the diagnostic')
    edited, slots = select_panel(rows, cases)
    controls = [row for row in rows if row['baseline']]
    if len(controls) != len(cases) or {row['case_id'] for row in controls} != set(cases):
        raise ValueError('Exactly one original unchanged control is required per diagnostic case')
    selected = sorted(edited+deepcopy(controls), key=lambda row: row['sample_id'])
    for row in selected:
        row['split'] = 'train'
        row['diagnostic_role'] = 'weight_fitting_and_training_diagnostic'
        _verify_masks(row, dependencies)
        for key in ('prepared_manifest', 'selection_manifest'):
            if key in row:
                _dependency(dependencies, row[key], row[key+'_sha256'])
        if proposals[row['sample_id']].get('coverage_passed') is not True:
            raise ValueError('Selected frozen focus failed coverage; crops must not be recentered')
    seed = config['training']['seed']
    stage_id = 'training_diagnostic/'+source_stage['outer_fold_id']+'/seed-'+str(seed)
    ids = [row['sample_id'] for row in selected]
    stage = {'id': stage_id, 'protocol': 'training_diagnostic', 'stage': 'sampling_fit',
             'outer_fold_id': source_stage['outer_fold_id'], 'seed': seed,
             'source_stage_id': source_stage['id'], 'train_case_ids': sorted(cases),
             'train_sample_ids': ids, 'diagnostic_sample_ids': ids,
             'diagnostic_strata': deepcopy(slots),
             'train_control_ids': [row['sample_id'] for row in selected if row['baseline']],
             'validation_case_ids': [], 'validation_sample_ids': [],
             'final_test_case_ids': [], 'final_test_sample_ids': [],
             'reserved_case_ids': list(old['reserved_case_ids']),
             'final_test_evaluation_permitted': False,
             'checkpoint_policy': 'fresh same-seed network; declared final update; TRAIN-only diagnostic',
             'counts': {'train': {'cases': len(cases), 'edited': len(edited), 'controls': len(controls),
                                   'selected_samples': len(selected)},
                        'validation': {'selected_samples': 0}, 'final_test': {'selected_samples': 0}},
             'training_schedule': {'epochs': config['training']['epochs'],
                                   'steps_per_epoch': config['training']['steps_per_epoch'],
                                   'total_optimizer_updates': config['training']['epochs']*config['training']['steps_per_epoch']}}
    return {'schema_version': SCHEMA, 'config': config, 'stages': [stage], 'samples': selected,
            'proposals': {stage_id: {sid: deepcopy(proposals[sid]) for sid in ids}},
            'runtime': deepcopy(old['runtime']), 'selection_slots': slots,
            'parent_directory': str(parent), 'parent_lock_sha256': parent_hash,
            'dependency_sha256': dependencies, 'reserved_case_ids': list(old['reserved_case_ids']),
            'reserved_test_evaluated': False, 'held_out_evaluated': False, 'classifier_started': False,
            'target_semantics': old['target_semantics'], 'inputs': old['inputs'],
            'scope': 'TRAIN-only memorization/sampling diagnostic; no generalization claim',
            'comparison_contract': 'Sampling correction only; same model/loss/intensity/resolution; new small TRAIN panel',
            'generalization_evaluation_permitted': False, 'full_fold_training_authorized_by_gate': False}


def freeze(config, content):
    lock = plan(config)
    payloads = {name: (ROOT/name).read_bytes() for name in CODE_FILES}
    output = Path(config['experiment']['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').touch()
    (output/'input.ini').write_bytes(content)
    lock['code_sha256'] = {}
    for name, data in payloads.items():
        path = output/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        lock['code_sha256'][name] = hashlib.sha256(data).hexdigest()
    lock['input_sha256'] = hashlib.sha256(content).hexdigest()
    base.write_json(output/'experiment-lock.json', lock)
    (output/'INCOMPLETE').unlink()
    return lock


def inspect_lock(config, content):
    output = Path(config['experiment']['output_directory'])
    lock = _read_json(output/'experiment-lock.json')
    if (lock['schema_version'] != SCHEMA or lock['config'] != config
            or lock['input_sha256'] != hashlib.sha256(content).hexdigest()
            or base.sha256(output/'input.ini') != lock['input_sha256']):
        raise ValueError('Frozen diagnostic input changed; use a fresh output revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Frozen diagnostic source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if base.sha256(ROOT/name) != digest or base.sha256(output/'source'/name) != digest:
            raise ValueError('Pinned implementation changed: '+name)
    for name, digest in lock['dependency_sha256'].items():
        if Path(name).suffix == '.json':
            _read_json(name, digest)
        elif base.sha256(name) != digest:
            raise ValueError('Pinned source payload changed: '+name)
    parent = Path(lock['parent_directory'])
    old = _read_json(parent/'experiment-lock.json')
    reference.inspect_lock(old['config'], (parent/'input.ini').read_bytes())
    stage = lock['stages'][0]
    if (lock.get('held_out_evaluated') is not False or lock.get('reserved_test_evaluated') is not False
            or stage['validation_sample_ids'] or stage['final_test_sample_ids']
            or any(row['split'] != 'train' or row['case_id'] not in stage['train_case_ids'] for row in lock['samples'])):
        raise ValueError('Frozen diagnostic partition changed')
    return lock


def require_smoke(config, lock):
    output = Path(config['experiment']['output_directory'])
    smoke = _read_json(output/'runtime-smoke.json')
    if (smoke.get('passed') is not True or smoke.get('gpu_visible') is not True
            or smoke.get('reload_parity') is not True
            or smoke.get('source_lock_sha256') != base.sha256(output/'experiment-lock.json')
            or smoke.get('train_traces') != 1 or smoke.get('predict_traces') != 1):
        raise ValueError('Own production-patch GPU smoke must pass before the diagnostic')
    return smoke
