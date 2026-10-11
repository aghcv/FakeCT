"""Expanded training siblings with unchanged focused-change evaluation panels.

This additive control changes training anatomy variants, not the network, loss,
resolution, held-out siblings or total optimizer budget. A reporting block is
224 updates, NOT a complete pass over the expanded population. Cases retain the
same family roles, and the reserved infant family remains unopened.
"""
from copy import deepcopy
import configparser
import hashlib
import math
from pathlib import Path
import random

import fakect_change_experiment as reference
import fakect_multiscale_experiment as base

ROOT = base.ROOT
SCHEMA = 'fakect.focused-change-population/1'
NEW_CODE = ('src/fakect_change_population_experiment.py',
            'src/fakect_change_population_learning.py',
            'scripts/train_change_population.py',
            'scripts/slurm/fakect_change_population_gpu.sh',
            'scripts/slurm/fakect_change_population_report.sh')
CODE_FILES = tuple(dict.fromkeys(reference.CODE_FILES + NEW_CODE))
FIELDS = {
    'experiment': {'schema_version', 'output_directory'},
    'reference': {'experiment_directory'},
    'population': {'manifest', 'siblings_per_case'},
    'training': {'reporting_blocks', 'updates_per_block', 'patch_shape_kji',
                 'base_filters', 'learning_rate', 'change_weight', 'dice_weight',
                 'changed_fraction', 'inference_overlap', 'seed'},
    'validation': {'minimum_dice', 'minimum_precision', 'minimum_recall',
                   'maximum_control_change_fraction'},
}
MATCHED_TRAINING = ('patch_shape_kji', 'base_filters', 'learning_rate',
                    'change_weight', 'dice_weight', 'changed_fraction',
                    'inference_overlap', 'seed')
SOURCE_FIELDS = ('source_atn', 'source_shape_kji', 'spacing_ijk_mm',
                 'original_reference_mask_sha256', 'source_atn_units',
                 'source_atn_dtype', 'image_units')
ROW_FIELDS = ('sample_id', 'case_id', 'variant_id', 'anatomy_family', 'baseline',
              'variant_npz', 'variant_npz_sha256', 'source_npz', 'source_npz_sha256',
              'mask_npz', 'mask_npz_sha256', 'edit_crop_origin_ijk',
              'edit_crop_shape_kji') + SOURCE_FIELDS


def load_config(path):
    p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                 empty_lines_in_values=False)
    p.read_string(Path(path).read_text())
    if p.defaults() or set(p.sections()) != set(FIELDS):
        raise ValueError('Expected experiment, reference, population, training, validation')
    c = {s: dict(p[s]) for s in FIELDS}
    if any(set(c[s]) != FIELDS[s] for s in FIELDS):
        raise ValueError('Unsupported or missing expanded-population fields')
    if any('\n' in v or '\r' in v for s in c.values() for v in s.values()):
        raise ValueError('Every input value must occupy one line')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported focused-change population schema')
    for section, key in (('experiment', 'output_directory'),
                         ('reference', 'experiment_directory'), ('population', 'manifest')):
        value = Path(c[section][key]).expanduser()
        c[section][key] = str((value if value.is_absolute() else ROOT/value).resolve())
    out = Path(c['experiment']['output_directory'])
    parent = Path(c['reference']['experiment_directory'])
    population = Path(c['population']['manifest']).parent
    for protected in (parent, population):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError('Experiment output must be separate from reference and population')
    c['population']['siblings_per_case'] = int(c['population']['siblings_per_case'])
    if not 1 <= c['population']['siblings_per_case'] <= 256:
        raise ValueError('siblings_per_case must be in [1,256] for this bounded control')
    t = c['training']
    for key in ('reporting_blocks', 'updates_per_block', 'base_filters', 'inference_overlap', 'seed'):
        t[key] = int(t[key])
    if (t['reporting_blocks'] != 64 or t['updates_per_block'] != 224
            or not 2 <= t['base_filters'] <= 8 or not 0 <= t['seed'] <= 2**31-1):
        raise ValueError('Use 64 reporting blocks of 224 updates and valid filter/seed settings')
    t['patch_shape_kji'] = [int(v.strip()) for v in t['patch_shape_kji'].split(',')]
    if (len(t['patch_shape_kji']) != 3 or any(n < 16 or n % 8 for n in t['patch_shape_kji'])
            or math.prod(t['patch_shape_kji']) > 1_048_576
            or not 0 <= t['inference_overlap'] < min(t['patch_shape_kji'])):
        raise ValueError('Invalid bounded patch shape or inference overlap')
    for key in ('learning_rate', 'change_weight', 'dice_weight', 'changed_fraction'):
        t[key] = float(t[key])
        if not math.isfinite(t[key]):
            raise ValueError('Nonfinite training.'+key)
    if (not 0 < t['learning_rate'] <= .1 or not 1 <= t['change_weight'] <= 64
            or not 0 <= t['dice_weight'] <= 10 or not 0 <= t['changed_fraction'] <= 1):
        raise ValueError('Invalid learning rate, change/Dice weight or changed sampling fraction')
    for key in c['validation']:
        c['validation'][key] = float(c['validation'][key])
        if not math.isfinite(c['validation'][key]) or not 0 <= c['validation'][key] <= 1:
            raise ValueError('Validation thresholds must be in [0,1]')
    return c


def _dependency(deps, path, *, expected=None):
    path = Path(path).resolve()
    if str(path) in deps:
        digest = deps[str(path)]
        if expected is not None and digest != expected:
            raise ValueError('Conflicting dependency declarations: '+str(path))
        return digest
    digest = base.sha256(path)
    if expected is not None and digest != expected:
        raise ValueError('Pinned dependency changed: '+str(path))
    deps[str(path)] = digest
    return digest


def _population_rows(c, old):
    manifest = Path(c['population']['manifest'])
    document = base.checked_json(manifest)
    if document.get('schema_version') != 'fakect.sibling-population/1' or document.get('complete') is not True:
        raise ValueError('Expanded sibling population must be complete and use schema sibling-population/1')
    rows = document.get('samples')
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError('Population manifest requires a nonempty samples list')
    ids = [r.get('sample_id') for r in rows]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(ids) != len(set(ids)):
        raise ValueError('Population sample IDs must be unique nonempty strings')
    controls = {r['case_id']: r for r in old['samples'] if r['baseline']}
    allowed = {case for stage in old['stages'] for case in stage['train_case_ids']+stage['final_test_case_ids']}
    if set(controls) != allowed:
        raise ValueError('Reference lacks exactly one original control per development case')
    grouped = {case: [] for case in sorted(allowed)}
    for row in rows:
        if any(key not in row for key in ROW_FIELDS):
            raise ValueError('Expanded row lacks a required ContextVolume descriptor')
        case = row['case_id']
        if case not in allowed or case in old['reserved_case_ids']:
            raise ValueError('Expanded population may contain only existing development cases')
        control = controls[case]
        if row['anatomy_family'] != control['anatomy_family']:
            raise ValueError('Expanded sibling changed anatomy-family identity')
        if any(row[key] != control[key] for key in SOURCE_FIELDS):
            raise ValueError('Expanded sibling changed original source or reference mask: '+row['sample_id'])
        if row['baseline'] is True:
            if row['sample_id'] != control['sample_id']:
                raise ValueError('Expanded baseline must be the original control')
            continue
        if row['baseline'] is not False or row['variant_id'] == 'baseline':
            raise ValueError('Edited population rows require baseline=false and a variant ID')
        grouped[case].append(deepcopy(row))
    count = c['population']['siblings_per_case']
    selected = {}
    for case, candidates in grouped.items():
        if len(candidates) < count:
            raise ValueError('Insufficient completed new siblings for '+case)
        candidates.sort(key=lambda row: row['sample_id'])
        # Pure metadata sampling, fixed by seed; no outcome/severity filtering.
        random.Random(str(c['training']['seed'])+':'+case).shuffle(candidates)
        selected[case] = sorted(candidates[:count], key=lambda row: row['sample_id'])
    return document, selected, controls


def plan(c):
    parent = Path(c['reference']['experiment_directory'])
    old = base.checked_json(parent/'experiment-lock.json')
    reference.inspect_lock(old['config'], (parent/'input.ini').read_bytes())
    if old.get('schema_version') != reference.SCHEMA or len(old['stages']) != 5:
        raise ValueError('Reference must contain exactly the five original family holdouts')
    if any(c['training'][key] != old['config']['training'][key] for key in MATCHED_TRAINING):
        raise ValueError('Population control must preserve reference network/loss/sampling settings')
    if c['validation'] != old['config']['validation']:
        raise ValueError('Population control must preserve the predeclared localization gates')
    population, selected, controls = _population_rows(c, old)
    deps = {}
    _dependency(deps, parent/'experiment-lock.json')
    _dependency(deps, parent/'input.ini')
    population_digest = _dependency(deps, c['population']['manifest'])
    samples = {r['sample_id']: deepcopy(r) for r in old['samples']}
    old_ids = set(samples)
    for rows in selected.values():
        for row in rows:
            if row['sample_id'] in old_ids:
                raise ValueError('Expanded edited rows must be new siblings, not the fixed reference panel')
            samples[row['sample_id']] = row
    proposals, stages = {}, []
    count = c['population']['siblings_per_case']
    for original in old['stages']:
        if (original['protocol'] != 'family_holdout' or original['stage'] != 'outer_refit'
                or original['validation_case_ids'] or len(original['train_case_ids']) != 8
                or len(original['final_test_case_ids']) != 2):
            raise ValueError('Reference partitions do not match the fixed 8/2 development folds')
        folder = parent/'stages'/original['id']
        result = base.checked_json(folder/'results.json')
        if (result.get('status') != 'completed' or result.get('reserved_test_evaluated') is not False
                or result.get('epochs_completed') != 64 or result.get('optimizer_updates') != 14336
                or {r['sample_id'] for r in result['records']} != set(original['final_test_sample_ids'])):
            raise ValueError('Reference must be a complete fixed-budget localization result')
        _dependency(deps, folder/'results.json')
        stage = deepcopy(original)
        train_controls = list(original['train_control_ids'])
        stage['train_sample_ids'] = [r['sample_id'] for case in stage['train_case_ids'] for r in selected[case]]+train_controls
        stage['all_train_sample_ids'] = list(stage['train_sample_ids'])
        stage['all_final_test_sample_ids'] = list(stage['final_test_sample_ids'])
        stage['validation_sample_ids'] = []
        stage['all_validation_sample_ids'] = []
        stage['counts'] = {'train': {'cases': 8, 'edited': 8*count, 'controls': 8,
                                     'selected_samples': 8*count+8},
                           'validation': {'cases': 0, 'selected_samples': 0},
                           'final_test': {'cases': 2, 'edited': 12, 'controls': 2,
                                          'selected_samples': len(stage['final_test_sample_ids'])},
                           'reserved': {'cases': 2, 'selected_samples': 0}}
        stage['training_schedule'] = {
            'reporting_blocks': 64, 'updates_per_block': 224, 'total_optimizer_updates': 14336,
            'edited_updates': 12288, 'control_updates': 2048,
            'edited_updates_per_case': 1536, 'control_updates_per_case': 256,
            'edited_updates_per_block': 192, 'control_updates_per_block': 32,
            'minimum_edited_visits': 2*(768//count), 'maximum_edited_visits': 2*math.ceil(768/count),
            'policy': 'Case-balanced without-replacement edited cycles; fixed 1/7 unchanged controls',
            'reporting_block_is_complete_population_epoch': False,
            'final_checkpoint_policy': 'declared final optimizer update; no held-out selection'}
        stage['checkpoint_policy'] = 'fresh same-seed network; declared final 14,336 updates only'
        stage['classifier_policy'] = 'classifier not started'
        stage['locator_policy'] = 'frozen original fold locator; new training images only; no QA recentering'
        rotation_folder = Path(old['parent_directory'])/'stages'/stage['id']
        locator_result = base.checked_json(rotation_folder/'results.json')
        _dependency(deps, rotation_folder/'results.json')
        locator = locator_result['locator']
        stage['locator_checkpoint'] = locator['checkpoint']
        stage['locator_checkpoint_sha256'] = _dependency(deps, locator['checkpoint'],
                                                        expected=locator['checkpoint_sha256'])
        fixed = stage['final_test_sample_ids']+train_controls
        proposals[stage['id']] = {sid: deepcopy(old['proposals'][stage['id']][sid]) for sid in fixed}
        stages.append(stage)
    needed = {sid for stage in stages for key in ('train_sample_ids', 'final_test_sample_ids') for sid in stage[key]}
    rows = [samples[sid] for sid in sorted(needed)]
    if any(row['case_id'] in old['reserved_case_ids'] for row in rows):
        raise ValueError('Reserved infants cannot enter the expanded change experiment')
    # Per-case metadata is independently pinned; payload hashes are retained in
    # every sample descriptor and checked by the bounded reader before use.
    for row in rows:
        for path_key, hash_key in (('prepared_manifest', 'prepared_manifest_sha256'),
                                   ('selection_manifest', 'selection_manifest_sha256')):
            if path_key in row:
                _dependency(deps, row[path_key], expected=row[hash_key])
    return {'schema_version': SCHEMA, 'config': c, 'stages': stages, 'samples': rows,
            'proposals': proposals, 'runtime': deepcopy(old['runtime']),
            'parent_directory': str(parent), 'parent_lock_sha256': base.sha256(parent/'experiment-lock.json'),
            'population_manifest': c['population']['manifest'],
            'population_manifest_sha256': population_digest,
            'population_schema_version': population['schema_version'],
            'dependency_sha256': deps, 'reserved_case_ids': old['reserved_case_ids'],
            'reserved_test_evaluated': False, 'classifier_started': False,
            'target_semantics': old['target_semantics'], 'inputs': old['inputs'],
            'comparison_contract': 'Only TRAIN sibling diversity changes; fixed held-out 60 edited scans, 10 controls and 14,336 updates',
            'scope': 'Development family holdout; fixed optimizer budget; no clinical validation'}


def freeze(c, content):
    lock = plan(c)
    payloads = {name: (ROOT/name).read_bytes() for name in CODE_FILES}
    out = Path(c['experiment']['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'INCOMPLETE').touch()
    (out/'input.ini').write_bytes(content)
    lock['code_sha256'] = {}
    for name, data in payloads.items():
        path = out/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        lock['code_sha256'][name] = hashlib.sha256(data).hexdigest()
    lock['input_sha256'] = hashlib.sha256(content).hexdigest()
    base.write_json(out/'experiment-lock.json', lock)
    (out/'INCOMPLETE').unlink()
    return lock


def inspect_lock(c, content):
    out = Path(c['experiment']['output_directory'])
    lock = base.checked_json(out/'experiment-lock.json')
    if (lock['schema_version'] != SCHEMA or lock['config'] != c
            or lock['input_sha256'] != hashlib.sha256(content).hexdigest()
            or base.sha256(out/'input.ini') != lock['input_sha256']):
        raise ValueError('Frozen population experiment input changed; use a fresh revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Frozen population experiment source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if base.sha256(ROOT/name) != digest or base.sha256(out/'source'/name) != digest:
            raise ValueError('Pinned implementation changed: '+name)
    for name, digest in lock['dependency_sha256'].items():
        if name.endswith('.json') and Path(str(name)+'.sha256').exists():
            base.checked_json(name, digest)
        elif base.sha256(name) != digest:
            raise ValueError('Pinned dependency changed: '+name)
    parent = base.checked_json(Path(lock['parent_directory'])/'experiment-lock.json')
    reference.inspect_lock(parent['config'], (Path(lock['parent_directory'])/'input.ini').read_bytes())
    return lock


def require_smoke(c, lock):
    out = Path(c['experiment']['output_directory'])
    smoke = base.checked_json(out/'runtime-smoke.json')
    if (smoke.get('passed') is not True or smoke.get('gpu_visible') is not True
            or smoke.get('reload_parity') is not True
            or smoke.get('source_lock_sha256') != base.sha256(out/'experiment-lock.json')
            or smoke.get('train_traces') != 1 or smoke.get('predict_traces') != 1):
        raise ValueError('Own production-patch GPU smoke must pass before fitting')
    return smoke
