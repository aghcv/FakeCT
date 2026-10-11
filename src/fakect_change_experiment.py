"""Frozen CT-only net-change localization, before downstream severity learning."""
from copy import deepcopy
import configparser
import hashlib
import math
from pathlib import Path

import fakect_multiscale_experiment as base
import fakect_coa_rotation_experiment as rotation

ROOT = base.ROOT
SCHEMA = 'fakect.focused-change/1'
NEW_CODE = ('src/fakect_change_experiment.py', 'src/fakect_change_learning.py',
            'src/fakect_change_network.py', 'src/fakect_change_data.py',
            'src/fakect_change_report.py', 'src/fakect_two_stage_3d.py',
            'src/fakect_surface_overlay.py', 'src/fakect_volume_preview.py',
            'scripts/research/train_focused_change.py', 'scripts/research/slurm/fakect_change_gpu.sh',
            'scripts/research/slurm/fakect_change_report.sh')
CODE_FILES = tuple(dict.fromkeys(rotation.CODE_FILES + NEW_CODE))
FIELDS = {
    'experiment': {'schema_version', 'output_directory'},
    'reference': {'experiment_directory'},
    'evaluation': {'families'},
    'training': {'epochs', 'patch_shape_kji', 'patches_per_sample', 'base_filters',
                 'learning_rate', 'change_weight', 'dice_weight', 'changed_fraction',
                 'inference_overlap', 'seed'},
    'validation': {'minimum_dice', 'minimum_precision', 'minimum_recall',
                   'maximum_control_change_fraction'},
}


def load_config(path):
    p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                 empty_lines_in_values=False)
    p.read_string(Path(path).read_text())
    if p.defaults() or set(p.sections()) != set(FIELDS):
        raise ValueError('Expected exactly experiment, reference, evaluation, training, validation')
    c = {s: dict(p[s]) for s in FIELDS}
    if any(set(c[s]) != FIELDS[s] for s in FIELDS):
        raise ValueError('Unsupported or missing focused-change fields')
    if any('\n' in v or '\r' in v for s in c.values() for v in s.values()):
        raise ValueError('Every input value must occupy one line')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported focused-change schema')
    for s in ('experiment', 'reference'):
        k = 'output_directory' if s == 'experiment' else 'experiment_directory'
        v = Path(c[s][k]).expanduser()
        c[s][k] = str((v if v.is_absolute() else ROOT/v).resolve())
    out, parent = Path(c['experiment']['output_directory']), Path(c['reference']['experiment_directory'])
    if out == parent or out in parent.parents or parent in out.parents:
        raise ValueError('Output must be separate from the frozen reference')
    families = [v.strip() for v in c['evaluation']['families'].split(',')]
    if not families or any(not v for v in families) or len(set(families)) != len(families):
        raise ValueError('Specify unique held-out families')
    c['evaluation']['families'] = families
    t = c['training']
    t['patch_shape_kji'] = [int(v.strip()) for v in t['patch_shape_kji'].split(',')]
    if (len(t['patch_shape_kji']) != 3 or any(n < 16 or n % 8 for n in t['patch_shape_kji'])
            or math.prod(t['patch_shape_kji']) > 1_048_576):
        raise ValueError('Patch shape must be divisible by8, >=16 and <=1,048,576 voxels')
    for k, lo, hi in [('epochs', 1, 300), ('patches_per_sample', 1, 16),
                      ('base_filters', 2, 8), ('seed', 0, 2**31-1), ('inference_overlap', 0, 64)]:
        t[k] = int(t[k])
        if not lo <= t[k] <= hi:
            raise ValueError('Invalid training.'+k)
    if t['inference_overlap'] > min(t['patch_shape_kji'])//2:
        raise ValueError('Inference overlap must not exceed half the smallest patch dimension')
    for k in ('learning_rate', 'change_weight', 'dice_weight', 'changed_fraction'):
        t[k] = float(t[k])
        if not math.isfinite(t[k]) or t[k] < 0:
            raise ValueError('Invalid training.'+k)
    if (not 0 < t['learning_rate'] <= .1 or not 1 <= t['change_weight'] <= 64
            or t['dice_weight'] > 10 or not 0 <= t['changed_fraction'] <= 1):
        raise ValueError('Invalid learning rate, change weight or changed sampling fraction')
    for k in c['validation']:
        c['validation'][k] = float(c['validation'][k])
        if not math.isfinite(c['validation'][k]) or not 0 <= c['validation'][k] <= 1:
            raise ValueError('Validation thresholds must be in [0,1]')
    return c


def plan(c):
    parent = Path(c['reference']['experiment_directory'])
    old = base.checked_json(parent/'experiment-lock.json')
    rotation.inspect_lock(old['config'], (parent/'input.ini').read_bytes())
    if any(p > n for p,n in zip(c['training']['patch_shape_kji'],old['runtime_geometry']['input.detail']['shape_kji'])):
        raise ValueError('Training patch must fit the inherited focus')
    candidates = [s for s in old['stages'] if s['protocol'] == 'family_holdout' and s['stage'] == 'outer_refit']
    stages = []
    for family in c['evaluation']['families']:
        matches = [s for s in candidates if s['outer_fold_id'] == family]
        if len(matches) != 1:
            raise ValueError('No unique completed parent family fold: '+family)
        stages.append(deepcopy(matches[0]))
    deps = {str(parent/'experiment-lock.json'): base.sha256(parent/'experiment-lock.json'),
            str(parent/'input.ini'): base.sha256(parent/'input.ini')}
    sample_by_id = {r['sample_id']: r for r in old['samples']}
    controls = {r['case_id']: r for r in old['baseline_samples']}
    proposals, baselines = {}, []
    for stage in stages:
        folder = parent/'stages'/stage['id']
        result = base.checked_json(folder/'results.json')
        if result['status'] != 'completed' or result['reserved_test_evaluated'] is not False:
            raise ValueError('Parent fold must be complete and reserved data unopened')
        deps[str(folder/'results.json')] = base.sha256(folder/'results.json')
        stage_proposals = {}
        for name, key, role in [('proposal-audit.json', 'train_sample_ids', 'train'),
                                ('final-test/proposal-audit.json', 'final_test_sample_ids', 'final_test')]:
            audit = base.checked_json(folder/name)
            rows = audit['samples']
            if ({r['sample_id'] for r in rows} != set(stage[key])
                    or len(rows) != len(stage[key]) or audit['selection_allowed'] is not False
                    or any(r['split'] != role for r in rows)):
                raise ValueError('Parent proposal roles or exact sample panel changed')
            deps[str(folder/name)] = base.sha256(folder/name)
            stage_proposals.update({r['sample_id']: deepcopy(r) for r in rows})
        for role, cases, ids in [('train', stage['train_case_ids'], stage['train_sample_ids']),
                                  ('final_test', stage['final_test_case_ids'], stage['final_test_sample_ids'])]:
            baseline_ids = []
            for case in cases:
                if case not in controls:
                    raise ValueError('Missing unchanged control: '+case)
                control = deepcopy(controls[case])
                if not control['baseline']:
                    raise ValueError('Control must be an original baseline')
                baseline_ids.append(control['sample_id'])
                if control['sample_id'] not in sample_by_id:
                    sample_by_id[control['sample_id']] = control
                    baselines.append(control)
                # Deterministic paired diagnostic framing, never target-guided.
                first = min(sid for sid in ids if sample_by_id[sid]['case_id'] == case)
                proposal = deepcopy(stage_proposals[first])
                proposal.update(sample_id=control['sample_id'], split=role,
                                control_crop_reference_sample_id=first,
                                control_crop_policy='fixed paired focus from lexicographically first sibling')
                stage_proposals[control['sample_id']] = proposal
            stage[role+'_control_ids'] = baseline_ids
        stage['train_sample_ids'] += stage['train_control_ids']
        stage['final_test_sample_ids'] += stage['final_test_control_ids']
        proposals[stage['id']] = stage_proposals
    needed = {sid for s in stages for key in ('train_sample_ids','final_test_sample_ids') for sid in s[key]}
    samples = [deepcopy(sample_by_id[sid]) for sid in sorted(needed)]
    if any(r['case_id'] in old['reserved_case_ids'] for r in samples):
        raise ValueError('Reserved infants cannot enter the change experiment')
    return {'schema_version': SCHEMA, 'config': c, 'stages': stages, 'samples': samples,
            'proposals': proposals, 'runtime': deepcopy(old['runtime_geometry']),
            'parent_directory': str(parent), 'parent_lock_sha256': base.sha256(parent/'experiment-lock.json'),
            'dependency_sha256': deps, 'reserved_case_ids': old['reserved_case_ids'],
            'reserved_test_evaluated': False, 'classifier_started': False,
            'target_semantics': 'net final-vs-original: unchanged=0, added=1, removed=2',
            'inputs': 'normalized sibling attenuation and scan validity only',
            'scope': 'Development family holdout; final fixed epoch; no clinical validation'}


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
    if lock['config'] != c or lock['input_sha256'] != hashlib.sha256(content).hexdigest():
        raise ValueError('Frozen input changed; use a fresh revision')
    if base.sha256(out/'input.ini') != lock['input_sha256'] or set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Frozen input or source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if base.sha256(ROOT/name) != digest or base.sha256(out/'source'/name) != digest:
            raise ValueError('Pinned source changed: '+name)
    for name, digest in lock['dependency_sha256'].items():
        base.checked_json(name) if name.endswith('.json') else None
        if base.sha256(name) != digest:
            raise ValueError('Pinned parent changed: '+name)
    parent = base.checked_json(Path(lock['parent_directory'])/'experiment-lock.json')
    rotation.inspect_lock(parent['config'], (Path(lock['parent_directory'])/'input.ini').read_bytes())
    return lock


def require_smoke(c, lock):
    out = Path(c['experiment']['output_directory'])
    s = base.checked_json(out/'runtime-smoke.json')
    if (s.get('passed') is not True or s.get('source_lock_sha256') != base.sha256(out/'experiment-lock.json')
            or s.get('gpu_visible') is not True or s.get('reload_parity') is not True
            or s.get('train_traces') != 1 or s.get('predict_traces') != 1):
        raise ValueError('Own production-patch GPU smoke must pass before fitting')
    return s
