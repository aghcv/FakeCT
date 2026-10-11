"""Immutable orchestration for family and phantom held-out CoA experiments."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import fakect_coa_rotation_design as design
import fakect_coa_pilot_experiment as pilot
import fakect_multiscale_experiment as base
import fakect_multiscale_experiment_v3 as geometry

ROOT = base.ROOT
SCORE = base.SCORE
load_config = design.load_config
CODE_FILES = tuple(dict.fromkeys(pilot.CODE_FILES + (
    'src/fakect_coa_rotation_design.py', 'src/fakect_coa_rotation_experiment.py',
    'src/fakect_coa_rotation_learning.py', 'src/fakect_coa_rotation_locator.py',
    'src/fakect_coa_rotation_report.py', 'scripts/research/train_coa_rotations.py',
    'src/fakect_coa_poster_figures.py',
    'scripts/research/slurm/fakect_coa_rotations_gpu.sh',
)))


def plan(c):
    document = design.plan(c)
    rc = geometry.load_config(c['reference']['geometry_config'])
    # Run the existing achieved-edit, source-provenance and annotation checks.
    # This reads metadata/stat/hash evidence, not model predictions or image arrays.
    validated = geometry.plan(rc)
    if (rc['data']['manifest'] != c['data']['manifest'] or
            rc['data']['annotations'] != c['data']['annotations']):
        raise ValueError('Geometry reference and rotation must use the same frozen cohort')
    reference = Path(c['reference']['experiment_directory'])
    old_lock = base.checked_json(reference/'experiment-lock.json')
    if old_lock['parent_config'] != rc:
        raise ValueError('Reference geometry differs from the completed focused pilot')
    dataset = base.checked_json(c['data']['manifest'])
    labels = base.checked_json(c['data']['annotations'])
    joined = base.join_annotations(dataset['samples'], labels['samples'])
    reserved = set(document['reserved_case_ids'])
    originals = [row for row in joined if row['baseline'] and row['case_id'] not in reserved]
    selected = document['samples']
    if set(row['case_id'] for row in originals) != set(row['case_id'] for row in selected):
        raise ValueError('Each development phantom needs exactly one locator reference')
    if len(originals) != len({row['case_id'] for row in originals}):
        raise ValueError('Duplicate original locator references')
    paths = [Path(c['data'][key]) for key in ('project', 'manifest', 'annotations')]
    paths += [Path(c['reference']['geometry_config']), reference/'experiment-lock.json', reference/'input.ini']
    dependencies = {str(path): base.sha256(path) for path in paths}
    dependencies.update({row['path']: row['sha256'] for row in labels['source_provenance']})
    for stage in document['stages']:
        assignments = stage['case_assignment']
        seen = {}
        conflicts = []
        for row in originals:
            partition = assignments[row['case_id']]
            # Actual shared source or identical reference masks may never cross roles,
            # including the explicitly exploratory single-phantom protocols.
            for identity in (('source', row['source_atn']['path']),
                             ('reference', row['original_reference_mask_sha256'],
                              tuple(row['spacing_ijk_mm']))):
                if identity in seen and seen[identity] != partition:
                    conflicts.append({'identity_type':identity[0], 'case_id':row['case_id'],
                                      'partition':partition,'other_partition':seen[identity]})
                seen[identity] = partition
        if conflicts and stage['protocol']=='family_holdout':
            raise ValueError('Shared source anatomy crosses primary partitions in '+stage['id'])
        stage['execution_eligible'] = not conflicts
        stage['isolation_conflicts'] = conflicts
        if conflicts:
            stage['exclusion_reason'] = 'Known shared source or identical reference mask crosses roles; secondary fit prohibited'
    document.update(runtime_geometry=rc, original_samples=originals,
                    dependency_sha256=dependencies,
                    dataset_sha256=validated['dataset_sha256'],
                    annotation_sha256=validated['annotation_sha256'],
                    annotation_acceptance=validated['annotation_acceptance'],
                    fresh_locator_required=True, existing_locator_checkpoint_reused=False,
                    cache_policy='Bounded LRU attenuation/validity crops; no whole-body sibling export')
    return document


def freeze(c, content):
    document = plan(c)
    sources = {name: (ROOT/name).read_bytes() for name in CODE_FILES}
    out = Path(c['experiment']['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'INCOMPLETE').touch()
    (out/'input.ini').write_bytes(content)
    hashes = {}
    for name, payload in sources.items():
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    document.update(config=deepcopy(c), code_sha256=hashes,
                    input_sha256=hashlib.sha256(content).hexdigest())
    base.write_json(out/'experiment-lock.json', document)
    (out/'INCOMPLETE').unlink()
    return document


def inspect_lock(c, content):
    out = Path(c['experiment']['output_directory'])
    lock = base.checked_json(out/'experiment-lock.json')
    if (lock['config'] != c or lock['input_sha256'] != hashlib.sha256(content).hexdigest()
            or base.sha256(out/'input.ini') != lock['input_sha256']):
        raise ValueError('Frozen inputs changed; use a new experiment revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Frozen source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if base.sha256(ROOT/name) != digest or base.sha256(out/'source'/name) != digest:
            raise ValueError('Pinned implementation changed: '+name)
    for path, digest in lock['dependency_sha256'].items():
        if base.sha256(path) != digest:
            raise ValueError('Pinned metadata changed: '+path)
    for row in lock['samples']+lock['original_samples']:
        source = row['source_atn']
        stat = Path(source['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source['bytes'], source['mtime_ns']):
            raise ValueError('Source attenuation changed: '+source['path'])
    return lock


def stage_by_id(lock, stage_id):
    rows = [stage for stage in lock['stages'] if stage['id'] == stage_id]
    if len(rows) != 1:
        raise ValueError('Unknown or duplicate rotation stage: '+stage_id)
    return rows[0]


def runtime_config(c, lock, stage, output):
    rc = deepcopy(lock['runtime_geometry'])
    rc['experiment']['output_directory'] = str(output)
    rc['model'].update(seed=stage['seed'], learning_rate=c['training']['classifier_learning_rate'])
    rc['training'].update(locator_learning_rate=c['training']['locator_learning_rate'],
                          host_memory_limit_mb=c['training']['host_memory_limit_mb'])
    rc['pilot'].update(require_gpu=c['training']['require_gpu'],
                      gpu_memory_limit_mb=c['training']['gpu_memory_limit_mb'])
    return rc


def require_smoke(c, lock):
    out = Path(c['experiment']['output_directory'])
    smoke = base.checked_json(out/'runtime-smoke.json')
    lock_digest = base.sha256(out/'experiment-lock.json')
    detail = lock['runtime_geometry']['input.detail']
    per_sample = detail['proposals']*math.prod(detail['shape_kji'])*2*4
    sample_count = c['training']['cache_limit_mb']*1024**2//per_sample
    probe = smoke.get('full_cache_memory_probe', {})
    if (smoke.get('status') != 'passed' or smoke.get('checkpoint_parity') is not True or
            smoke.get('source_lock_sha256') != base.sha256(out/'experiment-lock.json') or
            smoke.get('full_cache_memory_probe', {}).get('model_update_while_resident') is not True or
            smoke.get('overview_shape') != lock['runtime_geometry']['input.overview']['shape_kji'] or
            smoke.get('detail_shape') != detail['shape_kji'] or
            smoke.get('traces') != {'train':1,'predict':1} or
            probe.get('source_lock_sha256') != lock_digest or probe.get('physical_pages_touched') is not True or
            probe.get('sample_count') != sample_count or probe.get('resident_bytes') != sample_count*per_sample or
            probe.get('passed') is not True or probe.get('host_guard_passed') is not True or
            (c['training']['require_gpu'] and smoke.get('gpu_visible') is not True)):
        raise ValueError('Own production-size smoke must pass before fitting')
    for field in ('locator_gradient_norm','classifier_gradient_norm'):
        value = smoke.get(field)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Both smoke models require finite nonzero gradients')
    for field, budget in (('host_peak_rss_mib',c['training']['host_memory_limit_mb']),
                          ('gpu_peak_mib',c['training']['gpu_memory_limit_mb'])):
        value = smoke.get(field,0 if not c['training']['require_gpu'] and field=='gpu_peak_mib' else None)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= budget:
            raise ValueError('Smoke exceeded or failed to record '+field)
    return smoke


def select_refit_budgets(lock, stage, root):
    """Select budgets exclusively from inner validation, never outer outcomes."""
    parents = stage.get('parent_inner_stage_ids', [])
    if not parents:
        raise ValueError('Outer refit requires completed inner selection stages')
    results = []
    for stage_id in parents:
        parent = stage_by_id(lock, stage_id)
        if (parent['outer_fold_id'] != stage['outer_fold_id'] or
                parent['stage'] != 'inner_selection' or parent['seed'] != stage['seed']):
            raise ValueError('Refit parent belongs to another outer fold or seed')
        row = base.checked_json(Path(root)/'stages'/stage_id/'results.json')
        if row.get('status') != 'completed' or row.get('final_test_evaluated') is not False:
            raise ValueError('Inner selection must complete without outer testing')
        if row.get('stage_id') != stage_id or row.get('source_lock_sha256') != base.sha256(Path(root)/'experiment-lock.json'):
            raise ValueError('Inner result belongs to another experiment')
        results.append(row)
    # Robust budget summary fixed in the protocol; no best-test-fold cherry-picking.
    import numpy as np
    return {'locator_epochs': max(1, int(np.floor(np.median([r['locator_selected_epoch'] for r in results])+.5))),
            'classifier_epochs': max(1, int(np.floor(np.median([r['classifier_selected_epoch'] for r in results])+.5))),
            'source_inner_stage_ids': parents,
            'rule': 'Rounded median inner selected epochs; fresh final-epoch refit, no outer early stopping'}


def report(c, lock):
    from fakect_coa_rotation_report import write_rotation_report
    out = Path(c['experiment']['output_directory'])
    records, receipts, status = [], [], []
    for stage in lock['stages']:
        folder = out/'stages'/stage['id']
        path = folder/'results.json'
        if path.exists() and not (folder/'INCOMPLETE').exists():
            row = base.checked_json(path)
            if (row.get('source_lock_sha256') != base.sha256(out/'experiment-lock.json') or
                    row.get('stage_id') != stage['id'] or row.get('protocol') != stage['protocol'] or
                    row.get('status') != 'completed' or row.get('reserved_test_evaluated') is not False or
                    (folder/'failure.json').exists()):
                raise ValueError('Stage result belongs to a different frozen plan')
            status.append({'stage_id':stage['id'], 'status':row['status']})
            records.extend(row.get('final_predictions', []))
            if stage['stage'] != 'outer_refit':
                receipts.append({'protocol': stage['protocol'], 'fold_id': stage['outer_fold_id'],
                                 'stage_id':stage['id'],
                                 'seed': stage['seed'], 'validation_case_ids':stage['validation_case_ids']})
        elif (folder/'failure.json').exists():
            status.append({'stage_id': stage['id'], 'status':'failed'})
            failure_path = folder/'failure.json'
            if base.sha256(failure_path) != Path(str(failure_path)+'.sha256').read_text().strip():
                raise ValueError('Failure receipt checksum changed')
            receipts.append({'stage_id':stage['id'],'status':'failed',
                             'reason':json.loads(failure_path.read_text()).get('message')})
        else:
            status.append({'stage_id': stage['id'], 'status':('excluded_known_shared_reference' if not stage['execution_eligible']
                           else 'running' if folder.exists() else 'pending'),
                           'reason':stage.get('exclusion_reason')})
            if not stage['execution_eligible']:
                receipts.append({'stage_id':stage['id'],'status':'blocked','reason':stage['exclusion_reason']})
    from fakect_coa_rotation_report import rotation_manifest
    revision = 1
    while (out/f'review-r{revision}').exists():
        revision += 1
    review = out/f'review-r{revision}'
    result = write_rotation_report(review, records, rotation_manifest(lock),
        validation_receipts=receipts,
        provenance={'source_lock_sha256':base.sha256(out/'experiment-lock.json'),
                    'stage_status':status, 'reserved_case_ids':lock['reserved_case_ids']})
    target = review.name+'/report.html'
    (out/'report.html').write_text('<!doctype html><meta http-equiv="refresh" content="0;url='+target+'"><a href="'+target+'">Open rotating evaluation report</a>')
    return result
