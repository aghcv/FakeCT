"""Immutable classifier-only continuation from a reviewed spatial locator.

No data are synthesized and no locator optimization is performed. The exact
development panel, label definitions, crop geometry and locator are inherited
from a pinned parent experiment and its successful checkpoint review.
"""
from copy import deepcopy
import configparser
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import fakect_multiscale_experiment_v3 as parent
from fakect_multiscale_experiment import ROOT, SCORE, sha256, checked_json, write_json

SCHEMA = 'fakect.coa-classifier/1'
FIELDS = {'experiment':{'schema_version','output_directory'},
          'locator':{'experiment_directory','review_directory'},
          'training':{'epochs','learning_rate','seed','host_memory_limit_mb','gpu_memory_limit_mb','require_gpu'}}
CODE_FILES = parent.CODE_FILES + (
    'src/fakect_coa_classifier_experiment.py','src/fakect_coa_classifier_learning.py',
    'src/fakect_coa_classifier_report.py','scripts/train_coa_classifier.py',
    'scripts/slurm/fakect_coa_classifier_gpu.sh')
REVIEW_FILES = ('review-provenance','results','proposal-audit','locator-robustness','experiment-lock')


def load_config(path):
    p = configparser.ConfigParser(interpolation=None,inline_comment_prefixes=('#',),empty_lines_in_values=False)
    p.read_string(Path(path).read_text())
    if p.defaults() or set(p.sections()) != set(FIELDS):
        raise ValueError('Expected exactly [experiment], [locator], [training]')
    c = {section:dict(p[section]) for section in FIELDS}
    if any(set(c[s]) != keys for s,keys in FIELDS.items()) or c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported classifier schema or input fields')
    for section,key in [('experiment','output_directory'),('locator','experiment_directory'),('locator','review_directory')]:
        value = Path(c[section][key]).expanduser()
        c[section][key] = str((value if value.is_absolute() else ROOT/value).resolve())
    t = c['training']
    for key,lo,hi in [('epochs',1,100),('seed',0,2**31-1),('host_memory_limit_mb',512,12288),('gpu_memory_limit_mb',512,12288)]:
        t[key] = int(t[key])
        if not lo <= t[key] <= hi:
            raise ValueError(f'training.{key} outside [{lo},{hi}]')
    t['learning_rate'] = float(t['learning_rate'])
    if not math.isfinite(t['learning_rate']) or t['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive and finite')
    t['require_gpu'] = p.getboolean('training','require_gpu')
    out = Path(c['experiment']['output_directory'])
    if any(out == Path(c['locator'][k]) or Path(c['locator'][k]) in out.parents for k in c['locator']):
        raise ValueError('Continuation output must be separate from parent and review directories')
    return c


def validate_samples(lock):
    """Reject test leakage and cross-partition relatives before any image read."""
    rows = lock['samples']
    ids = [r['sample_id'] for r in rows]
    reserved = set(lock['reserved_test_sample_ids'])
    if not rows or len(ids) != len(set(ids)) or reserved.intersection(ids):
        raise ValueError('Duplicate, empty or reserved test sample panel')
    folds = lock['design']['folds']
    if len(folds) != 1:
        raise ValueError('Classifier continuation requires the single frozen development fold')
    assignment = folds[0]['family_assignment']
    families = lock['design']['families']
    groups = {key:{} for key in ('case_id','anatomy_family','source_atn','original_reference_mask_sha256')}
    for row in rows:
        split = row['split']; family = row['anatomy_family']
        if split not in ('train','validation') or assignment.get(family) != split:
            raise ValueError('Test or inconsistent family assignment in classifier panel')
        if row['case_id'] not in families.get(family,[]):
            raise ValueError('Case does not belong to its frozen anatomy family')
        for key,mapping in groups.items():
            value = row.get(key)
            if key == 'source_atn' and isinstance(value,dict):
                value = value.get('path')
            if key == 'source_atn' and value:
                value = str(Path(value).resolve())
            if not value:
                raise ValueError('Missing partition identity: '+key)
            if value in mapping and mapping[value] != split:
                raise ValueError('Shared '+key+' crosses training/validation partitions')
            mapping[value] = split
        label = row['annotation']
        if (label.get('sample_id') != row['sample_id'] or label.get('case_id') != row['case_id'] or
                label.get('anatomy_family') != family or type(label.get('coa_benchmark_positive')) is not bool or
                label.get('errors') or not math.isfinite(float(label[SCORE])) or not 0 <= label[SCORE] <= 1):
            raise ValueError('Invalid condition annotation identity or score')
    for split in ('train','validation'):
        group = [r for r in rows if r['split']==split]
        counts = lock['pilot_counts'][split]
        if (len(group) != counts['samples'] or sum(r['annotation']['coa_benchmark_positive'] for r in group) != counts['positive'] or
                {r['annotation']['coa_benchmark_positive'] for r in group} != {False,True}):
            raise ValueError('Panel class counts changed or a development split lacks a class')
    return rows


def validate_review(pc, lock, prepared, documents):
    """Validate exact joins and original coverage/robustness policy metadata."""
    rows = validate_samples(lock)
    by_id = {r['sample_id']:r for r in rows}
    if documents['experiment-lock'] != lock:
        raise ValueError('Review belongs to a different parent experiment')
    result, audit, robust = (documents[n] for n in ('results','proposal-audit','locator-robustness'))
    if (result.get('status') != 'locator_ready_for_classifier' or result.get('coverage_passed') is not True or
            result.get('robustness_passed') is not True or result.get('classifier_trained') is not False or
            result.get('test_evaluated') is not False or result.get('evaluation_only') is not True or
            result.get('failed_samples') or result.get('passing_samples') != len(rows) or result.get('total_samples') != len(rows)):
        raise ValueError('Locator review is incomplete or not ready for classifier training')
    if audit.get('passed') is not True or audit.get('failed_sample_ids'):
        raise ValueError('Review coverage gate failed')
    proposal_rows = audit['samples']
    proposals = {r['sample_id']:r for r in proposal_rows}
    prepared_rows = prepared['samples']
    preparation = {r['sample_id']:r for r in prepared_rows}
    if (len(proposals) != len(proposal_rows) or len(preparation) != len(prepared_rows) or
            set(proposals) != set(by_id) or set(preparation) != set(by_id)):
        raise ValueError('Proposal/preparation samples must join the frozen panel exactly')
    for key,row in proposals.items():
        if row['split'] != by_id[key]['split'] or row.get('coverage_passed') is not True or not parent.coverage_gate(row,pc):
            raise ValueError('Individual proposal coverage or split failed: '+key)
        ov = {'image':SimpleNamespace(shape=pc['input.overview']['shape_kji']),
              'spacing_ijk_mm':preparation[key]['overview_spacing_ijk_mm']}
        origins,boxes = parent.proposal_origins(row['predicted_box'],ov,pc)
        actual = np.asarray(row['origins_kji'])
        if (actual.shape != np.asarray(origins).shape or not np.array_equal(actual,origins) or
                not np.allclose(boxes,row['proposal_boxes'],atol=1e-7,rtol=0)):
            raise ValueError('Reviewed detail geometry does not follow image-predicted bounds')
    for key in ('passed','response_gate_passed','native_coverage_gate_passed','image_dependence_gate_passed'):
        if robust.get(key) is not True:
            raise ValueError('Locator robustness gate failed: '+key)
    if robust.get('test_evaluated') is not False:
        raise ValueError('Robustness review must exclude reserved test data')
    originals = {r['sample_id']:r for r in rows if r['baseline']}
    expected = {(key,axis+sign) for key in originals for axis in ('k','j','i') for sign in ('-','+')}
    probes = robust['probes']
    actual = {(r['sample_id'],r['probe']) for r in probes}
    if actual != expected or len(actual) != len(probes):
        raise ValueError('Fixed robustness probes must join every development original exactly')
    for probe in probes:
        if probe['split'] != originals[probe['sample_id']]['split'] or not probe['eligible']:
            raise ValueError('Ineligible or misassigned fixed robustness probe')
        if probe['split']=='validation':
            if (probe.get('coverage_passed') is not True or not parent.coverage_gate(probe['native_coverage'],pc) or
                    not math.isfinite(probe['relative_response_error']) or not 0 <= probe['relative_response_error'] <= 1):
                raise ValueError('Validation robustness probe failed coverage or movement')
    valid = [r for r in probes if r['split']=='validation']
    for axis in range(3):
        values = [r['relative_response_error'] for r in valid if r['axis']==axis]
        if not values or np.mean(values) > pc['robustness']['maximum_relative_error']:
            raise ValueError('Validation mean movement error exceeds threshold')
    identities = robust['identities']
    if len(identities) != len(originals) or {r['sample_id'] for r in identities} != set(originals):
        raise ValueError('Blank-image identity controls do not match originals')
    if any(r['split'] != originals[r['sample_id']]['split'] for r in identities):
        raise ValueError('Blank-image control split mismatch')
    paired = valid+[r for r in identities if r['split']=='validation']
    real = np.asarray([r['real_error_mm'] for r in paired],float)
    blank = np.asarray([r['blank_error_mm'] for r in paired],float)
    if (not len(real) or not np.isfinite(real).all() or not np.isfinite(blank).all() or
            np.any(real<0) or np.any(blank<0) or blank.mean() <= 0 or real.mean() > .75*blank.mean()):
        raise ValueError('Matched blank-image dependence screen failed')
    return proposals


def source_contract(c):
    source = Path(c['locator']['experiment_directory'])
    review = Path(c['locator']['review_directory'])
    if (review/'failure.json').exists():
        raise ValueError('Failed review cannot authorize classifier continuation')
    pc = parent.load_config(source/'input.ini')
    if Path(pc['experiment']['output_directory']).resolve() != source:
        raise ValueError('Parent input points to another experiment directory')
    lock = parent.inspect_lock(pc,(source/'input.ini').read_bytes())
    validate_samples(lock)
    prepared = checked_json(source/'preparation.json')
    if prepared.get('complete') is not True or prepared['lock_sha256'] != sha256(source/'experiment-lock.json'):
        raise ValueError('Parent preparation is incomplete or mismatched')
    docs = {name:checked_json(review/(name+'.json')) for name in REVIEW_FILES}
    p = docs['review-provenance']; result = docs['results']
    checkpoint = source/'model/locator.keras'
    digest = sha256(checkpoint)
    history = checked_json(source/'locator-history.json')
    best = min(history['epochs'],key=lambda x:x['validation_loss'])
    if (Path(p['parent_experiment']).resolve() != source or p['parent_lock_sha256'] != sha256(source/'experiment-lock.json') or
            Path(p['checkpoint']).resolve() != checkpoint or p['checkpoint_sha256'] != digest or result['checkpoint_sha256'] != digest or
            p['selected_epoch'] != best['epoch'] or result['selected_epoch'] != best['epoch'] or
            p.get('evaluation_only') is not True or p.get('test_evaluated') is not False):
        raise ValueError('Review checkpoint, selected epoch or parent provenance mismatch')
    proposals = validate_review(pc,lock,prepared,docs)
    smoke = checked_json(source/'runtime-smoke.json')
    if (smoke.get('status') != 'passed' or smoke.get('checkpoint_parity') is not True or
            smoke.get('classifier_shape_kji') != pc['input.detail']['shape_kji'] or
            smoke.get('classifier_traces') != {'train':1,'predict':1}):
        raise ValueError('The inherited classifier geometry needs a passing GPU compatibility check')
    paths = [source/name for name in ('input.ini','experiment-lock.json','preparation.json',
                                    'locator-history.json','runtime-smoke.json','model/locator.keras')]
    paths += [review/(name+'.json') for name in REVIEW_FILES]
    for name,digest in p['source_sha256'].items():
        snapshot = (review/'source'/name).resolve()
        if review not in snapshot.parents or sha256(snapshot) != digest:
            raise ValueError('Review implementation snapshot changed: '+name)
        paths.append(snapshot)
    return {'parent_config':pc,'samples':lock['samples'],'proposals':proposals,
        'locator_checkpoint':{'path':str(checkpoint),'sha256':sha256(checkpoint),'selected_epoch':best['epoch']},
        'dependency_sha256':{str(path):sha256(path) for path in paths},
        'reserved_test_sample_ids':lock['reserved_test_sample_ids'],'pilot_counts':lock['pilot_counts'],
        'design':lock['design'],'annotation_acceptance':lock['annotation_acceptance']}


def plan(c):
    return {'schema_version':SCHEMA,'ready':True,'config':deepcopy(c),**source_contract(c),
        'scope':'classifier_only_engineering_pilot',
        'training_contract':{'locator_frozen':True,'overview_encoder_frozen':True,
            'initialization':'Fresh classifier weights and shuffle generator use the configured seed; interrupted optimizer/RNG state is not resumed.',
            'checkpoint_selection':'Minimum validation binary cross entropy; reserved test families are never consumed.',
            'proposal_contract':'Recompute saved locator predictions and require exact sampled crop origins before classifier gradients.',
            'learning_gate':'Training BCE improves20%; edited-only sensitivity/specificity>=.75 and positive-negative probability gap>=.15.'}}


def freeze(c,input_bytes):
    document = plan(c)
    sources = {name:(ROOT/name).read_bytes() for name in CODE_FILES}
    out = Path(c['experiment']['output_directory'])
    out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').touch()
    (out/'input.ini').write_bytes(input_bytes)
    hashes = {}
    for name,content in sources.items():
        target = out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content); hashes[name] = hashlib.sha256(content).hexdigest()
    document.update(code_sha256=hashes,input_sha256=hashlib.sha256(input_bytes).hexdigest())
    write_json(out/'experiment-lock.json',document)
    (out/'INCOMPLETE').unlink()
    return document


def inspect_lock(c,input_bytes):
    out = Path(c['experiment']['output_directory'])
    lock = checked_json(out/'experiment-lock.json')
    if (lock['schema_version'] != SCHEMA or lock['config'] != c or
            lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest() or sha256(out/'input.ini') != lock['input_sha256']):
        raise ValueError('Classifier input changed; use a new experiment revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Classifier source inventory changed')
    for name,digest in lock['code_sha256'].items():
        if sha256(ROOT/name) != digest or sha256(out/'source'/name) != digest:
            raise ValueError('Pinned classifier source changed: '+name)
    for name,digest in lock['dependency_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Pinned classifier dependency changed: '+name)
    current = plan(c)
    if any(lock.get(key) != value for key,value in current.items()):
        raise ValueError('Classifier parent, review or sample contract changed')
    return lock


def runtime_config(c,lock):
    value = deepcopy(lock['parent_config'])
    value['experiment']['output_directory'] = c['experiment']['output_directory']
    t = c['training']
    value['model'].update(learning_rate=t['learning_rate'],seed=t['seed'])
    value['pilot'].update(classifier_epochs=t['epochs'],require_gpu=t['require_gpu'],gpu_memory_limit_mb=t['gpu_memory_limit_mb'])
    value['training'].update(host_memory_limit_mb=t['host_memory_limit_mb'],maximum_traces=1)
    return value
