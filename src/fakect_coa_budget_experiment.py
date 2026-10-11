"""Immutable classifier-budget control with completed fold-local locator crops."""
from copy import deepcopy
import configparser
import hashlib
import json
from pathlib import Path

import fakect_coa_rotation_experiment as rotation
import fakect_multiscale_experiment as base

ROOT = base.ROOT
SCHEMA = 'fakect.coa-budget-control/1'
CODE_FILES = tuple(dict.fromkeys(rotation.CODE_FILES + (
    'src/fakect_coa_budget_experiment.py', 'src/fakect_coa_budget_learning.py',
    'src/fakect_coa_budget_report.py', 'scripts/train_coa_budget.py',
    'scripts/slurm/fakect_coa_budget_gpu.sh',
)))
FIELDS = {'experiment': {'schema_version', 'output_directory'},
          'reference': {'experiment_directory'},
          'training': {'classifier_epochs', 'checkpoint_interval'}}


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(FIELDS):
        raise ValueError('Expected exactly [experiment], [reference], [training]')
    c = {section: dict(parser[section]) for section in FIELDS}
    if any(set(c[section]) != fields for section, fields in FIELDS.items()):
        raise ValueError('Unsupported classifier-budget fields')
    if any('\n' in value or '\r' in value for section in c.values() for value in section.values()):
        raise ValueError('Budget input values must occupy one line')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported classifier-budget schema')
    for section, key in [('experiment', 'output_directory'), ('reference', 'experiment_directory')]:
        p = Path(c[section][key]).expanduser()
        c[section][key] = str((p if p.is_absolute() else ROOT/p).resolve())
    for key in FIELDS['training']:
        c['training'][key] = int(c['training'][key])
        if not 1 <= c['training'][key] <= 1000:
            raise ValueError('training.'+key+' must be in [1, 1000]')
    if c['training']['classifier_epochs'] != 200:
        raise ValueError('This control revision requires classifier_epochs = 200')
    if c['training']['classifier_epochs'] % c['training']['checkpoint_interval']:
        raise ValueError('checkpoint_interval must divide classifier_epochs')
    out, parent = Path(c['experiment']['output_directory']), Path(c['reference']['experiment_directory'])
    if out == parent or parent in out.parents or out in parent.parents:
        raise ValueError('Control output must be separate from its parent experiment')
    return c


def parent_lock(c):
    parent = Path(c['reference']['experiment_directory'])
    document = base.checked_json(parent/'experiment-lock.json')
    rotation.inspect_lock(document['config'], (parent/'input.ini').read_bytes())
    rotation.require_smoke(document['config'], document)
    return document


def checked_audit(path, rows, role, *, training=False):
    """Require the exact reviewed panel and roles; QA never chooses new crops."""
    audit = base.checked_json(path)
    samples = audit.get('samples', [])
    expected = {r['sample_id'] for r in rows}
    actual = [r['sample_id'] for r in samples]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Reviewed proposal panel differs from frozen samples')
    if any(r['split'] != role for r in samples) or audit.get('selection_allowed') is not False:
        raise ValueError('Reviewed crop roles or selection policy changed')
    if training and (audit.get('passed') is not True or not all(r['coverage_passed'] for r in samples)):
        raise ValueError('Reviewed training crops did not pass coverage')
    return {r['sample_id']: r for r in samples}, samples


def plan(c):
    parent = Path(c['reference']['experiment_directory'])
    old = parent_lock(c)
    stages = [deepcopy(s) for s in old['stages']
              if s['protocol'] == 'family_holdout' and s['stage'] == 'outer_refit']
    if len(stages) != 5 or len({s['outer_fold_id'] for s in stages}) != 5 or {s['seed'] for s in stages} != {42}:
        raise ValueError('This control requires the five reviewed seed-42 primary folds')
    if old['reserved_case_ids'] != ['260612', '260617']:
        raise ValueError('Reserved infant policy changed')
    dependencies = {}
    for name in ['experiment-lock.json', 'input.ini', 'runtime-smoke.json']:
        p = parent/name
        dependencies[str(p)] = base.sha256(p)
    for stage in stages:
        if not stage['execution_eligible'] or stage['validation_case_ids']:
            raise ValueError('Only eligible fixed-budget outer folds are permitted')
        folder = parent/'stages'/stage['id']
        if (folder/'INCOMPLETE').exists() or (folder/'failure.json').exists():
            raise ValueError('Parent outer fold did not finish cleanly')
        result = base.checked_json(folder/'results.json')
        if (result['status'] != 'completed' or result['stage_id'] != stage['id']
                or result['source_lock_sha256'] != base.sha256(parent/'experiment-lock.json')
                or result['reserved_test_evaluated'] is not False
                or result['final_test_evaluated'] is not True):
            raise ValueError('Parent result does not belong to the completed fold')
        locator = base.checked_json(folder/'locator/locator-result.json')
        classifier = base.checked_json(folder/'classifier/receipt.json')
        if locator != result['locator'] or classifier != result['classifier']:
            raise ValueError('Parent model receipts differ from completed result')
        expected_originals = {r['sample_id'] for r in old['original_samples']
                              if r['case_id'] in stage['train_case_ids']}
        if (set(locator['training_sample_ids']) != expected_originals
                or locator['validation_sample_ids'] or locator['outer_evaluation_used']
                or locator['reserved_test_evaluated']
                or set(locator['prior_training_sample_ids']) != expected_originals
                or set(classifier['training_sample_ids']) != set(stage['train_sample_ids'])
                or classifier['validation_sample_ids']):
            raise ValueError('Parent training or locator-prior isolation changed')
        for receipt in (locator, classifier):
            if base.sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']:
                raise ValueError('Parent learned checkpoint changed')
        train = [{**r, 'split':'train'} for r in old['samples'] if r['case_id'] in stage['train_case_ids']]
        test = [{**r, 'split':'final_test'} for r in old['samples'] if r['case_id'] in stage['final_test_case_ids']]
        checked_audit(folder/'proposal-audit.json', train, 'train', training=True)
        checked_audit(folder/'final-test/proposal-audit.json', test, 'final_test')
        if ({r['sample_id'] for r in result['final_predictions']} != set(stage['final_test_sample_ids'])
                or any(r['case_id'] in old['reserved_case_ids'] for r in train+test)):
            raise ValueError('Parent held-out or reserved sample panel changed')
        stage.update(locator_policy='reuse_completed_fold_local_locator_and_exact_proposals',
                     classifier_policy='same_fresh_initialization_and_optimizer',
                     checkpoint_policy='predeclared_final_epoch; no outer checkpoint selection',
                     parent_inner_stage_ids=[],
                     expected_initial_weights_sha256=classifier['fresh_initial_weights_sha256'])
        for name in ['stage-input.json', 'results.json', 'proposal-audit.json',
                     'final-test/proposal-audit.json', 'final-test/predictions.json',
                     'locator/locator-result.json', 'classifier/receipt.json',
                     'classifier/initialization.json']:
            p = folder/name
            base.checked_json(p)
            dependencies[str(p)] = base.sha256(p)
        for receipt in (locator, classifier):
            dependencies[receipt['checkpoint']] = receipt['checkpoint_sha256']
    from fakect_coa_rotation_report import rotation_manifest
    manifest = rotation_manifest(old)
    manifest['folds'] = [f for f in manifest['folds'] if f['protocol'] == 'family_holdout']
    for fold in manifest['folds']:
        fold['selection_stages'] = []
    manifest['stage_catalog'] = [s for s in manifest['stage_catalog'] if s['stage'] == 'outer_refit']
    return {'schema_version':SCHEMA, 'config':deepcopy(c),
            'parent_directory':str(parent), 'parent_lock_sha256':base.sha256(parent/'experiment-lock.json'),
            'stages':stages, 'reserved_case_ids':old['reserved_case_ids'],
            'samples':old['samples'], 'runtime_geometry':old['runtime_geometry'],
            'fold_manifest':manifest, 'dependency_sha256':dependencies,
            'unchanged_training_settings':old['config']['training'],
            'classifier_epochs':c['training']['classifier_epochs'],
            'checkpoint_selection':'predeclared final epoch; intermediate checkpoints not evaluated',
            'scope':'Development cross-validation budget control; previous outer outcomes informed development; infant family unopened',
            'only_intended_change':'fresh final-classifier epoch budget; locator and crop proposals reused'}


def freeze(c, content):
    lock = plan(c)
    payloads = {name:(ROOT/name).read_bytes() for name in CODE_FILES}
    out = Path(c['experiment']['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'INCOMPLETE').touch()
    (out/'input.ini').write_bytes(content)
    hashes = {}
    for name, data in payloads.items():
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    lock.update(code_sha256=hashes, input_sha256=hashlib.sha256(content).hexdigest())
    base.write_json(out/'experiment-lock.json', lock)
    (out/'INCOMPLETE').unlink()
    return lock


def inspect_lock(c, content):
    out = Path(c['experiment']['output_directory'])
    if (out/'INCOMPLETE').exists():
        raise ValueError('Budget plan is incomplete')
    lock = base.checked_json(out/'experiment-lock.json')
    if (lock['config'] != c or lock['input_sha256'] != hashlib.sha256(content).hexdigest()
            or base.sha256(out/'input.ini') != lock['input_sha256']):
        raise ValueError('Frozen budget input changed; use a new revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Frozen budget source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if base.sha256(ROOT/name) != digest or base.sha256(out/'source'/name) != digest:
            raise ValueError('Pinned budget implementation changed: '+name)
    for path, digest in lock['dependency_sha256'].items():
        if base.sha256(path) != digest:
            raise ValueError('Pinned parent artifact changed: '+path)
    parent_lock(c)
    return lock


def runtime_settings(c, lock, stage, output):
    parent = base.checked_json(Path(lock['parent_directory'])/'experiment-lock.json')
    run_config = deepcopy(parent['config'])
    run_config['experiment']['output_directory'] = c['experiment']['output_directory']
    run_config['training']['classifier_epochs'] = c['training']['classifier_epochs']
    runtime = rotation.runtime_config(run_config, parent, stage, output)
    return run_config, runtime


def smoke(c, lock):
    from fakect_coa_rotation_learning import smoke as rotation_smoke
    run_config, _ = runtime_settings(c, lock, lock['stages'][0], c['experiment']['output_directory'])
    return rotation_smoke(run_config, lock)


def require_smoke(c, lock):
    run_config, _ = runtime_settings(c, lock, lock['stages'][0], c['experiment']['output_directory'])
    return rotation.require_smoke(run_config, lock)


def run_stage(c, lock, stage):
    from fakect_coa_budget_learning import fit_classifier
    import fakect_coa_rotation_learning as learning
    import fakect_coa_paired_steps as network
    root, parent = Path(c['experiment']['output_directory']), Path(lock['parent_directory'])
    output = root/'stages'/stage['id']
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').touch()
    run_config, runtime = runtime_settings(c, lock, stage, output)
    train = [{**r, 'split':'train'} for r in lock['samples'] if r['case_id'] in stage['train_case_ids']]
    parent_folder = parent/'stages'/stage['id']
    proposals, audit = checked_audit(parent_folder/'proposal-audit.json', train, 'train', training=True)
    base.write_json(output/'stage-input.json', {'stage':stage, 'classifier_epochs':c['training']['classifier_epochs'],
        'source_lock_sha256':base.sha256(root/'experiment-lock.json'),
        'locator_retrained':False, 'training_proposals_changed':False,
        'parent_proposal_audit_sha256':base.sha256(parent_folder/'proposal-audit.json')})
    base.write_json(output/'proposal-audit.json', {'passed':True, 'samples':audit, 'selection_allowed':False,
        'contract':'Exact reviewed parent fold-local image-only crops; no recentering'})
    tf, _ = learning.legacy.configure_tensorflow(runtime)
    receipt = fit_classifier(tf, runtime, train, proposals, output/'classifier', run_config,
        epochs=c['training']['classifier_epochs'], checkpoint_interval=c['training']['checkpoint_interval'],
        fixed_budget=True, expected_initial_weights_sha256=stage['expected_initial_weights_sha256'])
    if (receipt['selected_epoch'] != c['training']['classifier_epochs']
            or receipt['epochs_completed'] != c['training']['classifier_epochs']
            or receipt['validation_sample_ids'] or receipt['validation'] is not None
            or receipt['fresh_initial_weights_sha256'] != stage['expected_initial_weights_sha256']):
        raise ValueError('Control changed final-epoch policy, training roles or initialization')
    from fakect_coa_classifier_learning import checkpoint_prediction_parity
    original = base.checked_json(parent_folder/'classifier/initialization.json')
    initial = base.checked_json(output/'classifier/initialization.json')
    parity = checkpoint_prediction_parity(original['predictions'], initial['predictions'])
    if not parity['passed']:
        raise ValueError('Control initial predictions differ from the reviewed identical inputs')
    # No outer image payload has been read by the fitting path. The only scored
    # checkpoint is now fixed by the predeclared budget, before outer inference.
    test = [{**r, 'split':'final_test'} for r in lock['samples'] if r['case_id'] in stage['final_test_case_ids']]
    test_proposals, test_audit = checked_audit(parent_folder/'final-test/proposal-audit.json', test, 'final_test')
    test_dir = output/'final-test'
    test_dir.mkdir()
    base.write_json(test_dir/'proposal-audit.json', {'passed':all(r['coverage_passed'] for r in test_audit),
        'samples':test_audit, 'selection_allowed':False, 'qa_can_change_model':False,
        'contract':'Exact parent locator crops; final inference only; failures retained'})
    model = network.load_model(receipt['checkpoint'])
    detail = runtime['input.detail']
    steps = network.make_steps(model, detail['shape_kji'], num_proposals=detail['proposals'],
        regression_weight=run_config['training']['regression_weight'])
    guard = learning.legacy.RuntimeGuard(runtime, tf)
    inputs = learning.StageFineInputs(runtime, run_config['training']['cache_limit_mb'], test)
    predictions = learning._score(tf, steps, test, inputs, test_proposals, guard)
    coverage = {r['sample_id']:r['coverage_passed'] for r in test_audit}
    for row in predictions:
        row.update(protocol=stage['protocol'], fold_id=stage['outer_fold_id'], seed=stage['seed'],
                   outer_case=None, outer_family=row['family'],
                   locator_coverage_passed=coverage[row['sample_id']], pipeline_failure=not coverage[row['sample_id']])
    if {r['sample_id'] for r in predictions} != set(stage['final_test_sample_ids']):
        raise ValueError('Final prediction panel changed')
    inspect_lock(c, (root/'input.ini').read_bytes())
    result = {'status':'completed', 'stage_id':stage['id'], 'protocol':stage['protocol'],
        'source_lock_sha256':base.sha256(root/'experiment-lock.json'),
        'parent_lock_sha256':lock['parent_lock_sha256'],
        'locator_retrained':False, 'locator_proposals_changed':False,
        'classifier_selected_epoch':receipt['selected_epoch'], 'classifier':receipt,
        'initial_prediction_parity':parity, 'final_test_evaluated':True,
        'final_predictions':predictions, 'reserved_test_evaluated':False,
        'final_locator_failed_sample_ids':[r['sample_id'] for r in test_audit if not r['coverage_passed']],
        'budget':{'classifier_epochs':c['training']['classifier_epochs'], 'rule':'user-declared fixed final epoch'}}
    base.write_json(test_dir/'predictions.json', {'predictions':predictions,
        'checkpoint_sha256':receipt['checkpoint_sha256'], 'source_lock_sha256':result['source_lock_sha256'],
        'qa_can_change_model':False})
    base.write_json(output/'results.json', result)
    (output/'INCOMPLETE').unlink()
    return result
