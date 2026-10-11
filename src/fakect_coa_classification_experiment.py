"""Immutable classification-only loss control with unchanged graph and crops."""
from copy import deepcopy
import configparser
import hashlib
import json
from pathlib import Path

import fakect_coa_budget_experiment as budget
import fakect_coa_rotation_experiment as rotation
import fakect_multiscale_experiment as base

ROOT = base.ROOT
SCHEMA = 'fakect.coa-classification-control/1'
CODE_FILES = tuple(dict.fromkeys(budget.CODE_FILES + (
    'src/fakect_coa_classification_experiment.py',
    'src/fakect_coa_classification_learning.py',
    'src/fakect_coa_classification_report.py',
    'scripts/train_coa_classification.py',
    'scripts/slurm/fakect_coa_classification_gpu.sh',
    'scripts/slurm/fakect_coa_classification_report.sh',
)))
FIELDS = {'experiment': {'schema_version', 'output_directory'},
          'reference': {'experiment_directory', 'comparison_directory'},
          'training': {'classifier_epochs', 'checkpoint_interval', 'regression_weight'}}


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(FIELDS):
        raise ValueError('Expected exactly [experiment], [reference], [training]')
    c = {section: dict(parser[section]) for section in FIELDS}
    if any(set(c[section]) != fields for section, fields in FIELDS.items()):
        raise ValueError('Unsupported classification-control fields')
    if any('\n' in value or '\r' in value for section in c.values() for value in section.values()):
        raise ValueError('Classification input values must occupy one line')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported classification-control schema')
    for section, key in [('experiment', 'output_directory'), ('reference', 'experiment_directory'),
                         ('reference', 'comparison_directory')]:
        p = Path(c[section][key]).expanduser()
        c[section][key] = str((p if p.is_absolute() else ROOT/p).resolve())
    for key in ('classifier_epochs', 'checkpoint_interval'):
        c['training'][key] = int(c['training'][key])
        if not 1 <= c['training'][key] <= 1000:
            raise ValueError('training.'+key+' must be in [1, 1000]')
    if c['training']['classifier_epochs'] != 200:
        raise ValueError('This control revision requires classifier_epochs = 200')
    if c['training']['classifier_epochs'] % c['training']['checkpoint_interval']:
        raise ValueError('checkpoint_interval must divide classifier_epochs')
    c['training']['regression_weight'] = float(c['training']['regression_weight'])
    if c['training']['regression_weight'] != 0.0:
        raise ValueError('This control requires regression_weight = 0')
    out, parent = Path(c['experiment']['output_directory']), Path(c['reference']['experiment_directory'])
    if out == parent or parent in out.parents or out in parent.parents:
        raise ValueError('Control output must be separate from its parent experiment')
    comparison = Path(c['reference']['comparison_directory'])
    if out == comparison or comparison in out.parents or out in comparison.parents:
        raise ValueError('Control output must be separate from its comparison experiment')
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


def comparison_lock(c):
    root = Path(c['reference']['comparison_directory'])
    old = base.checked_json(root/'experiment-lock.json')
    if old.get('schema_version') != budget.SCHEMA:
        raise ValueError('Comparison must be a completed frozen 200-epoch budget control')
    budget.inspect_lock(old['config'], (root/'input.ini').read_bytes())
    budget.require_smoke(old['config'], old)
    if (Path(old['parent_directory']).resolve() != Path(c['reference']['experiment_directory']).resolve()
            or old['classifier_epochs'] != 200
            or old['unchanged_training_settings']['regression_weight'] != 1.0):
        raise ValueError('Comparison ancestor, epoch budget or supervised loss changed')
    return old


def plan(c):
    lock = budget.plan(c)
    comparison = Path(c['reference']['comparison_directory'])
    old = comparison_lock(c)
    if (old['stages'] != lock['stages'] or old['samples'] != lock['samples']
            or old['runtime_geometry'] != lock['runtime_geometry']
            or old['fold_manifest'] != lock['fold_manifest']
            or old['parent_lock_sha256'] != lock['parent_lock_sha256']):
        raise ValueError('Comparison folds, crops, labels or ancestry changed')
    dependencies = lock['dependency_sha256']
    for name in ('experiment-lock.json', 'input.ini', 'runtime-smoke.json'):
        path = comparison/name
        dependencies[str(path)] = base.sha256(path)
    for stage in lock['stages']:
        folder = comparison/'stages'/stage['id']
        if (folder/'INCOMPLETE').exists() or (folder/'failure.json').exists():
            raise ValueError('Every comparison fold must have finished cleanly')
        result = base.checked_json(folder/'results.json')
        receipt = base.checked_json(folder/'classifier/receipt.json')
        if (result.get('status') != 'completed' or result.get('stage_id') != stage['id']
                or result.get('source_lock_sha256') != base.sha256(comparison/'experiment-lock.json')
                or result.get('parent_lock_sha256') != lock['parent_lock_sha256']
                or result.get('reserved_test_evaluated') is not False
                or result.get('final_test_evaluated') is not True
                or result.get('classifier_selected_epoch') != 200
                or result.get('classifier') != receipt
                or receipt.get('selected_epoch') != 200 or receipt.get('epochs_completed') != 200
                or receipt.get('optimizer_updates') != 4800
                or receipt.get('validation_sample_ids') != [] or receipt.get('validation') is not None
                or set(receipt.get('training_sample_ids', [])) != set(stage['train_sample_ids'])
                or receipt.get('fresh_initial_weights_sha256') != stage['expected_initial_weights_sha256']
                or {r['sample_id'] for r in result['final_predictions']} != set(stage['final_test_sample_ids'])
                or len(result['final_predictions']) != len(stage['final_test_sample_ids'])):
            raise ValueError('Comparison completion, initialization or final-epoch contract changed')
        from fakect_coa_budget_report import checked_history
        checked_history(base.checked_json(folder/'classifier/history.json'), receipt, stage)
        ancestor = Path(lock['parent_directory'])/'stages'/stage['id']
        for name, role, training in [('proposal-audit.json', 'train', True),
                                     ('final-test/proposal-audit.json', 'final_test', False)]:
            cases = stage['train_case_ids'] if training else stage['final_test_case_ids']
            rows = [{**r, 'split':role} for r in lock['samples'] if r['case_id'] in cases]
            _, panel = checked_audit(folder/name, rows, role, training=training)
            _, expected = checked_audit(ancestor/name, rows, role, training=training)
            if panel != expected:
                raise ValueError('Comparison crop coordinates differ from the frozen ancestor')
        for name in ('stage-input.json', 'results.json', 'proposal-audit.json',
                     'final-test/proposal-audit.json', 'final-test/predictions.json',
                     'classifier/receipt.json', 'classifier/history.json', 'classifier/initialization.json'):
            path = folder/name
            base.checked_json(path)
            dependencies[str(path)] = base.sha256(path)
        if base.sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']:
            raise ValueError('Comparison classifier checkpoint changed')
        dependencies[receipt['checkpoint']] = receipt['checkpoint_sha256']
    intended = deepcopy(old['unchanged_training_settings'])
    intended.update(classifier_epochs=200, regression_weight=0.0)
    lock.update(schema_version=SCHEMA, comparison_directory=str(comparison),
                comparison_lock_sha256=base.sha256(comparison/'experiment-lock.json'),
                intended_training_settings=intended,
                regression_weight=0.0, severity_trained=False,
                scope='Development family cross-validation; previous outcomes informed development; infant family unopened',
                only_intended_change='regression_weight 1 to 0 at the same fixed 200 epochs; graph, initialization, optimizer, folds and crops retained')
    return lock


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
        raise ValueError('Classification plan is incomplete')
    lock = base.checked_json(out/'experiment-lock.json')
    if (lock['config'] != c or lock['input_sha256'] != hashlib.sha256(content).hexdigest()
            or base.sha256(out/'input.ini') != lock['input_sha256']):
        raise ValueError('Frozen classification input changed; use a new revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Frozen classification source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if base.sha256(ROOT/name) != digest or base.sha256(out/'source'/name) != digest:
            raise ValueError('Pinned classification implementation changed: '+name)
    for path, digest in lock['dependency_sha256'].items():
        if base.sha256(path) != digest:
            raise ValueError('Pinned parent artifact changed: '+path)
    parent_lock(c)
    comparison_lock(c)
    if lock.get('schema_version') != SCHEMA or lock.get('regression_weight') != 0.0 or lock.get('severity_trained') is not False:
        raise ValueError('Classification-only lock contract changed')
    return lock


def runtime_settings(c, lock, stage, output):
    parent = base.checked_json(Path(lock['parent_directory'])/'experiment-lock.json')
    run_config = deepcopy(parent['config'])
    run_config['experiment']['output_directory'] = c['experiment']['output_directory']
    run_config['training']['classifier_epochs'] = c['training']['classifier_epochs']
    run_config['training']['regression_weight'] = c['training']['regression_weight']
    runtime = rotation.runtime_config(run_config, parent, stage, output)
    return run_config, runtime


def smoke(c, lock):
    from fakect_coa_rotation_learning import smoke as rotation_smoke
    run_config, _ = runtime_settings(c, lock, lock['stages'][0], c['experiment']['output_directory'])
    result = rotation_smoke(run_config, lock)
    result.update(regression_weight=0.0, severity_trained=False)
    base.write_json(Path(c['experiment']['output_directory'])/'runtime-smoke.json', result)
    return result


def require_smoke(c, lock):
    run_config, _ = runtime_settings(c, lock, lock['stages'][0], c['experiment']['output_directory'])
    result = rotation.require_smoke(run_config, lock)
    if result.get('regression_weight') != 0.0 or result.get('severity_trained') is not False:
        raise ValueError('Own smoke must exercise the classification-only loss')
    return result


def run_stage(c, lock, stage):
    from fakect_coa_classification_learning import fit_classifier, sanitize_predictions
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
        'regression_weight':0.0, 'severity_trained':False,
        'comparison_lock_sha256':lock['comparison_lock_sha256'],
        'parent_proposal_audit_sha256':base.sha256(parent_folder/'proposal-audit.json')})
    base.write_json(output/'proposal-audit.json', {'passed':True, 'samples':audit, 'selection_allowed':False,
        'contract':'Exact reviewed parent fold-local image-only crops; no recentering'})
    tf, _ = learning.legacy.configure_tensorflow(runtime)
    receipt = fit_classifier(tf, runtime, train, proposals, output/'classifier', run_config,
        epochs=c['training']['classifier_epochs'], checkpoint_interval=c['training']['checkpoint_interval'],
        fixed_budget=True, expected_initial_weights_sha256=stage['expected_initial_weights_sha256'])
    if (receipt['selected_epoch'] != c['training']['classifier_epochs']
            or receipt['epochs_completed'] != c['training']['classifier_epochs']
            or receipt.get('optimizer_updates') != 4800
            or receipt.get('regression_weight') != 0.0 or receipt.get('severity_trained') is not False
            or receipt['validation_sample_ids'] or receipt['validation'] is not None
            or receipt['fresh_initial_weights_sha256'] != stage['expected_initial_weights_sha256']):
        raise ValueError('Control changed final-epoch policy, training roles or initialization')
    from fakect_coa_classifier_learning import checkpoint_prediction_parity
    original = base.checked_json(Path(lock['comparison_directory'])/'stages'/stage['id']/'classifier/initialization.json')
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
    predictions = sanitize_predictions(learning._score(tf, steps, test, inputs, test_proposals, guard))
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
        'comparison_lock_sha256':lock['comparison_lock_sha256'],
        'regression_weight':0.0, 'severity_trained':False,
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
