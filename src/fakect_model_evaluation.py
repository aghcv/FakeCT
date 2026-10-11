"""INI-driven family holdouts, outer folds and independent model evaluation.

This runner consumes published binary segmentation datasets. It never edits
phantoms; pathology severity/condition heads require a separate target contract.
"""
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

import numpy as np

from fakect_model_experiment import (
    _canonical, _sha256, fit_model_experiment, freeze_model_experiment,
    inspect_experiment_lock, plan_model_experiment,
)
from fakect_evaluation_design import build_evaluation_design

SCHEMA = 'fakect.model-evaluation-lock/1'


def _write_json(path, value):
    """Atomically publish a complete evaluation metadata document.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination JSON file.
    value : JSON-compatible object
        Finite metadata to serialize.

    Returns
    -------
    None
        Writes a sibling temporary file and replaces the destination after
        serialization.

    Notes
    -----
    Publish one complete file; partial writes never masquerade as results.
    """
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _implementation():
    """Fingerprint evaluation, preprocessing and segmentation consumers.

    Parameters
    ----------
    None
        This function takes no explicit arguments.

    Returns
    -------
    dict of str to str
        Implementation filenames mapped to their SHA256 digests.
    """
    directory = Path(__file__).parent
    return {name: _sha256(directory/name) for name in (
        'fakect_model_evaluation.py', 'fakect_evaluation_design.py',
        'fakect_evaluation_metrics.py', 'fakect_model_experiment.py',
        'fakect_segmentation.py', 'fakect_model_resampling.py')}


def preview_evaluation(config):
    """Preview provisional family rotations without opening paired arrays.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.

    Returns
    -------
    dict
        Planning-only design, fit count, project hash and unverified-family warning.

    Raises
    ------
    ValueError
        Planning project is absent, incompatible, duplicated or does not cover the
        configured revisions.

    Notes
    -----
    Display provisional family rotations without publishing or loading pairs.
    """
    settings = config['evaluation']
    project_path = settings.get('planning_project')
    if not project_path:
        raise ValueError('[evaluation] planning_project is required for --preview-splits')
    project = json.loads(Path(project_path).read_text())
    if project.get('schema_version') != 'fakect.cohort-project/1':
        raise ValueError('Planning requires a fakect.cohort-project/1 inventory')
    wanted = set(config['experiment']['datasets'])
    cases, found = [], set()
    for case in project['cases']:
        name = case.get('registry_name')
        if name in wanted:
            if name in found:
                raise ValueError('Duplicate registry name in planning project')
            found.add(name)
            cases.append({'case_id': case['case_id'], 'anatomy_family': case['provisional_family']})
    if found != wanted:
        raise ValueError('Planning project does not cover every configured dataset name')
    design = build_evaluation_design(cases, settings, config['split']['seed'])
    return {'ready': False, 'planning_only': True, 'outputs_written': False,
            'families_verified': False, 'project_sha256': _sha256(project_path),
            'design': design, 'fit_count': _fit_count(design),
            'message': 'Provisional split preview only; dataset publication and family verification are still required.'}


def _fit_count(design):
    """Count inner and final model fits over all repeat seeds.

    Parameters
    ----------
    design : dict
        Evaluation design containing outer/inner folds and model seeds.

    Returns
    -------
    int
        Total planned training runs.
    """
    per_seed = sum(len(fold['inner_folds'])+1 for fold in design['folds'])
    return per_seed * len(design['seeds'])


def plan_evaluation(config):
    """Validate prepared datasets and build an independent family evaluation design.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.

    Returns
    -------
    dict
        Readiness, base verified plan, design, fit count and accumulated errors.

    Notes
    -----
    Does not create outputs or train; temporary fractions are used only for base verification before the explicit fold design.
    """
    base_config = copy.deepcopy(config)
    base_config.pop('evaluation', None)
    # Verification needs a valid temporary assignment. The evaluation design,
    # not the legacy fractions, determines every actual fit/test partition.
    base_config['split'].update(validation_fraction=.2, test_fraction=.2)
    base = plan_model_experiment(base_config)
    result = {'ready': base['ready'], 'outputs_written': False, 'model_fitted': False,
              'base_plan': base, 'errors': list(base['errors']),
              'task': 'binary_aorta_segmentation', 'design': None}
    if base['ready']:
        try:
            design = build_evaluation_design(base['datasets'], config['evaluation'], config['split']['seed'])
            result.update(design=design, fit_count=_fit_count(design))
        except ValueError as exc:
            result['errors'].append(str(exc))
            result['ready'] = False
    return result


def freeze_evaluation(config, input_bytes):
    """Freeze evaluation inputs, implementation and family folds in a new revision.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    input_bytes : bytes
        Exact captured INI bytes corresponding to ``config``; retained without rewriting
        when provided.

    Returns
    -------
    dict
        Evaluation lock path, design, fit count and publication flags.

    Raises
    ------
    ValueError
        Dataset or evaluation planning is not ready.
    FileExistsError
        The requested output directory already exists.

    Notes
    -----
    Writes exact input.ini, evaluation-lock.json and its SHA256.
    """
    plan = plan_evaluation(config)
    if not plan['ready']:
        raise ValueError('Evaluation is not ready: '+'; '.join(plan['errors']))
    output = Path(config['experiment']['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    lock = {'schema_version': SCHEMA, 'config': config, 'plan': plan,
            'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
            'implementation_sha256': _implementation(),
            'nested_selection': 'one configured architecture; median inner best epoch rounded up; refit all development families',
            'task': 'binary_aorta_segmentation'}
    lock['fingerprint'] = hashlib.sha256(_canonical(lock).encode()).hexdigest()
    (output/'input.ini').write_bytes(input_bytes)
    _write_json(output/'evaluation-lock.json', lock)
    (output/'evaluation-lock.sha256').write_text(_sha256(output/'evaluation-lock.json')+'\n')
    return {'ready': True, 'outputs_written': True, 'model_fitted': False,
            'lock_path': str(output/'evaluation-lock.json'), 'design': plan['design'],
            'fit_count': plan['fit_count']}


def _load_lock(config, input_bytes):
    """Verify an evaluation lock against current input bytes and pinned consumers.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    input_bytes : bytes
        Exact captured INI bytes corresponding to ``config``; retained without rewriting
        when provided.

    Returns
    -------
    dict
        Verified evaluation lock with design and base plan.

    Raises
    ------
    ValueError
        Lock schema/fingerprint, INI, implementation hash or reconstructed fold design
        differs.
    """
    output = Path(config['experiment']['output_directory'])
    path = output/'evaluation-lock.json'
    if _sha256(path) != (output/'evaluation-lock.sha256').read_text().strip():
        raise ValueError('Evaluation lock checksum changed')
    lock = json.loads(path.read_text())
    digest = lock.pop('fingerprint', None)
    if digest != hashlib.sha256(_canonical(lock).encode()).hexdigest() or lock.get('schema_version') != SCHEMA:
        raise ValueError('Invalid evaluation lock fingerprint/schema')
    lock['fingerprint'] = digest
    if lock['config'] != config or lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest():
        raise ValueError('Evaluation INI differs from frozen input; choose a new output directory')
    if _sha256(output/'input.ini') != lock['input_sha256']:
        raise ValueError('Frozen evaluation INI changed')
    if lock['implementation_sha256'] != _implementation():
        raise ValueError('Model/evaluation implementation changed; create a new evaluation revision')
    expected = build_evaluation_design(lock['plan']['base_plan']['datasets'],
        config['evaluation'], config['split']['seed'])
    if lock['plan']['design'] != expected:
        raise ValueError('Frozen fold design is inconsistent')
    return lock


def _fold_config(config, fold, seed, directory, *, fixed_epochs=None):
    """Construct a fold-specific model configuration without mutating the parent.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    fold : dict
        Fold record containing the complete family assignment.
    seed : int
        Model training repeat seed.
    directory : str or pathlib.Path
        Per-fold model experiment directory.
    fixed_epochs : int or None
        Externally selected final-refit budget; None retains validation-selected
        fitting.

    Returns
    -------
    dict
        Copied experiment with explicit family assignment, seed, output and optional
        fixed epoch budget.
    """
    result = copy.deepcopy(config)
    result.pop('evaluation', None)
    result['experiment']['output_directory'] = str(directory)
    result['model']['seed'] = seed
    result['split'] = {'mode': 'anatomy_family', 'seed': config['split']['seed'],
                       'assignment': dict(fold['family_assignment']),
                       'allow_empty_validation': fixed_epochs is not None}
    if fixed_epochs is not None:
        result['model']['epochs'] = fixed_epochs
        result['training_protocol'] = 'fixed_epochs'
    return result


def _run_directory(config, fold, seed):
    """Locate one outer-fold training repeat.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    fold : dict
        Outer fold identity record.
    seed : int
        Model repeat seed.

    Returns
    -------
    pathlib.Path
        Configured output/folds/fold-id/seed-N directory.
    """
    return Path(config['experiment']['output_directory'])/'folds'/fold['id']/f'seed-{seed}'


def _ensure_model(fold_config, base_plan, input_bytes):
    """Freeze, fit or verify a resumable fold model under pinned settings.

    Parameters
    ----------
    fold_config : dict
        Exact fold-specific experiment settings.
    base_plan : dict
        Ready plan with payloads already verified.
    input_bytes : bytes
        Exact captured INI bytes corresponding to ``config``; retained without rewriting
        when provided.

    Returns
    -------
    dict
        Verified existing model metadata or metadata from a newly completed fit.

    Raises
    ------
    ValueError
        An existing lock or saved model does not match the requested fold.

    Notes
    -----
    Creates missing fold locks and model outputs; existing completed models are checked before reuse.
    """
    directory = Path(fold_config['experiment']['output_directory'])
    path = directory/'experiment-lock.json'
    if not path.exists():
        freeze_model_experiment(fold_config, input_bytes, prepared_plan=base_plan)
    checked = inspect_experiment_lock(path)
    if (checked['lock']['config'] != fold_config or
            checked['lock']['input_config_sha256'] != hashlib.sha256(input_bytes).hexdigest()):
        raise ValueError('Existing fold lock differs from requested assignment/settings')
    metadata_path = directory/'model'/'model-metadata.json'
    if metadata_path.exists():
        metadata = _check_model(directory/'model', checked)
    else:
        metadata = fit_model_experiment(fold_config, input_bytes)
    return metadata


def _check_model(directory, checked):
    """Verify saved model completion and provenance for a frozen fold.

    Parameters
    ----------
    directory : pathlib.Path
        Model directory containing metadata and model.keras.
    checked : dict
        Verified experiment lock returned by inspect_experiment_lock.

    Returns
    -------
    dict
        Validated saved model metadata.

    Raises
    ------
    ValueError
        Model source/settings/hash, completed epochs or selected best epoch is
        inconsistent.
    """
    metadata = json.loads((directory/'model-metadata.json').read_text())
    if (metadata.get('source_manifest_sha256') != checked['sha256'] or
            Path(metadata.get('source_manifest', '')).resolve() != checked['path'] or
            metadata.get('settings') != checked['lock']['config']['model'] or
            metadata.get('model_file') != 'model.keras' or
            metadata.get('model_sha256') != _sha256(directory/'model.keras')):
        raise ValueError('Saved model does not match its frozen fold')
    if metadata.get('epochs_completed') != metadata['settings']['epochs']:
        raise ValueError('Saved model did not complete its configured epoch budget')
    if checked['lock']['config'].get('training_protocol') != 'fixed_epochs':
        best = metadata.get('best_epoch')
        if isinstance(best, bool) or not isinstance(best, int) or not 1 <= best <= metadata['settings']['epochs']:
            raise ValueError('Saved best epoch is outside its configured training budget')
    return metadata


def fit_evaluation(config, input_bytes):
    """Train isolated fold models using development partitions only.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    input_bytes : bytes
        Exact captured INI bytes corresponding to ``config``; retained without rewriting
        when provided.

    Returns
    -------
    dict
        Completed fold/seed identities and flags confirming test evaluation has not run.

    Notes
    -----
    Nested fits select the median inner best epoch rounded up, then refit all development families. Writes per-repeat selection.json; test payloads remain unused.

    Fit only training/validation partitions, with isolated outer models.
    """
    lock = _load_lock(config, input_bytes)
    completed = []
    for fold in lock['plan']['design']['folds']:
        for seed in lock['plan']['design']['seeds']:
            directory = _run_directory(config, fold, seed)
            selected_epochs = None
            inner_results = []
            for inner in fold['inner_folds']:
                inner_config = _fold_config(config, inner, seed, directory/inner['id'])
                metadata = _ensure_model(inner_config, lock['plan']['base_plan'], input_bytes)
                inner_results.append({'id': inner['id'], 'best_epoch': metadata['best_epoch'],
                                      'model_sha256': metadata['model_sha256']})
                _clear_session()
            if inner_results:
                selected_epochs = int(math.ceil(statistics.median(row['best_epoch'] for row in inner_results)))
            final_config = _fold_config(config, fold, seed, directory/'outer-model', fixed_epochs=selected_epochs)
            _ensure_model(final_config, lock['plan']['base_plan'], input_bytes)
            _write_json(directory/'selection.json', {'fold_id': fold['id'], 'seed': seed,
                'inner_results': inner_results, 'refit_epochs': selected_epochs,
                'test_used_for_selection': False})
            completed.append({'fold_id': fold['id'], 'seed': seed})
            _clear_session()
    return {'ready': True, 'model_fitted': True, 'test_evaluated': False, 'completed': completed}


def _clear_session():
    # TensorFlow is optional for planning. Release graphs between actual fits.
    """Release TensorFlow graph state between completed model fits.

    Parameters
    ----------
    None
        This function takes no explicit arguments.

    Returns
    -------
    None
        Clears the Keras session when TensorFlow is already imported; otherwise returns
        without importing it.
    """
    import sys
    if 'tensorflow' in sys.modules:
        sys.modules['tensorflow'].keras.backend.clear_session()


def _test_manifest(config, fold, seed):
    """Verify outer model and inner-only epoch selection before releasing test data.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    fold : dict
        Frozen outer fold with optional inner fold records.
    seed : int
        Model training repeat seed.

    Returns
    -------
    tuple of pathlib.Path and dict
        Outer-model directory and inspected experiment lock.

    Raises
    ------
    ValueError
        Outer/inner assignments, input identity, completion or final epoch selection
        disagree.
    """
    directory = _run_directory(config, fold, seed)/'outer-model'
    path = directory/'experiment-lock.json'
    checked = inspect_experiment_lock(path)
    saved = checked['lock']['config']
    epochs = saved['model']['epochs'] if fold['inner_folds'] else None
    expected = _fold_config(config, fold, seed, directory, fixed_epochs=epochs)
    if saved != expected or checked['lock']['input_config_sha256'] != _sha256(Path(config['experiment']['output_directory'])/'input.ini'):
        raise ValueError('Outer model assignment/settings do not match the evaluation design')
    if fold['inner_folds']:
        actual = []
        for inner in fold['inner_folds']:
            inner_dir = directory.parent/inner['id']
            inner_checked = inspect_experiment_lock(inner_dir/'experiment-lock.json')
            inner_expected = _fold_config(config, inner, seed, inner_dir)
            if (inner_checked['lock']['config'] != inner_expected or
                    inner_checked['lock']['input_config_sha256'] != checked['lock']['input_config_sha256']):
                raise ValueError('Inner model assignment changed')
            actual.append(_check_model(inner_dir/'model', inner_checked)['best_epoch'])
        if epochs != int(math.ceil(statistics.median(actual))):
            raise ValueError('Outer refit epochs disagree with inner-only selection')
    _check_model(directory/'model', checked)
    return directory, checked


def test_evaluation(config, input_bytes):
    """Evaluate all frozen outer test partitions after every required model is verified.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    input_bytes : bytes
        Exact captured INI bytes corresponding to ``config``; retained without rewriting
        when provided.

    Returns
    -------
    dict
        Summary/report paths and equal-family macro metrics from summarize_evaluation.

    Notes
    -----
    Writes checksummed per-repeat evaluation.json for missing results and validates existing results before reuse. No tuning is performed using test outcomes.
    """
    from fakect_evaluation_metrics import evaluate_segmentation_model
    lock = _load_lock(config, input_bytes)
    pending = []
    # Check all models first, avoiding a partial test release due to absent fits.
    for fold in lock['plan']['design']['folds']:
        for seed in lock['plan']['design']['seeds']:
            directory, checked = _test_manifest(config, fold, seed)
            pending.append((fold, seed, directory, checked))
    for fold, seed, directory, checked in pending:
        result_path = directory.parent/'evaluation.json'
        if result_path.exists():
            _check_result(_read_result(result_path), directory, checked, config, fold, seed)
            continue
        result = evaluate_segmentation_model(directory/'model', checked['path'],
                                              threshold=config['evaluation']['threshold'])
        result.update(fold_id=fold['id'], seed=seed)
        _write_json(result_path, result)
        result_path.with_suffix('.sha256').write_text(_sha256(result_path)+'\n')
        _clear_session()
    return summarize_evaluation(config, input_bytes)


def _read_result(path):
    """Read a completed evaluation result after verifying its checksum.

    Parameters
    ----------
    path : pathlib.Path
        Evaluation JSON path with a sibling .sha256 file.

    Returns
    -------
    dict
        Decoded evaluation record.

    Raises
    ------
    ValueError
        The result bytes differ from the recorded checksum.
    """
    if _sha256(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Evaluation result checksum changed')
    return json.loads(path.read_text())


def _check_result(result, directory, checked, config, fold, seed):
    """Validate saved test coverage and provenance against one frozen fold.

    Parameters
    ----------
    result : dict
        Decoded per-fold evaluation record.
    directory : pathlib.Path
        Outer-model experiment directory.
    checked : dict
        Inspected outer experiment lock.
    config : dict
        Typed workflow configuration parsed from INI sections.
    fold : dict
        Frozen fold record.
    seed : int
        Expected model repeat seed.

    Returns
    -------
    None
        Completes when identity, threshold, model/source hashes and tested samples
        match.

    Raises
    ------
    ValueError
        Result identity, preprocessing, model provenance or sample coverage differs.
    """
    if result['fold_id'] != fold['id'] or result['seed'] != seed or result['threshold'] != config['evaluation']['threshold']:
        raise ValueError('Result identity or threshold differs from frozen design')
    if (result.get('source_manifest') != str(checked['path']) or
            result.get('source_manifest_sha256') != checked['sha256'] or
            result.get('model_directory') != str((directory/'model').resolve()) or
            result.get('model_sha256') != _sha256(directory/'model'/'model.keras') or
            result.get('model_metadata_sha256') != _sha256(directory/'model'/'model-metadata.json') or
            result.get('preprocessing', {}) != checked['data'].get('preprocessing', {})):
        raise ValueError('Saved evaluation provenance differs from the current frozen fold/model')
    expected = {(s['variant_id'], str(s['case_id']), s['anatomy_family'])
                for s in checked['samples'] if s['split'] == 'test'}
    actual = {(s['variant_id'], str(s['case_id']), s['anatomy_family']) for s in result['samples']}
    if actual != expected or len(actual) != len(result['samples']):
        raise ValueError('Test result coverage differs from frozen outer samples')


def summarize_evaluation(config, input_bytes):
    """Validate all fold results and write anatomy-balanced evaluation reports.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    input_bytes : bytes
        Exact captured INI bytes corresponding to ``config``; retained without rewriting
        when provided.

    Returns
    -------
    dict
        JSON/HTML report paths and macro-average segmentation metrics.

    Notes
    -----
    Writes evaluation-summary.json and evaluation-report.html after checking fold/model provenance; siblings and repeat seeds are not treated as independent families.
    """
    from fakect_evaluation_metrics import summarize_evaluations
    lock = _load_lock(config, input_bytes)
    results = []
    for fold in lock['plan']['design']['folds']:
        for seed in lock['plan']['design']['seeds']:
            directory, checked = _test_manifest(config, fold, seed)
            result = _read_result(directory.parent/'evaluation.json')
            _check_result(result, directory, checked, config, fold, seed)
            results.append(result)
    summary = summarize_evaluations(results)
    summary.update(complete=True, strategy=config['evaluation']['strategy'],
                   task='binary_aorta_segmentation', evaluation_lock_sha256=_sha256(
                       Path(config['experiment']['output_directory'])/'evaluation-lock.json'))
    output = Path(config['experiment']['output_directory'])
    _write_json(output/'evaluation-summary.json', summary)
    # Exact, inspectable JSON is also embedded in the portable report.
    rows = ''.join('<tr><th>'+html.escape(str(name))+'</th><td>'+html.escape(str(value))+'</td></tr>'
                   for name, value in summary['macro_average'].items())
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>FakeCT evaluation</title>'
    page += '<style>body{max-width:1000px;margin:40px auto;font:16px system-ui;color:#203047}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
    page += '<h1>Held-out family evaluation</h1><p>Binary aorta segmentation. Equal weight per family; training repeats are averaged within family.</p><table>'+rows+'</table>'
    page += '<h2>Held-out families</h2><table><tr><th>Family</th><th>Cases</th><th>Dice</th><th>IoU</th><th>Precision</th><th>Recall</th></tr>'
    for family in summary['family_reports']:
        page += '<tr><td>'+html.escape(family['anatomy_family'])+'</td><td>'+html.escape(', '.join(family['case_ids']))+'</td>'
        page += ''.join(f'<td>{family[name]:.4f}</td>' for name in ('dice', 'iou', 'precision', 'recall'))+'</tr>'
    page += '</table>'
    page += '<p><a href="evaluation-summary.json">Download full results</a></p><details><summary>Families, repeats and weighting</summary><pre>'
    page += html.escape(json.dumps(summary, indent=2))+'</pre></details></html>'
    (output/'evaluation-report.html').write_text(page)
    return {'ready': True, 'test_evaluated': True, 'report': str(output/'evaluation-report.html'),
            'summary': str(output/'evaluation-summary.json'), 'macro_average': summary['macro_average']}


def train_fixed_segmentation(config, manifest_path):
    """Refit a fresh model for the epoch count selected by inner validation folds.

    Parameters
    ----------
    config : dict
        Typed workflow configuration parsed from INI sections.
    manifest_path : str or pathlib.Path
        Prepared dataset manifest or frozen experiment-lock path.

    Returns
    -------
    dict
        Fixed-epoch model metadata and history with test payloads untouched.

    Raises
    ------
    ValueError
        The lock is not a valid fixed refit, training is empty/nonfinite or the manifest
        changes.
    FileExistsError
        The model output directory is nonempty.

    Notes
    -----
    No validation partition or best-epoch selection is used here. Writes history, final model.keras and model-metadata.json.

    Fresh final nested-CV fit with an epoch budget selected by inner folds.
    """
    from fakect_segmentation import MODEL_SCHEMA, PatchDataset, _tensorflow, build_unet, model_settings
    settings = model_settings(config)
    dataset = PatchDataset(manifest_path, settings)
    checked = dataset.manifest
    if (checked.get('lock', {}).get('config', {}).get('training_protocol') != 'fixed_epochs' or
            checked['split_sizes']['validation'] != 0 or not checked['split_sizes']['train']):
        raise ValueError('Fixed refit requires an explicit fixed-epochs fold without validation')
    output = Path(config['train']['model_directory'])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Fixed-refit model directory is not empty')
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(settings['seed'])
    tf.config.experimental.enable_op_determinism()
    model = build_unet(settings)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(settings['epochs']):
        total, examples = 0., 0
        for images, masks, valid in dataset.batches('train', epoch):
            model.reset_metrics()
            values = model.train_on_batch(images, np.concatenate((masks, valid), axis=-1), return_dict=True)
            loss = float(values['loss'])
            if not math.isfinite(loss):
                raise ValueError('Nonfinite fixed-refit training loss')
            total += loss*len(images)
            examples += len(images)
        if not examples:
            raise ValueError('Fixed refit contains no training patches')
        history.append({'epoch': epoch+1, 'train_loss': total/examples, 'train_patches': examples})
        _write_json(output/'history.json', history)
        print(json.dumps(history[-1]), flush=True)
    if _sha256(checked['path']) != checked['sha256']:
        raise ValueError('Fold manifest changed during final refit')
    model.save(output/'model.keras')
    metadata = {'schema_version': MODEL_SCHEMA, 'architecture': settings['architecture'],
        'settings': settings, 'source_manifest': str(checked['path']),
        'source_manifest_sha256': checked['sha256'], 'model_file': 'model.keras',
        'model_sha256': _sha256(output/'model.keras'), 'module_sha256': _sha256(__file__),
        'model_selection': 'Final fixed epoch budget selected using inner validation only',
        'epochs_completed': settings['epochs'], 'history': history,
        'split_mode': 'anatomy_family', 'family_assignment': checked['data']['family_assignment'],
        'split_sizes': checked['split_sizes'], 'loaded_variants': sorted(dataset.loaded_variants),
        'test_payloads_loaded': False, 'test_evaluated': False,
        'weights_include_test_selection': False, 'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__, 'evaluation_scope': 'Synthetic anatomy-family holdout; binary target only.'}
    _write_json(output/'model-metadata.json', metadata)
    return metadata
