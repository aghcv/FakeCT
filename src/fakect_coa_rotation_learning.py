"""Fresh fold-local locator and focused classifier with bounded native reads."""
from collections import OrderedDict
import gc
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

import fakect_coa_rotation_experiment as exp
import fakect_multiscale_experiment_v3 as geometry
import fakect_multiscale_learning_v3 as legacy
from fakect_context_volume import ContextVolume
from fakect_coa_classifier_learning import weight_fingerprint
from fakect_coa_pilot_learning import FineInputs


def matched_pairs(rows):
    """One negative/positive batch per frozen companion-edit block."""
    grouped = {}
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Matched panel must be nonempty and unique')
    for row in rows:
        if row.get('baseline'):
            raise ValueError('Primary classifier comparisons use edited matched pairs')
        companion = {k:v for k,v in row['parameters'].items() if k != 'coa_narrow'}
        key = (row['case_id'], json.dumps(companion, sort_keys=True))
        grouped.setdefault(key, []).append(row)
    pairs = []
    for key in sorted(grouped):
        group = grouped[key]
        if (len(group) != 2 or len({r['split'] for r in group}) != 1 or
                len({r['anatomy_family'] for r in group}) != 1):
            raise ValueError('Matched pairs must share case, family, companion edits and role')
        negative = [r for r in group if r['annotation']['coa_benchmark_positive'] is False]
        positive = [r for r in group if r['annotation']['coa_benchmark_positive'] is True]
        if len(negative) != 1 or len(positive) != 1:
            raise ValueError('Every training pair must have exactly one negative and positive')
        pairs.append((negative[0], positive[0]))
    return pairs


class StageFineInputs(FineInputs):
    """Bounded LRU cache, with a stage-specific allowlist for held-out inference."""
    def __init__(self, runtime, limit_mb, allowed_rows):
        super().__init__(runtime, limit_mb)
        self.allowed = {r['sample_id']: r['split'] for r in allowed_rows}
        if len(self.allowed) != len(allowed_rows):
            raise ValueError('Duplicate stage input IDs')

    def classifier(self, row, proposal, *, cache=True):
        if self.allowed.get(row['sample_id']) != row['split']:
            raise ValueError('Input is outside the frozen stage partition')
        # FineInputs is a reusable image reader with a development-only role guard.
        # Adapt that reader's role locally, after enforcing this stage's allowlist.
        adapted = row if row['split'] in ('train','validation') else {**row, 'split':'validation'}
        return super().classifier(adapted, proposal, cache=cache)


def pair_inputs(inputs, pair, proposals):
    arrays = [inputs.classifier(row, proposals[row['sample_id']]) for row in pair]
    return {key: np.concatenate([a[key] for a in arrays], axis=0) for key in arrays[0]}


def _score(tf, steps, rows, inputs, proposals, guard):
    result = []
    for pair in matched_pairs(rows):
        values = legacy._tensor_inputs(tf, pair_inputs(inputs, pair, proposals))
        output = steps.predict(values)
        logits = np.asarray(output['coa_logit']).ravel()
        scores = np.asarray(output['local_volume_loss']).ravel()
        if not np.isfinite(logits).all() or not np.isfinite(scores).all():
            raise ValueError('Nonfinite held-out prediction')
        for row, logit, score in zip(pair, logits, scores):
            result.append({'sample_id':row['sample_id'], 'case_id':row['case_id'],
                'family':row['anatomy_family'], 'split':row['split'],
                'label':int(row['annotation']['coa_benchmark_positive']),
                'logit':float(logit), 'probability':float(1/(1+np.exp(-np.clip(logit,-80,80)))),
                'score_label':row['annotation'][exp.SCORE], 'score_prediction':float(score)})
        guard.check('classifier-score', steps=steps)
    return result


def proposals_for_rows(locator, network, rows, runtime, output, guard, *, selection_allowed):
    """All crops depend only on the image prediction; masks audit without recentering."""
    inputs = legacy.Inputs(runtime)
    proposals, records = {}, []
    for row in rows:
        guard.check('proposal-before')
        ov, image = inputs.coarse(row)
        box = network.predict_boxes(locator, image)[0]
        origins, boxes = geometry.proposal_origins(box, ov, runtime)
        volume = ContextVolume(row, verify_hashes=True)
        try:
            geometry.verify_no_cancellation(volume, row)
            coverage = geometry.coverage(volume, origins, runtime)
            passed = geometry.coverage_gate(coverage, runtime)
        finally:
            volume.close()
        record = {'sample_id':row['sample_id'], 'split':row['split'],
                  'predicted_box':box.tolist(), 'origins_kji':origins,
                  'proposal_boxes':boxes.tolist(), 'coverage_passed':passed, **coverage}
        records.append(record)
        proposals[row['sample_id']] = record
        guard.check('proposal-after')
    geometry.write_json(Path(output)/'proposal-audit.json', {
        'passed':all(r['coverage_passed'] for r in records), 'samples':records,
        'selection_allowed':selection_allowed,
        'contract':'Image-only proposals. QA never moves a crop. Held-out failures remain in evaluation.'})
    return proposals, records


def fit_classifier(tf, runtime, rows, proposals, output, c, *, epochs, fixed_budget=False):
    import fakect_coa_paired_steps as network
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    guard = legacy.RuntimeGuard({**runtime, 'experiment':{'output_directory':str(output)}}, tf)
    train = [r for r in rows if r['split']=='train']
    validation = [r for r in rows if r['split']=='validation']
    if not train or (fixed_budget and validation) or (not fixed_budget and not validation):
        raise ValueError('Selection requires validation; fixed-budget refitting excludes it')
    if any(r['split'] not in ('train','validation') for r in rows):
        raise ValueError('Classifier fitting cannot read final-test rows')
    pairs = matched_pairs(train)
    inputs = StageFineInputs(runtime, c['training']['cache_limit_mb'], rows)
    detail = runtime['input.detail']
    model = network.build_classifier(detail['shape_kji'], num_proposals=detail['proposals'],
                                    base_filters=runtime['model']['base_filters'], seed=runtime['model']['seed'])
    initial_hash = weight_fingerprint(model)
    steps = network.make_steps(model, detail['shape_kji'], num_proposals=detail['proposals'],
        learning_rate=c['training']['classifier_learning_rate'], regression_weight=c['training']['regression_weight'])
    initial = _score(tf, steps, train+validation, inputs, proposals, guard)
    geometry.write_json(output/'initialization.json', {'seed':runtime['model']['seed'],
        'weights_sha256':initial_hash, 'fresh_weights':True, 'predictions':initial})
    best, selected_epoch, history, selected_records = math.inf, None, [], None
    rng = np.random.default_rng(runtime['model']['seed'])
    checkpoint = output/'classifier.keras'
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(pairs))
        losses, norms = [], []
        for index in order:
            pair = pairs[int(index)]
            guard.check('classifier-train-before', steps=steps)
            values = legacy._tensor_inputs(tf, pair_inputs(inputs, pair, proposals))
            targets = {'coa_logit':tf.constant([[0.],[1.]], tf.float32),
                'local_volume_loss':tf.constant([[r['annotation'][exp.SCORE]] for r in pair], tf.float32)}
            value = steps.train(values, targets)
            losses.append(float(value['loss'])); norms.append(float(value['gradient_global_norm']))
            guard.check('classifier-train-after', steps=steps)
        # Validation only governs selection. Training metrics are descriptive.
        validation_records = _score(tf, steps, validation, inputs, proposals, guard) if validation else []
        train_records = _score(tf, steps, train, inputs, proposals, guard)
        train_metrics = legacy.metrics(train_records)
        val_metrics = legacy.metrics(validation_records)
        selection = val_metrics['bce'] if val_metrics else None
        if fixed_budget or selection < best:
            model.save(checkpoint)
            selected_epoch = epoch
            selected_records = train_records+validation_records
            if selection is not None:
                best = selection
        record = {'epoch':epoch, 'train_loss':float(np.mean(losses)),
                  'gradient_norm_mean':float(np.mean(norms)), 'training':train_metrics,
                  'validation':val_metrics, 'selection_bce':selection,
                  'training_pair_order':[[r['sample_id'] for r in pairs[int(i)]] for i in order],
                  'optimizer_updates':epoch*len(pairs), 'runtime':guard.check('classifier-epoch',save=True)}
        history.append(record)
        geometry.write_json(output/'history.json', {'epochs':history, 'selected_epoch':selected_epoch,
            'checkpoint_selection':'final fixed-budget epoch' if fixed_budget else 'minimum validation BCE'})
        print(json.dumps({'phase':'classifier','epoch':epoch, 'selected_epoch':selected_epoch,
                          'training':train_metrics,'validation':val_metrics}), flush=True)
    if weight_fingerprint(model) == initial_hash:
        raise ValueError('Classifier weights did not change')
    del steps, model
    gc.collect()
    model = network.load_model(checkpoint)
    steps = network.make_steps(model, detail['shape_kji'], num_proposals=detail['proposals'],
        learning_rate=c['training']['classifier_learning_rate'], regression_weight=c['training']['regression_weight'])
    predictions = _score(tf, steps, train+validation, inputs, proposals, guard)
    from fakect_coa_classifier_learning import checkpoint_prediction_parity
    parity = checkpoint_prediction_parity(selected_records,predictions)
    if not parity['passed']:
        raise ValueError('Reloaded classifier differs from its selected-epoch predictions')
    # Check saved-model reload on independent single-scan inference as well as batch2.
    row = train[0]
    values = inputs.classifier(row, proposals[row['sample_id']])
    before = model(values, training=False)
    other = network.load_model(checkpoint)
    after = other(values, training=False)
    for key in before:
        np.testing.assert_allclose(before[key],after[key],atol=1e-6,rtol=1e-5)
    receipt = {'checkpoint':str(checkpoint), 'checkpoint_sha256':geometry.sha256(checkpoint),
               'selected_epoch':selected_epoch, 'epochs_completed':epochs,
               'optimizer_updates':epochs*len(pairs), 'fresh_initial_weights_sha256':initial_hash,
               'training_sample_ids':[r['sample_id'] for r in train],
               'validation_sample_ids':[r['sample_id'] for r in validation],
               'checkpoint_parity':parity, 'cache':inputs.stats(),
               'training':legacy.metrics([r for r in predictions if r['split']=='train']),
               'validation':legacy.metrics([r for r in predictions if r['split']=='validation'])}
    geometry.write_json(output/'receipt.json', receipt)
    del other, model, steps, inputs
    gc.collect()
    return receipt


def run_stage(c, lock, stage):
    from fakect_coa_rotation_locator import fit_locator
    root = Path(c['experiment']['output_directory'])
    output = root/'stages'/stage['id']
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').touch()
    runtime = exp.runtime_config(c, lock, stage, output)
    train_ids, validation_ids = set(stage['train_case_ids']), set(stage['validation_case_ids'])
    original_rows = [{**r,'split':'train' if r['case_id'] in train_ids else 'validation'}
                     for r in lock['original_samples'] if r['case_id'] in train_ids|validation_ids]
    rows = [{**r,'split':'train' if r['case_id'] in train_ids else 'validation'}
            for r in lock['samples'] if r['case_id'] in train_ids|validation_ids]
    fixed = stage['stage']=='outer_refit'
    budget = exp.select_refit_budgets(lock, stage, root) if fixed else {
        'locator_epochs':c['training']['locator_epochs'], 'classifier_epochs':c['training']['classifier_epochs']}
    geometry.write_json(output/'stage-input.json', {'stage':stage,'budget':budget,
        'source_lock_sha256':geometry.sha256(root/'experiment-lock.json')})
    tf, locator_network = legacy.configure_tensorflow(runtime)
    guard = legacy.RuntimeGuard(runtime,tf)
    locator_receipt = fit_locator(tf,runtime,original_rows,output/'locator',
        epochs=budget['locator_epochs'], fixed_budget=fixed,
        allow_provisional_family_overlap=stage['protocol'] != 'family_holdout')
    locator = locator_network.load_model(locator_receipt['checkpoint'])
    locator.trainable = False
    before_locator = weight_fingerprint(locator)
    proposals, audit = proposals_for_rows(locator,locator_network,rows,runtime,output,guard,selection_allowed=not fixed)
    if not all(r['coverage_passed'] for r in audit):
        raise ValueError('Fold-local locator failed training/validation coverage; inspect proposal-audit.json')
    classifier_receipt = fit_classifier(tf,runtime,rows,proposals,output/'classifier',c,
        epochs=budget['classifier_epochs'],fixed_budget=fixed)
    if weight_fingerprint(locator) != before_locator:
        raise ValueError('Locator changed during classifier fitting')
    result = {'status':'completed','stage_id':stage['id'],'protocol':stage['protocol'],
        'source_lock_sha256':geometry.sha256(root/'experiment-lock.json'),
        'locator_selected_epoch':locator_receipt['selected_epoch'],
        'classifier_selected_epoch':classifier_receipt['selected_epoch'],
        'locator':locator_receipt,'classifier':classifier_receipt,
        'final_test_evaluated':False,'final_predictions':[],
        'reserved_test_evaluated':False,'budget':budget}
    if stage['stage'] != 'inner_selection':
        # Open outer images only after both learned models and their selection are frozen.
        test_ids = set(stage['final_test_case_ids'])
        test_rows = [{**r,'split':'final_test'} for r in lock['samples'] if r['case_id'] in test_ids]
        test_dir = output/'final-test'
        test_dir.mkdir()
        test_proposals, test_audit = proposals_for_rows(locator,locator_network,test_rows,runtime,test_dir,guard,selection_allowed=False)
        import fakect_coa_paired_steps as network
        model = network.load_model(classifier_receipt['checkpoint'])
        steps = network.make_steps(model,runtime['input.detail']['shape_kji'],
            num_proposals=runtime['input.detail']['proposals'], regression_weight=c['training']['regression_weight'])
        inputs = StageFineInputs(runtime,c['training']['cache_limit_mb'],test_rows)
        predictions = _score(tf,steps,test_rows,inputs,test_proposals,guard)
        cov = {r['sample_id']:r['coverage_passed'] for r in test_audit}
        for row in predictions:
            row.update(protocol=stage['protocol'], fold_id=stage['outer_fold_id'], seed=stage['seed'],
                       outer_case=stage.get('target_case_id'), outer_family=row['family'],
                       locator_coverage_passed=cov[row['sample_id']],pipeline_failure=not cov[row['sample_id']])
        result.update(final_test_evaluated=True,final_predictions=predictions,
                      final_locator_failed_sample_ids=[r['sample_id'] for r in test_audit if not r['coverage_passed']])
        geometry.write_json(test_dir/'predictions.json',{'predictions':predictions,
            'checkpoint_sha256':classifier_receipt['checkpoint_sha256'],
            'source_lock_sha256':result['source_lock_sha256'], 'qa_can_change_model':False})
    exp.inspect_lock(c,(root/'input.ini').read_bytes())
    geometry.write_json(output/'results.json',result)
    (output/'INCOMPLETE').unlink()
    return result


def smoke(c, lock):
    """Exercise both real-size graphs and bounded cache pages before real folds."""
    import fakect_coa_paired_steps as network
    import fakect_multiscale_network_v3 as locator_network
    from fakect_tensorflow_steps import make_locator_steps
    from fakect_coa_pilot_learning import full_cache_memory_probe
    root = Path(c['experiment']['output_directory'])
    runtime = exp.runtime_config(c,lock,lock['stages'][0],root)
    tf,_ = legacy.configure_tensorflow(runtime)
    guard = legacy.RuntimeGuard(runtime,tf)
    overview_shape = runtime['input.overview']['shape_kji']
    locator = locator_network.build_locator(overview_shape,base_filters=runtime['model']['base_filters'])
    steps = make_locator_steps(locator_network.locator_outputs(locator),overview_shape,
        learning_rate=c['training']['locator_learning_rate'],loss_function=locator_network.locator_loss)
    x = tf.concat((tf.random.stateless_uniform((1,*overview_shape,1),seed=(42,0)),
                   tf.ones((1,*overview_shape,1))),axis=-1)
    target = tf.constant([[.5,.5,.5,.2,.2,.2]],tf.float32)
    value = steps.train(x,target)
    before = steps.predict(x)
    with tempfile.TemporaryDirectory(dir=str(root),prefix='rotation-smoke-') as temporary:
        path = Path(temporary)/'locator.keras';locator.save(path)
        restored = locator_network.load_model(path)
        after = locator_network.locator_outputs(restored)(x,training=False)
        for key in before:
            np.testing.assert_allclose(before[key],after[key],atol=1e-6,rtol=1e-5)
    locator_gradient = float(value['gradient_global_norm'])
    del locator,steps,x,restored,before,after
    gc.collect()
    detail = runtime['input.detail'];shape = detail['shape_kji']
    model = network.build_classifier(shape,base_filters=runtime['model']['base_filters'])
    steps = network.make_steps(model,shape,learning_rate=c['training']['classifier_learning_rate'],
        regression_weight=c['training']['regression_weight'])
    values = {f'detail_{i}':tf.concat((tf.random.stateless_uniform((2,*shape,1),seed=(42,i)),
                                   tf.ones((2,*shape,1))),axis=-1) for i in range(detail['proposals'])}
    targets = {'coa_logit':tf.constant([[0.],[1.]]),'local_volume_loss':tf.constant([[.1],[.4]])}
    for _ in range(4):
        value = steps.train(values,targets);steps.predict(values)
        guard.check('smoke-classifier',steps=steps)
    needed = detail['proposals']*math.prod(shape)*2*4
    count = int(c['training']['cache_limit_mb']*1024**2)//needed
    arrays,probe = full_cache_memory_probe(runtime,guard,sample_count=count,
        cache_limit_mb=c['training']['cache_limit_mb'])
    value = steps.train(values,targets)
    probe['model_update_while_resident'] = True
    guard.check('smoke-resident-update',steps=steps,save=True)
    before = steps.predict(values)
    with tempfile.TemporaryDirectory(dir=str(root),prefix='rotation-smoke-') as temporary:
        path = Path(temporary)/'classifier.keras';model.save(path)
        restored = network.load_model(path);after = restored(values,training=False)
        for key in before:
            np.testing.assert_allclose(before[key],after[key],atol=1e-6,rtol=1e-5)
    result = {'status':'passed','synthetic_data_only':True,'checkpoint_parity':True,
        'source_lock_sha256':geometry.sha256(root/'experiment-lock.json'),
        'locator_gradient_norm':locator_gradient,'classifier_gradient_norm':float(value['gradient_global_norm']),
        'full_cache_memory_probe':probe,'tensorflow_version':tf.__version__,
        'gpu_visible':bool(tf.config.list_logical_devices('GPU')),
        'overview_shape':overview_shape,'detail_shape':shape,'traces':steps.tracing_counts(),
        **guard.check('smoke-reload',steps=steps,save=True)}
    if locator_gradient <= 0 or result['classifier_gradient_norm'] <= 0:
        raise ValueError('Smoke must demonstrate nonzero gradients for both models')
    geometry.write_json(root/'runtime-smoke.json',result)
    del arrays
    return result
