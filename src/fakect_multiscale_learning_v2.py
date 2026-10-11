"""Revision-two pilot with stable TensorSpecs and measured resource guards."""
from collections import OrderedDict
import json
from pathlib import Path
import resource
import time

import numpy as np

import fakect_multiscale_experiment_v2 as exp
from fakect_context_volume import ContextVolume
from fakect_multiscale_volume import build_overview


def overview(row, c):
    s = c['input.overview']
    return build_overview(row, s['cache_directory'], shape_kji=tuple(s['shape_kji']),
                          max_chunk_voxels=s['max_chunk_voxels'], verify_hashes=True)


def _preview(row, ov, box, output, *, predicted=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    image = ov['image']
    center = np.clip((box[:3]*image.shape).astype(int), 0, np.array(image.shape)-1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for axis, ax in enumerate(axes):
        other = [i for i in range(3) if i != axis]
        physical_extent = np.array(image.shape)*np.array(ov['spacing_ijk_mm'][::-1])
        ax.imshow(np.take(image, int(center[axis]), axis=axis), cmap='gray', vmin=0, vmax=.5, origin='lower',
                  extent=(0,physical_extent[other[1]],0,physical_extent[other[0]]))
        for b, color, label in [(box, '#ffdf00', 'Supervision'), (predicted, '#00d8ff', 'Prediction')]:
            if b is None:
                continue
            lo = (b[:3]-b[3:]/2)*physical_extent
            size = b[3:]*physical_extent
            slice_mm = (center[axis]+.5)*ov['spacing_ijk_mm'][::-1][axis]
            intersects = lo[axis] <= slice_mm <= lo[axis]+size[axis]
            ax.add_patch(Rectangle((lo[other[1]], lo[other[0]]), size[other[1]], size[other[0]],
                                   fill=False, edgecolor=color, linewidth=1.5,
                                   linestyle='-' if intersects else '--',
                                   label=label if intersects else label+' (out-of-plane projection)'))
        ax.set_title(('Axial', 'Coronal', 'Sagittal')[axis])
        ax.set_xlabel(('k','j','i')[other[1]]+' (mm from source origin)')
        ax.set_ylabel(('k','j','i')[other[0]]+' (mm)')
    axes[0].legend(loc='lower left', fontsize=7)
    fig.suptitle(row['sample_id'] + ' · ' + row['split'] + '\nYellow: mask supervision; cyan: image prediction')
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def prepare(c, lock):
    out = Path(c['experiment']['output_directory'])
    if (out/'preparation.json').exists():
        return exp.checked_json(out/'preparation.json')
    previews = out/'previews'
    previews.mkdir(exist_ok=True)
    rows = []
    started = time.monotonic()
    for row in lock['samples']:
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Pilot cannot prepare test samples')
        ov = overview(row, c)
        volume = ContextVolume(row, verify_hashes=True)
        exp.verify_no_cancellation(volume, row)
        target = exp.bbox_target(volume, ov)
        origins, _ = exp.proposal_origins(target, ov, c)
        # Oracle coverage is a geometry feasibility diagnostic, not an inference result.
        ideal = exp.coverage(volume, origins, c)
        record = {'sample_id': row['sample_id'], 'split': row['split'],
                  'target_box_kji': target.tolist(), 'overview_composition_key': ov['composition_key'],
                  'overview_cache_key': ov['cache_key'], 'stats': ov['stats'],
                  'overview_spacing_ijk_mm': list(ov['spacing_ijk_mm']),
                  'ideal_box_coverage_qa_only': ideal, 'no_cancellation_verified': True}
        rows.append(record)
        if row['baseline']:
            _preview(row, ov, target, previews/(row['case_id']+'-overview.png'))
        print(json.dumps({'stage':'prepare', 'completed':len(rows), 'total':len(lock['samples']),
                          'sample_id':row['sample_id']}), flush=True)
        del volume, ov
    result = {'schema_version':'fakect.multiscale-prepared/1', 'complete':True,
              'lock_sha256':exp.sha256(out/'experiment-lock.json'), 'samples':rows,
              'elapsed_seconds':time.monotonic()-started,
              'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'storage':'Shared native ATN + immutable crop; shared base overview cache; no per-sibling full image export.'}
    exp.inspect_lock(c, (out/'input.ini').read_bytes())
    exp.write_json(out/'preparation.json', result)
    exp.report(c)
    return result


class Inputs:
    """At most two overview arrays and one transient native-crop reader."""
    def __init__(self, c):
        self.c, self.cache = c, OrderedDict()

    def coarse(self, row):
        key = row['sample_id']
        if key not in self.cache:
            if len(self.cache) >= 2:
                self.cache.popitem(last=False)
            self.cache[key] = overview(row, self.c)
        self.cache.move_to_end(key)
        ov = self.cache[key]
        x = exp.normalized_input(ov['image'], ov['valid_fraction'], self.c)[None]
        return ov, x

    def classifier(self, row, proposal):
        _, x = self.coarse(row)
        result = {'overview':x, 'proposal_boxes':np.asarray(proposal['proposal_boxes'],np.float32)[None]}
        volume = ContextVolume(row, verify_hashes=True)
        detail = self.c['input.detail']
        for i, origin in enumerate(proposal['origins_kji']):
            block = volume.read_block(origin, detail['shape_kji'], detail['spacing_ijk_mm'])
            result['detail_'+str(i)] = exp.normalized_input(block['image'], block['valid'], self.c)[None]
        return result


def metrics(records):
    """Fixed .5 decision; edited-negative separation is reported explicitly."""
    if not records:
        return None
    y = np.array([r['label'] for r in records], float)
    p = np.array([r['probability'] for r in records], float)
    decision = p >= .5
    positive, negative = y == 1, y == 0
    eps = np.finfo(np.float32).eps
    bce = float(np.mean(-y*np.log(np.clip(p,eps,1-eps))-(1-y)*np.log(np.clip(1-p,eps,1-eps))))
    return {'samples':len(records), 'bce':bce, 'accuracy':float(np.mean(decision==y)),
            'sensitivity':float(np.mean(decision[positive])) if positive.any() else None,
            'specificity':float(np.mean(~decision[negative])) if negative.any() else None,
            'score_mae':float(np.mean([abs(r['score_prediction']-r['score_label']) for r in records])),
            'positive_probability_mean':float(p[positive].mean()) if positive.any() else None,
            'negative_probability_mean':float(p[negative].mean()) if negative.any() else None}


def configure_tensorflow(c):
    import tensorflow as tf
    import fakect_multiscale_network_v2 as network
    gpus = tf.config.list_physical_devices('GPU')
    if c['pilot']['require_gpu'] and not gpus:
        raise ValueError('No GPU visible; submit the GPU pilot script')
    if gpus:
        tf.config.set_visible_devices(gpus[:1], 'GPU')
        tf.config.set_logical_device_configuration(gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=c['pilot']['gpu_memory_limit_mb'])])
    tf.keras.utils.set_random_seed(c['model']['seed'])
    return tf, network


def _score_classifier(model, rows, inputs, proposals):
    result = []
    for row in rows:
        outputs = model(inputs.classifier(row, proposals[row['sample_id']]), training=False)
        logit = float(np.asarray(outputs['coa_logit']).ravel()[0])
        prob = float(1/(1+np.exp(-np.clip(logit, -80, 80))))
        score = float(np.asarray(outputs['local_volume_loss']).ravel()[0])
        if not np.isfinite([logit,score]).all():
            raise ValueError('Nonfinite classifier output')
        result.append({'sample_id':row['sample_id'], 'case_id':row['case_id'],
                       'family':row['anatomy_family'], 'split':row['split'], 'role':row['pilot_role'],
                       'label':int(row['annotation']['coa_benchmark_positive']), 'probability':prob,
                       'score_label':row['annotation'][exp.SCORE], 'score_prediction':score})
    return result


def memory_snapshot(tf=None):
    """Linux RSS from proc; ru_maxrss is KiB on this deployment."""
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key = line.split(':', 1)[0]
        if key in ('VmRSS', 'VmHWM'):
            values[key] = float(line.split()[1])/1024
    result = {'rss_mib':values['VmRSS'], 'host_peak_rss_mib':values['VmHWM']}
    if tf is not None and tf.config.list_logical_devices('GPU'):
        info = tf.config.experimental.get_memory_info('GPU:0')
        result.update(gpu_current_mib=info['current']/1024**2, gpu_peak_mib=info['peak']/1024**2)
    return result


class RuntimeGuard:
    def __init__(self, c, tf):
        self.c, self.tf = c, tf
        self.path = Path(c['experiment']['output_directory'])/'training-runtime.json'
        self.records = []
        self.calls = 0

    def check(self, stage, *, steps=None, save=False):
        self.calls += 1
        record = {'stage':stage, 'sample_number':self.calls, **memory_snapshot(self.tf)}
        if steps is not None:
            record['traces'] = steps.tracing_counts()
        reason = None
        if record['host_peak_rss_mib'] > self.c['training']['host_memory_limit_mb']:
            reason = 'Host memory exceeded sampled application budget'
        if record.get('gpu_peak_mib',0) > self.c['pilot']['gpu_memory_limit_mb']:
            reason = 'GPU memory exceeded budget'
        if steps is not None and any(v > self.c['training']['maximum_traces'] for v in record['traces'].values()):
            reason = 'TensorFlow graph retraced; stop before accumulating more graphs'
        # Store warm-up detail and periodic evidence, not every live tensor/batch.
        if save or reason or self.calls <= 64 or self.calls % 32 == 0:
            self.records.append(record)
            exp.write_json(self.path, {'status':'failed' if reason else 'within_budget',
                'reason':reason, 'host_limit_mib':self.c['training']['host_memory_limit_mb'],
                'gpu_limit_mib':self.c['pilot']['gpu_memory_limit_mb'],
                'guard_semantics':'Sampled current/peak RSS before and after batches, not a hard host allocation limit.',
                'records':self.records})
        if reason:
            raise RuntimeError(reason)
        return record


def _tensor_inputs(tf, values):
    return {key:tf.convert_to_tensor(value, dtype=tf.float32) for key,value in values.items()}


def runtime_smoke(c):
    """Separate-process GPU check: real overview dimensions, synthetic inputs."""
    from fakect_tensorflow_steps import make_locator_steps, make_classifier_steps
    import tempfile
    tf,network = configure_tensorflow(c)
    shape = c['input.overview']['shape_kji']
    locator = network.build_locator(shape,c['model']['base_filters'],c['model']['seed'])
    steps = make_locator_steps(network.locator_outputs(locator),shape,
        learning_rate=c['training']['locator_learning_rate'],
        loss_function=lambda y,o:network.locator_loss(y,o['box'],logits=o['box_logits']))
    samples = []
    for i in range(32):
        x = tf.random.stateless_uniform((1,*shape,2),seed=(c['model']['seed'],i))
        y = tf.constant([[.60+.002*(i%4),.5,.5,.15,.12,.10]],tf.float32)
        value = steps.train(x,y)
        steps.predict(x)
        steps.assert_stable_traces()
        snapshot = memory_snapshot(tf)
        if snapshot['host_peak_rss_mib'] > c['training']['host_memory_limit_mb']:
            raise RuntimeError('GPU smoke exceeded the host memory guard')
        samples.append({'step':i+1,'loss':float(value['loss']),**snapshot,'traces':steps.tracing_counts()})
    growth = samples[-1]['rss_mib']-samples[7]['rss_mib']
    if growth > 512:
        raise RuntimeError(f'GPU smoke RSS grew {growth:.1f} MiB after warmup; investigate before training')
    classifier = network.build_classifier(shape,(16,16,16),base_filters=2,locator=locator)
    cls = make_classifier_steps(classifier,shape,(16,16,16))
    inputs={'overview':x,'proposal_boxes':tf.constant([[[.5,.5,.5,.15,.15,.15]]*2],tf.float32),
            'detail_0':tf.ones((1,16,16,16,2)), 'detail_1':tf.zeros((1,16,16,16,2))}
    for i in range(4):
        cls.train(inputs,{'coa_logit':tf.constant([[float(i%2)]]),'local_volume_loss':tf.constant([[.3]])})
    before=cls.predict(inputs)
    cls.assert_stable_traces()
    with tempfile.TemporaryDirectory(prefix='fakect-v2-gpu-') as directory:
        path=Path(directory)/'smoke.keras';classifier.save(path)
        after=network.load_model(path)(inputs,training=False)
        for key in before:
            np.testing.assert_allclose(before[key],after[key],atol=1e-6,rtol=1e-5)
    result={'status':'passed','synthetic_data_only':True,'tensorflow_version':tf.__version__,
            'overview_shape_kji':shape,'warmup_rss_growth_mib':growth,'samples':samples,
            'classifier_traces':cls.tracing_counts(),'checkpoint_parity':True,**memory_snapshot(tf)}
    exp.write_json(Path(c['experiment']['output_directory'])/'runtime-smoke.json',result)
    return result


def box_metrics(rows, predictions, targets, extents):
    errors, overlaps = [], []
    for row in rows:
        key = row['sample_id']
        b, t = np.asarray(predictions[key]), np.asarray(targets[key]).reshape(6)
        errors.append(float(np.linalg.norm((b[:3]-t[:3])*extents[key])))
        low = np.maximum(b[:3]-b[3:]/2,t[:3]-t[3:]/2)
        high = np.minimum(b[:3]+b[3:]/2,t[:3]+t[3:]/2)
        intersection = float(np.prod(np.maximum(high-low,0)))
        overlaps.append(intersection/(np.prod(b[3:])+np.prod(t[3:])-intersection))
    return {'samples':len(rows), 'mean_center_error_mm':float(np.mean(errors)),
            'max_center_error_mm':float(np.max(errors)), 'mean_box_iou':float(np.mean(overlaps))}


def _score_stable(tf, steps, rows, inputs, proposals, guard):
    records = []
    for row in rows:
        guard.check('classifier-evaluation-before',steps=steps)
        x = _tensor_inputs(tf,inputs.classifier(row,proposals[row['sample_id']]))
        output = steps.predict(x)
        logit = float(np.asarray(output['coa_logit']).ravel()[0])
        score = float(np.asarray(output['local_volume_loss']).ravel()[0])
        if not np.isfinite([logit,score]).all():
            raise ValueError('Nonfinite classifier output')
        records.append({'sample_id':row['sample_id'],'case_id':row['case_id'],
            'family':row['anatomy_family'],'split':row['split'],'role':row['pilot_role'],
            'label':int(row['annotation']['coa_benchmark_positive']),
            'probability':float(1/(1+np.exp(-np.clip(logit,-80,80)))),
            'score_label':row['annotation'][exp.SCORE],'score_prediction':score})
        guard.check('classifier-evaluation-after',steps=steps)
    return records


def fit(c, lock):
    from fakect_tensorflow_steps import make_locator_steps, make_classifier_steps
    out = Path(c['experiment']['output_directory'])
    prepared = exp.checked_json(out/'preparation.json')
    if prepared.get('complete') is not True or prepared['lock_sha256'] != exp.sha256(out/'experiment-lock.json'):
        raise ValueError('Invalid prepared revision')
    if any(r['split'] not in ('train','validation') for r in lock['samples']):
        raise ValueError('Pilot cannot consume test samples')
    prepared_ids = [r['sample_id'] for r in prepared['samples']]
    if len(set(prepared_ids)) != len(prepared_ids) or set(prepared_ids) != {r['sample_id'] for r in lock['samples']}:
        raise ValueError('Prepared samples must join exactly')
    targets = {r['sample_id']:np.asarray(r['target_box_kji'],np.float32)[None] for r in prepared['samples']}
    for target in targets.values():
        b=target.reshape(-1)
        if (b.shape!=(6,) or not np.isfinite(b).all() or np.any(b[3:]<=0) or
                np.any(b[:3]-b[3:]/2 < -1e-6) or np.any(b[:3]+b[3:]/2 > 1+1e-6)):
            raise ValueError('Invalid anatomy supervision')
    model_dir = out/'model'
    model_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    tf,network = configure_tensorflow(c)
    guard = RuntimeGuard(c,tf)
    guard.check('tensorflow-configured',save=True)
    inputs = Inputs(c)
    train = [r for r in lock['samples'] if r['split']=='train']
    validation = [r for r in lock['samples'] if r['split']=='validation']
    locator_train = [r for r in train if r['baseline']]
    locator_validation = [r for r in validation if r['baseline']]
    if not locator_train or not locator_validation:
        raise ValueError('Locator requires original training and validation references')
    shape = c['input.overview']['shape_kji']
    extents = {r['sample_id']:np.asarray(shape)*np.asarray(r['overview_spacing_ijk_mm'][::-1]) for r in prepared['samples']}
    prior = np.mean([targets[r['sample_id']][0] for r in locator_train],axis=0)
    prior_predictions = {r['sample_id']:prior for r in lock['samples']}
    diagnostics = {'prior_box':prior.tolist(),'prior_source_sample_ids':[r['sample_id'] for r in locator_train],
        'prior_training':box_metrics(locator_train,prior_predictions,targets,extents),
        'prior_validation':box_metrics(locator_validation,prior_predictions,targets,extents),
        'note':'Training-only constant prior is an initialization and comparator, not image localization.'}
    exp.write_json(out/'locator-diagnostics.json',diagnostics)
    settings = c['model']
    locator = network.build_locator(shape,settings['base_filters'],settings['seed'],prior_box=prior)
    view = network.locator_outputs(locator)
    locator_steps = make_locator_steps(view,shape,learning_rate=c['training']['locator_learning_rate'],
        loss_function=lambda y,o: network.locator_loss(y,o['box'],logits=o['box_logits']))
    history,best = [],float('inf')
    rng = np.random.default_rng(settings['seed'])
    for epoch in range(c['pilot']['locator_epochs']):
        losses,norms = [],[]
        for index in rng.permutation(len(locator_train)):
            row = locator_train[int(index)]
            _,x = inputs.coarse(row)
            guard.check('locator-train-before',steps=locator_steps)
            result = locator_steps.train(tf.convert_to_tensor(x),tf.convert_to_tensor(targets[row['sample_id']]))
            losses.append(float(result['loss']))
            norms.append(float(result['gradient_global_norm']))
            guard.check('locator-train-after',steps=locator_steps)
        predictions,validation_loss = {},[]
        for row in locator_train+locator_validation:
            _,x=inputs.coarse(row)
            output=locator_steps.predict(tf.convert_to_tensor(x))
            predictions[row['sample_id']]=np.asarray(output['box'])[0]
            if row['split']=='validation':
                loss=network.locator_loss(targets[row['sample_id']],output['box'],logits=output['box_logits'])
                validation_loss.append(float(tf.reduce_mean(loss)))
            guard.check('locator-evaluation',steps=locator_steps)
        record={'epoch':epoch+1,'train_loss':float(np.mean(losses)),
            'gradient_norm_mean':float(np.mean(norms)),'validation_loss':float(np.mean(validation_loss)),
            'training_geometry':box_metrics(locator_train,predictions,targets,extents),
            'validation_geometry':box_metrics(locator_validation,predictions,targets,extents),
            'runtime':guard.check('locator-epoch',steps=locator_steps,save=True)}
        if not np.isfinite([record['train_loss'],record['validation_loss']]).all():
            raise ValueError('Nonfinite locator training')
        if record['validation_loss'] < best:
            best=record['validation_loss'];locator.save(model_dir/'locator.keras')
        history.append(record)
        exp.write_json(out/'locator-history.json',{'epochs':history,'loss':'raw-parameter KL + coordinate regression'})
        print(json.dumps({'stage':'locator',**record}),flush=True)
    trace_summary={'locator':locator_steps.tracing_counts()}
    # Release the training graph and optimizer before constructing classifier.
    del locator_steps,view,locator
    import gc
    gc.collect()
    locator=network.load_model(model_dir/'locator.keras')
    proposals,audits,predictions,blank_differences={},{},{},[]
    audit_rows=[];failures=[]
    for row in lock['samples']:
        ov,x=inputs.coarse(row)
        predicted=network.predict_boxes(locator,x)[0]
        predictions[row['sample_id']]=predicted
        origins,boxes=exp.proposal_origins(predicted,ov,c)
        volume=ContextVolume(row,verify_hashes=True)
        exp.verify_no_cancellation(volume,row)
        cov=exp.coverage(volume,origins,c)
        ok=exp.coverage_gate(cov,c)
        if not ok:failures.append(row['sample_id'])
        record={'sample_id':row['sample_id'],'split':row['split'],'predicted_box':predicted.tolist(),
                'origins_kji':origins,'proposal_boxes':boxes.tolist(),**cov,'coverage_passed':ok,
                'center_error_mm':float(np.linalg.norm((predicted[:3]-targets[row['sample_id']][0,:3])*extents[row['sample_id']]))}
        proposals[row['sample_id']]=record;audit_rows.append(record)
        if row['baseline']:
            _preview(row,ov,targets[row['sample_id']][0],out/'previews'/(row['case_id']+'-locator.png'),predicted=predicted)
            blank=x.copy();blank[...,0]=0
            blank_box=network.predict_boxes(locator,blank)[0]
            blank_differences.append({'sample_id':row['sample_id'],
                'center_shift_mm':float(np.linalg.norm((blank_box[:3]-predicted[:3])*extents[row['sample_id']])),
                'note':'Blank intensity with validity unchanged; descriptive input-dependence check only.'})
        del volume
        guard.check('proposal-audit',save=True)
    diagnostics.update(learned_training=box_metrics(locator_train,predictions,targets,extents),
                       learned_validation=box_metrics(locator_validation,predictions,targets,extents),
                       blank_image_response=blank_differences)
    exp.write_json(out/'locator-diagnostics.json',diagnostics)
    exp.write_json(out/'proposal-audit.json',{'passed':not failures,'failed_sample_ids':failures,'samples':audit_rows,
        'proof':'No-cancellation counts plus 100% net-removal coverage; anatomy coverage required for ALL samples.',
        'inference':'Image-predicted crops only. Ground truth audits them without moving them.'})
    exp.report(c)
    if failures:
        result={'status':'stopped_at_locator_coverage_gate','classifier_trained':False,'test_evaluated':False,
                'failed_samples':failures,'next_step':'Inspect residual localization and coverage errors.'}
    else:
        classifier=network.build_classifier(shape,c['input.detail']['shape_kji'],
            num_proposals=c['input.detail']['proposals'],base_filters=settings['base_filters'],seed=settings['seed'],locator=locator)
        steps=make_classifier_steps(classifier,shape,c['input.detail']['shape_kji'],
            num_proposals=c['input.detail']['proposals'],learning_rate=settings['learning_rate'])
        initial=_score_stable(tf,steps,train,inputs,proposals,guard)
        history,best=[],float('inf')
        for epoch in range(c['pilot']['classifier_epochs']):
            losses,norms=[],[]
            for index in rng.permutation(len(train)):
                row=train[int(index)];guard.check('classifier-train-before',steps=steps)
                x=_tensor_inputs(tf,inputs.classifier(row,proposals[row['sample_id']]))
                y=_tensor_inputs(tf,{'coa_logit':np.array([[row['annotation']['coa_benchmark_positive']]],np.float32),
                    'local_volume_loss':np.array([[row['annotation'][exp.SCORE]]],np.float32)})
                values=steps.train(x,y)
                losses.append(float(values['loss']));norms.append(float(values['gradient_global_norm']))
                guard.check('classifier-train-after',steps=steps)
            valid_records=_score_stable(tf,steps,validation,inputs,proposals,guard)
            record={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'gradient_norm_mean':float(np.mean(norms)),
                    'validation':metrics(valid_records),'runtime':guard.check('classifier-epoch',steps=steps,save=True)}
            if record['validation']['bce']<best:
                best=record['validation']['bce'];classifier.save(model_dir/'classifier.keras')
            history.append(record)
            exp.write_json(out/'classifier-history.json',{'initial_training':metrics(initial),'epochs':history})
            print(json.dumps({'stage':'classifier',**record}),flush=True)
        trace_summary['classifier']=steps.tracing_counts()
        del steps,classifier
        gc.collect()
        classifier=network.load_model(model_dir/'classifier.keras')
        # Same stable inference interface; optimizer built but no further training occurs.
        selected=make_classifier_steps(classifier,shape,c['input.detail']['shape_kji'],
            num_proposals=c['input.detail']['proposals'],learning_rate=settings['learning_rate'])
        records=_score_stable(tf,selected,train+validation,inputs,proposals,guard)
        training=[r for r in records if r['split']=='train']
        validation_records=[r for r in records if r['split']=='validation']
        tm,em=metrics(training),metrics([r for r in training if r['role']!='original'])
        passed=(tm['bce']<.8*metrics(initial)['bce'] and em['sensitivity']>=.75 and em['specificity']>=.75 and
                em['positive_probability_mean']-em['negative_probability_mean']>=.15)
        result={'status':'learning_check_passed' if passed else 'learning_check_failed','classifier_trained':True,
                'test_evaluated':False,'training':tm,'training_edited_only':em,'validation':metrics(validation_records),
                'initial_training':metrics(initial),'predictions':records}
        trace_summary['selected_classifier']=selected.tracing_counts()
    result.update(elapsed_seconds=time.monotonic()-started,tensorflow_version=tf.__version__,
        **memory_snapshot(tf),gpu_limit_mib=c['pilot']['gpu_memory_limit_mb'],
        host_limit_mib=c['training']['host_memory_limit_mb'],training_traces=trace_summary,
        checkpoint_sha256={p.name:exp.sha256(p) for p in model_dir.glob('*.keras')})
    exp.inspect_lock(c,(out/'input.ini').read_bytes())
    exp.write_json(out/'results.json',result);exp.report(c)
    return result
