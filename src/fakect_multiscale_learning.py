"""Bounded, staged multi-case learning check; no held-out test execution."""
from collections import OrderedDict
import json
from pathlib import Path
import resource
import time

import numpy as np

import fakect_multiscale_experiment as exp
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
            ax.add_patch(Rectangle((lo[other[1]], lo[other[0]]), size[other[1]], size[other[0]],
                                   fill=False, edgecolor=color, linewidth=1.5, label=label))
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
    import fakect_multiscale_network as network
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


def fit(c, lock):
    """Fit locator, enforce image-proposal coverage, then attempt classifier pilot."""
    out = Path(c['experiment']['output_directory'])
    prepared = exp.checked_json(out/'preparation.json')
    if prepared['lock_sha256'] != exp.sha256(out/'experiment-lock.json'):
        raise ValueError('Prepared data refer to a different experiment')
    if any(r['split'] not in ('train','validation') for r in lock['samples']):
        raise ValueError('Pilot fit cannot consume test samples')
    prepared_ids = [r['sample_id'] for r in prepared['samples']]
    if (prepared.get('complete') is not True or len(set(prepared_ids)) != len(prepared_ids) or
            set(prepared_ids) != {r['sample_id'] for r in lock['samples']}):
        raise ValueError('Prepared samples must join the pilot exactly')
    for r in prepared['samples']:
        b = np.asarray(r['target_box_kji'])
        if (b.shape != (6,) or not np.isfinite(b).all() or np.any(b[3:] <= 0) or
                np.any(b[:3]-b[3:]/2 < -1e-6) or np.any(b[:3]+b[3:]/2 > 1+1e-6)):
            raise ValueError('Invalid prepared anatomy supervision')
    model_dir = out/'model'
    model_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    tf, network = configure_tensorflow(c)
    settings = c['model']
    targets = {r['sample_id']:np.asarray(r['target_box_kji'], np.float32)[None] for r in prepared['samples']}
    inputs = Inputs(c)
    train = [r for r in lock['samples'] if r['split']=='train']
    validation = [r for r in lock['samples'] if r['split']=='validation']
    locator_train = [r for r in train if r['baseline']]
    locator_validation = [r for r in validation if r['baseline']]
    shape = c['input.overview']['shape_kji']
    locator = network.compile_locator(network.build_locator(shape, settings['base_filters'], settings['seed']),
                                       learning_rate=settings['learning_rate'])
    history, best = [], float('inf')
    rng = np.random.default_rng(settings['seed'])
    for epoch in range(c['pilot']['locator_epochs']):
        loss = []
        for index in rng.permutation(len(locator_train)):
            row = locator_train[int(index)]
            _, x = inputs.coarse(row)
            locator.reset_metrics()
            value = locator.train_on_batch({'overview':x}, targets[row['sample_id']])
            loss.append(float(np.asarray(value).ravel()[0]))
        validation_loss = []
        for row in locator_validation:
            _, x = inputs.coarse(row)
            box = network.predict_boxes(locator, x)
            validation_loss.append(float(tf.reduce_mean(network.locator_loss(targets[row['sample_id']], box)).numpy()))
        record = {'epoch':epoch+1, 'train_loss':float(np.mean(loss)),
                  'validation_loss':float(np.mean(validation_loss))}
        if not np.isfinite(list(record.values())).all():
            raise ValueError('Nonfinite locator training')
        history.append(record)
        if record['validation_loss'] < best:
            best = record['validation_loss']
            locator.save(model_dir/'locator.keras')
        exp.write_json(out/'locator-history.json', {'epochs':history})
        print(json.dumps({'stage':'locator', **record}), flush=True)
    locator = network.load_model(model_dir/'locator.keras')
    proposals, audits = {}, []
    failures = []
    for row in lock['samples']:
        ov, x = inputs.coarse(row)
        predicted = network.predict_boxes(locator, x)[0]
        origins, boxes = exp.proposal_origins(predicted, ov, c)
        volume = ContextVolume(row, verify_hashes=True)
        exp.verify_no_cancellation(volume, row)
        cov = exp.coverage(volume, origins, c)
        ok = ((cov['changed_coverage'] is None or cov['changed_coverage'] >= c['pilot']['minimum_changed_coverage'])
              and (cov['removed_coverage'] is None or cov['removed_coverage'] == 1.0))
        if not ok:
            failures.append(row['sample_id'])
        extent = np.array(ov['image'].shape)*np.array(ov['spacing_ijk_mm'][::-1])
        error_mm = float(np.linalg.norm((predicted[:3]-targets[row['sample_id']][0,:3])*extent))
        record = {'sample_id':row['sample_id'], 'split':row['split'],
                  'predicted_box':predicted.tolist(), 'origins_kji':origins,
                  'proposal_boxes':boxes.tolist(), 'center_error_mm':error_mm, **cov, 'coverage_passed':ok}
        proposals[row['sample_id']] = record
        audits.append(record)
        if row['baseline']:
            _preview(row, ov, targets[row['sample_id']][0], out/'previews'/(row['case_id']+'-locator.png'),
                      predicted=predicted)
        del volume
    audit = {'passed':not failures, 'failed_sample_ids':failures, 'samples':audits,
             'proof':'Supported one-dilation/two-erosion recipe; net counts equal all step counts. Exact coverage of ALL net removed voxels conservatively covers CoA removals without spatial per-step annotations.',
             'inference':'Every box was predicted from attenuation. Ground truth only evaluated coverage.'}
    exp.write_json(out/'proposal-audit.json', audit)
    if failures:
        result = {'status':'stopped_at_locator_coverage_gate', 'classifier_trained':False,
                  'test_evaluated':False, 'failed_samples':failures,
                  'next_step':'Review predicted boxes; improve locator before classifier training.'}
    else:
        classifier = network.compile_classifier(network.build_classifier(shape, c['input.detail']['shape_kji'],
            num_proposals=c['input.detail']['proposals'], base_filters=settings['base_filters'],
            seed=settings['seed'], locator=locator), learning_rate=settings['learning_rate'])
        history, best = [], float('inf')
        initial = _score_classifier(classifier, train, inputs, proposals)
        for epoch in range(c['pilot']['classifier_epochs']):
            losses = []
            for index in rng.permutation(len(train)):
                row = train[int(index)]
                x = inputs.classifier(row, proposals[row['sample_id']])
                y = {'coa_logit':np.array([[int(row['annotation']['coa_benchmark_positive'])]], np.float32),
                     'local_volume_loss':np.array([[row['annotation'][exp.SCORE]]], np.float32)}
                classifier.reset_metrics()
                value = classifier.train_on_batch(x, y, return_dict=True)
                losses.append(float(value['loss']))
                if not np.isfinite(losses[-1]):
                    raise ValueError('Nonfinite classifier training')
            valid_records = _score_classifier(classifier, validation, inputs, proposals)
            record = {'epoch':epoch+1, 'train_loss':float(np.mean(losses)), 'validation':metrics(valid_records)}
            if record['validation']['bce'] < best:
                best = record['validation']['bce']
                classifier.save(model_dir/'classifier.keras')
            history.append(record)
            exp.write_json(out/'classifier-history.json', {'initial_training':metrics(initial), 'epochs':history})
            print(json.dumps({'stage':'classifier', **record}), flush=True)
        classifier = network.load_model(model_dir/'classifier.keras')
        records = _score_classifier(classifier, train+validation, inputs, proposals)
        training = [r for r in records if r['split']=='train']
        validation_results = [r for r in records if r['split']=='validation']
        edited_train = [r for r in training if r['role'] != 'original']
        tm, em = metrics(training), metrics(edited_train)
        passed = (tm['bce'] < .8*metrics(initial)['bce'] and em['sensitivity'] >= .75
                  and em['specificity'] >= .75 and
                  em['positive_probability_mean']-em['negative_probability_mean'] >= .15)
        result = {'status':'learning_check_passed' if passed else 'learning_check_failed',
                  'classifier_trained':True, 'test_evaluated':False,
                  'training':tm, 'training_edited_only':em, 'validation':metrics(validation_results),
                  'initial_training':metrics(initial), 'predictions':records,
                  'learning_gate':'Training BCE >=20% better than initialization; edited-only train sensitivity/specificity >=.75 and mean positive-negative probability gap >=.15. Engineering gate, not generalization claim.'}
    gpu_peak = None
    if tf.config.list_logical_devices('GPU'):
        gpu_peak = tf.config.experimental.get_memory_info('GPU:0')['peak']/1024**2
    result.update(elapsed_seconds=time.monotonic()-started, tensorflow_version=tf.__version__,
                  host_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  gpu_peak_mib=gpu_peak, gpu_limit_mib=c['pilot']['gpu_memory_limit_mb'],
                  checkpoint_sha256={p.name:exp.sha256(p) for p in model_dir.glob('*.keras')})
    exp.inspect_lock(c, (out/'input.ini').read_bytes())
    exp.write_json(out/'results.json', result)
    exp.report(c)
    return result
