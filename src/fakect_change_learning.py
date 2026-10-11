"""Train CT-only difference maps and evaluate complete frozen focus grids."""
from pathlib import Path
import gc
import resource
import time

import numpy as np

import fakect_change_experiment as exp
import fakect_change_network as network
from fakect_change_data import FocusedChangeVolume
from fakect_multiscale_experiment import write_json, sha256


def _tensorflow():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise ValueError('Production localization requires a TensorFlow GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf


def _tensors(tf, block):
    return (tf.convert_to_tensor(block['x'][None], tf.float32),
            tf.convert_to_tensor(block['labels'][None], tf.int32),
            tf.convert_to_tensor(block['valid'][None], tf.float32))


def _steps(c, model):
    t = c['training']
    return network.make_steps(model, learning_rate=t['learning_rate'],
                              change_weight=t['change_weight'], dice_weight=t['dice_weight'])


def smoke(c, lock):
    tf = _tensorflow()
    t, stage = c['training'], lock['stages'][0]
    rows = {r['sample_id']: r for r in lock['samples']}
    row = next(rows[sid] for sid in stage['train_sample_ids'] if not rows[sid]['baseline'])
    proposal = lock['proposals'][stage['id']][row['sample_id']]
    model = network.build_model(tuple(t['patch_shape_kji']), t['base_filters'], t['seed'])
    steps = _steps(c, model)
    with FocusedChangeVolume(row, lock['runtime'], proposal['origins_kji']) as volume:
        rng = np.random.default_rng(t['seed'])
        values, target_counts = None, {'added':0,'removed':0}
        for mode in ('added','removed','uniform'):
            tile = volume.training_patch(t['patch_shape_kji'],rng,mode=mode)
            block = volume.read_patch(**tile)
            target_counts['added'] += int((block['labels']==1).sum())
            target_counts['removed'] += int((block['labels']==2).sum())
            x, y, v = _tensors(tf, block)
            values = steps.train(x, y, v)
            steps.predict(x)
        before = np.asarray(steps.predict(x))
        out = Path(c['experiment']['output_directory'])
        model.save(out/'smoke/model.keras')
        restored = network.load_model(out/'smoke/model.keras')
        parity = bool(np.allclose(before, np.asarray(restored(x, training=False)), rtol=1e-6, atol=1e-6))
    metrics = {k: float(np.asarray(v)) for k, v in values.items()}
    peak = tf.config.experimental.get_memory_info('GPU:0')['peak']/1024**2
    result = {'passed': parity and metrics['gradient_global_norm'] > 0 and np.isfinite(list(metrics.values())).all()
                       and all(n>0 for n in target_counts.values()),
              'gpu_visible': True, 'reload_parity': parity,
              'train_traces': steps.train.experimental_get_tracing_count(),
              'predict_traces': steps.predict.experimental_get_tracing_count(),
              'metrics': metrics, 'gpu_peak_mib': peak,
              'host_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'source_lock_sha256': sha256(out/'experiment-lock.json'),
              'tensorflow_version': tf.__version__, 'sample_id': row['sample_id'],
              'patch_shape_kji': t['patch_shape_kji'], 'baseline_is_input': False,
              'supervised_change_voxels':target_counts}
    if result['gpu_peak_mib'] > 12288 or result['host_peak_rss_mib'] > 16384:
        result['passed'] = False
    result['passed'] = bool(result['passed'])
    write_json(out/'runtime-smoke.json', result)
    return result


def _sample_focus_masks(reader, index):
    full = reader.whole_focus(index)
    origin = reader.origins_kji[index]
    scale = reader.spacing_ijk_mm[::-1]/reader.volume._native_spacing_kji
    coords = [(int(o)+np.arange(int(n),dtype=np.float64)+.5)*s-.5
              for o,n,s in zip(origin,reader.shape_kji,scale)]
    nearest = [np.floor(c+.5).astype(np.int64) for c in coords]
    after = reader.volume._nearest_mask(nearest)*full['valid']
    before = ((after.astype(bool) & (full['labels'] != 1)) | (full['labels'] == 2)).astype(np.uint8)
    return full['labels'], full['valid'], before, after


def evaluate_sample(tf, steps, row, proposal, runtime, c, output):
    """Average all overlapping tiles/proposals, then score each voxel once."""
    from fakect_change_report import change_metrics
    t = c['training']
    with FocusedChangeVolume(row, runtime, proposal['origins_kji']) as reader:
        origin = reader.origins_kji.min(axis=0)
        shape = (reader.origins_kji + reader.shape_kji).max(axis=0)-origin
        if int(np.prod(shape)) > 24_000_000:
            raise ValueError('Common focus grid exceeds the explicit 24-million-voxel budget')
        dims = tuple(map(int, shape))
        sums = np.zeros(dims+(3,), np.float32)
        counts = np.zeros(dims, np.uint16)
        labels, valid, before, after = [np.zeros(dims, np.uint8) for _ in range(4)]
        seen = np.zeros(dims, bool)
        for index in range(len(reader.origins_kji)):
            l,v,b,a = _sample_focus_masks(reader,index)
            lo = reader.origins_kji[index]-origin
            sl = tuple(slice(int(o),int(o+n)) for o,n in zip(lo,reader.shape_kji))
            if np.any(seen[sl] & ((labels[sl]!=l) | (before[sl]!=b) | (after[sl]!=a))):
                raise ValueError('Overlapping focus targets disagree on their common physical grid')
            labels[sl], valid[sl], before[sl], after[sl] = l,v,b,a
            seen[sl] = True
            del l,v,b,a
        for tile in reader.inference_tiles(t['patch_shape_kji'], [t['inference_overlap']]*3):
            block = reader.read_patch(**tile)
            x = tf.convert_to_tensor(block['x'][None], tf.float32)
            p = np.asarray(tf.nn.softmax(steps.predict(x),axis=-1))[0]
            lo = reader.origins_kji[tile['proposal_index']]+tile['offset_kji']-origin
            sl = tuple(slice(int(o),int(o+n)) for o,n in zip(lo,tile['shape_kji']))
            sums[sl] += p
            counts[sl] += 1
        if np.any(seen & (counts==0)):
            raise ValueError('Inference left focus voxels uncovered')
        sums /= np.maximum(counts,1)[...,None]
        predicted = np.argmax(sums,axis=-1).astype(np.uint8)
        predicted *= valid
        metrics = change_metrics(labels,predicted,valid,reader.spacing_ijk_mm)
        fp = int(((predicted!=0)&(valid!=0)).sum()) if row['baseline'] else None
        reference_count = int(before.sum())
        record = {'sample_id':row['sample_id'], 'case_id':row['case_id'],
                  'anatomy_family':row['anatomy_family'], 'baseline':row['baseline'],
                  'spacing_ijk_mm':reader.spacing_ijk_mm.tolist(), 'origin_kji':origin.tolist(),
                  'focus_origins_kji':reader.origins_kji.tolist(), 'shape_kji':list(dims),
                  'metrics':metrics, 'original_aorta_voxels':reference_count,
                  'control_change_fraction':fp/max(reference_count,1) if fp is not None else None,
                  'locator_coverage_passed':bool(proposal['coverage_passed']),
                  'locator_added_coverage':proposal['added_coverage'] if not row['baseline'] else None,
                  'locator_removed_coverage':proposal['removed_coverage'] if not row['baseline'] else None,
                  'baseline_is_input':False, 'prediction_source':'final epoch CT-only model',
                  'reader_stats':reader.stats()}
        name = row['sample_id'].replace(':','__')+'.npz'
        path = output/name
        with path.open('xb') as stream:
            np.savez_compressed(stream,labels=labels,predicted=predicted,valid=valid,
                before=before,after=after,added_probability=sums[...,1].astype(np.float16),
                removed_probability=sums[...,2].astype(np.float16),
                spacing_ijk_mm=reader.spacing_ijk_mm,origin_kji=origin,sample_id=np.asarray(row['sample_id']))
        record.update(prediction_npz='predictions/'+name,prediction_sha256=sha256(path))
        return record


def validation_gate(c, records):
    """Predeclared engineering checks; a passing gate does not start Model 2."""
    v = c['validation']
    checks, details = {}, {}
    for key in ('added','removed'):
        cases = {}
        for row in records:
            m = row['metrics']['classes'][key]
            if not m['target_empty']:
                cases.setdefault(row['case_id'],[]).append(m)
        scores = {}
        for metric in ('dice','precision','recall'):
            case_values = [np.mean([r[metric] if r[metric] is not None else 0. for r in rows])
                           for rows in cases.values()]
            scores[metric] = float(np.mean(case_values)) if case_values else None
            checks[key+'_'+metric] = scores[metric] is not None and scores[metric] >= v['minimum_'+metric]
        details[key] = scores
    controls = [r for r in records if r['baseline']]
    checks['unchanged_controls'] = bool(controls) and all(
        r['original_aorta_voxels'] > 0 and r['control_change_fraction'] <= v['maximum_control_change_fraction']
        for r in controls)
    edited = [r for r in records if not r['baseline']]
    checks['locator_coverage'] = bool(edited) and all(r['locator_coverage_passed'] for r in edited)
    return {'passed':bool(all(checks.values())), 'checks':checks, 'case_macro_metrics':details,
            'thresholds':v, 'classifier_started':False,
            'scope':'Engineering localization gate on development anatomy; no clinical trust claim'}


def run_stage(c, lock, stage):
    tf = _tensorflow()
    t, runtime = c['training'], lock['runtime']
    out = Path(c['experiment']['output_directory'])/'stages'/stage['id']
    out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').touch()
    rows = {r['sample_id']:r for r in lock['samples']}
    train = [rows[sid] for sid in stage['train_sample_ids']]
    evaluate = [rows[sid] for sid in stage['final_test_sample_ids']]
    proposals = lock['proposals'][stage['id']]
    write_json(out/'stage-input.json', {'stage':stage,'config':c,
        'source_lock_sha256':sha256(Path(c['experiment']['output_directory'])/'experiment-lock.json'),
        'classifier_started':False})
    model = network.build_model(tuple(t['patch_shape_kji']), t['base_filters'], t['seed'])
    steps = _steps(c,model)
    rng = np.random.default_rng(t['seed'])
    history, verified = [], {}
    for epoch in range(1,t['epochs']+1):
        started, values = time.monotonic(), []
        for index in rng.permutation(len(train)):
            row = train[int(index)]
            with FocusedChangeVolume(row,runtime,proposals[row['sample_id']]['origins_kji'],
                                     verify_hashes=row['sample_id'] not in verified) as reader:
                stats = (reader.volume._variant_stat,reader.volume._mask_stat)
                if row['sample_id'] in verified and verified[row['sample_id']] != stats:
                    raise ValueError('Pinned training payload changed between epochs')
                verified[row['sample_id']] = stats
                for patch in range(t['patches_per_sample']):
                    mode = ('added' if patch%2==0 else 'removed') if rng.random()<t['changed_fraction'] else 'uniform'
                    tile = reader.training_patch(t['patch_shape_kji'],rng,mode=mode)
                    block = reader.read_patch(**tile)
                    x,y,v = _tensors(tf,block)
                    metrics = steps.train(x,y,v)
                    values.append({k:float(np.asarray(value)) for k,value in metrics.items()})
            del reader,block,x,y,v,metrics
        steps.assert_stable_traces()
        record = {'epoch':epoch,'loss':float(np.mean([r['loss'] for r in values])),
                  'ce':float(np.mean([r['cross_entropy'] for r in values])),
                  'dice_loss':float(np.mean([r['foreground_dice_loss'] for r in values])),
                  'updates':int(steps.optimizer.iterations.numpy()),'seconds':time.monotonic()-started}
        history.append(record)
        write_json(out/'history.json',{'epochs':history,'held_out_scored_during_training':False})
        print(f"{stage['outer_fold_id']} epoch {epoch}/{t['epochs']} loss={record['loss']:.5f} updates={record['updates']}",flush=True)
    model.save(out/'change-model.keras')
    # Reload parity on the last training patch; no held-out checkpoint selection.
    sample = train[0]
    with FocusedChangeVolume(sample,runtime,proposals[sample['sample_id']]['origins_kji'],verify_hashes=True) as reader:
        block = reader.read_patch(0,[0,0,0],t['patch_shape_kji'])
        x = tf.convert_to_tensor(block['x'][None],tf.float32)
        expected = np.asarray(steps.predict(x))
        restored = network.load_model(out/'change-model.keras')
        if not np.allclose(expected,np.asarray(restored(x,training=False)),rtol=1e-6,atol=1e-6):
            raise ValueError('Final checkpoint reload parity failed')
    del restored,expected,reader,block,x
    predictions = out/'predictions'
    predictions.mkdir()
    records = [evaluate_sample(tf,steps,row,proposals[row['sample_id']],runtime,c,predictions) for row in evaluate]
    gate = validation_gate(c,records)
    result = {'status':'completed','stage_id':stage['id'],'config':c,'records':records,
              'gate':gate,'epochs_completed':t['epochs'],'optimizer_updates':int(steps.optimizer.iterations.numpy()),
              'training_samples':len(train),'evaluation_samples':len(evaluate),
              'training_case_ids':stage['train_case_ids'],'held_out_case_ids':stage['final_test_case_ids'],
              'reserved_test_evaluated':False,'classifier_started':False,
              'source_lock_sha256':sha256(Path(c['experiment']['output_directory'])/'experiment-lock.json'),
              'checkpoint':str(out/'change-model.keras'),'checkpoint_sha256':sha256(out/'change-model.keras'),
              'checkpoint_selection':'declared final epoch only',
              'scope':'CT-only baseline-relative net change localization; development family holdout',
              'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'gpu_peak_mib':tf.config.experimental.get_memory_info('GPU:0')['peak']/1024**2}
    write_json(out/'results.json',result)
    (out/'INCOMPLETE').unlink()
    gc.collect()
    return result
