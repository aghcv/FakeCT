#!/usr/bin/env python3
"""GPU smoke test and measured runtime budget for the matched comparison.

Tiny synthetic files exercise real training/save/reload/cached scoring. Timing
uses existing training/validation splits only; no test payload is opened. The
diagnostic checkpoint is used solely for throughput, never as comparison fit.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
import fakect_context_cached_evaluation as cached
import fakect_context_comparison_learning as comparison
import fakect_context_learning as learning
from fakect_context_volume import ContextVolume


def stats(values):
    return dict(count=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
        p90=float(np.percentile(values, 90)), minimum=float(np.min(values)), maximum=float(np.max(values)))


def tiny_fixture(directory):
    directory.mkdir()
    shape, spacing = (12, 13, 14), [1., 1., 1.]
    rows = []
    for case, split in [('small-train', 'train'), ('small-validation', 'validation')]:
        source = directory/f'{case}.bin'
        before = np.zeros(shape, np.uint8)
        before[3:9, 4:9, 4:10] = 1
        native = np.full(shape, .01, np.float32)+before*.2
        (native/10).astype('<f4').tofile(source)
        native = np.fromfile(source, '<f4').reshape(shape)*np.float32(10.)
        stamp = source.stat()
        for baseline in (True, False):
            name = 'baseline' if baseline else 'edited'
            sid = case+':'+name
            after, image = before.copy(), native.copy()
            if not baseline:
                after[8:10, 5:7, 5:8] = 1
                image[8:10, 5:7, 5:8] = .21
            variant, masks = directory/f'{sid}-image.npz', directory/f'{sid}-masks.npz'
            np.savez_compressed(variant, image=image)
            np.savez_compressed(masks, before=before, after=after,
                added=after & ~before, removed=before & ~after,
                crop_origin_ijk=np.zeros(3, int), spacing_ijk_mm=spacing, sample_id=np.array(sid))
            rows.append(dict(sample_id=sid, case_id=case, anatomy_family=case, split=split,
                variant_id=name, baseline=baseline, source_atn=dict(path=str(source),
                bytes=stamp.st_size, mtime_ns=stamp.st_mtime_ns), source_shape_kji=list(shape),
                spacing_ijk_mm=spacing, variant_npz=str(variant),
                variant_npz_sha256=learning.sha256(variant), edit_crop_origin_ijk=[0, 0, 0],
                edit_crop_shape_kji=list(shape), mask_npz=str(masks), mask_npz_sha256=learning.sha256(masks)))
    return dict(samples=rows)


def tiny_smoke(directory):
    lock = tiny_fixture(directory/'tiny-data')
    config = dict(model=dict(patch_size_kji=[8, 8, 8], target_spacing_mm=[1, 1, 1],
        depth=1, base_filters=2, epochs=2, seed=77, augmentation_flip=False,
        intensity_scale_range=[1, 1]), comparison=dict(updates_per_epoch=4,
        validation_every_epochs=1, volume_cache_size=2))
    training, evaluation, history = {}, {}, {}
    for arm in comparison.ARMS:
        training[arm] = comparison.fit_arm(config, lock, arm, directory/arm/'model')
        evaluation[arm] = comparison.evaluate_arm(config, lock, arm,
            directory/arm/'model', directory/arm/'evaluation', split='validation')
        history[arm] = json.loads((directory/arm/'model'/'history.json').read_text())
    one, two = comparison.ARMS
    checks = dict(equal_initial_weights=training[one]['initial_weights_sha256'] == training[two]['initial_weights_sha256'],
        equal_optimizer_updates=training[one]['optimizer_updates'] == training[two]['optimizer_updates'],
        equal_spatial_events=[r['physical_patch_events_sha256'] for r in history[one]] ==
                             [r['physical_patch_events_sha256'] for r in history[two]],
        baseline_isolation=not training[one]['training_sibling_payloads_loaded'],
        both_reloaded_and_evaluated=all(evaluation[arm]['complete'] for arm in comparison.ARMS))
    for arm in comparison.ARMS:
        model, _ = comparison.load_arm(config, lock, arm, directory/arm/'model')
        for row, record in zip([r for r in lock['samples'] if r['split'] == 'validation'], evaluation[arm]['records']):
            with ContextVolume(row) as volume:
                direct = learning.score_volume(model, volume, learning.settings_for(config))
            if any(direct[key] != record[key] for key in direct):
                raise RuntimeError('Tiny actual TensorFlow cached/direct score mismatch')
    checks['actual_tensorflow_cached_direct_exact'] = True
    if not all(checks.values()):
        raise RuntimeError(f'Tiny comparison smoke failed: {checks}')
    return dict(checks=checks, optimizer_updates=training[one]['optimizer_updates'],
        initial_weights_sha256=training[one]['initial_weights_sha256'],
        validation_dice={arm:evaluation[arm]['summary']['family_macro']['dice'] for arm in comparison.ARMS},
        scope='Tiny synthetic functional check; not segmentation accuracy')


def stratified_origins(volume, settings, count=16):
    """Eight head-to-foot levels, center-body and padded lateral-edge blocks."""
    shape, patch = volume.model_shape(settings['target_spacing_mm']), settings['patch_size_kji']
    axes = [list(range(0, n, p)) for n, p in zip(shape, patch)]
    selected = []
    for index in np.linspace(0, len(axes[0])-1, count//2).round().astype(int):
        for edge in (False, True):
            origin = (axes[0][index], axes[1][len(axes[1])//2],
                      axes[2][-1] if edge else axes[2][len(axes[2])//2])
            if origin not in selected:
                selected.append(origin)
    return selected


def time_tile(inference, volume, origin, settings):
    start = time.monotonic()
    arrays = learning._read(volume, origin, settings)
    loaded = time.monotonic()
    x, y, valid = learning.prepare_block(*arrays, settings)
    prepared = time.monotonic()
    probability, logits = learning._predict_pair(inference, x)
    predicted = time.monotonic()
    counts = learning.counts_for(y[0, ..., 0], probability >= settings['threshold'], valid[0, ..., 0])
    totals = learning.loss_totals(y[0, ..., 0], probability, valid[0, ..., 0], logits=logits)
    metrics = learning.metrics_from_counts(counts, math.prod(settings['target_spacing_mm']))
    loss = learning.loss_from_totals(totals)
    finished = time.monotonic()
    return dict(origin_kji=list(origin), loader_seconds=loaded-start,
        preprocessing_seconds=prepared-loaded, forward_seconds=predicted-prepared,
        metrics_seconds=finished-predicted, total_seconds=finished-start,
        valid_fraction=float(np.mean(arrays[2])), truth_voxels=metrics['true_voxels'], loss=loss)


def equality_checks(model, baseline_row, sibling_row, settings):
    records = []
    inference = learning._inference_model(model)
    with ContextVolume(baseline_row) as baseline, ContextVolume(sibling_row) as sibling:
        proof, support = cached.prove_difference(baseline, sibling)
        origins = list(learning.tile_origins(baseline.model_shape(settings['target_spacing_mm']), settings['patch_size_kji']))
        affected = [o for o in origins if cached._affected(o, settings['patch_size_kji'],
            settings['target_spacing_mm'], baseline.spacing_ijk_mm, support)]
        untouched = [o for o in origins if not cached._affected(o, settings['patch_size_kji'],
            settings['target_spacing_mm'], baseline.spacing_ijk_mm, support)]
        for changed, selected in [(False, untouched[:2]), (True, affected[:2])]:
            for origin in selected:
                first, second = learning._read(baseline, origin, settings), learning._read(sibling, origin, settings)
                equal = [bool(np.array_equal(a, b)) for a, b in zip(first, second)]
                if not changed and not all(equal):
                    raise RuntimeError('Proof marked a differing real tile reusable')
                x1, _, _ = learning.prepare_block(*first, settings)
                x2, _, _ = learning.prepare_block(*second, settings)
                p1, l1 = learning._predict_pair(inference, x1)
                p2, l2 = learning._predict_pair(inference, x2)
                prediction_equal = bool(np.array_equal(p1, p2) and np.array_equal(l1, l2))
                if not changed and not prediction_equal:
                    raise RuntimeError('Identical real tile inference differed')
                records.append(dict(origin_kji=list(origin), conservatively_affected=changed,
                    image_equal=equal[0], mask_equal=equal[1], validity_equal=equal[2],
                    logits_and_probability_equal=prediction_equal))
    return dict(baseline_sample_id=baseline_row['sample_id'], sibling_sample_id=sibling_row['sample_id'],
        proof=proof, affected_tiles=len(affected), reused_tiles=len(untouched), checks=records)


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').write_text('Benchmark incomplete.\n')
    tf = learning._tensorflow()
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Benchmark requires a GPU')
    started = time.monotonic()
    tiny = tiny_smoke(output)
    learning._signed_write(output/'tiny-smoke.json', tiny)
    print(json.dumps(dict(phase='tiny_smoke_complete', **tiny)), flush=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    lock_path = Path(args.lock).resolve()
    lock = json.loads(lock_path.read_text())
    settings = learning.settings_for({'model':lock['model_settings']})
    if settings['patch_size_kji'] != [216, 160, 160] or settings['target_spacing_mm'] != [.6, .6, .6]:
        raise ValueError('Expected successful 216x160x160 @0.6mm diagnostic architecture')
    checkpoint = Path(args.checkpoint).resolve()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    inference = learning._inference_model(model)
    rows = learning.inspect_rows(lock)
    validation = [r for r in rows if r['split'] == 'validation' and r['baseline']]
    timing, initialization = [], []
    for row in validation:
        begin = time.monotonic()
        with ContextVolume(row) as volume:
            initialization.append(dict(sample_id=row['sample_id'], load_seconds=time.monotonic()-begin))
            origins = stratified_origins(volume, settings)
            if not timing:
                warmup = [time_tile(inference, volume, origin, settings) for origin in origins[:2]]
                learning._write(output/'inference-warmup.json', warmup)
            for origin in origins:
                record = dict(sample_id=row['sample_id'], **time_tile(inference, volume, origin, settings))
                timing.append(record)
                print(json.dumps(dict(phase='tile_timing', **record)), flush=True)
    if len(timing) < 24:
        raise ValueError('Need at least 24 timed stratified tiles')
    sibling = next(r for r in rows if r['split'] == 'validation' and not r['baseline'] and
                   r['case_id'] == validation[0]['case_id'] and r.get('mask_counts', {}).get('added', 0) > 0)
    exact = equality_checks(model, validation[0], sibling, settings)
    learning._signed_write(output/'tile-timing.json', dict(records=timing, equality=exact))
    # Optimizer throughput uses training data only and does not save weights.
    train_row = next(r for r in rows if r['split'] == 'train' and r['baseline'])
    step = learning._training_step(model, settings)
    updates = []
    with ContextVolume(train_row) as volume:
        points = volume.foreground_points(settings['target_spacing_mm'], max_points=100_000, seed=42)
        origins = learning.sample_origins(volume.model_shape(settings['target_spacing_mm']),
            settings['patch_size_kji'], points, 10, np.random.default_rng(42), .5)
        for index, origin in enumerate(origins):
            begin = time.monotonic()
            arrays = learning._read(volume, origin, settings)
            loaded = time.monotonic()
            prepared = learning.prepare_block(*arrays, settings, rng=np.random.default_rng(100+index))
            prep_end = time.monotonic()
            loss = float(step(*prepared))  # Conversion synchronizes GPU work.
            finished = time.monotonic()
            record = dict(sample_id=train_row['sample_id'], origin_kji=list(origin),
                warmup=index < 2, loader_seconds=loaded-begin, preprocessing_seconds=prep_end-loaded,
                optimizer_seconds=finished-prep_end, total_seconds=finished-begin, loss=loss)
            if not math.isfinite(loss):
                raise ValueError('Real timing optimizer produced nonfinite loss')
            updates.append(record)
            print(json.dumps(dict(phase='optimizer_timing', **record)), flush=True)
    fields = ['loader_seconds', 'preprocessing_seconds', 'forward_seconds', 'metrics_seconds', 'total_seconds']
    summary = {key:stats([r[key] for r in timing]) for key in fields}
    training_summary = {key:stats([r[key] for r in updates if not r['warmup']])
        for key in ('loader_seconds', 'preprocessing_seconds', 'optimizer_seconds', 'total_seconds')}
    # Full FOV counts are preserved by exact reuse. Include a timing margin;
    # file initialization/proof overhead is measured separately by the plan.
    mean_fit = training_summary['total_seconds']['mean']*512*20
    mean_validation = summary['total_seconds']['mean']*2036*4
    result = dict(schema_version='fakect.context-comparison-benchmark/1', complete=True,
        scope='GPU functional/throughput diagnostic; no generalization result',
        tensorflow_version=tf.__version__, devices=[d.name for d in tf.config.list_logical_devices()],
        gpu_allocator_memory=learning.gpu_allocator_memory(tf),
        lock_path=str(lock_path), lock_sha256=learning.sha256(lock_path),
        checkpoint_path=str(checkpoint), checkpoint_sha256=learning.sha256(checkpoint),
        script_sha256=learning.sha256(__file__), component_sha256=comparison._component_hashes(),
        tiny_smoke=tiny, settings=settings, initialization=initialization,
        inference_timing=summary, optimizer_timing=training_summary, optimizer_records=updates,
        real_equality_check=exact, test_payloads_loaded=False,
        real_validation_sample_ids=[r['sample_id'] for r in validation]+[sibling['sample_id']],
        real_optimizer_sample_id=train_row['sample_id'],
        projected_seconds_per_arm=dict(training_10240_updates=mean_fit,
            validation_8144_forwards=mean_validation,
            train_plus_validation=mean_fit+mean_validation,
            train_plus_validation_with_25_percent_margin=(mean_fit+mean_validation)*1.25),
        projection_limitations='Excludes test scoring, per-variant initialization/proof and checkpoint I/O; GPU type and filesystem load vary. 2036 validation forwards from the separate exact loader plan.',
        elapsed_seconds=time.monotonic()-started)
    learning._signed_write(output/'benchmark.json', result)
    (output/'INCOMPLETE').unlink()
    print(json.dumps(dict(phase='benchmark_complete', summary=result)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/qa/context-comparison-benchmark-r1')
    parser.add_argument('--lock', default='outputs/experiments/coa/context-grid-diagnostic-r2/experiment-lock.json')
    parser.add_argument('--checkpoint', default='outputs/experiments/coa/context-grid-diagnostic-r2/model/checkpoint.keras')
    run(parser.parse_args())
