"""Matched baseline-versus-sibling experiments, without changing pinned models.

Both arms use identical initialized weights, case order, physical patch origins,
augmentation seeds, and update budgets. Only the chosen training anatomy differs.
Baseline masks determine sampling for both arms, never a sibling lesion mask.
Validation chooses checkpoints; test payloads are opened only by evaluate_arm.
"""
from collections import Counter, OrderedDict
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

import fakect_context_learning as learning


ARMS = ('baseline_only', 'baseline_plus_siblings')
MODEL_SCHEMA = 'fakect.context-comparison-model/1'
RESULT_SCHEMA = 'fakect.context-comparison-results/1'
DEFAULTS = dict(updates_per_epoch=512, validation_every_epochs=5, volume_cache_size=2)
CODE_FILES = ('fakect_context_comparison_learning.py', 'fakect_context_learning.py',
              'fakect_context_volume.py', 'fakect_context_cached_evaluation.py')


def settings_for(config):
    section = dict(config['comparison']) if 'comparison' in config else {}
    unknown = set(section)-set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown comparison settings: {sorted(unknown)}')
    values = {**DEFAULTS, **section}
    return {key: learning._int(value, key, 1, 2 if key == 'volume_cache_size' else None)
            for key, value in values.items()}


def _arm(arm):
    if arm not in ARMS:
        raise ValueError(f'arm must be one of {ARMS}')
    return arm


def training_cases(lock, settings):
    """Validate metadata without touching image or mask payloads."""
    rows = learning.inspect_rows(lock)
    cases = {}
    for row in rows:
        if row['split'] == 'train':
            cases.setdefault(row['case_id'], []).append(row)
    if not cases or not any(row['split'] == 'validation' for row in rows):
        raise ValueError('Comparison needs training and validation partitions')
    if settings['updates_per_epoch'] % len(cases):
        raise ValueError('updates_per_epoch must be divisible by the training case count')
    slots = settings['updates_per_epoch']//len(cases)
    result = {}
    for case_id, members in sorted(cases.items()):
        members = sorted(members, key=lambda row: row['sample_id'])
        baselines = [row for row in members if row.get('baseline') is True]
        if len(baselines) != 1:
            raise ValueError(f'{case_id} requires exactly one baseline')
        if slots % len(members):
            raise ValueError('Per-case updates must be divisible by its complete variant count')
        baseline = baselines[0]
        for row in members:
            for key in ('source_shape_kji', 'spacing_ijk_mm', 'source_atn', 'anatomy_family'):
                if row.get(key) != baseline.get(key):
                    raise ValueError(f'{case_id}: siblings have incompatible {key}')
        result[case_id] = dict(baseline=baseline, variants=members, slots=slots)
    return result


class VolumeCache:
    """Small LRU of verified volumes; closes every volume on eviction or exit."""
    def __init__(self, maximum=2, loaded=None):
        self.maximum = learning._int(maximum, 'volume_cache_size', 1, 2)
        self.loaded = loaded if loaded is not None else set()
        self.volumes = OrderedDict()

    def get(self, row):
        key = row['sample_id']
        if key in self.volumes:
            self.volumes.move_to_end(key)
            return self.volumes[key]
        # Evict first: never temporarily allocate maximum+1 native masks.
        if len(self.volumes) >= self.maximum:
            _, volume = self.volumes.popitem(last=False)
            learning._close(volume)
        volume = learning._volume(row)
        self.loaded.add(key)
        self.volumes[key] = volume
        return volume

    def close(self):
        while self.volumes:
            _, volume = self.volumes.popitem(last=False)
            learning._close(volume)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def epoch_events(cases, model_settings, epoch, cache):
    """Yield matching spatial events for either arm, one case at a time.

Within a case, a seeded variant permutation is repeated in contiguous blocks.
Thus all variants get identical update counts with one load per variant, while
physical origins and augmentation draws remain independent of variant choice.
    """
    seed = model_settings['seed']
    case_ids = sorted(cases)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 101])).permutation(len(case_ids))
    for case_index in order:
        case_index = int(case_index)
        case = cases[case_ids[case_index]]
        baseline = case['baseline']
        volume = cache.get(baseline)
        spacing = model_settings['target_spacing_mm']
        points = volume.foreground_points(spacing, max_points=100_000,
            seed=int(np.random.SeedSequence([seed, epoch, case_index, 102]).generate_state(1)[0]))
        origins = learning.sample_origins(volume.model_shape(spacing),
            model_settings['patch_size_kji'], points, case['slots'],
            np.random.default_rng(np.random.SeedSequence([seed, epoch, case_index, 103])),
            model_settings['foreground_fraction'])
        del volume  # Do not keep an evicted baseline's native arrays alive.
        permutation = np.random.default_rng(np.random.SeedSequence([seed, epoch, case_index, 104])).permutation(len(case['variants']))
        indices = np.repeat(permutation, case['slots']//len(case['variants']))
        for slot, (origin, index) in enumerate(zip(origins, indices)):
            yield dict(case_id=baseline['case_id'], slot=slot, origin_kji=list(origin),
                augmentation_seed=[seed, epoch, case_index, slot, 105],
                baseline=baseline, sibling=case['variants'][int(index)])


def _digest_event(digest, document):
    digest.update((json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode())


def weights_digest(model):
    """Digest values and shape/dtype in canonical weight order, not layer names."""
    digest = hashlib.sha256()
    for index, variable in enumerate(model.weights):
        value = np.ascontiguousarray(np.asarray(variable))
        _digest_event(digest, dict(index=index, shape=list(value.shape), dtype=value.dtype.str))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _component_hashes():
    return {name: learning.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}


def _score(model, rows, settings, loaded, phase):
    from fakect_context_cached_evaluation import score_rows
    def progress(event):
        print(json.dumps(dict(phase=phase, **event), allow_nan=False), flush=True)
    return score_rows(model, rows, settings, loaded=loaded, progress_callback=progress)


def fit_arm(config, lock, arm, output_directory):
    """Fit one fresh matched arm; no resume or in-place replacement."""
    _arm(arm)
    settings, comparison = learning.settings_for(config), settings_for(config)
    if settings['max_train_batches_per_epoch']:
        raise ValueError('Comparison updates are controlled only by updates_per_epoch')
    cases = training_cases(lock, comparison)
    rows = learning.inspect_rows(lock)
    validation = [row for row in rows if row['split'] == 'validation']
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'INCOMPLETE').write_text('Comparison fit is incomplete; use a fresh revision after failure.\n')
    component_hashes = _component_hashes()
    tf = learning._tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(settings['seed'])
    tf.config.experimental.enable_op_determinism()
    model = learning.build_model(settings)
    initial_digest = weights_digest(model)
    learning._signed_write(directory/'initialization.json', dict(arm=arm,
        settings=settings, comparison=comparison, lock_fingerprint=learning._fingerprint(lock),
        component_sha256=component_hashes, initial_weights_sha256=initial_digest))
    step = learning._training_step(model, settings)
    history, loaded = [], set()
    best_value, best_epoch = -math.inf, None
    start = time.monotonic()
    for epoch in range(settings['epochs']):
        epoch_start = time.monotonic()
        print(json.dumps(dict(phase='epoch_start', arm=arm, epoch=epoch+1,
            updates=comparison['updates_per_epoch'])), flush=True)
        losses, selections = [], Counter()
        physical_digest, selection_digest = hashlib.sha256(), hashlib.sha256()
        # A new bounded cache each epoch; validation cannot retain native arrays.
        with VolumeCache(comparison['volume_cache_size'], loaded) as cache:
            for event in epoch_events(cases, settings, epoch, cache):
                row = event['baseline'] if arm == 'baseline_only' else event['sibling']
                raw = learning._read(cache.get(row), event['origin_kji'], settings)
                rng = np.random.default_rng(np.random.SeedSequence(event['augmentation_seed']))
                value = float(step(*learning.prepare_block(*raw, settings, rng=rng)))
                if not math.isfinite(value):
                    raise ValueError('Comparison training produced nonfinite loss')
                losses.append(value)
                selections[row['sample_id']] += 1
                spatial = {key: event[key] for key in ('case_id', 'slot', 'origin_kji', 'augmentation_seed')}
                _digest_event(physical_digest, spatial)
                _digest_event(selection_digest, dict(sample_id=row['sample_id'], **spatial))
                if len(losses) % 32 == 0:
                    print(json.dumps(dict(phase='training_progress', arm=arm, epoch=epoch+1,
                        updates_completed=len(losses), train_loss=float(np.mean(losses)),
                        elapsed_seconds=time.monotonic()-epoch_start)), flush=True)
        if len(losses) != comparison['updates_per_epoch']:
            raise ValueError('Training schedule did not meet its matched update budget')
        train_seconds = time.monotonic()-epoch_start
        assessment = dict(validation_performed=False)
        if (epoch+1) % comparison['validation_every_epochs'] == 0 or epoch+1 == settings['epochs']:
            records = _score(model, validation, settings, loaded, f'{arm}:validation:epoch-{epoch+1}')
            summary = learning.summarize(records)
            score = summary['family_macro']['dice']
            if not math.isfinite(score):
                raise ValueError('Validation score is nonfinite')
            path = directory/f'validation-epoch-{epoch+1:03d}.json'
            learning._signed_write(path, dict(epoch=epoch+1, arm=arm, records=records, summary=summary))
            assessment = dict(validation_performed=True, validation_family_macro_dice=score,
                validation_sample_macro_dice=summary['sample_macro']['dice'],
                validation_samples=len(records), validation_tiles=sum(r['tiles'] for r in records),
                validation_file=path.name, validation_sha256=learning.sha256(path))
            if score > best_value:
                best_value, best_epoch = score, epoch+1
                model.save(directory/'checkpoint.keras')
        traces = int(step.experimental_get_tracing_count())
        if traces > 2:
            raise RuntimeError('Training graph retraced unexpectedly')
        record = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), train_batches=len(losses),
            train_seconds=train_seconds, epoch_seconds=time.monotonic()-epoch_start,
            train_function_traces=traces, physical_patch_events_sha256=physical_digest.hexdigest(),
            selected_patch_events_sha256=selection_digest.hexdigest(),
            selected_variant_counts=dict(sorted(selections.items())), **assessment)
        history.append(record)
        learning._write(directory/'history.json', history)
        print(json.dumps(dict(arm=arm, **record), allow_nan=False), flush=True)
    forbidden = {row['sample_id'] for row in rows if row['split'] == 'test' or
        (arm == 'baseline_only' and row['split'] == 'train' and row.get('baseline') is not True)}
    if forbidden & loaded:
        raise RuntimeError('Forbidden training payload was loaded')
    if component_hashes != _component_hashes():
        raise ValueError('Implementation changed while training; cannot freeze model')
    metadata = dict(schema_version=MODEL_SCHEMA, complete=True, arm=arm,
        scope='engineering; provisional atlas coverage; synthetic unseen-family comparison',
        settings=settings, comparison=comparison, lock_fingerprint=learning._fingerprint(lock),
        component_sha256=component_hashes, initial_weights_sha256=initial_digest,
        initialization_sha256=learning.sha256(directory/'initialization.json'),
        model_sha256=learning.sha256(directory/'checkpoint.keras'),
        history_sha256=learning.sha256(directory/'history.json'), epochs_completed=len(history),
        optimizer_updates=sum(record['train_batches'] for record in history),
        best_epoch=best_epoch, best_selection_value=best_value,
        model_selection='maximum validation family-macro whole-FOV Dice; earliest checkpoint wins ties',
        loaded_samples=sorted(loaded), test_payloads_loaded=False,
        training_sibling_payloads_loaded=any(row['sample_id'] in loaded for row in rows
            if row['split'] == 'train' and row.get('baseline') is not True),
        model_parameters=int(model.count_params()), elapsed_seconds=time.monotonic()-start,
        tensorflow_version=tf.__version__, numpy_version=np.__version__,
        scipy_version=package_version('scipy'), keras_version=package_version('keras'),
        python_version=platform.python_version(),
        devices=[device.name for device in tf.config.list_logical_devices()],
        gpu_allocator_memory=learning.gpu_allocator_memory(tf),
        optimizer='Adam', optimizer_state_saved=False, deterministic_operations=True,
        loss='valid-voxel mean sigmoid cross-entropy from logits + soft Dice (smoothing=1)',
        input='attenuation only; fixed clipping bounds',
        sampling='Same balanced case order and baseline-derived physical patches in both arms; same augmentation seeds; sibling variants equally represented',
        inference_policy='Nonoverlapping whole-FOV tiles; exact cached reuse of unchanged sibling tiles; no target crop; padding excluded',
        split_assignment=[learning._identity(row) for row in rows])
    learning._signed_write(directory/'training.json', metadata)
    (directory/'INCOMPLETE').unlink()
    return metadata


def load_arm(config, lock, arm, model_directory):
    _arm(arm)
    directory = Path(model_directory)
    path = directory/'training.json'
    if (directory/'INCOMPLETE').exists():
        raise ValueError('Comparison fit is incomplete')
    if Path(str(path)+'.sha256').read_text().strip() != learning.sha256(path):
        raise ValueError('Comparison training metadata checksum differs')
    metadata = json.loads(path.read_text())
    expected = dict(schema_version=MODEL_SCHEMA, complete=True, arm=arm,
        settings=learning.settings_for(config), comparison=settings_for(config),
        lock_fingerprint=learning._fingerprint(lock), component_sha256=_component_hashes(),
        model_sha256=learning.sha256(directory/'checkpoint.keras'),
        history_sha256=learning.sha256(directory/'history.json'),
        initialization_sha256=learning.sha256(directory/'initialization.json'))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Comparison model provenance differs from its frozen experiment')
    history = json.loads((directory/'history.json').read_text())
    for record in history:
        if record.get('validation_performed'):
            if learning.sha256(directory/record['validation_file']) != record['validation_sha256']:
                raise ValueError('Validation checkpoint-selection evidence changed')
    model = learning._tensorflow().keras.models.load_model(directory/'checkpoint.keras', compile=False)
    image, probability = learning._single_io(model)
    shape = (None,)+tuple(metadata['settings']['patch_size_kji'])+(1,)
    if tuple(image.shape) != shape or tuple(probability.shape) != shape:
        raise ValueError('Comparison checkpoint shape differs from its frozen settings')
    return model, metadata


def evaluate_arm(config, lock, arm, model_directory, output_directory, *, split='test'):
    if split not in ('validation', 'test'):
        raise ValueError('Comparison evaluation split must be validation or test')
    rows = [row for row in learning.inspect_rows(lock) if row['split'] == split]
    if not rows:
        raise ValueError(f'No {split} samples')
    settings = learning.settings_for(config)
    model, metadata = load_arm(config, lock, arm, model_directory)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'INCOMPLETE').write_text('Comparison evaluation is incomplete.\n')
    loaded = set()
    records = _score(model, rows, settings, loaded, f'{arm}:{split}')
    result = dict(schema_version=RESULT_SCHEMA, complete=True, arm=arm, split=split,
        scope='unseen-family synthetic segmentation' if split == 'test' else 'validation synthetic segmentation',
        settings=settings, comparison=settings_for(config), lock_fingerprint=learning._fingerprint(lock),
        component_sha256=_component_hashes(), model_sha256=metadata['model_sha256'],
        training_metadata_sha256=learning.sha256(Path(model_directory)/'training.json'),
        inference_policy=metadata['inference_policy'], records=records, summary=learning.summarize(records),
        loaded_samples=sorted(loaded), test_payloads_loaded=split == 'test')
    learning._signed_write(directory/'results.json', result)
    (directory/'INCOMPLETE').unlink()
    return result
