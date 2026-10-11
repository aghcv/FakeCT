"""Fresh fold-local anatomical locators for CoA development rotations.

Only original baselines in the supplied training and inner-validation rows are
opened. Outer evaluation families cannot initialize a prior, contribute a
gradient or select an epoch. Final refits consume training originals alone and
save the final preselected epoch, regardless of the observed training loss.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import numpy as np

import fakect_multiscale_experiment_v3 as geometry
import fakect_multiscale_learning_v3 as legacy
from fakect_context_volume import ContextVolume
from fakect_coa_classifier_learning import weight_fingerprint


def validate_rows(rows, *, fixed_budget=False, allow_provisional_family_overlap=False):
    """Reject evaluation payloads and shared split identities before reading."""
    rows = list(rows)
    if type(fixed_budget) is not bool:
        raise ValueError('fixed_budget must be a boolean')
    if type(allow_provisional_family_overlap) is not bool:
        raise ValueError('allow_provisional_family_overlap must be an explicit boolean')
    ids = [row.get('sample_id') for row in rows]
    if not rows or any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('Locator needs unique nonempty original sample identities')
    cases, families, source_paths, mask_identities = {}, {}, {}, {}
    for row in rows:
        split = row.get('split')
        if split not in ('train', 'validation'):
            raise ValueError('Locator cannot open held-out or test samples')
        if row.get('baseline') is not True or row.get('pilot_role', 'original') != 'original':
            raise ValueError('Locator fitting requires original baseline samples only')
        case, family = row.get('case_id'), row.get('anatomy_family')
        if not isinstance(case, str) or not case or not isinstance(family, str) or not family:
            raise ValueError('Locator original needs case and anatomy family identities')
        if case in cases:
            raise ValueError('Locator needs exactly one original baseline per case')
        cases[case] = (family, split)
        if family in families and families[family] != split and not allow_provisional_family_overlap:
            raise ValueError('Locator anatomy family crosses training and validation')
        families[family] = split
        source = row.get('source_atn')
        path = source.get('path') if isinstance(source, dict) else source
        if path:
            identity = str(Path(path).resolve())
            if identity in source_paths and source_paths[identity] != split:
                raise ValueError('Locator attenuation source crosses partitions')
            source_paths[identity] = split
        digest = row.get('original_reference_mask_sha256')
        if digest:
            identity = (digest, tuple(row.get('spacing_ijk_mm', ())))
            if identity in mask_identities and mask_identities[identity] != split:
                raise ValueError('Locator duplicate reference anatomy crosses partitions')
            mask_identities[identity] = split
    train = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'validation']
    if not train or (fixed_budget and validation) or (not fixed_budget and not validation):
        raise ValueError('Fixed-budget refit needs training originals only; inner selection needs training and validation originals')
    return rows, train, validation


def _target(volume, overview):
    value = geometry.bbox_target(volume, overview)[None]
    box = value.reshape(-1)
    if (box.shape != (6,) or not np.isfinite(box).all() or np.any(box[3:] <= 0)
            or np.any(box[:3] - box[3:] / 2 < -1e-6)
            or np.any(box[:3] + box[3:] / 2 > 1 + 1e-6)):
        raise ValueError('Invalid fold-local anatomy supervision')
    return value


def fit_locator(tf, runtime, rows, output, *, epochs, fixed_budget=False,
                allow_provisional_family_overlap=False):
    """Fit a fresh locator; TensorFlow GPU configuration belongs to the caller.

    ``fixed_budget=False`` selects minimum equally weighted native and six fixed
    translated inner-validation losses. ``True`` refuses validation rows and
    retains the final epoch. No earlier locator checkpoint is loaded for fitting.
    Only an explicitly exploratory secondary protocol may allow provisional
    family overlap; exact shared source/reference anatomy still fails closed.
    """
    rows, train, validation = validate_rows(rows, fixed_budget=fixed_budget,
        allow_provisional_family_overlap=allow_provisional_family_overlap)
    if type(epochs) is not int or not 1 <= epochs <= 500:
        raise ValueError('Locator epoch budget must be an integer in [1,500]')
    import fakect_multiscale_network_v3 as network
    from fakect_tensorflow_steps import make_locator_steps

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'locator.keras'
    if checkpoint.exists() or (out / 'locator-history.json').exists():
        raise ValueError('Fold locator already started; use a new immutable stage')
    c = deepcopy(runtime)
    c['experiment']['output_directory'] = str(out)
    guard = legacy.RuntimeGuard(c, tf)
    inputs = legacy.Inputs(c)
    targets, extents = {}, {}
    started = time.monotonic()
    shape = c['input.overview']['shape_kji']
    for row in rows:
        guard.check('rotation-locator-supervision-before')
        overview, _ = inputs.coarse(row)
        volume = ContextVolume(row, verify_hashes=True)
        try:
            targets[row['sample_id']] = _target(volume, overview)
        finally:
            volume.close()
        extents[row['sample_id']] = np.asarray(shape) * np.asarray(overview['spacing_ijk_mm'][::-1])
        guard.check('rotation-locator-supervision-after')

    prior = np.mean([targets[row['sample_id']][0] for row in train], axis=0)
    seed = c['model']['seed']
    tf.keras.backend.clear_session()
    gc.collect()
    locator = network.build_locator(shape, c['model']['base_filters'], seed, prior_box=prior)
    initial_hash = weight_fingerprint(locator)
    view = network.locator_outputs(locator)
    steps = make_locator_steps(view, shape, learning_rate=c['training']['locator_learning_rate'],
                               loss_function=network.locator_loss)
    guard.check('rotation-locator-fresh', save=True)
    rng = np.random.default_rng(seed)
    panel = []
    for row in validation:
        _, image = inputs.coarse(row)
        for name, _, _, transformed, target, _, error in legacy.fixed_probes(image, targets[row['sample_id']], c):
            if error:
                raise ValueError('Ineligible inner-validation translation probe '+row['sample_id']+':'+name+': '+error)
            panel.append((transformed, target))
            guard.check('rotation-locator-validation-probe')

    selection_policy = ('Final preselected epoch; training originals only, no validation selection.' if fixed_budget else
                        'Minimum 0.5 native inner-validation loss + 0.5 mean six fixed translation losses.')
    provenance = {'fresh_initialization': True, 'initial_weights_sha256': initial_hash,
                  'seed': seed, 'fixed_budget': fixed_budget, 'epochs_requested': epochs,
                  'prior_box': prior.tolist(), 'prior_training_sample_ids': [row['sample_id'] for row in train],
                  'training_sample_ids': [row['sample_id'] for row in train],
                  'validation_sample_ids': [row['sample_id'] for row in validation],
                  'allow_provisional_family_overlap': allow_provisional_family_overlap,
                  'overlap_family_ids': sorted({row['anatomy_family'] for row in train}.intersection(
                      row['anatomy_family'] for row in validation)),
                  'outer_evaluation_used': False, 'reserved_test_evaluated': False,
                  'checkpoint_selection': selection_policy,
                  'prior_scope': 'Original training means initialize extent only; image heatmap determines centers.'}
    geometry.write_json(out / 'initialization.json', provenance)
    history, best, selected_epoch, selected_predictions, selected_hash = [], float('inf'), None, None, None
    for epoch in range(epochs):
        losses, norms, order = [], [], []
        for index in rng.permutation(len(train)):
            row = train[int(index)]
            order.append(row['sample_id'])
            _, image = inputs.coarse(row)
            target = targets[row['sample_id']]
            for augmentation_index in range(1 + c['augmentation']['samples_per_case']):
                if augmentation_index == 0:
                    transformed, truth = image, target
                else:
                    a = c['augmentation']
                    transformed, truth, _ = legacy.augment_overview(image, target, rng,
                        translate_fraction_kji=a['translate_fraction_kji'], scale_range=a['scale_range'],
                        crop_fraction_max=a['crop_fraction_max'])
                guard.check('rotation-locator-train-before', steps=steps)
                result = steps.train(tf.convert_to_tensor(transformed), tf.convert_to_tensor(truth))
                losses.append(float(result['loss']))
                norms.append(float(result['gradient_global_norm']))
                guard.check('rotation-locator-train-after', steps=steps)
        predictions, native_losses = {}, []
        for row in rows:
            _, image = inputs.coarse(row)
            value = steps.predict(tf.convert_to_tensor(image))
            predictions[row['sample_id']] = np.asarray(value['box'])[0]
            if row['split'] == 'validation':
                native_losses.append(float(tf.reduce_mean(network.locator_loss(targets[row['sample_id']], value))))
            guard.check('rotation-locator-native-evaluation', steps=steps)
        probe_losses = []
        for image, truth in panel:
            value = steps.predict(tf.convert_to_tensor(image))
            probe_losses.append(float(tf.reduce_mean(network.locator_loss(truth, value))))
            guard.check('rotation-locator-probe-evaluation', steps=steps)
        native = float(np.mean(native_losses)) if native_losses else None
        translated = float(np.mean(probe_losses)) if probe_losses else None
        selection = None if fixed_budget else .5 * native + .5 * translated
        record = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                  'gradient_norm_mean': float(np.mean(norms)), 'training_sample_order': order,
                  'optimizer_updates_this_epoch': len(losses),
                  'validation_loss': selection, 'validation_identity_loss': native,
                  'validation_translation_loss': translated,
                  'training_geometry': legacy.box_metrics(train, predictions, targets, extents),
                  'validation_geometry': legacy.box_metrics(validation, predictions, targets, extents) if validation else None,
                  'predicted_boxes': {key: value.tolist() for key, value in predictions.items()},
                  'runtime': guard.check('rotation-locator-epoch', steps=steps, save=True)}
        numeric = [record['train_loss'], record['gradient_norm_mean']] + ([] if fixed_budget else [selection])
        if not np.isfinite(numeric).all():
            raise ValueError('Nonfinite fold-local locator learning')
        if (fixed_budget and epoch + 1 == epochs) or (not fixed_budget and selection < best):
            locator.save(checkpoint)
            guard.check('rotation-locator-checkpoint-saved', steps=steps)
            best = selection if selection is not None else best
            selected_epoch = epoch + 1
            selected_hash = weight_fingerprint(locator)
            selected_predictions = {key: value.copy() for key, value in predictions.items()}
        history.append(record)
        geometry.write_json(out / 'locator-history.json', {'epochs': history, 'selected_epoch': selected_epoch,
                            'checkpoint_selection': selection_policy, **provenance})
        print(json.dumps({'stage': 'rotation-locator', 'output': str(out),
                          **{key: value for key, value in record.items() if key not in ('predicted_boxes', 'training_sample_order')}}), flush=True)
    traces = steps.tracing_counts()
    updates = int(np.asarray(steps.optimizer.iterations))
    final_hash = weight_fingerprint(locator)
    if final_hash == initial_hash or selected_epoch is None:
        raise RuntimeError('Fold locator weights did not change or no checkpoint was selected')
    del steps, view, locator, panel, inputs
    gc.collect()
    restored = network.load_model(checkpoint)
    if weight_fingerprint(restored) != selected_hash:
        raise ValueError('Reloaded fold locator weights differ from the selected epoch')
    guard.check('rotation-locator-checkpoint-reloaded', save=True)
    reader, maximum_error = legacy.Inputs(c), 0.
    for row in rows:
        _, image = reader.coarse(row)
        observed = np.asarray(network.predict_boxes(restored, image))[0]
        expected = selected_predictions[row['sample_id']]
        if not np.allclose(observed, expected, atol=1e-6, rtol=1e-5):
            raise ValueError('Reloaded fold locator prediction changed: '+row['sample_id'])
        maximum_error = max(maximum_error, float(np.max(np.abs(observed - expected))))
        guard.check('rotation-locator-reload-parity')
    result = {'checkpoint': str(checkpoint), 'checkpoint_sha256': geometry.sha256(checkpoint),
              'selected_epoch': selected_epoch, 'epochs_completed': len(history), 'history': history,
              'optimizer_steps': updates, 'traces': traces, 'final_weights_sha256': final_hash,
              'selected_weights_sha256': selected_hash, 'elapsed_seconds': time.monotonic() - started,
              'reload_parity': {'passed': True, 'maximum_absolute_error': maximum_error,
                                'absolute_tolerance': 1e-6, 'relative_tolerance': 1e-5}, **provenance}
    geometry.write_json(out / 'locator-result.json', result)
    del restored, reader
    gc.collect()
    return result
