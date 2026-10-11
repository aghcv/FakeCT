"""Bounded held-out segmentation evaluation and anatomy-balanced summaries.

Only a single decoded image/mask pair and one predicted axial slice are kept
at a time. These are segmentation metrics, not disease-classification or
clinical severity measurements. Repeated seeds are repeated model fits, not
additional independent anatomies.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from fakect_segmentation import (MAX_VOLUME_VOXELS, inspect_manifest,
                                 load_segmentation, predict_axial_slice)


METRICS = ('dice', 'iou', 'precision', 'recall')
COUNTS = ('true_positive', 'false_positive', 'false_negative', 'true_negative')
EMPTY_POLICY = ('Dice and IoU are 1 when truth and prediction are both empty; '
                'precision and recall are also 1 only for that empty-both case. '
                'An undefined precision or recall with a nonempty opposite mask is 0.')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _metrics(tp, fp, fn, tn):
    empty = tp + fp + fn == 0
    return {'dice': 2 * tp / (2 * tp + fp + fn) if not empty else 1.,
            'iou': tp / (tp + fp + fn) if not empty else 1.,
            'precision': tp / (tp + fp) if tp + fp else float(empty),
            'recall': tp / (tp + fn) if tp + fn else float(empty)}


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _shape(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or
            any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or n <= 0 for n in value)):
        raise ValueError('Evaluation requires positive integer shape_kji dimensions')
    result = tuple(int(n) for n in value)
    if math.prod(result) > MAX_VOLUME_VOXELS:
        raise ValueError('Evaluation volume exceeds the bounded segmentation volume limit')
    return result


def evaluate_segmentation_model(model_directory, manifest_path, *, threshold=0.5):
    """Evaluate only frozen test samples, without writing or tuning anything.

    The model must have been trained from this exact manifest/experiment lock.
    Array and archive hashes are checked before inference; padded image tiles
    are cropped by the standard predictor and never enter the metric counts.
    """
    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError('Evaluation threshold must be finite and strictly between 0 and 1')
    try:
        threshold = float(threshold)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Evaluation threshold must be finite and strictly between 0 and 1') from exc
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Evaluation threshold must be finite and strictly between 0 and 1')
    manifest = inspect_manifest(manifest_path)
    test_samples = [row for row in manifest['samples'] if row['split'] == 'test']
    if not test_samples:
        raise ValueError('Held-out evaluation requires at least one test sample')
    directory = Path(model_directory).resolve()
    metadata_path = directory / 'model-metadata.json'
    metadata_sha256 = _sha256(metadata_path)
    model, metadata = load_segmentation(directory)
    if (not metadata.get('source_manifest') or
            Path(metadata['source_manifest']).resolve() != manifest['path'].resolve() or
            metadata.get('source_manifest_sha256') != manifest['sha256']):
        raise ValueError('Model source manifest path/hash differs from the evaluation manifest')
    model_path = directory / metadata['model_file']
    model_sha256 = _sha256(model_path)
    if model_sha256 != metadata['model_sha256']:
        raise ValueError('Saved model hash differs from recorded model provenance')
    rows = []
    identifiers = set()
    for sample in test_samples:
        variant = _identity(sample.get('variant_id'), 'variant_id')
        if variant in identifiers:
            raise ValueError('A test variant may appear only once in an evaluation')
        identifiers.add(variant)
        case = _identity(sample.get('case_id', manifest['data'].get('case_id')), 'case_id')
        family = _identity(sample.get('anatomy_family', manifest['data'].get('anatomy_family')),
                           'anatomy_family')
        source_shape = _shape(sample.get('source_shape_kji', sample.get('shape_kji', manifest['shape_kji'])))
        output_shape = _shape(sample.get('shape_kji', manifest['shape_kji']))
        path = Path(sample['_path'])
        if _sha256(path) != sample['sha256']:
            raise ValueError(f'Sample file hash changed: {variant}')
        with np.load(path, allow_pickle=False) as archive:
            image, mask = archive['image'], archive['mask']
        if image.dtype != np.float32 or image.shape != source_shape or not np.isfinite(image).all():
            raise ValueError('Sample image must be finite float32 matching source shape_kji')
        if mask.dtype != np.uint8 or mask.shape != source_shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError('Sample mask must be uint8 {0,1} matching source shape_kji')
        for array, name in ((image, 'image_sha256'), (mask, 'mask_sha256')):
            if hashlib.sha256(array.tobytes(order='C')).hexdigest() != sample[name]:
                raise ValueError(f'Sample {name} disagrees with its payload')
        del array
        if 'target_spacing_ijk_mm' in sample:
            from fakect_model_resampling import resample_pair
            image, mask = resample_pair(image, mask, sample['source_spacing_ijk_mm'],
                                        sample['target_spacing_ijk_mm'])
        if image.shape != output_shape or mask.shape != output_shape:
            raise ValueError('Evaluated shape disagrees with frozen experiment geometry')
        tp = fp = fn = tn = 0
        for k in range(image.shape[0]):
            probability = np.asarray(predict_axial_slice(model, image[k], metadata['settings']))
            if (probability.shape != image.shape[1:] or probability.dtype.kind not in 'fiu' or
                    not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
                raise ValueError('Model produced invalid test probabilities or slice shape')
            predicted, truth = probability >= threshold, mask[k].astype(bool)
            tp += int(np.count_nonzero(predicted & truth))
            fp += int(np.count_nonzero(predicted & ~truth))
            fn += int(np.count_nonzero(~predicted & truth))
            tn += int(np.count_nonzero(~predicted & ~truth))
        rows.append({'variant_id': variant, 'case_id': case, 'anatomy_family': family,
                     'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
                     'true_negative': tn, 'valid_voxels': tp + fp + fn + tn,
                     **_metrics(tp, fp, fn, tn)})
        # Explicitly release this pair before decompressing the next volume.
        del image, mask, probability, predicted, truth
    if (_sha256(manifest['path']) != manifest['sha256'] or
            _sha256(metadata_path) != metadata_sha256 or _sha256(model_path) != model_sha256):
        raise ValueError('Model or source manifest changed during held-out evaluation')
    return {'schema_version': 'fakect.segmentation-evaluation/1',
            'model_directory': str(directory), 'model_sha256': model_sha256,
            'model_metadata_sha256': metadata_sha256,
            'source_manifest': str(manifest['path']), 'source_manifest_sha256': manifest['sha256'],
            'threshold': threshold, 'split': 'test', 'test_samples': len(rows),
            'only_test_payloads_loaded': True, 'empty_mask_policy': EMPTY_POLICY,
            'evaluation_scope': 'Held-out synthetic anatomy segmentation; not clinical disease classification.',
            'preprocessing': manifest['data'].get('preprocessing', {}),
            'samples': rows}


def _mean(rows):
    return {metric: math.fsum(row[metric] for row in rows) / len(rows) for metric in METRICS}


def _checked_row(row):
    if not isinstance(row, dict):
        raise ValueError('Evaluation samples must be mappings')
    for name in ('variant_id', 'case_id', 'anatomy_family'):
        _identity(row.get(name), name)
    for name in (*COUNTS, 'valid_voxels'):
        value = row.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if row['valid_voxels'] <= 0 or sum(row[name] for name in COUNTS) != row['valid_voxels']:
        raise ValueError('Confusion counts must sum to positive valid_voxels')
    expected = _metrics(*(row[name] for name in COUNTS))
    for name in METRICS:
        value = row.get(name)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.number)) or
                not np.isrealobj(value) or not math.isfinite(float(value)) or not 0 <= value <= 1):
            raise ValueError(f'{name} must be a finite metric between 0 and 1')
        if not math.isclose(value, expected[name], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'{name} disagrees with its confusion counts')
    return row


def summarize_evaluations(evaluations):
    """Average variants, then cases, then seeds, then independent families.

    Input records need fold_id, a nonnegative integer seed, and sample rows from
    evaluate_segmentation_model. Each outer test sample must appear exactly
    once per seed. The same folds and sample identities must recur for every
    seed. The caller additionally verifies the planned fold/model provenance.
    No confidence interval treating siblings or seeds as patients is computed.
    """
    if not isinstance(evaluations, (list, tuple)) or not evaluations:
        raise ValueError('At least one completed evaluation is required')
    observed, fold_samples, by_seed, identities, case_families = set(), {}, {}, {}, {}
    thresholds = set()
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            raise ValueError('Evaluations must be mappings')
        fold = _identity(evaluation.get('fold_id'), 'fold_id')
        seed = evaluation.get('seed')
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        seed = int(seed)
        if (fold, seed) in observed:
            raise ValueError('Duplicate fold_id/seed evaluation')
        observed.add((fold, seed))
        if 'threshold' in evaluation:
            value = evaluation['threshold']
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int)) or
                    not math.isfinite(value) or not 0 < value < 1):
                raise ValueError('Evaluation threshold must be finite and strictly between 0 and 1')
            thresholds.add(value)
        samples = evaluation.get('samples')
        if not isinstance(samples, list) or not samples:
            raise ValueError('Every evaluation requires nonempty test samples')
        records = by_seed.setdefault(seed, {})
        membership = set()
        for sample in samples:
            row = _checked_row(sample)
            variant, case, family = (row[name] for name in ('variant_id', 'case_id', 'anatomy_family'))
            if variant in records:
                raise ValueError('Duplicate out-of-fold variant_id/seed; each sample is tested once per seed')
            if variant in identities and identities[variant] != (case, family):
                raise ValueError('Sample case/family identity changes between seeds')
            if case in case_families and case_families[case] != family:
                raise ValueError('A case cannot belong to different anatomy families')
            identities[variant], case_families[case], records[variant] = (case, family), family, row
            membership.add(variant)
        if fold in fold_samples and fold_samples[fold] != membership:
            raise ValueError('Test sample membership of a fold changes between seeds')
        fold_samples[fold] = membership
    if len(thresholds) > 1:
        raise ValueError('Cannot aggregate evaluations with different prediction thresholds')
    seeds = sorted(by_seed)
    folds = sorted(fold_samples)
    if observed != {(fold, seed) for fold in folds for seed in seeds}:
        raise ValueError('Every fold must have completed evaluations for the same seeds')
    expected = set(identities)
    if any(set(records) != expected for records in by_seed.values()):
        raise ValueError('Each seed must evaluate the same held-out sample identities')
    families = sorted(set(case_families.values()))
    per_seed = []
    family_across_seeds = {family: [] for family in families}
    for seed in seeds:
        cases = {}
        for row in by_seed[seed].values():
            cases.setdefault(row['case_id'], []).append(row)
        family_reports = []
        for family in families:
            case_reports = [{'case_id': case, 'samples': len(rows), **_mean(rows)}
                            for case, rows in sorted(cases.items()) if case_families[case] == family]
            report = {'anatomy_family': family, 'cases': case_reports,
                      'case_count': len(case_reports), **_mean(case_reports)}
            family_reports.append(report)
            family_across_seeds[family].append(report)
        per_seed.append({'seed': seed, 'family_reports': family_reports,
                         'macro_average': _mean(family_reports)})
    family_reports = [{'anatomy_family': family,
                       'case_ids': sorted(case for case, owner in case_families.items() if owner == family),
                       'case_count': family_across_seeds[family][0]['case_count'],
                       'variants_per_seed': sum(owner == family for _, owner in identities.values()),
                       'seeds': seeds, **_mean(family_across_seeds[family])} for family in families]
    return {'schema_version': 'fakect.segmentation-evaluation-summary/1',
            'weighting': 'Equal variant weight within case, equal case weight within family, '
                         'equal seed weight within family, then equal family weight overall.',
            'empty_mask_policy': EMPTY_POLICY,
            'uncertainty': 'No confidence intervals; siblings and repeated fits are not independent anatomies.',
            'counts': {'folds': len(folds), 'seeds': len(seeds), 'evaluations': len(evaluations),
                       'families': len(families), 'cases': len(case_families),
                       'unique_variants': len(identities), 'sample_evaluations': len(identities) * len(seeds)},
            'family_reports': family_reports, 'macro_average': _mean(family_reports),
            'per_seed': per_seed}
