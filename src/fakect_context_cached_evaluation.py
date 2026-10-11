"""Exact whole-volume sibling scoring with case-local unchanged-tile reuse.

Every scan voxel is evaluated under the same nonoverlapping tiling as
``fakect_context_learning.score_volume``. A baseline's additive tile statistics
can be reused only where BOTH the image and target are proven unchanged. No
prediction volume is cached and no ground-truth bounding box limits inference.
The cache lives for one call only; the caller must keep model weights fixed.
"""
from collections import OrderedDict
import math

import numpy as np

import fakect_context_learning as learning
from fakect_context_volume import ContextVolume


MAX_COMPARISON_VOXELS = 64_000_000
COMPARISON_CHUNK_VOXELS = 2_000_000


class UnprovenReuse(ValueError):
    """Valid input whose equality cannot be established within the budget."""


def _groups(rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Evaluation requires at least one sample')
    groups, seen = OrderedDict(), set()
    for row in rows:
        if row['sample_id'] in seen:
            raise ValueError('Evaluation sample IDs must be unique')
        seen.add(row['sample_id'])
        groups.setdefault(row['case_id'], []).append(row)
    for case, members in groups.items():
        if len([row for row in members if row.get('baseline') is True]) != 1:
            raise ValueError(f'Case {case} needs exactly one baseline for cached evaluation')
        if len({(row['anatomy_family'], row['split']) for row in members}) != 1:
            raise ValueError('A case cannot cross anatomy families or evaluation partitions')
    return rows, groups


def _same_source(first, second):
    """Use exactly the pinned source identity contract of ContextVolume."""
    if (first.case_id != second.case_id or
            first.source_atn != second.source_atn or
            first._source_stat != second._source_stat or
            first.source_shape_kji != second.source_shape_kji or
            first.spacing_ijk_mm != second.spacing_ijk_mm):
        raise UnprovenReuse('Source identity or native geometry differs')
    first._check_current()
    second._check_current()


def _bounds(volume, kind):
    origin = (volume.edit_crop_origin_ijk if kind == 'image' else volume.mask_origin_ijk)[::-1]
    array = volume.edit_image if kind == 'image' else getattr(volume, kind)
    return np.asarray(origin), np.asarray(origin) + np.asarray(array.shape)


def _native_chunk(volume, kind, lo, hi):
    if kind == 'image':
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        value = np.array(volume._base[slices], dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError('Source comparison slab contains nonfinite attenuation')
        value *= np.float32(10. / volume.spacing_ijk_mm[0])
        array = volume.edit_image
    else:
        value = np.zeros(tuple(hi - lo), dtype=np.uint8)
        array = getattr(volume, kind)
    crop_lo, crop_hi = _bounds(volume, kind)
    overlap_lo, overlap_hi = np.maximum(lo, crop_lo), np.minimum(hi, crop_hi)
    if np.all(overlap_hi > overlap_lo):
        dst = tuple(slice(int(a - o), int(b - o)) for a, b, o in zip(overlap_lo, overlap_hi, lo))
        src = tuple(slice(int(a - o), int(b - o)) for a, b, o in zip(overlap_lo, overlap_hi, crop_lo))
        value[dst] = array[src]
    return value


def _difference_box(first, second, kind):
    """Exact native bounding box of differing values, with bounded allocation.

    Different crop origins are compared in the shared physical native grid.
    Outside a mask crop the value is zero; outside an image crop it is the
    actual source attenuation, scaled exactly as in ContextVolume.read_block.
    """
    a_lo, a_hi = _bounds(first, kind)
    b_lo, b_hi = _bounds(second, kind)
    lo, hi = np.minimum(a_lo, b_lo), np.maximum(a_hi, b_hi)
    extent = hi - lo
    if math.prod(map(int, extent)) > MAX_COMPARISON_VOXELS:
        raise UnprovenReuse('Compact-crop comparison union exceeds the native voxel budget')
    plane = int(extent[1]) * int(extent[2])
    if plane > COMPARISON_CHUNK_VOXELS:
        raise UnprovenReuse('Compact-crop comparison plane exceeds the allocation budget')
    depth = max(1, COMPARISON_CHUNK_VOXELS // plane)
    minimum, maximum = None, None
    for k in range(int(lo[0]), int(hi[0]), depth):
        chunk_lo, chunk_hi = lo.copy(), hi.copy()
        chunk_lo[0], chunk_hi[0] = k, min(k + depth, hi[0])
        difference = _native_chunk(first, kind, chunk_lo, chunk_hi) != _native_chunk(second, kind, chunk_lo, chunk_hi)
        if np.any(difference):
            # Axis reductions avoid allocating a coordinate for every changed voxel.
            nonzero = [np.flatnonzero(np.any(difference, axis=tuple(j for j in range(3) if j != axis)))
                       for axis in range(3)]
            low = chunk_lo + np.array([values[0] for values in nonzero])
            high = chunk_lo + np.array([values[-1] + 1 for values in nonzero])
            minimum = low if minimum is None else np.minimum(minimum, low)
            maximum = high if maximum is None else np.maximum(maximum, high)
    first._check_current()
    second._check_current()
    return None if minimum is None else (minimum, maximum)


def prove_difference(first, second):
    """Prove shared reference and return conservative resampling support."""
    _same_source(first, second)
    if _difference_box(first, second, 'mask_before') is not None:
        raise UnprovenReuse('Physical before masks differ')
    image = _difference_box(first, second, 'image')
    mask = _difference_box(first, second, 'mask_after')
    boxes = []
    if image is not None:
        # A native center contributes to linear interpolation within one voxel
        # on either side. Include zero-weight end points conservatively.
        boxes.append((image[0].astype(float) - 1., image[1].astype(float)))
    if mask is not None:
        # Nearest-neighbor target selection uses floor(native_coordinate + .5).
        boxes.append((mask[0].astype(float) - .5, mask[1].astype(float) - .5))
    support = None if not boxes else (np.minimum.reduce([b[0] for b in boxes]),
                                      np.maximum.reduce([b[1] for b in boxes]))
    return dict(status='proven', native_image_difference_box_kji=_box_json(image),
                native_target_difference_box_kji=_box_json(mask),
                resampling_support_kji=_box_json(support),
                source_integrity_scope='same resolved path, size and mtime_ns; no whole-body content hash'), support


def _box_json(box):
    return None if box is None else [np.asarray(value).tolist() for value in box]


def _affected(origin, patch, spacing, native_spacing, support):
    if support is None:
        return False
    scale = np.asarray(spacing)[::-1] / np.asarray(native_spacing)[::-1]
    lower = (np.asarray(origin) + .5) * scale - .5
    upper = (np.asarray(origin) + np.asarray(patch) - .5) * scale - .5
    # The affine sampler evaluates local_index*scale + offset, whereas these
    # bounds use global indices. Algebraically equal endpoints can differ by
    # a few ulps after subtracting the native slab origin. Expand rather than
    # risk declaring a boundary tile unchanged because of arithmetic ordering.
    magnitude = np.maximum.reduce([np.ones(3), np.abs(lower), np.abs(upper),
                                   np.abs(support[0]), np.abs(support[1])])
    tolerance = 64 * np.finfo(np.float64).eps * magnitude
    return bool(np.all(upper >= support[0] - tolerance) and
                np.all(lower <= support[1] + tolerance))


def _valid_origin(volume, origin, settings):
    scale = np.asarray(settings['target_spacing_mm'])[::-1] / np.asarray(volume.spacing_ijk_mm)[::-1]
    first = (np.asarray(origin) + .5) * scale - .5
    return bool(np.all(first < np.asarray(volume.source_shape_kji) - .5))


def _origins(volume, settings):
    shape = volume.model_shape(settings['target_spacing_mm'])
    return shape, [origin for origin in learning.tile_origins(shape, settings['patch_size_kji'])
                   if _valid_origin(volume, origin, settings)]


def _proof_or_fallback(baseline, volume):
    try:
        return prove_difference(baseline, volume)
    except UnprovenReuse as error:
        return dict(status='recompute_all', reason=str(error)), None


def plan_rows(rows, settings):
    """Load/verify compact payloads and count exact sharing without a model."""
    rows, groups = _groups(rows)
    records = {}
    for members in groups.values():
        base_row = next(row for row in members if row['baseline'])
        with ContextVolume(base_row) as baseline:
            _, origins = _origins(baseline, settings)
            records[base_row['sample_id']] = dict(sample_id=base_row['sample_id'], case_id=base_row['case_id'],
                baseline_sample_id=base_row['sample_id'], total_tiles=len(origins), inferred_tiles=len(origins),
                reused_tiles=0, proof=dict(status='baseline'))
            for row in members:
                if row is base_row:
                    continue
                with ContextVolume(row) as volume:
                    proof, support = _proof_or_fallback(baseline, volume)
                    _, sample_origins = _origins(volume, settings)
                    count = len(sample_origins) if proof['status'] != 'proven' else sum(
                        _affected(origin, settings['patch_size_kji'], settings['target_spacing_mm'],
                                  volume.spacing_ijk_mm, support) for origin in sample_origins)
                    records[row['sample_id']] = dict(sample_id=row['sample_id'], case_id=row['case_id'],
                        baseline_sample_id=base_row['sample_id'], total_tiles=len(sample_origins),
                        inferred_tiles=count, reused_tiles=len(sample_origins)-count, proof=proof)
    result = [records[row['sample_id']] for row in rows]
    total = sum(row['total_tiles'] for row in result)
    inferred = sum(row['inferred_tiles'] for row in result)
    return dict(records=result, total_tiles=total, inferred_tiles=inferred,
                reused_tiles=total-inferred, forward_pass_reduction_factor=total/inferred)


def _score_tile(inference_model, volume, origin, settings):
    arrays = learning._read(volume, origin, settings)
    x, y, valid = learning.prepare_block(*arrays, settings)
    probability, logits = learning._predict_pair(inference_model, x)
    truth, validity = y[0, ..., 0], valid[0, ..., 0]
    return (learning.counts_for(truth, probability >= settings['threshold'], validity),
            learning.loss_totals(truth, probability, validity, logits=logits))


def _aggregate(statistics, settings):
    counts, totals = None, None
    for current, loss in statistics:
        counts = dict(current) if counts is None else {key: counts[key]+value for key, value in current.items()}
        totals = dict(loss) if totals is None else {key: totals[key]+value for key, value in loss.items()}
    if counts is None or counts['true_positive'] + counts['false_negative'] == 0:
        raise ValueError('Full-volume target is empty on the requested model grid')
    return dict(metrics=learning.metrics_from_counts(counts, math.prod(settings['target_spacing_mm'])),
                loss=learning.loss_from_totals(totals))


def score_rows(model, rows, settings, loaded=None, progress_callback=None):
    """Score all rows exactly, sharing unchanged sibling tiles within each case.

    ``loaded`` is an optional set receiving every accessed sample ID. Callback
    receives one JSON-compatible event dict at sample start/completion and each
    100 processed tiles. Returned records preserve input order and are accepted
    by ``fakect_context_learning.summarize``. Incompatible but valid provenance
    falls back to a full scan; corrupt payloads still raise loader errors.
    """
    rows, groups = _groups(rows)
    inference_model = learning._inference_model(model)
    records = {}
    def emit(event, **values):
        if progress_callback is not None:
            progress_callback(dict(event=event, **values))
    def evaluate(row, volume, cache, proof, support):
        if loaded is not None:
            loaded.add(row['sample_id'])
        shape, origins = _origins(volume, settings)
        inferred, reused, statistics = 0, 0, []
        emit('sample_start', sample_id=row['sample_id'], total_tiles=len(origins), proof=proof)
        for index, origin in enumerate(origins):
            can_reuse = (proof['status'] == 'proven' and not _affected(origin, settings['patch_size_kji'],
                         settings['target_spacing_mm'], volume.spacing_ijk_mm, support))
            if can_reuse:
                volume._check_current()
                statistic = cache[origin]
                reused += 1
            else:
                statistic = _score_tile(inference_model, volume, origin, settings)
                inferred += 1
            statistics.append(statistic)
            if proof['status'] == 'baseline':
                cache[origin] = statistic
            if (index+1) % 100 == 0 or index+1 == len(origins):
                emit('tile_progress', sample_id=row['sample_id'], processed_tiles=index+1,
                     total_tiles=len(origins), inferred_tiles=inferred, reused_tiles=reused)
        record = {**learning._identity(row), **_aggregate(statistics, settings),
                  'tiles': len(origins), 'model_shape_kji': list(shape),
                  'evaluation_reuse': dict(baseline_sample_id=base_row['sample_id'],
                      inferred_tiles=inferred, reused_tiles=reused, total_tiles=len(origins), proof=proof)}
        emit('sample_complete', sample_id=row['sample_id'], **record['evaluation_reuse'])
        return record
    for members in groups.values():
        base_row = next(row for row in members if row['baseline'])
        cache = {}
        with ContextVolume(base_row) as baseline:
            records[base_row['sample_id']] = evaluate(base_row, baseline, cache, dict(status='baseline'), None)
            for row in members:
                if row is base_row:
                    continue
                with ContextVolume(row) as volume:
                    proof, support = _proof_or_fallback(baseline, volume)
                    records[row['sample_id']] = evaluate(row, volume, cache, proof, support)
    return [records[row['sample_id']] for row in rows]
