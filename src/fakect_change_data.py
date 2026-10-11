"""Bounded image-only patches and net baseline-relative aorta targets.

The frozen locator supplies origins in its unchanged high-resolution grid.
Images use the existing linear ATN resampler and normalization; before/after
masks are sampled nearest-neighbor at the SAME physical voxel centers, then
combined into 0=unchanged, 1=added, 2=removed. This includes removed voxels
outside the current aorta. Edit order, clinical condition and attenuation-delta
sign do not affect the net targets. Truth-guided patch sampling is explicitly
training-only and is separate from the target-independent inference schedule.
"""
from copy import deepcopy
from itertools import product

import numpy as np

from fakect_context_volume import ContextVolume, MAX_BLOCK_VOXELS, _triple, _product
from fakect_multiscale_experiment_v3 import normalized_input

MAX_FOCUS_VOXELS = 16_000_000


def net_change_labels(before, after, valid=None):
    """Return mutually exclusive net additions/removals from aligned masks."""
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or not before.ndim == 3:
        raise ValueError('Before/after masks must be aligned 3D arrays')
    if not all(np.all((value == 0) | (value == 1)) for value in (before, after)):
        raise ValueError('Before/after masks must be binary')
    result = np.zeros(before.shape, np.uint8)
    result[(after == 1) & (before == 0)] = 1
    result[(before == 1) & (after == 0)] = 2
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != result.shape or not np.all((valid == 0) | (valid == 1)):
            raise ValueError('Validity must be an aligned binary array')
        result *= valid.astype(np.uint8)
    return result


def _nearest_before(volume, indices_kji):
    """Mirror the pinned after-mask sampler without altering that module."""
    local = [values - int(origin) for values, origin in
             zip(indices_kji, volume.mask_origin_ijk[::-1])]
    inside = [(v >= 0) & (v < n) for v, n in zip(local, volume.mask_before.shape)]
    clipped = [np.clip(v, 0, n - 1) for v, n in zip(local, volume.mask_before.shape)]
    result = volume.mask_before[np.ix_(*clipped)].copy()
    result *= inside[0][:, None, None]
    result *= inside[1][None, :, None]
    result *= inside[2][None, None, :]
    return result


class FocusedChangeVolume:
    """One native source/crop reader, streamed at frozen predicted origins.

    Baseline rows use their original image and naturally have zero targets.
    ``baseline_control=True`` is an optional equivalent control constructed
    from an edited row: original ATN is restored in the edit crop and both
    target masks are the paired baseline. Its inherited focus origins must be
    described as a fixed paired-focus diagnostic, not a fresh baseline locator.
    Neither mask nor true difference enters the two predictor channels.
    """
    def __init__(self, row, runtime, origins_kji, *, verify_hashes=True,
                 baseline_control=False):
        if type(baseline_control) is not bool:
            raise ValueError('baseline_control must be Boolean')
        self.runtime = deepcopy(runtime)
        detail = self.runtime['input.detail']
        self.shape_kji = _triple(detail['shape_kji'], 'Focus shape', integer=True, positive=True)
        self.spacing_ijk_mm = _triple(detail['spacing_ijk_mm'], 'Focus spacing', positive=True)
        origins = np.asarray(origins_kji)
        count = detail['proposals']
        if (type(count) is not int or count < 1 or origins.shape != (count, 3)
                or origins.dtype.kind not in 'iu'):
            raise ValueError('Focus origins must match the reviewed integer proposal count')
        self.origins_kji = origins.astype(np.int64, copy=True)
        if count * _product(self.shape_kji) > MAX_FOCUS_VOXELS:
            raise ValueError('Focus exceeds the 16-million-voxel display/evaluation budget')
        low, high = self.runtime['model']['clip_min'], self.runtime['model']['clip_max']
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError('Invalid fixed attenuation normalization range')
        self.volume = ContextVolume(row, verify_hashes=verify_hashes)
        self.baseline_control = baseline_control
        self._point_cache = {}
        self._reads, self._read_voxels, self._peak_patch_voxels = 0, 0, 0
        try:
            if row['baseline'] and np.any(self.volume.mask_before != self.volume.mask_after):
                raise ValueError('A baseline row must have unchanged before/after masks')
            if baseline_control:
                origin = self.volume.edit_crop_origin_ijk[::-1]
                shape = self.volume.edit_image.shape
                sl = tuple(slice(int(o), int(o + n)) for o, n in zip(origin, shape))
                original = np.array(self.volume._base[sl], dtype=np.float32)
                original *= np.float32(10. / self.volume.spacing_ijk_mm[0])
                if not np.isfinite(original).all():
                    raise ValueError('Original control crop contains nonfinite attenuation')
                original.flags.writeable = False
                self.volume.edit_image = original
        except Exception:
            self.volume.close()
            raise

    def _patch_geometry(self, proposal_index, offset_kji, shape_kji):
        if type(proposal_index) is not int or not 0 <= proposal_index < len(self.origins_kji):
            raise ValueError('Invalid proposal index')
        offset = _triple(offset_kji, 'Patch offset', integer=True, nonnegative=True)
        shape = _triple(shape_kji, 'Patch shape', integer=True, positive=True)
        if np.any(offset + shape > self.shape_kji):
            raise ValueError('Patch extends outside its frozen focus proposal')
        if _product(shape) > MAX_BLOCK_VOXELS:
            raise ValueError('Patch exceeds the bounded source-reader voxel budget')
        return self.origins_kji[proposal_index] + offset, shape

    def read_patch(self, proposal_index, offset_kji, shape_kji):
        origin, shape = self._patch_geometry(proposal_index, offset_kji, shape_kji)
        block = self.volume.read_block(origin, shape, self.spacing_ijk_mm)
        scale = self.spacing_ijk_mm[::-1] / self.volume._native_spacing_kji
        coords = [(int(o) + np.arange(int(n), dtype=np.float64) + .5) * s - .5
                  for o, n, s in zip(origin, shape, scale)]
        nearest = [np.floor(c + .5).astype(np.int64) for c in coords]
        before = _nearest_before(self.volume, nearest)
        after = before if self.baseline_control else block['mask']
        labels = net_change_labels(before, after, block['valid'])
        x = normalized_input(block['image'], block['valid'], self.runtime)
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('Invalid normalized predictor input')
        size = _product(shape)
        self._reads += 1
        self._read_voxels += size
        self._peak_patch_voxels = max(self._peak_patch_voxels, size)
        metadata = {'sample_id': self.volume.sample_id, 'case_id': self.volume.case_id,
            'anatomy_family': self.volume.anatomy_family, 'proposal_index': proposal_index,
            'focus_origin_kji': self.origins_kji[proposal_index].tolist(),
            'offset_kji': list(map(int, offset_kji)), 'origin_kji': origin.tolist(),
            'shape_kji': shape.tolist(), 'spacing_ijk_mm': self.spacing_ijk_mm.tolist(),
            'baseline_control': self.baseline_control, 'baseline_row': self.volume.baseline,
            'grid_alignment': self.volume.metadata['grid_alignment'],
            'input_channels': ['normalized_attenuation', 'acquisition_validity'],
            'target_classes': {'0': 'unchanged', '1': 'added', '2': 'removed'},
            'target_semantics': 'net before/after difference after nearest-neighbor resampling'}
        return {'x': x, 'labels': labels, 'valid': block['valid'], 'metadata': metadata}

    def whole_focus(self, proposal_index):
        return self.read_patch(proposal_index, [0, 0, 0], self.shape_kji)

    def inference_tiles(self, shape_kji, overlap_kji=(0, 0, 0)):
        """Yield a deterministic complete schedule without consulting targets.

        Last tiles are shifted to end exactly at the proposal boundary. Any
        overlap must be averaged during reconstruction; it is not new evidence.
        Each record can be passed as keyword arguments to ``read_patch``.
        """
        shape = _triple(shape_kji, 'Tile shape', integer=True, positive=True)
        overlap = _triple(overlap_kji, 'Tile overlap', integer=True, nonnegative=True)
        if np.any(shape > self.shape_kji) or np.any(overlap >= shape):
            raise ValueError('Tiles must fit the focus and overlap less than their shape')
        if _product(shape) > MAX_BLOCK_VOXELS:
            raise ValueError('Tile exceeds the bounded source-reader voxel budget')
        axes = []
        for full, n, over in zip(self.shape_kji, shape, overlap):
            last = int(full - n)
            starts = list(range(0, last + 1, int(n - over)))
            if starts[-1] != last:
                starts.append(last)
            axes.append(starts)
        for index in range(len(self.origins_kji)):
            for offset in product(*axes):
                yield {'proposal_index': index, 'offset_kji': tuple(map(int, offset)),
                       'shape_kji': tuple(map(int, shape))}

    def _change_points(self, class_id, *, max_points=4096, seed=0):
        """Bounded native-change reservoir, verified on the sampled model grid."""
        key = (class_id, max_points, seed)
        if key in self._point_cache:
            return self._point_cache[key]
        if class_id not in (1, 2) or type(max_points) is not int or not 1 <= max_points <= 65536:
            raise ValueError('Invalid changed-point class or reservoir budget')
        v = self.volume
        v._check_current()
        if self.baseline_control:
            result = np.empty((0, 3), np.int64)
            self._point_cache[key] = result
            return result
        rng = np.random.default_rng(seed)
        selected, priorities = np.empty(0, np.int64), np.empty(0, np.float64)
        b, a = v.mask_before.ravel(), v.mask_after.ravel()
        limit = min(max_points * 16, 262144)
        for low in range(0, len(b), 262144):
            bb, aa = b[low:low + 262144], a[low:low + 262144]
            changed = ((aa == 1) & (bb == 0)) if class_id == 1 else ((bb == 1) & (aa == 0))
            indices = np.flatnonzero(changed) + low
            selected = np.concatenate((selected, indices))
            priorities = np.concatenate((priorities, rng.random(len(indices))))
            if len(selected) > limit:
                keep = np.argpartition(priorities, -limit)[-limit:]
                selected, priorities = selected[keep], priorities[keep]
        if not len(selected):
            result = np.empty((0, 3), np.int64)
        else:
            native = np.column_stack(np.unravel_index(selected, v.mask_before.shape)) + v.mask_origin_ijk[::-1]
            points = np.floor((native + .5) * v._native_spacing_kji / self.spacing_ijk_mm[::-1]).astype(np.int64)
            points = np.unique(points, axis=0)
            # Exactly read_block's arithmetic at nearest-neighbor ties.
            scale = self.spacing_ijk_mm[::-1] / v._native_spacing_kji
            coords = (points + .5) * scale - .5
            near = np.floor(coords + .5).astype(np.int64)
            local = near - v.mask_origin_ijk[::-1]
            ok = (np.all((coords >= -.5) & (coords < np.asarray(v.source_shape_kji) - .5), axis=1)
                  & np.all((local >= 0) & (local < v.mask_before.shape), axis=1))
            points, local = points[ok], local[ok]
            bb, aa = v.mask_before[tuple(local.T)], v.mask_after[tuple(local.T)]
            changed = ((aa == 1) & (bb == 0)) if class_id == 1 else ((bb == 1) & (aa == 0))
            result = points[changed]
            if len(result) > max_points:
                result = result[rng.choice(len(result), max_points, replace=False)]
            result = result[np.lexsort(result.T[::-1])]
        result.flags.writeable = False
        self._point_cache[key] = result
        return result

    def training_offset(self, proposal_index, shape_kji, rng, *, mode='uniform'):
        """Return a training offset; changed classes fall back to uniform if absent.

        This function deliberately uses target information. It must never choose
        an inference crop or affect the held-out evaluation tile schedule.
        """
        shape = _triple(shape_kji, 'Training patch shape', integer=True, positive=True)
        self._patch_geometry(proposal_index, [0, 0, 0], shape)
        if mode not in ('uniform', 'added', 'removed'):
            raise ValueError('Training sampling mode must be uniform, added or removed')
        maximum = self.shape_kji - shape
        if mode != 'uniform':
            points = self._change_points(1 if mode == 'added' else 2)
            local = points - self.origins_kji[proposal_index]
            local = local[np.all((local >= 0) & (local < self.shape_kji), axis=1)]
            if len(local):
                center = local[int(rng.integers(len(local)))]
                # Randomize the point's position within the patch while keeping
                # that sampled changed voxel present after edge clipping.
                jitter = np.asarray([rng.integers(int(n)) for n in shape])
                return tuple(map(int, np.clip(center - jitter, 0, maximum)))
        return tuple(int(rng.integers(int(n) + 1)) for n in maximum)

    def available_classes(self, proposal_index):
        """Training-only changed classes represented inside one predicted focus."""
        self._patch_geometry(proposal_index, [0, 0, 0], [1, 1, 1])
        classes = []
        for name, code in (('added', 1), ('removed', 2)):
            local = self._change_points(code) - self.origins_kji[proposal_index]
            if np.any(np.all((local >= 0) & (local < self.shape_kji), axis=1)):
                classes.append(name)
        return tuple(classes)

    def training_patch(self, shape_kji, rng, *, mode='uniform'):
        """Choose an eligible focus for the requested training target class.

        Returns the same geometry-only record as ``inference_tiles``. Baseline
        controls and absent classes fall back to uniform proposal/offset draws.
        Uniform draws never inspect masks or invoke the target-point reservoir.
        """
        if mode not in ('uniform', 'added', 'removed'):
            raise ValueError('Training sampling mode must be uniform, added or removed')
        self._patch_geometry(0, [0, 0, 0], shape_kji)
        candidates = list(range(len(self.origins_kji)))
        if mode != 'uniform':
            eligible = [i for i in candidates if mode in self.available_classes(i)]
            if eligible:
                candidates = eligible
        index = candidates[int(rng.integers(len(candidates)))]
        return {'proposal_index': index,
            'offset_kji': self.training_offset(index, shape_kji, rng, mode=mode),
            'shape_kji': tuple(map(int, shape_kji))}

    def stats(self):
        return {'patch_reads': self._reads, 'total_patch_voxels': self._read_voxels,
            'peak_patch_voxels': self._peak_patch_voxels,
            'focus_voxels': len(self.origins_kji) * _product(self.shape_kji),
            'cached_change_point_bytes': sum(a.nbytes for a in self._point_cache.values()),
            'input_channels': ['normalized_attenuation', 'acquisition_validity'],
            'baseline_control': self.baseline_control}

    def close(self):
        self._point_cache.clear()
        self.volume.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
