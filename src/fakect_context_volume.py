"""Read whole-body XCAT attenuation blocks with immutable sibling crop overlays.

The base ATN is memory mapped. Only the native slab needed for a requested block
is copied, and the frozen edited image (already cm^-1) is inserted before linear
interpolation. No full-body sibling image or full-body target is allocated.
Arrays and origins use k,j,i order except explicitly named ``*_ijk`` metadata.
Model and native grids share the lower face of voxel (0,0,0).
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.ndimage import affine_transform

MAX_BLOCK_VOXELS = 8_388_608
MAX_SOURCE_SLAB_VOXELS = 64_000_000
MAX_CROP_VOXELS = 32_000_000


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _triple(value, name, *, integer=False, positive=False, nonnegative=False):
    a = np.asarray(value)
    if a.shape != (3,) or (integer and a.dtype.kind not in 'iu'):
        raise ValueError(f'{name} must contain three {"integer " if integer else ""}values')
    try:
        a = a.astype(np.int64 if integer else np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid {name}') from exc
    if not np.isfinite(a).all() or (positive and np.any(a <= 0)) or (nonnegative and np.any(a < 0)):
        raise ValueError(f'Invalid {name}')
    return a


def _product(shape):
    result = 1
    for n in shape:
        result *= int(n)
    return result


def _budget(shape, maximum, name):
    if _product(shape) > maximum:
        raise ValueError(f'{name} exceeds the voxel budget ({maximum:,}); request smaller blocks')


def _stat(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def _checked_npz(path, expected_hash, verify_hashes):
    path = Path(path).expanduser().resolve()
    if not isinstance(expected_hash, str) or re.fullmatch(r'[0-9a-f]{64}', expected_hash) is None:
        raise ValueError(f'Missing or malformed checksum: {path.name}')
    stat = _stat(path)
    if verify_hashes and _sha256(path) != expected_hash:
        raise ValueError(f'Checksum changed: {path.name}')
    return path, stat


def _array_header(path, key):
    """Reject oversized arrays from their NPY header, before decompressing them."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + '.npy') as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError('Unsupported NPY header version')
    if dtype.hasobject:
        raise ValueError(f'{key} must not contain Python objects')
    return shape, dtype


def _crop_bounds(origin_ijk, shape_kji, source_shape_kji, name):
    if np.any(origin_ijk + np.asarray(shape_kji)[::-1] > source_shape_kji[::-1]):
        raise ValueError(f'{name} exceeds source bounds')


class ContextVolume:
    """One frozen sibling represented by a native base volume and local deltas.

    ``read_block`` returns float32 ``image`` in cm^-1, binary uint8 ``mask``
    and binary uint8 ``valid``. Image padding and targets outside the physical
    scan field of view are zero. Edge samples within the field of view use the
    nearest edge center for interpolation; this avoids artificial edge darkening.

    ``verify_hashes=False`` skips expensive NPZ content hashes only. Shape,
    binary-mask, provenance-stat and alignment checks remain mandatory.
    """

    def __init__(self, row, *, verify_hashes=True):
        if type(verify_hashes) is not bool:
            raise ValueError('verify_hashes must be Boolean')
        self.row = copy.deepcopy(row)
        self.sample_id = str(row['sample_id'])
        self.case_id = str(row['case_id'])
        self.anatomy_family = str(row['anatomy_family'])
        if type(row['baseline']) is not bool:
            raise ValueError('baseline must be Boolean')
        self.baseline = row['baseline']
        self.source_shape_kji = tuple(_triple(row['source_shape_kji'], 'Source shape', integer=True, positive=True))
        self.spacing_ijk_mm = tuple(_triple(row['spacing_ijk_mm'], 'Native spacing', positive=True))
        self._native_spacing_kji = np.asarray(self.spacing_ijk_mm)[::-1]
        source = row['source_atn']
        self.source_atn = Path(source['path']).expanduser().resolve()
        if (type(source['bytes']) is not int or type(source['mtime_ns']) is not int
                or source['bytes'] != 4 * _product(self.source_shape_kji) or source['mtime_ns'] <= 0):
            raise ValueError('ATN source provenance disagrees with float32 source geometry')
        self._source_stat = (source['bytes'], source['mtime_ns'])
        if _stat(self.source_atn) != self._source_stat:
            raise ValueError('ATN source stat changed')

        self.edit_crop_origin_ijk = _triple(row['edit_crop_origin_ijk'], 'Edit origin', integer=True, nonnegative=True)
        edit_shape = _triple(row['edit_crop_shape_kji'], 'Edit shape', integer=True, positive=True)
        _budget(edit_shape, MAX_CROP_VOXELS, 'Edit crop')
        _crop_bounds(self.edit_crop_origin_ijk, edit_shape, np.asarray(self.source_shape_kji), 'Edit crop')
        self.variant_npz, self._variant_stat = _checked_npz(row['variant_npz'], row['variant_npz_sha256'], verify_hashes)
        shape, dtype = _array_header(self.variant_npz, 'image')
        if shape != tuple(edit_shape) or dtype.kind != 'f':
            raise ValueError('Edited image shape/type disagrees with native edit crop')
        with np.load(self.variant_npz, allow_pickle=False) as data:
            self.edit_image = np.asarray(data['image'], dtype=np.float32)
        if not np.isfinite(self.edit_image).all():
            raise ValueError('Edited image contains nonfinite attenuation')
        self.edit_image.flags.writeable = False

        self.mask_npz, self._mask_stat = _checked_npz(row['mask_npz'], row['mask_npz_sha256'], verify_hashes)
        mask_shape, _ = _array_header(self.mask_npz, 'after')
        mask_shape = _triple(mask_shape, 'Mask shape', integer=True, positive=True)
        _budget(mask_shape, MAX_CROP_VOXELS, 'Mask crop')
        for key in ('before', 'after', 'added', 'removed'):
            shape, dtype = _array_header(self.mask_npz, key)
            if shape != tuple(mask_shape) or dtype.kind not in 'bui':
                raise ValueError(f'{key} shape/type disagrees with mask crop')
        with np.load(self.mask_npz, allow_pickle=False) as data:
            self.mask_origin_ijk = _triple(data['crop_origin_ijk'], 'Mask origin', integer=True, nonnegative=True)
            mask_spacing = _triple(data['spacing_ijk_mm'], 'Mask spacing', positive=True)
            if not np.allclose(mask_spacing, self.spacing_ijk_mm, rtol=0, atol=1e-9):
                raise ValueError('Mask spacing disagrees with native source')
            if 'sample_id' in data and (data['sample_id'].shape != () or str(data['sample_id']) != self.sample_id):
                raise ValueError('Mask sample ID disagrees with row')
            masks = {}
            for key in ('before', 'after', 'added', 'removed'):
                value = data[key]
                if not np.all((value == 0) | (value == 1)):
                    raise ValueError(f'{key} must be binary')
                masks[key] = value.astype(np.uint8)
        _crop_bounds(self.mask_origin_ijk, mask_shape, np.asarray(self.source_shape_kji), 'Mask crop')
        b, a = masks['before'].astype(bool), masks['after'].astype(bool)
        if (not np.array_equal(masks['added'], a & ~b)
                or not np.array_equal(masks['removed'], b & ~a)):
            raise ValueError('Broad before/after masks disagree with added/removed masks')
        self.mask_before, self.mask_after = masks['before'], masks['after']
        for value in (self.mask_before, self.mask_after, self.mask_origin_ijk, self.edit_crop_origin_ijk):
            value.flags.writeable = False
        self._base = np.memmap(self.source_atn, dtype='<f4', mode='r', shape=self.source_shape_kji)
        self.metadata = {
            'image_units': 'cm^-1', 'image_method': 'whole-body ATN with attenuation_copy_proxy crop overlay',
            'source_shape_kji': list(map(int, self.source_shape_kji)),
            'spacing_ijk_mm': list(map(float, self.spacing_ijk_mm)),
            'grid_alignment': 'shared lower voxel face; model center = (index + 0.5) * spacing',
            'physical_extent_ijk_mm': (np.asarray(self.source_shape_kji)[::-1] * np.asarray(self.spacing_ijk_mm)).tolist(),
            'source_integrity_scope': 'size and mtime_ns; no whole-body content hash',
        }
        self._check_current()

    def _check_current(self):
        for path, expected in ((self.source_atn, self._source_stat), (self.variant_npz, self._variant_stat),
                               (self.mask_npz, self._mask_stat)):
            if _stat(path) != expected:
                raise ValueError(f'Pinned source stat changed: {path.name}')

    def model_shape(self, target_spacing_ijk_mm):
        spacing = _triple(target_spacing_ijk_mm, 'Target spacing', positive=True)[::-1]
        shape = np.ceil(np.asarray(self.source_shape_kji) * self._native_spacing_kji / spacing)
        if np.any(shape > np.iinfo(np.int64).max):
            raise ValueError('Target grid shape is not representable')
        return tuple(int(v) for v in shape)

    def _nearest_mask(self, indices_kji):
        local = [values - int(origin) for values, origin in zip(indices_kji, self.mask_origin_ijk[::-1])]
        ok = [(v >= 0) & (v < n) for v, n in zip(local, self.mask_after.shape)]
        clip = [np.clip(v, 0, n - 1) for v, n in zip(local, self.mask_after.shape)]
        result = self.mask_after[np.ix_(*clip)].copy()
        result *= ok[0][:, None, None]
        result *= ok[1][None, :, None]
        result *= ok[2][None, None, :]
        return result

    def read_block(self, origin_kji, shape_kji, target_spacing_ijk_mm):
        origin = _triple(origin_kji, 'Block origin', integer=True)
        shape = _triple(shape_kji, 'Block shape', integer=True, positive=True)
        _budget(shape, MAX_BLOCK_VOXELS, 'Output block')
        spacing = _triple(target_spacing_ijk_mm, 'Target spacing', positive=True)[::-1]
        scale = spacing / self._native_spacing_kji
        self._check_current()
        coords = [(int(o) + np.arange(int(n), dtype=np.float64) + .5) * s - .5
                  for o, n, s in zip(origin, shape, scale)]
        valid_axes = [(c >= -.5) & (c < n - .5) for c, n in zip(coords, self.source_shape_kji)]
        valid = (valid_axes[0][:, None, None] & valid_axes[1][None, :, None]
                 & valid_axes[2][None, None, :]).astype(np.uint8)
        if not np.any(valid):
            return {'image': np.zeros(tuple(shape), np.float32), 'mask': np.zeros(tuple(shape), np.uint8),
                    'valid': valid}
        nearest = [np.floor(c + .5).astype(np.int64) for c in coords]
        mask = self._nearest_mask(nearest)
        mask *= valid
        # Compute the native slab only from in-FOV sample centers. Include both
        # linear-interpolation neighbors; an edit crop boundary is not an image boundary.
        lo = np.array([max(0, int(np.floor(c[ok].min()))) for c, ok in zip(coords, valid_axes)])
        hi = np.array([min(n, int(np.ceil(c[ok].max())) + 1)
                       for c, ok, n in zip(coords, valid_axes, self.source_shape_kji)])
        _budget(hi - lo, MAX_SOURCE_SLAB_VOXELS, 'Native source slab')
        slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        slab = np.array(self._base[slices], dtype=np.float32, copy=True)
        if not np.isfinite(slab).all():
            raise ValueError('Source slab contains nonfinite attenuation')
        slab *= np.float32(10. / self.spacing_ijk_mm[0])
        edit_lo = self.edit_crop_origin_ijk[::-1]
        edit_hi = edit_lo + np.asarray(self.edit_image.shape)
        overlap_lo, overlap_hi = np.maximum(lo, edit_lo), np.minimum(hi, edit_hi)
        if np.all(overlap_hi > overlap_lo):
            dst = tuple(slice(int(l - o), int(h - o)) for l, h, o in zip(overlap_lo, overlap_hi, lo))
            src = tuple(slice(int(l - o), int(h - o)) for l, h, o in zip(overlap_lo, overlap_hi, edit_lo))
            slab[dst] = self.edit_image[src]
        offset = (origin.astype(np.float64) + .5) * scale - .5 - lo
        image = affine_transform(slab, np.diag(scale), offset=offset, output_shape=tuple(shape),
                                 order=1, mode='nearest', prefilter=False, output=np.float32)
        image *= valid
        self._check_current()
        return {'image': image, 'mask': mask, 'valid': valid}

    def foreground_points(self, target_spacing_ijk_mm, max_points=4096, seed=0):
        """Deterministic foreground-centered sampling candidates in model k,j,i.

        A bounded uniform native-foreground reservoir is mapped to the model
        grid, deduplicated, and checked against nearest-neighbor target sampling.
        Therefore thin targets erased by coarse sampling are not falsely offered
        as foreground. May return fewer than ``max_points`` distinct locations.
        """
        if type(max_points) is not int or max_points <= 0 or max_points > 1_000_000:
            raise ValueError('max_points must be between 1 and 1,000,000')
        spacing = _triple(target_spacing_ijk_mm, 'Target spacing', positive=True)[::-1]
        self._check_current()
        rng = np.random.default_rng(seed)
        limit = min(max_points * 16, 1_000_000)
        selected = np.empty(0, np.int64)
        priorities = np.empty(0, np.float64)
        flat = self.mask_after.ravel()
        for low in range(0, len(flat), 262144):
            indices = np.flatnonzero(flat[low:low + 262144]) + low
            keys = rng.random(len(indices))
            selected = np.concatenate((selected, indices))
            priorities = np.concatenate((priorities, keys))
            if len(selected) > limit:
                keep = np.argpartition(priorities, -limit)[-limit:]
                selected, priorities = selected[keep], priorities[keep]
        if not len(selected):
            return np.empty((0, 3), np.int64)
        native = np.column_stack(np.unravel_index(selected, self.mask_after.shape)) + self.mask_origin_ijk[::-1]
        points = np.floor((native + .5) * self._native_spacing_kji / spacing).astype(np.int64)
        points = np.unique(points, axis=0)
        # Verify proposals using exactly read_block's arithmetic. Rearranging
        # multiply/divide operations can select a neighboring voxel at a tie.
        scale = spacing / self._native_spacing_kji
        coordinates = (points + .5) * scale - .5
        nearest = np.floor(coordinates + .5).astype(np.int64)
        local = nearest - self.mask_origin_ijk[::-1]
        valid_fov = np.all((coordinates >= -.5) &
                           (coordinates < np.asarray(self.source_shape_kji) - .5), axis=1)
        keep = valid_fov & np.all((local >= 0) & (local < np.asarray(self.mask_after.shape)), axis=1)
        loc = local[keep]
        points = points[keep][self.mask_after[tuple(loc.T)].astype(bool)]
        if len(points) > max_points:
            points = points[rng.choice(len(points), max_points, replace=False)]
        return points[np.lexsort(points.T[::-1])]

    def close(self):
        """Release the read-only base-volume mapping."""
        if getattr(self, '_base', None) is not None:
            self._base._mmap.close()
            self._base = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
