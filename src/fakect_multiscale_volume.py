"""Bounded, anti-aliased anatomical overviews of compact FakeCT siblings.

``build_overview`` pools native float32 XCAT attenuation into integer box cells.
Cells share the native grid's lower faces. Partial edge cells are averaged over
only their in-FOV voxels; the rest of the requested shape is zero padding.
``valid_fraction`` describes that distinction explicitly. No anatomy masks or
complete native sibling image are loaded.

The immutable cache stores float64 base sums, not learned features. A streamed
crop difference is added before division and any downstream normalization.
Every changed native voxel contributes to its whole box cell, including cells
that cross the crop boundary. Float64 summation gives composed-image agreement
within ordinary float32 rounding, not a guarantee of bitwise network equality.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

import numpy as np

SCHEMA = 'fakect.multiscale-overview/1'
MAX_OVERVIEW_VOXELS = 2_000_000
MAX_CHUNK_VOXELS = 4_000_000
MAX_CROP_VOXELS = 32_000_000
FILTER = 'integer-box-sum-float64/1'


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _stat(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _triple(value, name, *, integer=False, zero=False):
    try:
        values = list(value)
        if len(values) != 3:
            raise ValueError()
        if integer:
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
                raise ValueError()
            result = tuple(int(v) for v in values)
            if any(v > np.iinfo(np.int32).max for v in result):
                raise ValueError()
        else:
            if any(isinstance(v, (bool, np.bool_)) for v in values):
                raise ValueError()
            result = tuple(float(v) for v in values)
        if any(not math.isfinite(v) or (v < 0 if zero else v <= 0) for v in result):
            raise ValueError()
        return result
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must contain three {"integer " if integer else ""}{"nonnegative" if zero else "positive"} values') from None


def _checksum(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'Missing or malformed {name} checksum')
    return value


def _header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError('Unsupported NPY header version')


@contextlib.contextmanager
def _image_stream(path, expected_shape):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count('image.npy') != 1:
            raise ValueError('Variant must contain exactly one image.npy')
        with archive.open('image.npy') as stream:
            shape, fortran, dtype = _header(stream)
            if tuple(shape) != expected_shape or dtype.kind != 'f' or dtype.itemsize not in (2, 4, 8):
                raise ValueError('Edited image shape/type disagrees with native edit crop')
            if archive.getinfo('image.npy').file_size != stream.tell() + math.prod(shape) * dtype.itemsize:
                raise ValueError('Edited image payload length disagrees with header')
            yield stream, dtype, fortran


def _chunks(shape, budget, *, fortran=False):
    """Yield C- or F-storage-contiguous boxes, each no larger than budget."""
    ordered = shape[::-1] if fortran else shape
    i = min(ordered[2], budget)
    j = min(ordered[1], budget // i)
    k = min(ordered[0], budget // (i * j))
    for a in range(0, ordered[0], k):
        for b in range(0, ordered[1], j):
            for c in range(0, ordered[2], i):
                starts = (a, b, c)
                stops = tuple(min(o + n, end) for o, n, end in zip(starts, (k, j, i), ordered))
                if fortran:
                    starts, stops = starts[::-1], stops[::-1]
                yield starts, stops


def _add_pooled(sums, values, native_start, factors):
    """Reduce a bounded chunk into all intersected box cells, then accumulate."""
    reduced = values
    for axis, (offset, factor) in enumerate(zip(native_start, factors)):
        n = reduced.shape[axis]
        first_boundary = factor - offset % factor
        starts = np.r_[0, np.arange(first_boundary, n, factor, dtype=np.int64)]
        reduced = np.add.reduceat(reduced, starts, axis=axis, dtype=np.float64)
    lower = tuple(o // f for o, f in zip(native_start, factors))
    out = tuple(slice(o, o + n) for o, n in zip(lower, reduced.shape))
    sums[out] += reduced


def _read_source(base, lo, hi, scale, stats):
    shape = tuple(h - l for l, h in zip(lo, hi))
    data = np.array(base[tuple(slice(l, h) for l, h in zip(lo, hi))], dtype=np.float32, copy=True)
    # The copy owns its memory; release mapped source pages so a sequential
    # whole-body pass does not retain gigabytes of file-backed RSS on Linux.
    if hasattr(base._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
        base._mmap.madvise(mmap.MADV_DONTNEED)
    if not np.isfinite(data).all():
        raise ValueError('Source chunk contains nonfinite attenuation')
    data *= scale
    if not np.isfinite(data).all():
        raise ValueError('Converted source attenuation is not finite')
    count = math.prod(shape)
    stats['source_voxels_read'] += count
    stats['source_bytes_read'] += count * 4
    stats['peak_source_chunk_voxels'] = max(stats['peak_source_chunk_voxels'], count)
    return data


def _load_cache(directory, identity, shape, stats):
    manifest_path, sums_path = directory / 'metadata.json', directory / 'base-sums.npy'
    try:
        if manifest_path.stat().st_size > 32_768:
            raise ValueError('Oversized overview cache manifest')
        before = (_stat(manifest_path), _stat(sums_path))
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('identity') != identity or manifest.get('schema') != SCHEMA:
            raise ValueError('Overview cache identity changed')
        if _sha256(sums_path) != _checksum(manifest.get('sums_sha256'), 'Cache'):
            raise ValueError('Overview cache checksum changed')
        with sums_path.open('rb') as stream:
            header_shape, fortran, dtype = _header(stream)
            if tuple(header_shape) != shape or dtype != np.dtype('float64') or fortran:
                raise ValueError('Overview cache array shape/type changed')
            if sums_path.stat().st_size != stream.tell() + math.prod(shape) * 8:
                raise ValueError('Overview cache array length changed')
        sums = np.load(sums_path, allow_pickle=False)
        if not np.isfinite(sums).all():
            raise ValueError('Nonfinite overview cache')
        if before != (_stat(manifest_path), _stat(sums_path)):
            raise ValueError('Overview cache changed during load')
        stats['base_cache_hit'] = True
        stats['base_cache_bytes_read'] = sums_path.stat().st_size
        return sums
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Incomplete or invalid overview cache: {directory}') from exc


def _save_cache(directory, identity, sums):
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.overview-', dir=directory.parent))
    try:
        np.save(temporary / 'base-sums.npy', sums, allow_pickle=False)
        metadata = dict(schema=SCHEMA, identity=identity, sums_sha256=_sha256(temporary / 'base-sums.npy'))
        (temporary / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
        try:
            os.rename(temporary, directory)
        except OSError:
            # Another process may have published the same immutable base. Its
            # contents are checked by the caller; never replace an existing one.
            if not directory.is_dir():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_overview(row, cache_directory, shape_kji=(160, 96, 96),
                   max_chunk_voxels=4_000_000, verify_hashes=True):
    """Return a fixed-shape sibling overview without a full native allocation.

    ``row`` follows the frozen context-dataset manifest. Only its source and
    variant fields are required; masks are intentionally never opened.
    ``image`` is float32 cm^-1. ``valid`` is one for any occupied box cell;
    ``valid_fraction`` is its in-FOV fraction (zero for padded cells). Integer
    ``native_factors_kji`` multiply native spacing to give nominal grid spacing.
    The lower origin is zero and ``center_axes_ijk_mm`` gives true in-FOV cell
    centroids, including partial boundary cells, and nominal padded centers.

    Hash verification covers the crop before and after use. Legacy raw sources
    are pinned by size and mtime; if source_atn.sha256 is supplied it is also
    checked before and after use. Derived cache hashes are always checked.
    Chunk budgets limit each image/chunk/reduction array, not total process RSS;
    returned metadata declares a conservative temporary-array estimate.
    """
    started = time.monotonic()
    if type(verify_hashes) is not bool:
        raise ValueError('verify_hashes must be Boolean')
    if type(max_chunk_voxels) is not int or not 1 <= max_chunk_voxels <= MAX_CHUNK_VOXELS:
        raise ValueError(f'max_chunk_voxels must be between 1 and {MAX_CHUNK_VOXELS:,}')
    output_shape = _triple(shape_kji, 'Overview shape', integer=True)
    if math.prod(output_shape) > MAX_OVERVIEW_VOXELS:
        raise ValueError(f'Overview exceeds the voxel budget ({MAX_OVERVIEW_VOXELS:,})')
    native_shape = _triple(row['source_shape_kji'], 'Source shape', integer=True)
    spacing = _triple(row['spacing_ijk_mm'], 'Native spacing')
    if row.get('image_units', 'cm^-1') != 'cm^-1' or row.get('source_atn_units', 'per_pixel') != 'per_pixel':
        raise ValueError('Unsupported attenuation units')
    if row.get('source_atn_dtype', '<f4') != '<f4':
        raise ValueError('Raw source must be little-endian float32')
    source = row['source_atn']
    source_path = Path(source['path']).expanduser().resolve()
    if (type(source['bytes']) is not int or source['bytes'] != 4 * math.prod(native_shape)
            or type(source['mtime_ns']) is not int or source['mtime_ns'] <= 0):
        raise ValueError('ATN source provenance disagrees with float32 source geometry')
    source_stat = (source['bytes'], source['mtime_ns'])
    source_hash = _checksum(source['sha256'], 'Source') if 'sha256' in source else None
    crop_shape = _triple(row['edit_crop_shape_kji'], 'Edit shape', integer=True)
    crop_origin = _triple(row['edit_crop_origin_ijk'], 'Edit origin', integer=True, zero=True)[::-1]
    if math.prod(crop_shape) > MAX_CROP_VOXELS:
        raise ValueError(f'Edit crop exceeds the voxel budget ({MAX_CROP_VOXELS:,})')
    if any(o + n > end for o, n, end in zip(crop_origin, crop_shape, native_shape)):
        raise ValueError('Edit crop exceeds source bounds')
    variant_path = Path(row['variant_npz']).expanduser().resolve()
    variant_hash = _checksum(row['variant_npz_sha256'], 'Variant')
    variant_stat = _stat(variant_path)
    factors = tuple((n + limit - 1) // limit for n, limit in zip(native_shape, output_shape))
    occupied = tuple((n + f - 1) // f for n, f in zip(native_shape, factors))
    scale = np.float32(10. / spacing[0])
    if not np.isfinite(scale):
        raise ValueError('Native spacing produces nonfinite unit conversion')

    def check_current(*, hashes=False):
        if _stat(source_path) != source_stat:
            raise ValueError('ATN source stat changed')
        if _stat(variant_path) != variant_stat:
            raise ValueError('Variant stat changed')
        if hashes and verify_hashes:
            if _sha256(variant_path) != variant_hash:
                raise ValueError('Variant checksum changed')
            if source_hash is not None and _sha256(source_path) != source_hash:
                raise ValueError('ATN source checksum changed')
        if _stat(source_path) != source_stat or _stat(variant_path) != variant_stat:
            raise ValueError('Source changed during integrity verification')

    check_current(hashes=True)
    # Validate the compressed array header before opening a raw map or building
    # any cache. Streaming below supports both C and Fortran stored arrays.
    with _image_stream(variant_path, crop_shape):
        pass
    identity = dict(schema=SCHEMA, filter=FILTER, source_path=str(source_path),
                    source_bytes=source_stat[0], source_mtime_ns=source_stat[1], source_sha256=source_hash,
                    source_shape_kji=list(native_shape), native_spacing_ijk_mm=list(spacing),
                    shape_kji=list(output_shape), factors_kji=list(factors), image_units='cm^-1',
                    boundary='divide by actual in-FOV voxel count; pad zero', sum_dtype='float64')
    key = _identity(identity)
    cache_path = Path(cache_directory).expanduser().resolve() / ('base-' + key)
    stats = dict(base_cache_hit=False, base_cache_bytes_read=0, source_bytes_read=0,
                 source_voxels_read=0, peak_source_chunk_voxels=0, overlay_voxels_read=0,
                 peak_overlay_chunk_voxels=0, max_chunk_voxels=max_chunk_voxels)
    base = np.memmap(source_path, dtype='<f4', mode='r', shape=native_shape)
    try:
        if cache_path.exists():
            sums = _load_cache(cache_path, identity, output_shape, stats)
        else:
            sums = np.zeros(output_shape, np.float64)
            for lo, hi in _chunks(native_shape, max_chunk_voxels):
                values = _read_source(base, lo, hi, scale, stats)
                _add_pooled(sums, values, lo, factors)
            del values
            check_current(hashes=True)
            _save_cache(cache_path, identity, sums)
            # Verify even a newly published or concurrently created cache.
            verified = _load_cache(cache_path, identity, output_shape, {})
            if not np.allclose(verified, sums, rtol=1e-12, atol=1e-12):
                raise ValueError('Concurrent overview cache has inconsistent sums')
            # Concurrent builders may use different summation chunk boundaries.
            # Use the canonical published sums after checking numerical parity.
            sums = verified
        with _image_stream(variant_path, crop_shape) as (stream, dtype, fortran):
            for lo, hi in _chunks(crop_shape, max_chunk_voxels, fortran=fortran):
                chunk_shape = tuple(h - l for l, h in zip(lo, hi))
                count = math.prod(chunk_shape)
                payload = stream.read(count * dtype.itemsize)
                if len(payload) != count * dtype.itemsize:
                    raise ValueError('Truncated edited image payload')
                overlay = np.frombuffer(payload, dtype=dtype).reshape(chunk_shape, order='F' if fortran else 'C')
                overlay = np.asarray(overlay, dtype=np.float32)
                if not np.isfinite(overlay).all():
                    raise ValueError('Edited image contains nonfinite attenuation')
                native_lo = tuple(o + l for o, l in zip(crop_origin, lo))
                native_hi = tuple(o + h for o, h in zip(crop_origin, hi))
                source_values = _read_source(base, native_lo, native_hi, scale, stats)
                difference = np.subtract(overlay, source_values, dtype=np.float64)
                _add_pooled(sums, difference, native_lo, factors)
                stats['overlay_voxels_read'] += count
                stats['peak_overlay_chunk_voxels'] = max(stats['peak_overlay_chunk_voxels'], count)
            del payload, overlay, source_values, difference
        check_current(hashes=True)
    finally:
        base._mmap.close()

    widths = [np.clip(n - np.arange(out, dtype=np.int64) * f, 0, f)
              for n, f, out in zip(native_shape, factors, output_shape)]
    counts = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
    valid = (counts > 0).astype(np.uint8)
    np.divide(sums, counts, out=sums, where=counts > 0)
    sums[counts == 0] = 0
    image = sums.astype(np.float32)
    fraction = (counts / float(math.prod(factors))).astype(np.float32)
    centers = [(np.arange(out) * f + np.where(width > 0, width, f) / 2) * s
               for out, f, width, s in zip(output_shape, factors, widths, spacing[::-1])]
    composition = dict(base_cache_key=key, variant_path=str(variant_path), variant_sha256=variant_hash,
                       variant_stat=list(variant_stat), edit_crop_origin_ijk=list(crop_origin[::-1]),
                       edit_crop_shape_kji=list(crop_shape))
    stats['elapsed_seconds'] = time.monotonic() - started
    stats['temporary_array_budget_bytes'] = max_chunk_voxels * 48 + math.prod(output_shape) * 64
    stats['output_array_bytes'] = image.nbytes + valid.nbytes + fraction.nbytes
    return dict(image=image, valid=valid, valid_fraction=fraction,
                spacing_ijk_mm=[s * f for s, f in zip(spacing, factors[::-1])],
                native_factors_kji=list(factors), occupied_shape_kji=list(occupied),
                physical_extent_ijk_mm=[s * n for s, n in zip(spacing, native_shape[::-1])],
                center_axes_ijk_mm=centers[::-1], cache_key=key, composition_key=_identity(composition),
                cache_directory=str(cache_path), stats=stats,
                metadata=dict(schema=SCHEMA, image_units='cm^-1', filter=FILTER,
                              native_spacing_ijk_mm=list(spacing), source_shape_kji=list(native_shape),
                              grid_origin_lower_face_ijk_mm=[0., 0., 0.],
                              coordinate_rule='box lower face = index * nominal spacing; partial last cell centroid is clipped to in-FOV support',
                              padding_rule='zero outside scan; partial boxes normalized over in-FOV voxels',
                              source_integrity_scope='sha256, size and mtime_ns' if source_hash and verify_hashes else 'size and mtime_ns; no verified whole-body content hash',
                              variant_hash_verified=verify_hashes, composition=composition))
