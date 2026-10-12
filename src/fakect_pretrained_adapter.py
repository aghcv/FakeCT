"""Bounded native CT-to-HU/NIfTI adapter for an ROI-assisted teacher benchmark.

Only a small authorized pilot is exported.  Shared native ATN files retain the
whole-body context; ContextVolume inserts each frozen local sibling overlay.
No whole-body sibling image is created.  The atlas thoracic slab determines a
full-X/Y image crop plus physical z context.  Target and scoring-domain masks
retain the stricter T12--T1 slab, independently of that context.

NIfTI-1 stores i fastest, matching C-order native k,j,i bytes; dimensions in
the header are i,j,k.  Both qform and sform describe RAS voxel CENTER positions.
Format authority: https://nifti.nimh.nih.gov/nifti-1/documentation/faq.html
and https://github.com/NIFTI-Imaging/nifti_clib/blob/master/nifti2/nifti1.h
"""
from __future__ import annotations

from contextlib import ExitStack
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from fakect_context_volume import ContextVolume, MAX_BLOCK_VOXELS, _array_header


SCHEMA = 'fakect.pretrained-ct-adapter/1'
MAX_EXPORT_VOXELS = 320_000_000


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _triple(value, name, *, integer=False, positive=False):
    a = np.asarray(value)
    if a.shape != (3,) or (integer and a.dtype.kind not in 'iu'):
        raise ValueError(name + ' must contain three valid numbers')
    a = a.astype(np.int64 if integer else float)
    if not np.all(np.isfinite(a)) or (positive and np.any(a <= 0)):
        raise ValueError(name + ' must be finite and valid')
    return a


def _checked(path, digest, verify):
    path = Path(path).resolve()
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('Missing frozen adapter input checksum')
    if verify and _sha(path) != digest:
        raise ValueError('Frozen adapter input changed: ' + path.name)
    return path


def attenuation_to_hu(attenuation_cm_inverse, mu_water_cm_inverse):
    """Calibrated HU = 1000*(mu/mu_water - 1); no guessed tissue rescaling."""
    water = float(mu_water_cm_inverse)
    if not np.isfinite(water) or water <= 0:
        raise ValueError('Water attenuation must be finite and positive')
    image = np.asarray(attenuation_cm_inverse, dtype=np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError('CT attenuation must be finite')
    result = image.copy()
    result /= np.float32(water)
    result -= np.float32(1.)
    result *= np.float32(1000.)
    if not np.all(np.isfinite(result)):
        raise ValueError('Calibrated HU is not finite')
    return result


def crop_ras_affine(evidence, origin_ijk):
    """Translate the verified full-volume voxel-center affine for a crop."""
    affine = np.asarray(evidence['orientation']['affine_ras_ijk'], dtype=float)
    origin = _triple(origin_ijk, 'Crop origin', integer=True)
    if (affine.shape != (4, 4) or not np.all(np.isfinite(affine)) or
            not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-9)):
        raise ValueError('Require a finite homogeneous verified RAS affine')
    shift = np.eye(4)
    shift[:3, 3] = origin
    return affine @ shift


def _slab_indices(selection, evidence, spacing, source_k, context_mm):
    slab = np.asarray(selection['z_slab_mm'], dtype=float)
    boundary = _triple(evidence['orientation']['grid_boundary_origin_raw_mm'], 'RAW boundary')
    if slab.shape != (2,) or not np.all(np.isfinite(slab)) or slab[0] >= slab[1]:
        raise ValueError('Require a finite ordered atlas thoracic slab')
    lo = max(0, int(math.ceil((slab[0] - context_mm - boundary[2]) / spacing[2] - .5)))
    hi = min(int(source_k), int(math.floor((slab[1] + context_mm - boundary[2]) / spacing[2] - .5)) + 1)
    if lo >= hi:
        raise ValueError('Atlas thoracic slab does not overlap the native CT')
    return lo, hi


def plan_thorax_crop(row, selection_manifest, evidence, *, context_mm=50.,
                     max_voxels=MAX_EXPORT_VOXELS):
    """Full-X/Y native image crop with a fixed atlas thoracic z slab + context."""
    context_mm = float(context_mm)
    if not np.isfinite(context_mm) or not 0 <= context_mm <= 100:
        raise ValueError('Thoracic z context must be in [0,100] mm')
    if type(max_voxels) is not int or not 1 <= max_voxels <= MAX_EXPORT_VOXELS:
        raise ValueError('Invalid bounded export voxel budget')
    if row.get('image_units') != 'cm^-1' or row.get('source_atn_units') != 'per_pixel':
        raise ValueError('Require physical cm^-1 overlay and per-pixel native ATN units')
    shape = _triple(row['source_shape_kji'], 'Source shape', integer=True, positive=True)
    spacing = _triple(row['spacing_ijk_mm'], 'Native spacing', positive=True)
    calibrated_spacing = _triple(evidence['orientation']['spacing_ijk_mm'], 'Evidence spacing', positive=True)
    if not np.allclose(spacing, calibrated_spacing, rtol=0, atol=1e-9):
        raise ValueError('Evidence and native CT spacing disagree')
    if (evidence['orientation'].get('status') != 'supported_by_generator_notes_and_named_surface_landmarks' or
            list(evidence['orientation'].get('shape_kji', [])) != list(shape)):
        raise ValueError('Require verified orientation evidence matching the full native shape')
    scale = float(evidence['calibration']['source_atn_to_cm_inverse_scale'])
    water = float(evidence['calibration']['mu_water_cm_inverse'])
    if not np.isfinite(scale) or not np.isclose(scale, 10 / spacing[0], rtol=1e-9, atol=1e-9):
        raise ValueError('Evidence per-pixel attenuation conversion disagrees with source')
    if not np.isfinite(water) or water <= 0:
        raise ValueError('Invalid calibrated water attenuation')
    if str(evidence['case_id']) != str(row['case_id']):
        raise ValueError('Calibration belongs to a different phantom')
    lo, hi = _slab_indices(selection_manifest, evidence, spacing, shape[0], context_mm)
    crop_shape = [hi - lo, int(shape[1]), int(shape[2])]
    if math.prod(crop_shape) > max_voxels or max(crop_shape) > 32767:
        raise ValueError('Thoracic export exceeds the bounded NIfTI-1 voxel/dimension budget')
    origin = [0, 0, lo]
    affine = crop_ras_affine(evidence, origin)
    if not np.allclose(np.linalg.norm(affine[:3, :3], axis=0), spacing, rtol=0, atol=1e-6):
        raise ValueError('Verified RAS affine scale disagrees with native spacing')
    return {'schema_version': SCHEMA, 'sample_id': row['sample_id'], 'case_id': str(row['case_id']),
        'shape_kji': crop_shape, 'shape_ijk': crop_shape[::-1], 'origin_ijk': origin,
        'spacing_ijk_mm': spacing.tolist(), 'affine_ras_ijk': affine.tolist(),
        'image_units': 'HU', 'mu_water_cm_inverse': water,
        'context_mm': context_mm, 'atlas_z_slab_mm': list(selection_manifest['z_slab_mm']),
        'crop_scope': 'ROI-assisted atlas thoracic T12--T1 slab plus z context; full native X/Y',
        'resampling': 'none; native resolution and center alignment retained',
        'hu_formula': '1000 * (attenuation_cm_inverse / mu_water_cm_inverse - 1)',
        'orientation_evidence': evidence['orientation'], 'calibration_evidence': evidence['calibration']}


def _paste(destination, source, source_origin_ijk, destination_origin_ijk):
    """Paste only native-grid overlap, with origins explicitly in i,j,k."""
    src_lo, dst_lo = np.asarray(source_origin_ijk)[::-1], np.asarray(destination_origin_ijk)[::-1]
    lo = np.maximum(src_lo, dst_lo)
    hi = np.minimum(src_lo + source.shape, dst_lo + destination.shape)
    if np.all(hi > lo):
        src = tuple(slice(int(l-o), int(h-o)) for l,h,o in zip(lo,hi,src_lo))
        dst = tuple(slice(int(l-o), int(h-o)) for l,h,o in zip(lo,hi,dst_lo))
        destination[dst] = source[src]


class ThoracicCoreProvider:
    """Native baseline/sibling aorta-core truth with branches excluded.

    Atlas core voxels remain outside the edit crop. Inside it, retained original
    core voxels and their inherited descendants define current core membership.
    Every newly introduced core voxel must inherit an original selected core
    anchor through frozen target_origin_index. Original disconnected islands
    sharing an aorta ID remain excluded, preserving the atlas component policy.
    The original tight X/Y atlas crop does not clip permitted core growth.
    """
    def __init__(self, row, selection_manifest, evidence, *, verify_hashes=True):
        self.row = dict(row)
        self.evidence = evidence
        self.selection = selection_manifest
        self.spacing = _triple(row['spacing_ijk_mm'], 'Native spacing', positive=True)
        self.source_shape = _triple(row['source_shape_kji'], 'Native shape', integer=True, positive=True)
        self.domain_k = _slab_indices(selection_manifest, evidence, self.spacing, self.source_shape[0], 0.)
        selection_path = Path(row['selection_manifest']).resolve()
        if verify_hashes and _sha(selection_path) != row['selection_manifest_sha256']:
            raise ValueError('Frozen thoracic anatomy selection changed')
        target_path = Path(selection_manifest['mask_npz'])
        if not target_path.is_absolute():
            target_path = selection_path.parent / target_path
        target_path = _checked(target_path, selection_manifest['mask_npz_sha256'], verify_hashes)
        if math.prod(_array_header(target_path, 'core_mask')[0]) > 32_000_000:
            raise ValueError('Atlas core crop exceeds the bounded mask budget')
        with np.load(target_path, allow_pickle=False) as data:
            raw_core = data['core_mask']
            if raw_core.dtype.kind not in 'bui' or not np.all((raw_core == 0) | (raw_core == 1)):
                raise ValueError('Atlas aorta core mask must be binary')
            self.baseline_core = np.asarray(raw_core, dtype=np.uint8)
            self.core_origin = _triple(data['crop_origin_ijk'], 'Core origin', integer=True)
            if (list(data['source_shape_kji']) != list(self.source_shape) or
                    not np.allclose(data['spacing_ijk_mm'], self.spacing, rtol=0, atol=1e-9)):
                raise ValueError('Atlas core mask geometry disagrees with native source')
        if not np.all((self.baseline_core == 0) | (self.baseline_core == 1)):
            raise ValueError('Atlas aorta core mask must be binary')
        self.edit_origin = _triple(row['edit_crop_origin_ijk'], 'Edit origin', integer=True)
        edit_shape = tuple(_triple(row['edit_crop_shape_kji'], 'Edit shape', integer=True, positive=True))
        if math.prod(edit_shape) > 32_000_000:
            raise ValueError('Core transfer edit crop exceeds the bounded mask budget')
        anchors = np.asarray(selection_manifest['anchor_source_ids'], dtype=np.int64)
        if anchors.ndim != 1 or len(anchors) == 0:
            raise ValueError('Atlas aorta anchor IDs are required for core truth')
        source_path = _checked(row['source_npz'], row['source_npz_sha256'], verify_hashes)
        with np.load(source_path, allow_pickle=False) as data:
            original = data['original_labels']
            source_geometry = json.loads(str(data['geometry_json'].item()))
        if original.shape != edit_shape:
            raise ValueError('Original label crop disagrees with transfer geometry')
        if (list(source_geometry['crop_origin_ijk']) != list(self.edit_origin) or
                list(source_geometry['crop_shape_kji']) != list(edit_shape) or
                not np.allclose(source_geometry['spacing_ijk_mm'], self.spacing, rtol=0, atol=1e-9)):
            raise ValueError('Original label crop origin/spacing disagrees with transfer geometry')
        original_anchor = np.isin(original, anchors)
        atlas_reference = np.zeros(edit_shape, dtype=np.uint8)
        _paste(atlas_reference, self.baseline_core, self.core_origin, self.edit_origin)
        self._clamp_domain(atlas_reference, self.edit_origin)
        if not np.all(original_anchor[atlas_reference.astype(bool)]):
            raise ValueError('Original anchor labels do not match the frozen selected thoracic core')
        variant_path = _checked(row['variant_npz'], row['variant_npz_sha256'], verify_hashes)
        with np.load(variant_path, allow_pickle=False) as data:
            labels = data['edited_labels']
            origins = data['target_origin_index']
        if labels.shape != edit_shape or origins.shape != edit_shape or origins.dtype.kind not in 'iu':
            raise ValueError('Edited labels/ancestry disagree with native transfer geometry')
        current_anchor = np.isin(labels, anchors)
        added_anchor = current_anchor & ~original_anchor
        inherited = origins[added_anchor].astype(np.int64)
        if (np.any(inherited < 0) or np.any(inherited >= original.size) or
                not np.all(original_anchor.ravel()[inherited])):
            raise ValueError('New aortic anchor voxels do not inherit original aorta ancestry')
        # Whole-ID membership alone would reincorporate tiny disconnected aorta
        # islands intentionally omitted by the frozen largest-component core.
        valid_origins = (origins >= 0) & (origins < original.size)
        inherited_core = np.zeros(edit_shape, dtype=bool)
        inherited_core[valid_origins] = atlas_reference.ravel()[origins[valid_origins]].astype(bool)
        current = current_anchor & (atlas_reference.astype(bool) | inherited_core)
        added = current & ~atlas_reference.astype(bool)
        self.current_edit_core = current.astype(np.uint8)
        self._clamp_domain(self.current_edit_core, self.edit_origin)
        if row['baseline'] and not np.array_equal(self.current_edit_core, atlas_reference):
            raise ValueError('Baseline core transfer introduced geometric changes')
        self.baseline_core.flags.writeable = False
        self.current_edit_core.flags.writeable = False
        self.metadata = {'semantics': 'atlas thoracic aorta core; proximal branches excluded',
            'anchor_source_ids': anchors.tolist(), 'atlas_z_slab_mm': list(selection_manifest['z_slab_mm']),
            'core_xy_growth_policy': 'edit-derived core can extend beyond original tight atlas X/Y bounds',
            'new_core_ancestry_checked': int(np.sum(added)), 'truth_resolution': 'native',
            'original_selected_core_anchor_labels_checked': True,
            'original_anchor_voxels_outside_selected_core_in_edit_crop': int(np.sum(original_anchor & ~atlas_reference.astype(bool))),
            'mask_sha256': selection_manifest['mask_npz_sha256']}

    def _clamp_domain(self, mask, origin_ijk):
        ks = np.arange(mask.shape[0]) + int(origin_ijk[2])
        mask *= ((ks >= self.domain_k[0]) & (ks < self.domain_k[1]))[:, None, None]

    def read_block(self, origin_ijk, shape_kji):
        origin = _triple(origin_ijk, 'Target block origin', integer=True)
        shape = tuple(_triple(shape_kji, 'Target block shape', integer=True, positive=True))
        if math.prod(shape) > MAX_BLOCK_VOXELS:
            raise ValueError('Target block exceeds bounded read budget')
        before = np.zeros(shape, dtype=np.uint8)
        _paste(before, self.baseline_core, self.core_origin, origin)
        self._clamp_domain(before, origin)
        after = before.copy()
        _paste(after, self.current_edit_core, self.edit_origin, origin)
        self._clamp_domain(after, origin)
        domain = np.zeros(shape, dtype=np.uint8)
        ks = np.arange(shape[0]) + int(origin[2])
        js = np.arange(shape[1]) + int(origin[1])
        is_ = np.arange(shape[2]) + int(origin[0])
        domain[:] = (((ks >= self.domain_k[0]) & (ks < self.domain_k[1]))[:,None,None]
                     & ((js >= 0) & (js < self.source_shape[1]))[None,:,None]
                     & ((is_ >= 0) & (is_ < self.source_shape[2]))[None,None,:])
        return {'target': after, 'baseline': before, 'domain': domain,
                'changed': np.logical_xor(after, before).astype(np.uint8)}


def _nifti_header(shape_kji, affine, dtype, description):
    shape = _triple(shape_kji, 'NIfTI shape', integer=True, positive=True)
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError('NIfTI affine must be finite 4x4')
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError('NIfTI affine has invalid voxel scales')
    rotation = affine[:3, :3] / spacing
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6):
        raise ValueError('NIfTI qform cannot represent a sheared affine')
    qfac = 1. if np.linalg.det(rotation) > 0 else -1.
    rotation[:, 2] *= qfac
    quaternion = Rotation.from_matrix(rotation).as_quat()
    if quaternion[3] < 0:
        quaternion *= -1
    dtype = np.dtype(dtype)
    if dtype == np.dtype('float32'):
        datatype, bitpix = 16, 32
    elif dtype == np.dtype('uint8'):
        datatype, bitpix = 2, 8
    else:
        raise ValueError('Only float32 HU and uint8 masks are exported')
    if max(shape) > 32767:
        raise ValueError('NIfTI-1 dimensions exceed signed short range')
    header = bytearray(352)
    struct.pack_into('<i', header, 0, 348)
    struct.pack_into('<8h', header, 40, 3, *map(int, shape[::-1]), 1, 1, 1, 1)
    struct.pack_into('<hh', header, 70, datatype, bitpix)
    struct.pack_into('<8f', header, 76, qfac, *spacing, 1, 1, 1, 1)
    struct.pack_into('<fff', header, 108, 352., 1., 0.)
    header[123] = 2  # NIFTI_UNITS_MM.
    header[148:228] = description.encode('ascii', 'replace')[:79].ljust(80, b'\0')
    struct.pack_into('<hh', header, 252, 1, 1)  # Scanner anatomy in both forms.
    struct.pack_into('<3f', header, 256, *quaternion[:3])
    struct.pack_into('<3f', header, 268, *affine[:3, 3])
    struct.pack_into('<12f', header, 280, *affine[:3].ravel())
    header[344:348] = b'n+1\0'
    return bytes(header)


def _open_writer(stack, path, shape, affine, dtype, description):
    path = Path(path)
    if path.exists():
        raise ValueError('Refusing to overwrite a pretrained pilot NIfTI: ' + str(path))
    if not (str(path).endswith('.nii') or str(path).endswith('.nii.gz')):
        raise ValueError('NIfTI export requires .nii or .nii.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.partial')
    stream = stack.enter_context(temporary.open('xb'))
    writer = (stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=stream,
                                               compresslevel=1, mtime=0))
              if str(path).endswith('.gz') else stream)
    writer.write(_nifti_header(shape, affine, dtype, description))
    return writer, temporary, path


def export_thorax_nifti(row, selection_manifest, evidence, output_path, *,
                        context_mm=50., slab_slices=8, verify_hashes=True,
                        core_provider=None, mask_output_paths=None,
                        max_voxels=MAX_EXPORT_VOXELS):
    """Stream calibrated HU and optional core/domain/changed masks on one grid.

    ``mask_output_paths`` maps any of target/baseline/domain/changed to a fresh
    .nii[.gz] path. ``core_provider`` is a ThoracicCoreProvider; if absent it is
    constructed only when masks are requested. No target enters image export.
    """
    plan = plan_thorax_crop(row, selection_manifest, evidence, context_mm=context_mm,
                            max_voxels=max_voxels)
    if type(slab_slices) is not int or not 1 <= slab_slices <= 64:
        raise ValueError('slab_slices must be an integer in [1,64]')
    shape = tuple(plan['shape_kji'])
    if shape[1] * shape[2] > MAX_BLOCK_VOXELS:
        raise ValueError('One native X/Y slice exceeds bounded ContextVolume reads')
    slab_slices = min(slab_slices, MAX_BLOCK_VOXELS // (shape[1] * shape[2]))
    affine = np.asarray(plan['affine_ras_ijk'])
    mask_output_paths = {} if mask_output_paths is None else dict(mask_output_paths)
    if not set(mask_output_paths) <= {'target', 'baseline', 'domain', 'changed'}:
        raise ValueError('Unsupported pretrained pilot mask export')
    destinations = [Path(output_path).resolve()] + [Path(p).resolve() for p in mask_output_paths.values()]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Image/mask exports require distinct paths')
    if mask_output_paths and core_provider is None:
        core_provider = ThoracicCoreProvider(row, selection_manifest, evidence, verify_hashes=verify_hashes)
    staged = []
    stats = {'native_voxels_read': 0, 'peak_read_voxels': 0,
             'hu_min': None, 'hu_max': None, 'mask_voxel_counts': {k:0 for k in mask_output_paths}}
    try:
        with ExitStack() as stack:
            image_writer, temporary, final = _open_writer(stack, output_path, shape, affine,
                np.float32, 'syncoCT calibrated HU; atlas thorax + context; synthetic benchmark')
            staged.append((temporary, final))
            masks = {}
            for key, path in mask_output_paths.items():
                writer, temporary, final = _open_writer(stack, path, shape, affine, np.uint8,
                    'syncoCT native thoracic aorta ' + key + '; atlas core excludes branches')
                staged.append((temporary, final)); masks[key] = writer
            volume = stack.enter_context(ContextVolume(row, verify_hashes=verify_hashes))
            for offset in range(0, shape[0], slab_slices):
                block_shape = (min(slab_slices, shape[0] - offset), shape[1], shape[2])
                origin_ijk = [0, 0, plan['origin_ijk'][2] + offset]
                block = volume.read_block(origin_ijk[::-1], block_shape, row['spacing_ijk_mm'])
                if not np.all(block['valid']):
                    raise ValueError('Native thoracic image export unexpectedly includes invalid scan voxels')
                hu = attenuation_to_hu(block['image'], plan['mu_water_cm_inverse'])
                image_writer.write(np.asarray(hu, dtype='<f4').tobytes(order='C'))
                if masks:
                    target_blocks = core_provider.read_block(origin_ijk, block_shape)
                    for key, writer in masks.items():
                        data = np.asarray(target_blocks[key], dtype=np.uint8)
                        if data.shape != block_shape or not np.all((data == 0) | (data == 1)):
                            raise ValueError('Native core provider returned an invalid mask block')
                        writer.write(data.tobytes(order='C'))
                        stats['mask_voxel_counts'][key] += int(np.sum(data))
                count = math.prod(block_shape)
                stats['native_voxels_read'] += count
                stats['peak_read_voxels'] = max(stats['peak_read_voxels'], count)
                minimum, maximum = float(hu.min()), float(hu.max())
                stats['hu_min'] = minimum if stats['hu_min'] is None else min(stats['hu_min'], minimum)
                stats['hu_max'] = maximum if stats['hu_max'] is None else max(stats['hu_max'], maximum)
        for temporary, final in staged:
            # Link without replacing any destination created during a race.
            os.link(temporary, final)
            temporary.unlink()
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    plan['statistics'] = stats
    plan['artifacts'] = {str(final): {'sha256': _sha(final), 'bytes': final.stat().st_size}
                         for _, final in staged}
    if core_provider is not None:
        plan['target_metadata'] = core_provider.metadata
    return plan


def export_mask_nifti(mask_kji, output_path, *, origin_ijk, evidence):
    """Export a small native target crop with the same verified RAS convention."""
    mask = np.asarray(mask_kji)
    if mask.ndim != 3 or mask.size > 32_000_000 or not np.all((mask == 0) | (mask == 1)):
        raise ValueError('Mask export requires a bounded native binary k,j,i crop')
    affine = crop_ras_affine(evidence, origin_ijk)
    staged = None
    try:
        with ExitStack() as stack:
            writer, temporary, final = _open_writer(stack, output_path, mask.shape, affine,
                np.uint8, 'syncoCT native thoracic aorta core truth')
            staged = (temporary, final)
            writer.write(np.asarray(mask, np.uint8).tobytes(order='C'))
        os.link(temporary, final); temporary.unlink()
    except Exception:
        if staged is not None:
            staged[0].unlink(missing_ok=True)
        raise
    return {'path': str(final), 'sha256': _sha(final), 'bytes': final.stat().st_size,
            'shape_kji': list(mask.shape), 'origin_ijk': list(origin_ijk),
            'affine_ras_ijk': affine.tolist()}
