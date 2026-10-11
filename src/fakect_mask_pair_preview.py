"""Transfer frozen edit deltas onto a broader, independently selected mask.

This creates review pairs, not a training release. Native geometry is preserved;
an edit must belong to the new anatomical target before it can be transferred.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


def _binary(value, name):
    value = np.asarray(value)
    if value.ndim != 3 or not value.size or not np.all((value == 0) | (value == 1)):
        raise ValueError(f'{name} must be a nonempty binary 3D array')
    return value.astype(bool)


def _origin(value):
    value = np.asarray(value)
    if value.shape != (3,) or value.dtype.kind not in 'iu' or np.any(value < 0):
        raise ValueError('Crop origin must contain three nonnegative integer i,j,k indices')
    return value.astype(np.int64)


def _shape(value, name):
    value = np.asarray(value)
    if value.shape != (3,) or value.dtype.kind not in 'iu' or np.any(value <= 0):
        raise ValueError(f'{name} must contain three positive integer k,j,i sizes')
    return value.astype(np.int64)


def _bounds(origin_ijk, shape_kji, source_shape_kji):
    if np.any(origin_ijk + shape_kji[::-1] > source_shape_kji[::-1]):
        raise ValueError('Native crop leaves its declared source geometry')


def _act_identity(value):
    if (not isinstance(value, dict) or not value.get('path') or
            type(value.get('bytes')) is not int or value['bytes'] <= 0 or
            type(value.get('mtime_ns')) is not int or value['mtime_ns'] <= 0):
        raise ValueError('Missing or invalid original ACT provenance')
    return {'path': str(Path(value['path']).resolve()), 'bytes': value['bytes'], 'mtime_ns': value['mtime_ns']}


def transfer_pair(anatomy_mask, anatomy_origin_ijk, before, after,
                  edit_origin_ijk, origins, *, max_voxels=32_000_000):
    """Preserve broad anatomy outside edits; enforce native lineage membership.

    ``origins`` is the frozen variant's flat, zero-based index into its original
    native crop. Newly borrowed voxels may extend beyond the automatic crop.
    """
    anatomy = _binary(anatomy_mask, 'Anatomy mask')
    old_before, old_after = _binary(before, 'Original mask'), _binary(after, 'Edited mask')
    if old_before.shape != old_after.shape:
        raise ValueError('Original and edited crops differ')
    auto_origin, edit_origin = _origin(anatomy_origin_ijk), _origin(edit_origin_ijk)
    origins = np.asarray(origins)
    if (origins.shape != old_before.shape or origins.dtype.kind not in 'iu'
            or np.any(origins < -1) or np.any(origins >= old_before.size)):
        raise ValueError('Invalid frozen target lineage indices')
    if not np.array_equal(old_after, (origins >= 0) & old_before.ravel()[np.maximum(origins, 0)]):
        raise ValueError('Edited mask disagrees with its frozen target lineage')

    def native_points(mask):
        return np.argwhere(mask)[:, ::-1] + edit_origin

    def in_anatomy(points):
        local = points - auto_origin
        valid = np.all((local >= 0) & (local < np.asarray(anatomy.shape)[::-1]), axis=1)
        result = np.zeros(len(points), bool)
        q = local[valid]
        result[valid] = anatomy[tuple(q[:, ::-1].T)]
        return result

    added_points = native_points(old_after & ~old_before)
    removed_points = native_points(old_before & ~old_after)
    removed_outside = int(np.count_nonzero(~in_anatomy(removed_points)))
    borrowed_origins = origins[old_after & ~old_before]
    parent_points = (np.column_stack(np.unravel_index(borrowed_origins, old_before.shape))[:, ::-1]
                     + edit_origin)
    parents_outside = int(np.count_nonzero(~in_anatomy(parent_points)))
    if removed_outside or parents_outside:
        raise ValueError(f'Anatomical scope mismatch: {removed_outside} removed voxels and '
                         f'{parents_outside} dilation ancestors are outside the automatic mask; '
                         'review the branch extent/landmarks before transferring this edit')
    low = auto_origin.copy()
    high = auto_origin + np.asarray(anatomy.shape)[::-1]
    if len(added_points):
        low = np.minimum(low, added_points.min(axis=0))
        high = np.maximum(high, added_points.max(axis=0)+1)
    shape = tuple(map(int, (high-low)[::-1]))
    if int(np.prod(shape)) > max_voxels:
        raise ValueError('Expanded mask-pair preview exceeds its volume budget')
    broad_before = np.zeros(shape, bool)
    offset = (auto_origin-low)[::-1]
    slices = tuple(slice(int(o), int(o+n)) for o, n in zip(offset, anatomy.shape))
    broad_before[slices] = anatomy
    broad_after = broad_before.copy()
    if len(removed_points):
        broad_after[tuple((removed_points-low)[:, ::-1].T)] = False
    if len(added_points):
        broad_after[tuple((added_points-low)[:, ::-1].T)] = True
    added = broad_after & ~broad_before
    removed = broad_before & ~broad_after
    return {'before': broad_before, 'after': broad_after, 'added': added, 'removed': removed,
            'crop_origin_ijk': low, 'counts': {
                'before': int(broad_before.sum()), 'after': int(broad_after.sum()),
                'added': int(added.sum()), 'removed': int(removed.sum()),
                'old_lineage_added': len(added_points), 'old_lineage_removed': len(removed_points),
                'old_additions_already_in_broad_reference': int(in_anatomy(added_points).sum()),
                'old_selected_outside_broad_reference': int((~in_anatomy(native_points(old_before))).sum())}}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_preview(anatomy_npz, *, case_id, dataset_manifest, output_directory):
    """Use an explicit automatic mask NPZ and pinned existing paired manifest.

    Anatomy NPZ must match its checksummed sibling selection.json. Every variant
    is reviewed independently; failures are recorded instead of publishing a
    silently reduced training cohort. No image generation or model fitting runs.
    """
    anatomy_npz, manifest_path = Path(anatomy_npz).resolve(), Path(dataset_manifest).resolve()
    manifest_hash = sha256(manifest_path)
    sidecar = manifest_path.with_suffix('.sha256')
    if not sidecar.is_file() or sidecar.read_text().strip() != manifest_hash:
        raise ValueError('Paired manifest checksum is missing or changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 'fakect.two-stage-dataset/1':
        raise ValueError('Expected the frozen two-stage paired-data manifest')
    rows = [r for r in manifest['samples'] if r['case_id'] == str(case_id)]
    if not rows:
        raise ValueError('Case is absent from the paired manifest')
    selection_path = anatomy_npz.parent / 'selection.json'
    selection_sidecar = selection_path.with_suffix('.json.sha256')
    if not selection_path.is_file() or not selection_sidecar.is_file():
        raise ValueError('Automatic mask requires a checksummed selection.json')
    selection_hash = sha256(selection_path)
    if selection_sidecar.read_text().strip() != selection_hash:
        raise ValueError('Automatic selection manifest checksum changed')
    selection = json.loads(selection_path.read_text())
    if (selection.get('schema_version') != 'fakect.phantom-anatomy/1' or
            selection.get('complete') is not True or (anatomy_npz.parent/'INCOMPLETE').exists()):
        raise ValueError('Expected a complete automatic anatomical selection')
    if (selection.get('case_id') != str(case_id) or type(selection.get('frame')) is not int or
            selection['frame'] < 1):
        raise ValueError('Automatic selection has the wrong case/frame')
    if (not isinstance(selection.get('status'), str) or not selection['status'] or
            type(selection.get('provisional')) is not bool or
            type(selection.get('coordinate_reviewed')) is not bool):
        raise ValueError('Automatic selection lacks its review status')
    if selection.get('mask_npz') != anatomy_npz.name:
        raise ValueError('Automatic mask filename disagrees with its selection manifest')
    anatomy_hash = sha256(anatomy_npz)
    if selection.get('mask_npz_sha256') != anatomy_hash:
        raise ValueError('Automatic mask checksum disagrees with its selection manifest')
    with np.load(anatomy_npz, allow_pickle=False) as data:
        anatomy = _binary(data['mask'], 'Automatic mask')
        origin = _origin(data['crop_origin_ijk'])
        spacing = np.asarray(data['spacing_ijk_mm'], float)
        if data['case_id'].shape != () or str(data['case_id']) != str(case_id):
            raise ValueError('Automatic mask belongs to a different phantom')
        if data['frame'].shape != () or data['frame'].dtype.kind not in 'iu':
            raise ValueError('Automatic mask frame must be a scalar integer')
        frame = int(data['frame'])
        source_shape = _shape(data['source_shape_kji'], 'Source shape')
    if frame != selection['frame']:
        raise ValueError('Automatic mask frame disagrees with its selection manifest')
    _bounds(origin, np.asarray(anatomy.shape), source_shape)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid automatic-mask spacing')
    expected_geometry = selection.get('geometry', {})
    for key, value in (('shape_kji', source_shape), ('crop_low_ijk', origin),
                       ('crop_shape_kji', anatomy.shape), ('spacing_ijk_mm', spacing),
                       ('crop_high_ijk_exclusive', origin + np.asarray(anatomy.shape)[::-1])):
        if not np.array_equal(value, expected_geometry.get(key)):
            raise ValueError(f'Automatic mask {key} disagrees with its selection manifest')
    source_act = _act_identity(selection.get('act_source'))
    if any(not np.array_equal(spacing, r['geometry']['spacing_ijk_mm']) for r in rows):
        raise ValueError('Automatic mask and edit data must share the native grid')
    if any(not np.array_equal(source_shape, r['geometry']['source_shape_kji']) for r in rows):
        raise ValueError('Automatic mask and edit data have different source geometry')
    for row in rows:
        old_origin = _origin(row['geometry']['crop_origin_ijk'])
        old_shape = _shape(row['geometry']['crop_shape_kji'], 'Edit crop shape')
        _bounds(old_origin, old_shape, source_shape)
        if not np.array_equal(old_origin + old_shape[::-1], row['geometry']['crop_high_ijk_exclusive']):
            raise ValueError('Edit crop bounds disagree with its shape')
    prepared_hashes = {}
    for filename in {r['prepared_manifest'] for r in rows}:
        expected_hashes = {r['prepared_manifest_sha256'] for r in rows if r['prepared_manifest'] == filename}
        observed_hash = sha256(filename)
        if expected_hashes != {observed_hash}:
            raise ValueError('Prepared source manifest changed')
        prepared_hashes[filename] = observed_hash
        metadata = json.loads(Path(filename).read_text())
        if metadata['frame'] != frame or metadata['case_id'] != str(case_id):
            raise ValueError('Automatic mask and edit data use different case/frame')
        if _act_identity(metadata.get('source_files', {}).get('act')) != source_act:
            raise ValueError('Automatic mask and edit data use different original ACT source revisions')
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').write_text('Preparing review-only mask pairs\n')
    verified, results = set(), []
    for row in rows:
        result = {k: row[k] for k in ('sample_id', 'case_id', 'variant_id', 'baseline')}
        for key in ('source_npz', 'variant_npz'):
            pair = (row[key], row[key+'_sha256'])
            if pair not in verified:
                if sha256(pair[0]) != pair[1]:
                    raise ValueError('Frozen pair payload changed')
                verified.add(pair)
        with np.load(row['source_npz'], allow_pickle=False) as data:
            before = data['target_mask']
        with np.load(row['variant_npz'], allow_pickle=False) as data:
            after, origins = data['mask'], data['target_origin_index']
        expected_shape = tuple(row['geometry']['crop_shape_kji'])
        if before.shape != expected_shape or after.shape != expected_shape or origins.shape != expected_shape:
            raise ValueError('Frozen pair payload shape disagrees with its declared native crop')
        try:
            pair = transfer_pair(anatomy, origin, before, after,
                                 row['geometry']['crop_origin_ijk'], origins)
        except ValueError as error:
            result.update(status='requires_review', reason=str(error))
        else:
            filename = f'{len(results):04d}-masks.npz'
            np.savez_compressed(output/filename, **{k:pair[k] for k in ('before','after','added','removed','crop_origin_ijk')},
                                spacing_ijk_mm=spacing, sample_id=np.asarray(row['sample_id']))
            result.update(status='ready_for_visual_review', path=filename,
                          sha256=sha256(output/filename), counts=pair['counts'])
        results.append(result)
    document = {'schema_version': 'fakect.mask-pair-preview/1', 'case_id': str(case_id),
                'training_ready': False, 'anatomical_extent_reviewed': False,
                'source_anatomy_npz': str(anatomy_npz), 'source_anatomy_sha256': anatomy_hash,
                'source_selection_manifest': str(selection_path), 'source_selection_manifest_sha256': selection_hash,
                'source_selection_status': selection['status'],
                'source_selection_provisional': selection['provisional'],
                'source_selection_coordinate_reviewed': selection['coordinate_reviewed'],
                'source_act': source_act, 'source_shape_kji': source_shape.tolist(), 'frame': frame,
                'exporter_code_sha256': sha256(Path(__file__)),
                'source_paired_manifest': str(manifest_path), 'source_paired_manifest_sha256': manifest_hash,
                'spacing_ijk_mm': spacing.tolist(), 'samples': results,
                'semantics': 'Native frozen edit deltas transferred onto the broader automatic reference; no morphology rerun.',
                'limitations': ['Review the automatic anatomical extent before creating a training release.',
                                'This preview schema cannot be passed off as the old ROI-limited training dataset.',
                                'Net additions may shrink where old lineage additions already belonged to the broader original anatomy.']}
    target = output/'preview-manifest.json'
    if (sha256(anatomy_npz) != anatomy_hash or sha256(manifest_path) != manifest_hash or
            sha256(selection_path) != selection_hash or
            any(sha256(path) != expected for path, expected in prepared_hashes.items())):
        raise ValueError('A mask-preview input changed during export')
    target.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    target.with_suffix('.json.sha256').write_text(sha256(target)+'\n')
    (output/'INCOMPLETE').unlink()
    return document
