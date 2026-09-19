"""Immutable, portable prepared cohorts, independent of generating source/code.

Recorded hashes prove consistency with a frozen dataset, not anatomical realism.
Only standard-library and NumPy code is used: importing or changing a morphology
engine must never invalidate already prepared data. No registry API overwrites an
entry or published object; changes require another revision name.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

import numpy as np

DATASET_SCHEMA = 'fakect.training-dataset/1'
ENTRY_SCHEMA = 'fakect.cohort-registry-entry/1'
LINEAGE_SEMANTICS = ('Surviving selected target ancestry and its offspring. The original target is '
                     'selected tissue inside the main ROI; descendants outside that selector remain '
                     'positive, while unrelated same-tissue anatomy stays negative.')
ID_SEMANTICS = 'Membership in edited original target IDs over the complete crop; not ROI-clipped'
_HEX = re.compile(r'^[0-9a-f]{64}$')
_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError(f'Duplicate JSON key: {key}')
        value[key] = item
    return value


def _json(text):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def _digest(value, label):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f'Invalid SHA256 for {label}')
    return value


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value) or '..' in value:
        raise ValueError('Registry name must use letters, digits, dots, underscores or hyphens, without ..')
    return value


def _safe_file(root, relative, *, must_exist=True):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError(f'Unsafe artifact path: {relative!r}')
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in ('', '.', '..') for part in relative.split('/')):
        raise ValueError(f'Unsafe artifact path: {relative!r}')
    result = root.joinpath(*parsed.parts)
    current = root
    for part in parsed.parts:
        current = current/part
        if current.is_symlink():
            raise ValueError(f'Symlink artifact is unsafe: {relative}')
    if root not in result.resolve().parents or (must_exist and not result.is_file()):
        raise ValueError(f'Missing or unsafe artifact: {relative}')
    return result


def _shape(geometry):
    if not isinstance(geometry, dict) or geometry.get('array_order') != 'kji':
        raise ValueError('Dataset geometry must use native kji array order')
    shape = geometry.get('crop_shape_kji')
    spacing = geometry.get('spacing_ijk_mm')
    if (not isinstance(shape, list) or len(shape) != 3 or
            any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape)):
        raise ValueError('Invalid crop_shape_kji')
    if (not isinstance(spacing, list) or len(spacing) != 3 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or x <= 0 for x in spacing)):
        raise ValueError('Invalid spacing_ijk_mm')
    return tuple(shape)


def _binary(array, shape, name):
    if array.shape != shape or array.dtype.kind not in 'bu' or np.any((array != 0) & (array != 1)):
        raise ValueError(f'{name} must be a matching binary array')


def _npz(path):
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile) or len(set(archive.files)) != len(archive.files):
        if hasattr(archive, 'close'):
            archive.close()
        raise ValueError(f'Invalid or duplicate archive members: {path.name}')
    return archive


def _source_arrays(root, manifest, shape):
    with _npz(_safe_file(root, 'source.npz')) as data:
        required = {'original_labels', 'original_attenuation_per_pixel', 'target_mask', 'roi_mask', 'geometry_json'}
        if not required.issubset(data.files):
            raise ValueError('Source snapshot lacks required arrays')
        labels, attenuation = data['original_labels'], data['original_attenuation_per_pixel']
        target, roi = data['target_mask'], data['roi_mask']
        if (labels.shape != shape or labels.dtype.kind not in 'iuf' or
                not np.isfinite(labels).all() or np.any(labels <= -(2**31)) or
                np.any(labels >= 2**31) or not np.equal(labels, np.trunc(labels)).all()):
            raise ValueError('Source labels are invalid or contain diagnostic released voxels')
        if attenuation.shape != shape or attenuation.dtype.kind != 'f' or not np.isfinite(attenuation).all():
            raise ValueError('Source attenuation must be finite and match the crop')
        _binary(target, shape, 'Source target'); _binary(roi, shape, 'Source ROI')
        if _json(str(data['geometry_json'])) != manifest['geometry']:
            raise ValueError('Source geometry disagrees with dataset manifest')
        source_hashes = manifest.get('source_crop_sha256', {})
        for name, array in (('act', labels), ('atn', attenuation), ('selected', target.astype(bool))):
            if name in source_hashes and _array_hash(array) != source_hashes[name]:
                raise ValueError(f'Source crop hash mismatch: {name}')
        return labels, attenuation, target.astype(bool), roi.astype(bool)


def _sample_arrays(path, sample, manifest, shape, source):
    labels_before, attenuation_before, selected, roi = source
    with _npz(path) as data:
        required = {'image', 'mask', 'edited_labels', 'changed_mask', 'metadata_json'}
        if manifest.get('target_scope') == 'selected_lineage':
            required.update({'target_origin_index', 'edit_region_mask', 'scalar_changed_mask'})
        if not required.issubset(data.files):
            raise ValueError(f'Sample lacks required arrays: {sample["variant_id"]}')
        image, mask, labels, changed = (data[name] for name in ('image', 'mask', 'edited_labels', 'changed_mask'))
        if image.shape != shape or image.dtype != np.dtype('float32') or not np.isfinite(image).all():
            raise ValueError('Sample image must be finite float32 and match crop shape')
        _binary(mask, shape, 'Sample mask'); _binary(changed, shape, 'Changed mask')
        if mask.dtype != np.dtype('uint8'):
            raise ValueError('Sample foreground mask must use uint8')
        if labels.shape != shape or labels.dtype.kind != 'i' or np.any(labels == -(2**31)):
            raise ValueError('Sample labels are invalid or contain diagnostic released voxels')
        if not np.array_equal(changed, labels != labels_before):
            raise ValueError('Changed mask disagrees with original and edited labels')
        if _array_hash(mask) != sample['mask_sha256'] or _array_hash(image) != sample['image_sha256']:
            raise ValueError('Sample image/mask hash mismatch')
        if int(mask.sum()) != sample['foreground_voxels']:
            raise ValueError('Sample foreground count mismatch')
        metadata = _json(str(data['metadata_json']))
        expected = {key: value for key, value in sample.items() if key not in ('path', 'sha256', '_path', 'shape_kji')}
        actual = {key: value for key, value in metadata.items() if key not in ('path', 'sha256')}
        if actual != expected:
            raise ValueError('Sample embedded metadata disagrees with dataset manifest')
        region = roi
        if manifest.get('target_scope') == 'selected_lineage':
            origins = data['target_origin_index']
            if (origins.shape != shape or origins.dtype.kind != 'i' or
                    np.any(origins < -1) or np.any(origins >= selected.size)):
                raise ValueError('Invalid selected-lineage origins')
            expected_mask = (origins >= 0) & selected.ravel()[np.maximum(origins, 0)]
            region, scalar_changed = data['edit_region_mask'], data['scalar_changed_mask']
            _binary(region, shape, 'Edit region'); _binary(scalar_changed, shape, 'Scalar changed mask')
            if np.any(scalar_changed.astype(bool) & ~region.astype(bool)):
                raise ValueError('Scalar changes escape edit region')
            unmodified = ~scalar_changed.astype(bool)
        else:
            expected_mask = np.isin(labels, manifest['target_source_ids'])
            unmodified = ~changed.astype(bool)
        if not np.array_equal(mask, expected_mask):
            raise ValueError('Foreground mask violates its recorded target semantics')
        if np.any(changed.astype(bool) & ~region.astype(bool)):
            raise ValueError('Label changes escape edit region')
        original_image = np.asarray(attenuation_before/(manifest['geometry']['spacing_ijk_mm'][0]/10), dtype=np.float32)
        if not np.array_equal(image[unmodified], original_image[unmodified]):
            raise ValueError('Attenuation changed outside its recorded change mask')


def verify_prepared_manifest(path, verify_payloads=True):
    """Check frozen data without reading current source volumes or generator code.

    False checks manifest/metadata, inventory structure and file existence but
    never opens NPZ payloads (including test data). True verifies every artifact's
    SHA256 and the native image, mask, lineage and locality contracts.
    """
    path = Path(path).expanduser()
    if path.is_symlink() or path.name != 'dataset-manifest.json':
        raise ValueError('Expected a regular dataset-manifest.json file')
    path = path.resolve()
    root = path.parent
    if (root/'INCOMPLETE').exists() or (root/'INCOMPLETE').is_symlink():
        raise ValueError('Prepared dataset is marked INCOMPLETE')
    checksum_path = _safe_file(root, 'dataset-manifest.sha256')
    checksum = _digest(checksum_path.read_text().strip(), 'dataset manifest')
    if _hash(path) != checksum:
        raise ValueError('Dataset manifest checksum mismatch')
    manifest = _json(path.read_text())
    if manifest.get('schema_version') != DATASET_SCHEMA or manifest.get('complete') is not True:
        raise ValueError('Unsupported or incomplete prepared dataset')
    _digest(manifest.get('dataset_fingerprint'), 'dataset fingerprint')
    if manifest.get('image_units') != 'cm^-1' or manifest.get('image_method') != 'attenuation_copy_proxy':
        raise ValueError('Dataset image units/method are unsupported')
    scope = manifest.get('target_scope')
    if scope == 'selected_lineage':
        if manifest.get('target_semantics') != LINEAGE_SEMANTICS:
            raise ValueError('Unsupported selected-lineage target semantics')
        semantics = LINEAGE_SEMANTICS
    elif scope in (None, 'original_ids'):
        targets = manifest.get('target_source_ids')
        if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets) or
                any(isinstance(x, bool) or not isinstance(x, int) or x == 0 or not -(2**31) < x < 2**31 for x in targets)):
            raise ValueError('Original-ID targets must be distinct nonzero int32 labels')
        semantics = ID_SEMANTICS
    else:
        raise ValueError('Unsupported foreground mask semantics')
    shape = _shape(manifest.get('geometry'))
    inventory = manifest.get('artifacts_sha256')
    if not isinstance(inventory, dict) or not inventory or 'source.npz' not in inventory:
        raise ValueError('Dataset lacks a complete artifact inventory and source snapshot')
    for relative, expected in inventory.items():
        if relative in ('dataset-manifest.json', 'dataset-manifest.sha256', 'INCOMPLETE'):
            raise ValueError('Reserved file in dataset artifact inventory')
        artifact = _safe_file(root, relative)
        _digest(expected, relative)
        if (verify_payloads or artifact.suffix != '.npz') and _hash(artifact) != expected:
            raise ValueError(f'Dataset artifact checksum mismatch: {relative}')
    if 'artifact-manifest.json' in inventory:
        secondary = _json((root/'artifact-manifest.json').read_text())
        if not isinstance(secondary, dict) or any(inventory.get(k) != v for k, v in secondary.items()):
            raise ValueError('Artifact manifest disagrees with dataset inventory')
    snapshot = manifest.get('input_snapshot')
    if snapshot is not None and (snapshot != 'input.ini' or inventory.get(snapshot) != manifest.get('input_config_sha256')):
        raise ValueError('Input snapshot checksum disagrees with inventory')
    samples = manifest.get('samples')
    if (not isinstance(samples, list) or not samples or isinstance(manifest.get('sample_count'), bool) or
            manifest.get('sample_count') != len(samples)):
        raise ValueError('Dataset sample count mismatch or empty dataset')
    split_mode = manifest.get('split_mode')
    allowed = {'unassigned'} if split_mode == 'unassigned' else {'train', 'validation', 'test'}
    if split_mode not in ('scenario_only', 'unassigned'):
        raise ValueError('Unsupported prepared dataset split mode')
    ids, paths, group_splits, mask_splits = set(), set(), {}, {}
    normalized = []
    source = _source_arrays(root, manifest, shape) if verify_payloads else None
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError('Invalid sample record')
        identity, relative = sample.get('variant_id'), sample.get('path')
        if not isinstance(identity, str) or not identity or identity in ids or relative in paths:
            raise ValueError('Duplicate or invalid sample identity/path')
        ids.add(identity); paths.add(relative)
        sample_path = _safe_file(root, relative)
        if sample_path.suffix != '.npz' or sample.get('split') not in allowed:
            raise ValueError('Invalid sample archive or split')
        if inventory.get(relative) != sample.get('sha256'):
            raise ValueError('Sample checksum disagrees with artifact inventory')
        for key in ('mask_sha256', 'image_sha256'):
            _digest(sample.get(key), key)
        if (sample.get('image_method') != manifest['image_method'] or sample.get('image_units') != manifest['image_units'] or
                sample.get('mask_semantics') != semantics or sample.get('target_scope') != scope or
                sample.get('dataset_fingerprint') != manifest['dataset_fingerprint'] or
                any(sample.get(key) != manifest.get(key) for key in ('case_id', 'frame', 'anatomy_family', 'geometry'))):
            raise ValueError('Sample metadata disagrees with dataset contract')
        if sample.get('status', 'ok') != 'ok':
            raise ValueError('Failed sample cannot be registered')
        for groups, key in ((group_splits, sample.get('geometry_group')), (mask_splits, sample['mask_sha256'])):
            if not isinstance(key, str) or not key or groups.setdefault(key, sample['split']) != sample['split']:
                raise ValueError('Duplicate geometry crosses prepared splits')
        count = sample.get('foreground_voxels')
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= math.prod(shape):
            raise ValueError('Invalid sample foreground count')
        if verify_payloads:
            _sample_arrays(sample_path, sample, manifest, shape, source)
        normalized.append({**sample, '_path': str(sample_path), 'shape_kji': list(shape)})
    counts = {split: sum(sample['split'] == split for sample in samples) for split in allowed}
    if manifest.get('split_counts') != counts:
        raise ValueError('Recorded dataset split counts disagree with samples')
    return {**manifest, 'samples': normalized, 'manifest_sha256': checksum, '_manifest_path': str(path)}


def _registry_root(registry, create=False):
    root = Path(registry).expanduser().resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    for name in ('objects', 'entries', '.staging'):
        child = root/name
        if child.is_symlink():
            raise ValueError(f'Unsafe registry directory: {name}')
        if create:
            child.mkdir(exist_ok=True)
    return root


@contextmanager
def _registry_lock(root):
    lock = root/'.register.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError('Registry registration is already running; inspect .register.lock if a prior process was interrupted') from exc
    try:
        yield
    finally:
        lock.rmdir()


def load_registry_entry(registry, name, verify=True):
    """Read a pinned revision; verify=False never opens sample/source NPZs."""
    root = _registry_root(registry)
    entry_path = _safe_file(root, f'entries/{_name(name)}.json', must_exist=False)
    if not entry_path.is_file():
        raise FileNotFoundError(f'Registry entry has not been published: {name}')
    entry = _json(entry_path.read_text())
    if entry.get('schema_version') != ENTRY_SCHEMA or entry.get('name') != name:
        raise ValueError('Registry entry identity/schema mismatch')
    checksum = _digest(entry.get('entry_sha256'), 'registry entry')
    if hashlib.sha256(_canonical({k: v for k, v in entry.items() if k != 'entry_sha256'}).encode()).hexdigest() != checksum:
        raise ValueError('Registry entry checksum mismatch')
    digest = _digest(entry.get('manifest_sha256'), 'registry manifest')
    relative = f'objects/{digest}/dataset-manifest.json'
    if entry.get('manifest_relative_path') != relative:
        raise ValueError('Registry object path disagrees with pinned manifest hash')
    if not isinstance(entry.get('family_verified'), bool) or not isinstance(entry.get('anatomy_family'), str) or not entry['anatomy_family'].strip():
        raise ValueError('Registry anatomy-family policy is invalid')
    path = _safe_file(root, relative)
    manifest = verify_prepared_manifest(path, verify_payloads=verify)
    if digest != manifest['manifest_sha256'] or any(entry.get(key) != manifest.get(key) for key in ('dataset_fingerprint', 'case_id', 'frame', 'sample_count')):
        raise ValueError('Registered object disagrees with immutable receipt')
    return {**entry, 'manifest_path': str(path), 'registry_entry_path': str(entry_path), 'manifest': manifest}


def list_registry_entries(registry, verify=False):
    root = _registry_root(registry)
    if not root.exists():
        return []
    return [load_registry_entry(root, path.stem, verify=verify)
            for path in sorted((root/'entries').glob('*.json'))]


def register_dataset(registry, manifest_path, name, anatomy_family, family_verified=False):
    """Copy verified files once, then atomically publish an immutable receipt.

    The caller supplies family identity explicitly; case IDs alone do not prove
    independence. Repeated identical registration is safe. Different contents or
    family policy under the same name require a new revision name.
    """
    name = _name(name)
    if not isinstance(anatomy_family, str) or not anatomy_family.strip() or anatomy_family != anatomy_family.strip():
        raise ValueError('anatomy_family must be an explicit nonblank identifier')
    if not isinstance(family_verified, bool):
        raise ValueError('family_verified must be Boolean')
    manifest = verify_prepared_manifest(manifest_path)
    root = _registry_root(registry, create=True)
    checksum = manifest['manifest_sha256']
    source = Path(manifest['_manifest_path']).parent
    with _registry_lock(root):
        receipt = root/'entries'/(name+'.json')
        if receipt.exists() or receipt.is_symlink():
            existing = load_registry_entry(root, name)
            if any(existing[key] != value for key, value in (
                    ('manifest_sha256', checksum), ('anatomy_family', anatomy_family), ('family_verified', family_verified))):
                raise ValueError('Registry revision already exists with different contents or family policy; use a new name')
            return {**existing, 'already_registered': True}
        destination = root/'objects'/checksum
        staging = None
        try:
            if destination.exists() or destination.is_symlink():
                _safe_file(root, f'objects/{checksum}/dataset-manifest.json')
                existing = verify_prepared_manifest(destination/'dataset-manifest.json')
                if existing['manifest_sha256'] != checksum:
                    raise ValueError('Existing registry object is inconsistent')
            else:
                staging = Path(tempfile.mkdtemp(prefix=name+'-', dir=root/'.staging'))
                relatives = set(manifest['artifacts_sha256']) | {'dataset-manifest.json', 'dataset-manifest.sha256'}
                for relative in sorted(relatives):
                    src = _safe_file(source, relative)
                    dst = _safe_file(staging, relative, must_exist=False)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                copied = verify_prepared_manifest(staging/'dataset-manifest.json')
                if copied['manifest_sha256'] != checksum:
                    raise ValueError('Source manifest changed during registration')
                # No replacement is possible while the registry lock is held.
                staging.rename(destination)
                staging = None
            entry = {'schema_version': ENTRY_SCHEMA, 'name': name,
                     'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                     'manifest_relative_path': f'objects/{checksum}/dataset-manifest.json',
                     'manifest_sha256': checksum, 'dataset_fingerprint': manifest['dataset_fingerprint'],
                     'case_id': manifest.get('case_id'), 'frame': manifest.get('frame'),
                     'anatomy_family': anatomy_family, 'family_verified': family_verified,
                     'prepared_anatomy_family': manifest.get('anatomy_family'),
                     'sample_count': manifest['sample_count'], 'split_mode': manifest['split_mode'],
                     'target_scope': manifest.get('target_scope', 'original_ids'),
                     'source_crop_sha256': manifest.get('source_crop_sha256', {}),
                     'original_manifest_path': str(Path(manifest['_manifest_path'])),
                     'immutability': 'Published files are never overwritten by the registry workflow; checksums are verified before reuse.'}
            entry['entry_sha256'] = hashlib.sha256(_canonical(entry).encode()).hexdigest()
            temporary = root/'.staging'/(name+'.receipt.tmp')
            try:
                with temporary.open('x') as handle:
                    handle.write(json.dumps(entry, indent=2, allow_nan=False)+'\n')
                    handle.flush(); os.fsync(handle.fileno())
                os.link(temporary, receipt)  # Exclusive creation: never replaces a receipt.
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            if staging is not None:
                shutil.rmtree(staging)
    return {**load_registry_entry(root, name, verify=False), 'already_registered': False}
