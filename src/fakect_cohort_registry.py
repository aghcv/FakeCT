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
    """Serialize registry data deterministically without nonfinite JSON values.

    Parameters
    ----------
    value : JSON-compatible object
        Registry metadata to serialize.

    Returns
    -------
    str
        Sorted compact JSON.
    """
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(path):
    """Hash an immutable registry artifact in bounded chunks.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.

    Returns
    -------
    str
        SHA256 hexadecimal digest of complete file contents.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value):
    """Hash an array using contiguous native element bytes.

    Parameters
    ----------
    value : array_like
        Array data to fingerprint.

    Returns
    -------
    str
        SHA256 of contiguous C-order values; dtype/shape are separately validated by
        callers.
    """
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _pairs(items):
    """Decode JSON object pairs while rejecting duplicate keys.

    Parameters
    ----------
    items : iterable of tuple
        Decoded JSON key/value pairs.

    Returns
    -------
    dict
        Object mapping retaining each unique decoded key.

    Raises
    ------
    ValueError
        An object repeats a JSON key.
    """
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError(f'Duplicate JSON key: {key}')
        value[key] = item
    return value


def _json(text):
    """Decode strict JSON with duplicate-key and nonfinite-number rejection.

    Parameters
    ----------
    text : str or bytes
        JSON artifact contents.

    Returns
    -------
    object
        Decoded JSON data.

    Raises
    ------
    ValueError
        JSON is malformed, repeats keys or uses nonfinite number constants.
    """
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def _digest(value, label):
    """Validate a recorded lowercase SHA256 identifier.

    Parameters
    ----------
    value : str
        Expected 64-character hexadecimal digest.
    label : str
        Artifact label used in validation messages.

    Returns
    -------
    str
        The unchanged digest string.

    Raises
    ------
    ValueError
        The value is not a lowercase SHA256 digest.
    """
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f'Invalid SHA256 for {label}')
    return value


def _name(value):
    """Validate a portable registry revision name.

    Parameters
    ----------
    value : str
        1–128-character name containing letters, digits, dots, underscores or hyphens.

    Returns
    -------
    str
        Unchanged safe revision name.

    Raises
    ------
    ValueError
        The name is invalid or includes a parent-directory token.
    """
    if not isinstance(value, str) or not _NAME.fullmatch(value) or '..' in value:
        raise ValueError('Registry name must use letters, digits, dots, underscores or hyphens, without ..')
    return value


def _safe_file(root, relative, *, must_exist=True):
    """Resolve a confined artifact path without following symlink components.

    Parameters
    ----------
    root : pathlib.Path
        Resolved artifact root.
    relative : str
        Nonempty POSIX-relative artifact path without empty or traversal components.
    must_exist : bool
        Require the destination to be an existing regular file.

    Returns
    -------
    pathlib.Path
        Safe child path below the registry/dataset root.

    Raises
    ------
    ValueError
        The path is unsafe, a component is a symlink, or a required file is missing.
    """
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
    """Validate recorded native dataset shape and physical spacing.

    Parameters
    ----------
    geometry : dict
        Native geometry with array_order, crop_shape_kji and positive spacing_ijk_mm.

    Returns
    -------
    tuple of int
        Positive crop dimensions in k,j,i order.

    Raises
    ------
    ValueError
        Geometry, array order, shape or millimetre spacing is invalid.
    """
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
    """Check a paired binary array against its recorded geometry.

    Parameters
    ----------
    array : numpy.ndarray
        Mask payload to validate.
    shape : tuple of int
        Expected k,j,i dimensions.
    name : str
        Mask label used in validation errors.

    Returns
    -------
    None
        Completes when shape, Boolean/unsigned dtype and zero/one values agree.

    Raises
    ------
    ValueError
        The mask shape, dtype or values are incompatible.
    """
    if array.shape != shape or array.dtype.kind not in 'bu' or np.any((array != 0) & (array != 1)):
        raise ValueError(f'{name} must be a matching binary array')


def _npz(path):
    """Open a nonpickled NPZ archive while rejecting repeated members.

    Parameters
    ----------
    path : pathlib.Path
        Input file path.

    Returns
    -------
    numpy.lib.npyio.NpzFile
        Open archive; the caller must close it or use it as a context manager.

    Raises
    ------
    ValueError
        The file is not an NPZ archive or contains duplicate member names.
    """
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile) or len(set(archive.files)) != len(archive.files):
        if hasattr(archive, 'close'):
            archive.close()
        raise ValueError(f'Invalid or duplicate archive members: {path.name}')
    return archive


def _source_arrays(root, manifest, shape):
    """Validate and load a prepared dataset source snapshot.

    Parameters
    ----------
    root : pathlib.Path
        Dataset directory containing source.npz.
    manifest : dict
        Verified dataset metadata, geometry and optional native crop hashes.
    shape : tuple of int
        Expected native crop dimensions.

    Returns
    -------
    tuple of numpy.ndarray
        Original labels, per-pixel attenuation, Boolean selected target and Boolean ROI
        arrays in k,j,i order.

    Raises
    ------
    ValueError
        Required arrays, original labels, scalar values, geometry or recorded hashes
        disagree.
    """
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
    """Validate sample arrays against the frozen native source and target semantics.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.
    sample : dict
        Manifest sample record with hashes, target count and embedded metadata
        expectations.
    manifest : dict
        Dataset contract including target scope, source IDs and native geometry.
    shape : tuple of int
        Expected crop dimensions in k,j,i order.
    source : tuple of numpy.ndarray
        Original labels/attenuation and selected/ROI masks from ``_source_arrays``.

    Returns
    -------
    None
        Completes when image, mask, lineage, metadata and change-locality contracts
        hold.

    Raises
    ------
    ValueError
        Payload validity, checksums, ancestry, metadata or unchanged-region attenuation
        checks fail.
    """
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
    """Verify frozen dataset integrity independently of current source/generator code.

    Parameters
    ----------
    path : str or pathlib.Path
        Regular dataset-manifest.json file to verify.
    verify_payloads : bool
        Whether to read and validate NPZ payload bytes and array contracts, rather than
        metadata only.

    Returns
    -------
    dict
        Validated manifest with normalized sample paths/shapes, manifest hash and
        resolved manifest path.

    Raises
    ------
    ValueError
        Dataset schema, inventory, path safety, split integrity, metadata or requested
        payload validation fails.

    Notes
    -----
    Metadata-only verification still checks non-NPZ artifacts and existence; it never decodes source or sample NPZs.

    Check frozen data without reading current source volumes or generator code.

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
    """Resolve and optionally initialize a registry directory layout.

    Parameters
    ----------
    registry : str or pathlib.Path
        Root directory containing immutable registry objects and entries.
    create : bool
        Create missing registry/layout directories.

    Returns
    -------
    pathlib.Path
        Resolved root with objects, entries and staging directories when creation is
        requested.

    Raises
    ------
    ValueError
        An internal registry directory is a symlink.

    Notes
    -----
    Creates directories only when ``create`` is true.
    """
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
    """Hold an exclusive directory lock for registry publication.

    Parameters
    ----------
    root : pathlib.Path
        Registry root in which the lock directory is created.

    Returns
    -------
    context manager
        Yields None while registration owns the lock; removes the lock on context exit.

    Raises
    ------
    ValueError
        A registration lock already exists.

    Notes
    -----
    Creates and removes ``.register.lock``; the context does not permit concurrent publishers.
    """
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
    """Load a pinned registry revision and verify its immutable object identity.

    Parameters
    ----------
    registry : str or pathlib.Path
        Root directory containing immutable registry objects and entries.
    name : str
        Published safe revision name.
    verify : bool
        Verify NPZ payloads as well as metadata when true.

    Returns
    -------
    dict
        Receipt augmented with manifest/entry paths and the verified prepared manifest.

    Raises
    ------
    FileNotFoundError
        The named revision has not been published.
    ValueError
        Receipt integrity, family policy, object paths or dataset validation fails.

    Notes
    -----
    Read a pinned revision; verify=False never opens sample/source NPZs.
    """
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
    """List published dataset revisions in deterministic name order.

    Parameters
    ----------
    registry : str or pathlib.Path
        Root directory containing immutable registry objects and entries.
    verify : bool
        Perform full NPZ validation for every listed entry when true.

    Returns
    -------
    list of dict
        Loaded registry entries; an absent registry returns an empty list.
    """
    root = _registry_root(registry)
    if not root.exists():
        return []
    return [load_registry_entry(root, path.stem, verify=verify)
            for path in sorted((root/'entries').glob('*.json'))]


def register_dataset(registry, manifest_path, name, anatomy_family, family_verified=False):
    """Publish a verified prepared cohort as an immutable registry revision.

    Parameters
    ----------
    registry : str or pathlib.Path
        Root directory containing immutable registry objects and entries.
    manifest_path : str or pathlib.Path
        Prepared dataset manifest or frozen experiment-lock path.
    name : str
        New safe revision name, or an exactly matching published name.
    anatomy_family : str
        Explicit shared-source anatomy-family identity supplied by the caller.
    family_verified : bool
        Recorded provenance decision that the family identity has been reviewed.

    Returns
    -------
    dict
        Loaded registry receipt and ``already_registered`` flag.

    Raises
    ------
    ValueError
        Name/family policy, dataset integrity, existing revision contents or publication
        safety fails.

    Notes
    -----
    Copies and verifies files into a content-addressed object, then exclusively publishes a receipt. Existing names/objects are never overwritten.

    Copy verified files once, then atomically publish an immutable receipt.

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
