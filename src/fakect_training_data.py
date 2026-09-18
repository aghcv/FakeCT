"""Plan and prepare bounded image/mask pairs from independent original snapshots.

These are attenuation-copy-proxy experiments, not recovered CT or independent
patients. Scenario splits hold out complete variant crops; exact mask geometry
duplicates stay together. Original signed labels and source scalars are retained.
"""
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from fakect_morphology import apply_morphology, validate_edit_geometry, MAX_DISTANCE_MM
from fakect_erosion_guard import erosion_guard_spec
from fakect_roi import prepare_crop


SCHEMA = 'fakect.training-dataset/1'
SPLIT_WARNING = ('Scenario-only holdout: every split shares the same source anatomy. '
                 'This does not measure generalization to unseen anatomy or patients.')
IMAGE_WARNING = ('Attenuation-copy proxy in inverse centimetres; not HU, learned '
                 'background recovery, physical CT reconstruction, or validated patient imaging.')
_DATA_TRAIN_KEYS = ('target_source_ids', 'anatomy_family', 'operations', 'distances_mm',
                    'shape_ks', 'include_baseline', 'max_variants', 'image_method',
                    'split_mode', 'validation_fraction', 'test_fraction', 'split_seed')
_EROSION_GUARD_FIELDS = ('min_volume_ratio', 'preserve_connectivity', 'backoff_factor', 'max_backoff_steps')


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order='C')).hexdigest()


def _number_list(values, name, maximum=None):
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f'{name} must be a nonempty list')
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must contain positive finite numbers')
        number = float(value)
        if not math.isfinite(number) or number <= 0 or (maximum is not None and number > maximum):
            raise ValueError(f'{name} must contain positive finite numbers' +
                             (f' no greater than {maximum:g}' if maximum is not None else ''))
        result.append(number)
    if len(set(result)) != len(result):
        raise ValueError(f'{name} must not contain duplicate values')
    return tuple(sorted(result))


def plan_variants(config):
    """Return a deterministic baseline and Cartesian edit plan, without file I/O.

    The plan's splits are provisional until exact mask geometries are known.
    During preparation, equal masks become one split group; baseline-equivalent
    masks stay in training. All slices and patches inherit their crop's split.
    """
    train = config['train']
    if train['split_mode'] != 'scenario_only':
        raise ValueError('Only explicit scenario_only splitting is implemented')
    if train['image_method'] != 'attenuation_copy_proxy':
        raise ValueError('Only attenuation_copy_proxy image preparation is implemented')
    if not isinstance(train['include_baseline'], bool):
        raise ValueError('train.include_baseline must be Boolean')
    maximum = train['max_variants']
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('train.max_variants must be a positive integer')
    operations = tuple(train['operations'])
    if not operations or len(set(operations)) != len(operations) or any(
            operation not in ('erosion', 'dilation') for operation in operations):
        raise ValueError('train.operations must contain distinct erosion/dilation operations')
    distances = _number_list(train['distances_mm'], 'train.distances_mm', MAX_DISTANCE_MM)
    shape_ks = _number_list(train['shape_ks'], 'train.shape_ks')
    count = len(operations) * len(distances) * len(shape_ks) + int(train['include_baseline'])
    if count > maximum:
        raise ValueError(f'Planned {count} variants exceeds train.max_variants={maximum}')
    fractions = [float(train[name]) for name in ('validation_fraction', 'test_fraction')]
    if any(not math.isfinite(f) or f < 0 or f >= 1 for f in fractions) or sum(fractions) >= 1:
        raise ValueError('Validation/test fractions must be finite, nonnegative, and sum to less than one')
    seed = train['split_seed']
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('train.split_seed must be a nonnegative integer')
    family = train['anatomy_family']
    if not isinstance(family, str) or not family.strip():
        raise ValueError('train.anatomy_family must explicitly identify the shared source family')
    base = {'case_id': config['input']['case_id'], 'frame': config['input']['frame'],
            'anatomy_family': family, 'profile': config['edit']['profile'],
            'profile_axis': config['edit']['profile_axis'],
            'shape_window': list(config['edit']['shape_window'])}
    guard = erosion_guard_spec(config)
    variants = []
    if train['include_baseline']:
        variants.append({**base, 'variant_id': 'baseline', 'operation': 'none',
                         'distance_mm': 0., 'shape_k': float(config['edit']['shape_k']), 'split': 'train'})
    edited = []
    for operation, distance, shape_k in product(sorted(operations), distances, shape_ks):
        record = {**base, 'operation': operation, 'distance_mm': distance, 'shape_k': shape_k}
        if operation == 'erosion' and guard['enabled']:
            # Scenario identities bind the effective policy, including runtime
            # defaults. Disabled/absent safeguards preserve existing identities.
            record.update({key: guard[key] for key in _EROSION_GUARD_FIELDS})
        identifier = hashlib.sha256(_canonical(record).encode()).hexdigest()[:16]
        edited.append({**record, 'variant_id': operation + '-' + identifier, 'split': 'train'})
    # Hash ranking avoids dependence on filesystem order or global RNG state.
    ranked = sorted(edited, key=lambda record: hashlib.sha256(
        f"{seed}:{record['variant_id']}".encode()).digest())
    counts = [max(1, math.floor(len(edited) * fraction)) if fraction else 0 for fraction in fractions]
    if sum(counts) >= len(edited) and not variants:
        raise ValueError('Too few variants to leave training examples after requested holdouts')
    if sum(counts) > len(edited):
        raise ValueError('Too few variants for the requested nonempty holdouts')
    for record in ranked[:counts[0]]:
        record['split'] = 'validation'
    for record in ranked[counts[0]:sum(counts)]:
        record['split'] = 'test'
    return variants + edited


def _variant_config(config, variant):
    result = deepcopy(config)
    for key in ('operation', 'distance_mm', 'shape_k', 'shape_window', 'profile', 'profile_axis'):
        result['edit'][key] = variant[key]
    for key in _EROSION_GUARD_FIELDS:
        if variant['operation'] != 'erosion':
            result['edit'].pop(key, None)
        elif key in variant:
            result['edit'][key] = variant[key]
    return result


def _geometry(resolved):
    low, high = resolved['crop_low_ijk'], resolved['crop_high_ijk_exclusive']
    return {'array_order': 'kji', 'crop_origin_ijk': list(low),
            'crop_high_ijk_exclusive': list(high),
            'crop_shape_kji': list((np.asarray(high) - low)[::-1]),
            'source_shape_kji': list(resolved['shape_kji']),
            'spacing_ijk_mm': list(resolved['spacing_ijk_mm']),
            'roi_kind': resolved['roi_kind'], 'roi_nodes_ijk': resolved['roi_nodes_ijk'],
            'roi_radii_mm': resolved['roi_radii_mm'],
            'orientation': 'Native index grid; physical origin and anatomical orientation unverified'}


def _provenance(config, resolved):
    sources = {}
    for channel, path in sorted(resolved['source_files'].items()):
        path = Path(path)
        stat = path.stat()
        sources[channel] = {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    # Bind the actual files, rather than trusting a stale in-memory resolution.
    for key in ('catalog', 'audit'):
        if _file_hash(config['input'][key]) != resolved[key + '_sha256']:
            raise ValueError(f'{key} changed after source resolution')
    modules = ('fakect_training_data.py', 'fakect_morphology.py', 'fakect_reassignment.py', 'fakect_released.py', 'fakect_erosion_guard.py',
               'fakect_roi.py', 'fakect_tissues.py')
    metadata_sources = {str(resolved['case'][name]): _file_hash(resolved['case'][name])
                        for name in ('par_path', 'log_path') if name in resolved.get('case', {})}
    return {'input': config['input'], 'selection': config['selection'], 'roi': config['roi'],
            'edit': config['edit'], 'reassignment': config['reassignment'],
            'train': {key: config['train'][key] for key in _DATA_TRAIN_KEYS},
            'geometry': _geometry(resolved), 'source_ids': list(resolved['source_ids']),
            'catalog_sha256': resolved['catalog_sha256'], 'audit_sha256': resolved['audit_sha256'],
            'source_files': sources, 'source_metadata_sha256': metadata_sources,
            'source_integrity_scope': 'File size/mtime plus saved crop hashes; full source volumes are not hashed',
            'code_sha256': {name: _file_hash(Path(__file__).with_name(name)) for name in modules}}


def dataset_fingerprint(config, resolved):
    """Bind data semantics, source state, atlas, geometry and generating code.

    Training stage, model hyperparameters, run/output directories and display
    settings do not alter prepared data and are deliberately excluded.
    """
    return hashlib.sha256(_canonical(_provenance(config, resolved)).encode()).hexdigest()


def _target_ids(config, resolved):
    values = config['train']['target_source_ids']
    if not values or len(set(values)) != len(values):
        raise ValueError('train.target_source_ids must explicitly list distinct original signed labels')
    ids = []
    records = {record['original_id']: record for record in resolved['catalog']['records']}
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not -(2**31) <= value < 2**31:
            raise ValueError('train.target_source_ids must contain int32 original label IDs')
        if value not in records or value == 0 or records[value]['classification']['tissue_name'] != config['selection']['tissue']:
            raise ValueError(f'Training target ID {value} is absent or outside the selected tissue')
        ids.append(int(value))
    if not set(resolved['source_ids']).issubset(ids):
        raise ValueError('Edit source_ids must be a subset of train.target_source_ids; explicitly select the reviewed anatomy')
    return tuple(ids)


def validate_study_plan(config, resolved):
    """Validate every planned variant against resolved metadata, without crop I/O.

    Target identity, edit-candidate scope, all morphology policies, crop bounds
    and required halos must pass before a caller advertises a preparable plan.
    The returned whole-variant splits remain provisional until native geometry
    duplicates can be identified during preparation.
    """
    if config.get('edit', {}).get('assign_surrounding_tissue', True) is False:
        raise ValueError('Diagnostic released labels have unassigned attenuation and cannot form training pairs; use preview_roi.py for diagnostic edits or set assign_surrounding_tissue=true')
    variants = plan_variants(config)
    _target_ids(config, resolved)
    for variant in variants:
        validate_edit_geometry(resolved, _variant_config(config, variant))
    return variants


def _write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(_json_value(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _write_npz(path, **arrays):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def prepare_training_dataset(config, resolved, *, input_bytes=None):
    """Prepare a fresh, bounded, immutable crop dataset; never train a model.

    Each variant starts from the same read-only original arrays. Files are
    atomically published inside a newly reserved directory; the manifest is
    written last. A failure leaves an INCOMPLETE marker and is never accepted
    as a finished dataset. Existing directories are never overwritten. Optional
    ``input_bytes`` saves the exact CLI-captured input as ``input.ini`` and binds
    it into the artifact inventory; no newline, encoding, or comment rewriting
    occurs. The caller supplies bytes corresponding to the parsed configuration.
    """
    if input_bytes is not None and not isinstance(input_bytes, bytes):
        raise ValueError('input_bytes must contain the exact captured input bytes')
    variants = validate_study_plan(config, resolved)
    targets = tuple(map(int, config['train']['target_source_ids']))
    provenance = _provenance(config, resolved)
    fingerprint = hashlib.sha256(_canonical(provenance).encode()).hexdigest()
    output = Path(config['train']['dataset_directory']).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Training dataset directory exists; choose a new dataset_directory')
    arrays, sources = prepare_crop(resolved, config)
    from fakect_released import RELEASED_LABEL_ID
    if np.any(arrays['act'] == RELEASED_LABEL_ID):
        raise ValueError('Diagnostic released voxels have unassigned attenuation and cannot form training pairs')
    original_mask = np.isin(arrays['act'], targets)
    if not np.any(original_mask):
        raise ValueError('Training target is absent from the complete crop')
    geometry = _geometry(resolved)
    output.mkdir(parents=True, exist_ok=False)
    marker = output / 'INCOMPLETE'
    marker.write_text('Dataset preparation is incomplete; do not train from these files.\n')
    (output / 'variants').mkdir()
    if input_bytes is not None:
        temporary_input = output / 'input.ini.tmp'
        with temporary_input.open('xb') as handle:
            handle.write(input_bytes)
        temporary_input.replace(output / 'input.ini')
    _write_npz(output / 'source.npz', original_labels=arrays['act'], original_attenuation_per_pixel=arrays['atn'],
               original_tissue_labels=arrays['tissue'], roi_mask=arrays['roi'],
               target_mask=original_mask.astype(np.uint8),
               geometry_json=np.array(_canonical(geometry)),
               catalog_json=np.array(resolved['catalog_bytes'].decode('utf-8')))
    _write_json(output / 'data-config.json', provenance)
    groups, samples = {}, []
    baseline_mask_hash = _array_hash(original_mask.astype(np.uint8))
    for variant in variants:
        result = apply_morphology(arrays, resolved, _variant_config(config, variant))
        if np.any(result.get('attenuation_unassigned_mask', False)):
            raise ValueError('Diagnostic released voxels have unassigned attenuation and cannot form training pairs')
        labels = result['edited_labels']
        changed = result['changed_mask']
        mask = np.isin(labels, targets).astype(np.uint8)
        image = np.asarray(result['attenuation_proxy_per_pixel'] /
                           (resolved['spacing_ijk_mm'][0] / 10), dtype=np.float32)
        if not np.all(np.isfinite(image)):
            raise ValueError('Nonfinite attenuation after inverse-centimetre conversion')
        if np.any(changed & ~arrays['roi']) or not np.array_equal(labels[~changed], arrays['act'][~changed]):
            raise ValueError('Morphology violated the original-label locality contract')
        if not np.array_equal(result['attenuation_proxy_per_pixel'][~changed], arrays['atn'][~changed]):
            raise ValueError('Morphology modified attenuation outside accepted changes')
        mask_hash, image_hash = _array_hash(mask), _array_hash(image)
        if mask_hash == baseline_mask_hash:
            groups[mask_hash] = 'train'
        actual_split = groups.setdefault(mask_hash, variant['split'])
        group = f"{config['train']['anatomy_family']}:{mask_hash}"
        metadata = {**variant, 'planned_split': variant['split'], 'split': actual_split,
                    'geometry_group': group, 'mask_sha256': mask_hash, 'image_sha256': image_hash,
                    'target_source_ids': list(targets), 'foreground_voxels': int(mask.sum()),
                    'image_method': 'attenuation_copy_proxy', 'image_units': 'cm^-1',
                    'image_warning': IMAGE_WARNING, 'mask_semantics': 'Membership in edited original target IDs over the complete crop; not ROI-clipped',
                    'geometry': geometry, 'dataset_fingerprint': fingerprint,
                    'morphology': result['summary'], 'zero_change': not bool(changed.any())}
        path = output / 'variants' / (variant['variant_id'] + '.npz')
        _write_npz(path, image=image, mask=mask, edited_labels=labels, changed_mask=changed,
                   metadata_json=np.array(_canonical(metadata)))
        samples.append({**metadata, 'path': str(path.relative_to(output)), 'sha256': _file_hash(path)})
        del result, image, mask, labels, changed
    # Reject source/code/catalog changes during preparation before publishing a
    # complete manifest. Full-volume content is outside this bounded scope.
    if dataset_fingerprint(config, resolved) != fingerprint:
        raise ValueError('Data-generating inputs or code changed during preparation; partial outputs retained')
    artifacts = {str(path.relative_to(output)): _file_hash(path)
                 for path in sorted(output.rglob('*')) if path.is_file() and path != marker}
    split_counts = {name: sum(sample['split'] == name for sample in samples)
                    for name in ('train', 'validation', 'test')}
    geometry_counts = {name: len({sample['geometry_group'] for sample in samples if sample['split'] == name})
                       for name in split_counts}
    warnings = [SPLIT_WARNING, IMAGE_WARNING]
    if any(sample['planned_split'] != sample['split'] for sample in samples):
        warnings.append('Exact mask duplicates were reassigned to one geometry-group split; baseline-equivalent masks remain in training.')
    if not split_counts['validation']:
        warnings.append('No distinct validation geometry remains; fitting must not proceed with this dataset.')
    if config['train']['test_fraction'] > 0 and not split_counts['test']:
        warnings.append('No distinct test geometry remains after duplicate grouping; this dataset has no final test holdout.')
    manifest = {'schema_version': SCHEMA, 'complete': True, 'dataset_fingerprint': fingerprint,
                'input_snapshot': 'input.ini' if input_bytes is not None else None,
                'input_config_sha256': hashlib.sha256(input_bytes).hexdigest() if input_bytes is not None else None,
                'anatomy_family': config['train']['anatomy_family'], 'case_id': config['input']['case_id'],
                'frame': config['input']['frame'], 'split_mode': 'scenario_only',
                'split_warning': SPLIT_WARNING, 'image_method': 'attenuation_copy_proxy',
                'image_units': 'cm^-1', 'image_warning': IMAGE_WARNING,
                'target_source_ids': list(targets), 'geometry': geometry, 'source_files': sources,
                'catalog_sha256': resolved['catalog_sha256'], 'audit_sha256': resolved['audit_sha256'],
                'provenance': provenance, 'samples': samples, 'sample_count': len(samples),
                'split_counts': split_counts, 'geometry_group_counts': geometry_counts,
                'zero_change_count': sum(sample['zero_change'] for sample in samples),
                'original_sources_modified': False, 'variants_start_from_original': True,
                'warnings': warnings, 'artifacts_sha256': artifacts}
    manifest_path = output / 'dataset-manifest.json'
    _write_json(manifest_path, manifest)
    checksum = _file_hash(manifest_path)
    (output / 'dataset-manifest.sha256').write_text(checksum + '\n')
    marker.unlink()
    return {'manifest_path': str(manifest_path), 'manifest_sha256': checksum,
            'dataset_directory': str(output), 'dataset_fingerprint': fingerprint,
            'sample_count': len(samples), 'split_counts': split_counts,
            'geometry_group_counts': geometry_counts, 'zero_change_count': manifest['zero_change_count'],
            'warnings': warnings, 'samples': samples}


def validate_dataset(config, resolved, manifest_path=None):
    """Verify a frozen dataset and all artifact bytes before reuse or fitting.

    This checks test archive integrity as bytes; it does not load test arrays or
    use their values for fitting, normalization, or model selection.
    """
    path = Path(manifest_path) if manifest_path is not None else Path(config['train']['dataset_directory']) / 'dataset-manifest.json'
    path = path.expanduser().resolve()
    directory = path.parent
    if (directory / 'INCOMPLETE').exists():
        raise ValueError('Training dataset is marked INCOMPLETE')
    expected = (directory / 'dataset-manifest.sha256').read_text().strip()
    if _file_hash(path) != expected:
        raise ValueError('Dataset manifest checksum mismatch')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != SCHEMA or manifest.get('complete') is not True:
        raise ValueError('Unsupported or incomplete training dataset manifest')
    if manifest.get('dataset_fingerprint') != dataset_fingerprint(config, resolved):
        raise ValueError('Prepared dataset does not match current data configuration/source/catalog/code')
    snapshot = manifest.get('input_snapshot')
    if snapshot is not None and (snapshot != 'input.ini' or
            manifest['artifacts_sha256'].get(snapshot) != manifest.get('input_config_sha256')):
        raise ValueError('Input snapshot checksum disagrees with dataset artifact inventory')
    for relative, digest in manifest['artifacts_sha256'].items():
        artifact = (directory / relative).resolve()
        if directory not in artifact.parents or not artifact.is_file() or _file_hash(artifact) != digest:
            raise ValueError(f'Dataset artifact is missing, unsafe, or changed: {relative}')
    samples = manifest['samples']
    if len(samples) != manifest['sample_count']:
        raise ValueError('Dataset sample count mismatch')
    seen_ids, paths, group_splits, mask_splits = set(), set(), {}, {}
    for sample in samples:
        identity, relative = sample['variant_id'], sample['path']
        if identity in seen_ids or relative in paths:
            raise ValueError('Duplicate sample identity or path in dataset manifest')
        seen_ids.add(identity)
        paths.add(relative)
        if sample['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown dataset split')
        if manifest['artifacts_sha256'].get(relative) != sample['sha256']:
            raise ValueError('Sample checksum disagrees with artifact inventory')
        for groups, key in ((group_splits, sample['geometry_group']), (mask_splits, sample['mask_sha256'])):
            if groups.setdefault(key, sample['split']) != sample['split']:
                raise ValueError('Duplicate geometry crosses dataset splits')
    return manifest
