"""Model experiments over immutable prepared datasets, independent of generation.

The registry supplies reviewed dataset versions. Planning verifies their saved
artifacts and assigns complete anatomy families to splits. Fitting consumes the
frozen lock; it does not consult XCAT sources, ROI files, or generator code.
"""
import configparser
import hashlib
import json
import math
from pathlib import Path
import re

from fakect_segmentation import MAX_VOLUME_VOXELS, model_settings


SCHEMA = 'fakect.model-experiment/1'
LOCK_SCHEMA = 'fakect.model-experiment-lock/1'
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
EXPERIMENT_FIELDS = {'schema_version', 'registry', 'datasets', 'output_directory'}
SPLIT_FIELDS = {'mode', 'validation_fraction', 'test_fraction', 'seed'}
MODEL_FIELDS = {'architecture', 'patch_size', 'slice_axis', 'normalization',
                'clip_min', 'clip_max', 'epochs', 'batch_size', 'learning_rate', 'seed'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _integer(value, label, minimum=0):
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError(f'{label} must be an integer >= {minimum}')
    result = int(value)
    if result < minimum or result > 2**31-1:
        raise ValueError(f'{label} must be between {minimum} and 2147483647')
    return result


def load_model_experiment_config(path, *, repo_root=None):
    """Parse a model-only INI without resolving datasets or opening volumes."""
    root = Path(repo_root or REPOSITORY_ROOT).expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       strict=True, empty_lines_in_values=False)
    parser.optionxform = str
    try:
        with Path(path).expanduser().open(encoding='utf-8-sig') as stream:
            parser.read_file(stream)
    except configparser.Error as exc:
        raise ValueError(f'Invalid model experiment INI: {exc}') from exc
    if parser.defaults() or set(parser.sections()) != {'experiment', 'split', 'model'}:
        raise ValueError('Model experiments require only [experiment], [split], and [model]; no ROI or generation inputs')
    for section, fields in (('experiment', EXPERIMENT_FIELDS), ('split', SPLIT_FIELDS), ('model', MODEL_FIELDS)):
        found = set(parser[section])
        if found != fields:
            raise ValueError(f'[{section}] unknown={sorted(found-fields)}, missing={sorted(fields-found)}')
        if any('\n' in value for value in parser[section].values()):
            raise ValueError(f'[{section}] values must each occupy one line')
    experiment, split, model = (dict(parser[key]) for key in ('experiment', 'split', 'model'))
    if experiment['schema_version'] != SCHEMA:
        raise ValueError(f'experiment.schema_version must be {SCHEMA}')
    names = [name.strip() for name in experiment['datasets'].split(',')]
    if not names or any(not NAME.fullmatch(name) for name in names) or len(set(names)) != len(names):
        raise ValueError('experiment.datasets must be unique comma-separated registry names')
    experiment['datasets'] = names
    for name in ('registry', 'output_directory'):
        if not experiment[name].strip():
            raise ValueError(f'experiment.{name} cannot be blank')
        location = Path(experiment[name]).expanduser()
        experiment[name] = str((location if location.is_absolute() else root/location).resolve())
    if split['mode'] != 'anatomy_family':
        raise ValueError('split.mode must be anatomy_family; prepared scenario assignments are not reused')
    val, test = float(split['validation_fraction']), float(split['test_fraction'])
    if not math.isfinite(val+test) or val <= 0 or test <= 0 or val+test >= 1:
        raise ValueError('Validation/test fractions must be positive and sum to less than one')
    split.update(validation_fraction=val, test_fraction=test, seed=_integer(split['seed'], 'split.seed'))
    try:
        typed_model = {**model, 'patch_size': [_integer(v.strip(), 'model.patch_size', 4)
                                             for v in model['patch_size'].split(',')],
                       **{key: _integer(model[key], f'model.{key}', 0 if key == 'seed' else 1)
                          for key in ('epochs', 'batch_size', 'seed')},
                       **{key: float(model[key]) for key in ('clip_min', 'clip_max', 'learning_rate')}}
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid model setting: {exc}') from exc
    return {'experiment': experiment, 'split': split, 'model': model_settings({'model': typed_model})}


def _family_splits(families, settings):
    if len(families) < 3:
        raise ValueError('At least three verified independent anatomy families are required for train, validation, and test')
    ordered = sorted(families, key=lambda family: hashlib.sha256(
        f'{settings["seed"]}:{family}'.encode()).hexdigest())
    n_validation = max(1, int(len(ordered)*settings['validation_fraction']))
    n_test = max(1, int(len(ordered)*settings['test_fraction']))
    if n_validation+n_test >= len(ordered):
        raise ValueError('Family fractions leave no training family')
    return {family: ('validation' if i < n_validation else 'test' if i < n_validation+n_test else 'train')
            for i, family in enumerate(ordered)}


def _contract(manifest):
    geometry = manifest['geometry']
    spacing = geometry.get('spacing_ijk_mm')
    if (not isinstance(spacing, (list, tuple)) or len(spacing) != 3
            or any(not math.isfinite(float(v)) or float(v) <= 0 for v in spacing)):
        raise ValueError('Every prepared dataset needs positive finite spacing_ijk_mm')
    shape = geometry.get('crop_shape_kji')
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape)
            or math.prod(shape) > MAX_VOLUME_VOXELS):
        raise ValueError('Prepared crop_shape_kji exceeds the bounded segmentation volume contract')
    if manifest.get('image_units') != 'cm^-1' or geometry.get('array_order') != 'kji':
        raise ValueError('Model inputs require cm^-1 images in kji array order')
    orientation = geometry.get('orientation')
    if not isinstance(orientation, str) or not orientation:
        raise ValueError('Every prepared dataset must declare its native orientation convention')
    return {'image_units': manifest['image_units'], 'array_order': geometry['array_order'],
            'spacing_ijk_mm': list(spacing), 'orientation': orientation,
            'image_method': manifest.get('image_method'), 'target_scope': manifest.get('target_scope'),
            'target_semantics': manifest.get('target_semantics')}


def _source_identity(entry, manifest):
    """Known shared source identities cannot acquire a new family via a receipt."""
    result = [('case_id', str(manifest['case_id']))]
    sources = manifest.get('source_files', manifest.get('provenance', {}).get('source_files', {}))
    for channel, info in sources.items():
        if isinstance(info, dict) and info.get('path'):
            result.append((f'source_path_{channel}', str(Path(info['path']).expanduser().resolve())))
        if isinstance(info, dict) and info.get('sha256'):
            result.append((f'source_sha256_{channel}', info['sha256']))
    hashes = manifest.get('source_crop_sha256') or entry.get('source_crop_sha256') or {}
    if hashes:
        result.append(('source_crop', _canonical(hashes)))
    return result


def plan_model_experiment(config, *, verify_payloads=True):
    """Return all missing/unverified inputs together; never allocate outputs."""
    from fakect_cohort_registry import load_registry_entry
    settings = config['experiment']
    result = {'schema_version': SCHEMA, 'ready': False, 'model_fitted': False,
              'outputs_written': False, 'missing_datasets': [], 'unverified_datasets': [],
              'errors': [], 'datasets': [], 'samples': [], 'family_assignment': {},
              'split_counts': {'train': 0, 'validation': 0, 'test': 0},
              'warnings': ['Family verification is a recorded provenance decision; distinct case IDs alone do not prove independent anatomy.',
                           'Saved per-dataset scenario splits are ignored. All variants of each anatomy family receive one experiment split.']}
    families, identity_families, geometry_families = set(), {}, {}
    common = None
    for name in settings['datasets']:
        try:
            entry = load_registry_entry(settings['registry'], name, verify=verify_payloads)
        except (FileNotFoundError, KeyError):
            result['missing_datasets'].append(name)
            continue
        except (ValueError, OSError) as exc:
            result['errors'].append(f'{name}: {exc}')
            continue
        if not entry.get('family_verified', False):
            result['unverified_datasets'].append(name)
        family = entry['anatomy_family']
        families.add(family)
        try:
            manifest = entry.get('manifest')
            if manifest is None:
                from fakect_cohort_registry import verify_prepared_manifest
                manifest = verify_prepared_manifest(entry['manifest_path'], verify_payloads=verify_payloads)
            contract = _contract(manifest)
            if common is not None and common != contract:
                raise ValueError('Dataset units, spacing, orientation, image method, or target semantics differ; use a separately versioned harmonized dataset')
            common = contract
            for identity in _source_identity(entry, manifest):
                previous = identity_families.setdefault(identity, family)
                if previous != family:
                    raise ValueError(f'Shared source identity {identity[0]} is assigned to multiple anatomy families')
            source = {'name': name, 'anatomy_family': family, 'family_verified': entry.get('family_verified', False),
                      'manifest_path': str(Path(entry['manifest_path']).resolve()),
                      'manifest_sha256': entry['manifest_sha256'],
                      'dataset_fingerprint': entry['dataset_fingerprint'],
                      'registry_entry_path': str(Path(entry['registry_entry_path']).resolve()),
                      'registry_entry_sha256': _sha256(entry['registry_entry_path']),
                      'case_id': manifest['case_id'], 'frame': manifest['frame'], 'geometry': manifest['geometry']}
            rows = []
            for sample in manifest['samples']:
                shape = sample.get('shape_kji', manifest['geometry']['crop_shape_kji'])
                for key in (('sample_bytes', sample['sha256']),
                            ('mask_geometry', _canonical([shape, sample['mask_sha256'], contract['spacing_ijk_mm']]))):
                    previous = geometry_families.setdefault(key, family)
                    if previous != family:
                        raise ValueError('Duplicate sample bytes or target geometry occurs across anatomy families; refusing split leakage')
                rows.append({'variant_id': f'{name}:{sample["variant_id"]}',
                             'source_variant_id': sample['variant_id'], 'dataset_name': name,
                             'path': str(Path(sample['_path']).resolve()), 'anatomy_family': family,
                             'case_id': manifest['case_id'], 'frame': manifest['frame'],
                             'geometry_group': f'{family}:{sample["mask_sha256"]}',
                             'shape_kji': list(shape), 'sha256': sample['sha256'],
                             'mask_sha256': sample['mask_sha256'], 'image_sha256': sample['image_sha256']})
            result['datasets'].append(source)
            result['samples'].extend(rows)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            result['errors'].append(f'{name}: {exc}')
    if result['missing_datasets']:
        result['errors'].append('Prepare and register the missing dataset versions before locking this experiment')
    if result['unverified_datasets']:
        result['errors'].append('Verify source anatomy provenance and register reviewed family identities before making independent-family splits')
    try:
        result['family_assignment'] = _family_splits(families, config['split'])
    except ValueError as exc:
        result['errors'].append(str(exc))
    if not result['errors']:
        for sample in result['samples']:
            sample['split'] = result['family_assignment'][sample['anatomy_family']]
            result['split_counts'][sample['split']] += 1
        if not all(result['split_counts'].values()):
            result['errors'].append('The experiment requires nonempty train, validation, and test splits')
        else:
            result['ready'] = True
    result['input_contract'] = common
    result['family_count'] = len(families)
    result['sample_count'] = len(result['samples'])
    result['verified_payloads'] = verify_payloads
    return result


def freeze_model_experiment(config, input_bytes):
    """Create an immutable manifest of dataset versions, family splits and model."""
    plan = plan_model_experiment(config)
    if not plan['ready']:
        raise ValueError('Experiment is not ready: ' + '; '.join(plan['errors']))
    output = Path(config['experiment']['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    lock = {'schema_version': LOCK_SCHEMA, 'config': config,
            'input_config_sha256': hashlib.sha256(input_bytes).hexdigest(),
            'input_snapshot': 'input.ini', 'plan': plan,
            'model_directory': str(output/'model'),
            'integrity_scope': 'Prepared dataset and registry byte hashes; no dependency on raw sources or current generation code.'}
    lock['lock_fingerprint'] = hashlib.sha256(_canonical(lock).encode()).hexdigest()
    path = output/'experiment-lock.json'
    (output/'input.ini').write_bytes(input_bytes)
    path.write_text(json.dumps(lock, indent=2, allow_nan=False)+'\n')
    (output/'experiment-lock.sha256').write_text(_sha256(path)+'\n')
    return {'ready': True, 'lock_path': str(path), 'lock_sha256': _sha256(path),
            'family_assignment': plan['family_assignment'], 'split_counts': plan['split_counts'],
            'sample_count': plan['sample_count'], 'model_fitted': False, 'outputs_written': True}


def inspect_experiment_lock(path):
    """Validate frozen identities without opening any sample NPZ payloads.

    Train/validation samples are hash-checked when PatchDataset loads them.
    The test partition is reserved; its arrays are not opened during fitting.
    """
    from fakect_cohort_registry import load_registry_entry, verify_prepared_manifest
    path = Path(path).resolve()
    lock_hash = _sha256(path)
    if (path.parent/'experiment-lock.sha256').read_text().strip() != lock_hash:
        raise ValueError('Experiment lock checksum changed; create a new experiment')
    lock = json.loads(path.read_text())
    if lock.get('schema_version') != LOCK_SCHEMA:
        raise ValueError('Unsupported model experiment lock schema')
    fingerprint = lock.pop('lock_fingerprint', None)
    if fingerprint != hashlib.sha256(_canonical(lock).encode()).hexdigest():
        raise ValueError('Experiment lock fingerprint changed')
    lock['lock_fingerprint'] = fingerprint
    if _sha256(path.parent/'input.ini') != lock['input_config_sha256']:
        raise ValueError('Frozen experiment INI changed')
    plan = lock['plan']
    if not plan.get('ready') or any(not d['family_verified'] for d in plan['datasets']):
        raise ValueError('Experiment lock must contain reviewed independent anatomy families')
    by_name = {}
    for source in plan['datasets']:
        entry = load_registry_entry(lock['config']['experiment']['registry'], source['name'], verify=False)
        for key in ('manifest_sha256', 'dataset_fingerprint', 'anatomy_family', 'family_verified'):
            if entry[key] != source[key]:
                raise ValueError(f'Registry dataset identity changed: {source["name"]} {key}')
        if (_sha256(source['registry_entry_path']) != source['registry_entry_sha256']
                or str(Path(entry['registry_entry_path']).resolve()) != source['registry_entry_path']
                or str(Path(entry['manifest_path']).resolve()) != source['manifest_path']
                or _sha256(source['manifest_path']) != source['manifest_sha256']):
            raise ValueError(f'Registry receipt or prepared manifest changed: {source["name"]}')
        manifest = verify_prepared_manifest(source['manifest_path'], verify_payloads=False)
        by_name[source['name']] = {s['variant_id']: s for s in manifest['samples']}
    samples, counts, families = [], {'train': 0, 'validation': 0, 'test': 0}, {}
    for frozen in plan['samples']:
        sample = by_name[frozen['dataset_name']][frozen['source_variant_id']]
        for key in ('sha256', 'image_sha256', 'mask_sha256', 'shape_kji'):
            if sample[key] != frozen[key]:
                raise ValueError(f'Frozen sample metadata changed: {frozen["variant_id"]} {key}')
        if str(Path(sample['_path']).resolve()) != frozen['path']:
            raise ValueError('Frozen sample path changed')
        split, family = frozen['split'], frozen['anatomy_family']
        if split not in counts or families.setdefault(family, split) != split:
            raise ValueError('Anatomy family crosses experiment splits')
        counts[split] += 1
        samples.append({**frozen, '_path': Path(frozen['path'])})
    if counts != plan['split_counts'] or not all(counts.values()):
        raise ValueError('Invalid frozen split counts')
    contract = plan['input_contract']
    data = {'schema_version': LOCK_SCHEMA, 'split_mode': 'anatomy_family',
            'anatomy_family': sorted(families), 'geometry': {
                **contract, 'crop_shapes_kji': sorted({tuple(s['shape_kji']) for s in samples})},
            'image_method': contract['image_method'], 'image_units': contract['image_units'],
            'family_assignment': plan['family_assignment'], 'source_datasets': plan['datasets']}
    return {'path': path, 'sha256': lock_hash, 'data': data, 'samples': samples,
            'shape_kji': None, 'split_sizes': counts, 'lock': lock}


def fit_model_experiment(config, input_bytes):
    """Explicit model-only fitting against an already frozen experiment."""
    from fakect_segmentation import train_segmentation
    path = Path(config['experiment']['output_directory'])/'experiment-lock.json'
    checked = inspect_experiment_lock(path)
    lock = checked['lock']
    if lock['config'] != config or hashlib.sha256(input_bytes).hexdigest() != lock['input_config_sha256']:
        raise ValueError('Model experiment INI differs from its frozen lock; create a new output_directory and plan')
    for sample in checked['samples']:
        if sample['split'] != 'test' and _sha256(sample['_path']) != sample['sha256']:
            raise ValueError(f'Prepared training/validation sample changed: {sample["variant_id"]}')
    return train_segmentation({'model': config['model'],
                               'train': {'model_directory': lock['model_directory']}}, path)
