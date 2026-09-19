"""Generation-only recipes and an explicit reviewed-input freeze boundary."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from fakect_config import REPOSITORY_ROOT, _path

SCHEMA = 'fakect.recipe-cohort/1'
FIELDS = {'stage', 'preflight_directory', 'dataset_directory', 'freeze_directory',
          'anatomy_family', 'target_scope', 'include_baseline', 'max_variants',
          'image_method', 'parameters_reviewed'}


def parse_cohort_sections(parser, root=REPOSITORY_ROOT):
    from fakect_recipe_study_config import parse_recipe_study_sections
    if parser.defaults() or 'model' in parser:
        raise ValueError('Generation-only cohorts have no [DEFAULT] or [model]; use a separate model experiment')
    if 'train' not in parser or set(parser['train']) != FIELDS:
        actual = set(parser['train']) if 'train' in parser else set()
        raise ValueError(f'[train] unknown={sorted(actual-FIELDS)}, missing={sorted(FIELDS-actual)}')
    train = dict(parser['train'])
    if train['stage'] not in ('plan', 'preview', 'preflight', 'freeze', 'prepare'):
        raise ValueError('Cohort train.stage must be plan, preview, preflight, freeze, or prepare')
    if train['parameters_reviewed'] not in ('true', 'false'):
        raise ValueError('train.parameters_reviewed must be true or false')
    # Reuse the strict recipe/sweep parser; these compatibility defaults never
    # become a generation model or split contract in the returned config.
    adapted = deepcopy(parser)
    adapted['study']['schema_version'] = 'fakect.recipe-study/1'
    adapted['train'].pop('freeze_directory')
    adapted['train'].pop('parameters_reviewed')
    adapted['train'].update(stage='plan', model_directory=train['dataset_directory'],
                            split_mode='scenario_only', validation_fraction='0.2',
                            test_fraction='0.2', split_seed='0')
    adapted['model'] = dict(architecture='unet2d', patch_size='64,64', slice_axis='k',
                            normalization='fixed_clip', clip_min='0', clip_max='0.5',
                            epochs='1', batch_size='4', learning_rate='0.001', seed='0')
    config = parse_recipe_study_sections(adapted, root)
    config['study']['schema_version'] = SCHEMA
    config.pop('model')
    config['train'].pop('model_directory')
    config['train'].update(stage=train['stage'], split_mode='unassigned',
                           validation_fraction=0., test_fraction=0., split_seed=0,
                           parameters_reviewed=train['parameters_reviewed'] == 'true',
                           freeze_directory=_path(train['freeze_directory'], 'train.freeze_directory', Path(root)))
    return config


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_cohort(config, resolved, input_bytes):
    """Freeze an explicitly reviewed recipe only after matching native preflight."""
    from fakect_recipe_training import _fingerprint
    from fakect_training_data import _write_json
    if not config['roi']['coordinate_reviewed'] or not config['train']['parameters_reviewed']:
        raise ValueError('Review the ROI and native parameter outcomes, then set roi.coordinate_reviewed=true and train.parameters_reviewed=true')
    folder = config['train']['preflight_directory']
    if (folder/'INCOMPLETE').exists():
        raise ValueError('Preflight is INCOMPLETE')
    inventory = json.loads((folder/'artifact-manifest.json').read_text())
    for name, checksum in inventory.items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or _sha(path) != checksum:
            raise ValueError(f'Preflight artifact is unsafe or changed: {name}')
    if 'preflight.json' not in inventory:
        raise ValueError('Preflight inventory does not cover its result')
    preflight = json.loads((folder/'preflight.json').read_text())
    fingerprint = _fingerprint(config, resolved)
    if (not preflight.get('complete') or not preflight.get('execution_valid') or
            preflight.get('failed_variants') != 0 or preflight.get('dataset_fingerprint') != fingerprint):
        raise ValueError('Run native preflight successfully for these exact data-generating settings before freezing')
    output = config['train']['freeze_directory']
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').write_text('Input freeze is incomplete\n')
    (output/'input.ini').write_bytes(input_bytes)
    (output/'preflight.json').write_bytes((folder/'preflight.json').read_bytes())
    lock = {'schema_version': 'fakect.cohort-freeze/1', 'dataset_fingerprint': fingerprint,
            'case_id': config['input']['case_id'], 'frame': config['input']['frame'],
            'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
            'preflight_sha256': _sha(output/'preflight.json'),
            'variant_count': preflight['variant_count'],
            'coordinate_reviewed': True, 'parameters_reviewed': True}
    _write_json(output/'freeze.json', lock)
    (output/'freeze.sha256').write_text(_sha(output/'freeze.json')+'\n')
    (output/'INCOMPLETE').unlink()
    return {'stage': 'freeze', 'freeze_path': str(output/'freeze.json'), **lock}


def validate_cohort_freeze(config, resolved, input_bytes):
    from fakect_recipe_training import _fingerprint
    folder = config['train']['freeze_directory']
    if (folder/'INCOMPLETE').exists():
        raise ValueError('Cohort freeze is INCOMPLETE')
    try:
        lock = json.loads((folder/'freeze.json').read_text())
        valid = (lock.get('schema_version') == 'fakect.cohort-freeze/1' and
                 _sha(folder/'freeze.json') == (folder/'freeze.sha256').read_text().strip() and
                 input_bytes is not None and _sha(folder/'input.ini') == lock['input_sha256'] == hashlib.sha256(input_bytes).hexdigest() and
                 _sha(folder/'preflight.json') == lock['preflight_sha256'] and
                 lock['dataset_fingerprint'] == _fingerprint(config, resolved) and
                 config['roi']['coordinate_reviewed'] and config['train']['parameters_reviewed'])
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError('A complete matching input freeze is required; run --stage freeze first') from error
    if not valid:
        raise ValueError('Frozen cohort input, preflight, or generation dependencies changed; create a new reviewed revision')
    return lock
