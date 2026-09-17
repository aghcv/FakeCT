"""Strict staged study inputs sharing the existing ROI and morphology contract."""
import configparser
from pathlib import Path

from fakect_config import (REPOSITORY_ROOT, _float, _integer, _name, _parts, _path,
                           _IDENTIFIER, parse_preview_sections)
from fakect_segmentation import validate_clip_bounds


SCHEMA = 'fakect.study/1'
TRAIN_FIELDS = {'stage', 'dataset_directory', 'model_directory', 'target_source_ids',
                'anatomy_family', 'operations', 'distances_mm', 'shape_ks', 'include_baseline',
                'max_variants', 'image_method', 'split_mode', 'validation_fraction',
                'test_fraction', 'split_seed'}
MODEL_FIELDS = {'architecture', 'patch_size', 'slice_axis', 'normalization',
                'clip_min', 'clip_max', 'epochs', 'batch_size', 'learning_rate', 'seed'}
STAGES = {'preview', 'plan', 'prepare', 'fit'}


def load_study_config(path, *, repo_root=None):
    """Load one human-authored study; no images, TensorFlow or outputs are touched."""
    root = Path(repo_root or REPOSITORY_ROOT).expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       strict=True, empty_lines_in_values=False)
    parser.optionxform = str
    try:
        with Path(path).expanduser().open(encoding='utf-8-sig') as stream:
            parser.read_file(stream)
    except configparser.Error as error:
        raise ValueError(f'Invalid study input: {error}') from error
    if parser.defaults():
        raise ValueError('[DEFAULT] settings are unsupported')
    if 'study' not in parser or parser['study'].get('schema_version') != SCHEMA:
        raise ValueError(f'study.schema_version must be {SCHEMA}')
    for section, expected in [('train', TRAIN_FIELDS), ('model', MODEL_FIELDS)]:
        if section not in parser:
            raise ValueError(f'Missing [{section}] section')
        actual = set(parser[section])
        if actual != expected:
            raise ValueError(f'[{section}] invalid settings: unknown={sorted(actual-expected)}, missing={sorted(expected-actual)}')
        if any('\n' in value for value in parser[section].values()):
            raise ValueError(f'[{section}] settings must each be on one line')
    train = dict(parser['train'])
    model = dict(parser['model'])
    parser.remove_section('train')
    parser.remove_section('model')
    parser['study']['schema_version'] = 'fakect.edit/1'
    config = parse_preview_sections(parser, root)
    config['study']['schema_version'] = SCHEMA
    if train['stage'] not in STAGES:
        raise ValueError('train.stage must be preview, plan, prepare, or fit')
    target = tuple(_integer(v, 'train.target_source_ids', -(2**31), 2**31-1)
                   for v in _parts(train['target_source_ids'], 'train.target_source_ids'))
    if not target or len(set(target)) != len(target):
        raise ValueError('train.target_source_ids must contain explicit unique original signed IDs')
    operations = _parts(train['operations'], 'train.operations')
    if not operations or set(operations) - {'erosion', 'dilation'} or len(set(operations)) != len(operations):
        raise ValueError('train.operations must contain erosion and/or dilation without duplicates')
    distances = tuple(_float(v, 'train.distances_mm', positive=True)
                      for v in _parts(train['distances_mm'], 'train.distances_mm'))
    shapes = tuple(_float(v, 'train.shape_ks', positive=True)
                   for v in _parts(train['shape_ks'], 'train.shape_ks'))
    if not distances or len(set(distances)) != len(distances) or max(distances) > 50:
        raise ValueError('train.distances_mm must contain unique positive values <= 50 mm')
    if not shapes or len(set(shapes)) != len(shapes):
        raise ValueError('train.shape_ks must contain unique positive values')
    if config['edit']['profile'] == 'uniform' and len(shapes) != 1:
        raise ValueError('A uniform profile ignores shape_k; use one train.shape_ks value')
    if train['include_baseline'] not in ('true', 'false'):
        raise ValueError('train.include_baseline must be true or false')
    if train['image_method'] != 'attenuation_copy_proxy':
        raise ValueError('Only the explicitly experimental attenuation_copy_proxy image method is implemented')
    if train['split_mode'] != 'scenario_only':
        raise ValueError('This one-anatomy prototype supports scenario_only splits; independent-family studies need a future manifest')
    val = _float(train['validation_fraction'], 'train.validation_fraction', opacity=True)
    test = _float(train['test_fraction'], 'train.test_fraction', opacity=True)
    if val <= 0 or test <= 0 or val + test >= 1:
        raise ValueError('Use positive validation/test fractions whose sum is < 1')
    config['train'] = {
        'stage': train['stage'], 'dataset_directory': _path(train['dataset_directory'], 'train.dataset_directory', root),
        'model_directory': _path(train['model_directory'], 'train.model_directory', root),
        'target_source_ids': target, 'anatomy_family': _name(train['anatomy_family'], 'train.anatomy_family', _IDENTIFIER),
        'operations': operations, 'distances_mm': distances, 'shape_ks': shapes,
        'include_baseline': train['include_baseline'] == 'true',
        'max_variants': _integer(train['max_variants'], 'train.max_variants', 1, 1000),
        'image_method': train['image_method'], 'split_mode': train['split_mode'],
        'validation_fraction': val, 'test_fraction': test,
        'split_seed': _integer(train['split_seed'], 'train.split_seed', 0, 2**31-1),
    }
    if model['architecture'] != 'unet2d' or model['slice_axis'] != 'k':
        raise ValueError('The initial model supports architecture=unet2d and slice_axis=k')
    if model['normalization'] != 'fixed_clip':
        raise ValueError('model.normalization must be fixed_clip')
    patch = tuple(_integer(v, 'model.patch_size', 16, 512) for v in _parts(model['patch_size'], 'model.patch_size'))
    if len(patch) != 2 or any(v % 4 for v in patch):
        raise ValueError('model.patch_size needs two sizes divisible by four, in j,i order')
    clip_min, clip_max = validate_clip_bounds(model['clip_min'], model['clip_max'])
    config['model'] = {
        'architecture': model['architecture'], 'patch_size': patch, 'slice_axis': model['slice_axis'],
        'normalization': model['normalization'], 'clip_min': clip_min, 'clip_max': clip_max,
        'epochs': _integer(model['epochs'], 'model.epochs', 1, 10000),
        'batch_size': _integer(model['batch_size'], 'model.batch_size', 1, 256),
        'learning_rate': _float(model['learning_rate'], 'model.learning_rate', positive=True),
        'seed': _integer(model['seed'], 'model.seed', 0, 2**31-1),
    }
    return config
