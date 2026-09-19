"""Strict named-recipe studies and bounded, deterministic Cartesian sweeps.

Parsing and planning never read source voxels. Metadata validation checks every
complete recipe against one resolved crop; native preflight remains responsible
for target presence, overlaps and the actual accepted geometry.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re

from fakect_config import REPOSITORY_ROOT, _float, _integer, _name, _parts, _path, _IDENTIFIER
from fakect_recipe_config import parse_recipe_sections
from fakect_study_config import MODEL_FIELDS, parse_model_settings


SCHEMA = 'fakect.recipe-study/1'
STAGES = {'preview', 'plan', 'preflight', 'prepare', 'fit'}
MAX_VARIANTS = 1000
TRAIN_FIELDS = {'stage', 'dataset_directory', 'model_directory', 'preflight_directory',
                'target_scope', 'anatomy_family', 'include_baseline', 'max_variants',
                'image_method', 'split_mode', 'validation_fraction', 'test_fraction', 'split_seed'}
SWEEP_FIELDS = {'distance_mm', 'shape_k', 'iterations'}
_DISABLED_EDIT_FIELDS = {'assign_surrounding_tissue', 'min_volume_ratio', 'preserve_connectivity',
                         'backoff_factor', 'max_backoff_steps'}
_DECIMAL = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')


def _fields(parser, section, expected):
    if not parser.has_section(section):
        raise ValueError(f'Missing [{section}] section')
    actual = set(parser[section])
    if actual != expected:
        raise ValueError(f'[{section}] invalid settings: unknown={sorted(actual - expected)}, '
                         f'missing={sorted(expected - actual)}')
    if any('\n' in value for value in parser[section].values()):
        raise ValueError(f'[{section}] settings must each be on one line')


def _decimal_value(text, label, *, integer=False):
    text = text.strip()
    if integer:
        _integer(text, label)
    elif not _DECIMAL.fullmatch(text):
        raise ValueError(f'{label} must contain finite decimal numbers')
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f'{label} must contain finite decimal numbers') from error
    number = float(value)
    if not value.is_finite() or not math.isfinite(number) or (value != 0 and number == 0):
        raise ValueError(f'{label} values must be finite and representable without underflow')
    return value


def _axis_values(text, parameter, label):
    """Expand decimal endpoints exactly, without accumulated binary float error."""
    if ':' in text:
        parts = tuple(part.strip() for part in text.split(':'))
        if len(parts) != 3 or ',' in text or any(not part for part in parts):
            raise ValueError(f'{label} must be a list or one inclusive start:stop:step range')
        start, stop, step = (_decimal_value(part, label, integer=parameter == 'iterations') for part in parts)
        # Fractions preserve the decimal literals exactly even when endpoint
        # precision exceeds the default Decimal arithmetic context.
        start, stop, step = map(Fraction, (start, stop, step))
        if not step:
            raise ValueError(f'{label} range step must be nonzero')
        count = (stop - start) / step
        if count < 0:
            raise ValueError(f'{label} range step must move from start toward stop')
        if count.denominator != 1:
            raise ValueError(f'{label} inclusive range must reach its stop exactly')
        if count >= MAX_VARIANTS:
            raise ValueError(f'{label} expands beyond {MAX_VARIANTS} values')
        values = tuple(start + index * step for index in range(int(count) + 1))
    else:
        parts = _parts(text.strip(), label)
        if not parts:
            raise ValueError(f'{label} must contain at least one value')
        if len(parts) > MAX_VARIANTS:
            raise ValueError(f'{label} exceeds {MAX_VARIANTS} values')
        values = tuple(_decimal_value(part, label, integer=parameter == 'iterations') for part in parts)
    values = tuple(int(value) if parameter == 'iterations' else float(value) for value in values)
    return _checked_axis(values, parameter, label)


def _checked_axis(values, parameter, label):
    if not isinstance(values, (tuple, list)) or not values or len(values) > MAX_VARIANTS:
        raise ValueError(f'{label} requires 1..{MAX_VARIANTS} numeric values')
    result = []
    for value in values:
        if parameter == 'iterations':
            if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 10:
                raise ValueError(f'{label} values must be integers between 1 and 10')
            result.append(int(value))
        else:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{label} values must be finite numbers')
            if parameter == 'distance_mm' and not 0 <= value <= 50:
                raise ValueError(f'{label} values must be between 0 and 50 mm')
            if parameter == 'shape_k' and value <= 0:
                raise ValueError(f'{label} values must be positive')
            result.append(float(value))
    if len(set(result)) != len(result):
        raise ValueError(f'{label} must not contain duplicate values')
    return tuple(result)


def parse_recipe_study_sections(parser, root=REPOSITORY_ROOT):
    """Normalize a recipe study without mutating the caller's parser."""
    if parser.defaults():
        raise ValueError('[DEFAULT] settings are unsupported')
    if not parser.has_section('study') or parser['study'].get('schema_version') != SCHEMA:
        raise ValueError(f'study.schema_version must be {SCHEMA}')
    root = Path(root).expanduser().resolve()
    _fields(parser, 'train', TRAIN_FIELDS)
    _fields(parser, 'model', MODEL_FIELDS)
    train = {key: value.strip() for key, value in parser['train'].items()}
    model = {key: value.strip() for key, value in parser['model'].items()}
    sweep_sections = [name for name in parser.sections() if name.startswith('sweep.')]
    if not sweep_sections:
        raise ValueError('A recipe study requires at least one [sweep.EDIT] section')
    recipe_parser = deepcopy(parser)
    for section in ('train', 'model', *sweep_sections):
        recipe_parser.remove_section(section)
    recipe_parser['study']['schema_version'] = 'fakect.recipe/1'
    config = parse_recipe_sections(recipe_parser, root)
    config['study']['schema_version'] = SCHEMA
    if train['stage'] not in STAGES:
        raise ValueError('train.stage must be preview, plan, preflight, prepare, or fit')
    if train['target_scope'] != 'selected_lineage' or config['recipe'].get('roi_role') != 'selection':
        raise ValueError('Recipe studies require train.target_scope=selected_lineage and recipe.roi_role=selection')
    if train['include_baseline'] not in ('true', 'false'):
        raise ValueError('train.include_baseline must be true or false')
    if train['image_method'] != 'attenuation_copy_proxy':
        raise ValueError('Only the explicitly experimental attenuation_copy_proxy image method is implemented')
    if train['split_mode'] != 'scenario_only':
        raise ValueError('Recipe studies currently support scenario_only splits')
    validation = _float(train['validation_fraction'], 'train.validation_fraction', opacity=True)
    test = _float(train['test_fraction'], 'train.test_fraction', opacity=True)
    if validation <= 0 or test <= 0 or validation + test >= 1:
        raise ValueError('Use positive validation/test fractions whose sum is < 1')
    config['train'] = {key: train[key] for key in ('stage', 'target_scope', 'image_method', 'split_mode')}
    config['train'].update(
        anatomy_family=_name(train['anatomy_family'], 'train.anatomy_family', _IDENTIFIER),
        include_baseline=train['include_baseline'] == 'true',
        max_variants=_integer(train['max_variants'], 'train.max_variants', 1, MAX_VARIANTS),
        validation_fraction=validation, test_fraction=test,
        split_seed=_integer(train['split_seed'], 'train.split_seed', 0, 2**31-1))
    for key in ('dataset_directory', 'model_directory', 'preflight_directory'):
        config['train'][key] = _path(train[key], 'train.' + key, root)
    config['model'] = parse_model_settings(model)
    config['sweeps'] = {}
    for section in sweep_sections:
        name = _name(section[len('sweep.'):], section, _IDENTIFIER)
        if name not in config['edits']:
            raise ValueError(f'[{section}] references undefined edit {name!r}')
        actual = set(parser[section])
        if not actual or actual - SWEEP_FIELDS:
            raise ValueError(f'[{section}] requires one or more of {sorted(SWEEP_FIELDS)}; '
                             f'unknown={sorted(actual - SWEEP_FIELDS)}')
        if any('\n' in value for value in parser[section].values()):
            raise ValueError(f'[{section}] settings must each be on one line')
        axes = {key: _axis_values(value.strip(), key, f'{section}.{key}')
                for key, value in parser[section].items()}
        if config['edits'][name]['profile'] == 'uniform' and len(axes.get('shape_k', ())) > 1:
            raise ValueError(f'[{section}]: a uniform profile ignores shape_k; use one value')
        config['sweeps'][name] = axes
    return config


def load_recipe_study_config(path, *, repo_root=None):
    from fakect_study_config import load_study_config
    config = load_study_config(path, repo_root=repo_root)
    if config['study']['schema_version'] != SCHEMA:
        raise ValueError(f'study.schema_version must be {SCHEMA}')
    return config


def _disabled(edit):
    return {**{key: deepcopy(value) for key, value in edit.items() if key not in _DISABLED_EDIT_FIELDS},
            'operation': 'none', 'distance_mm': 0.}


def _canonical(value):
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _json(value):
    return json.dumps(_canonical(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def plan_recipe_variants(config):
    """Return at most 1000 complete recipes, with provisional whole-variant splits."""
    train = config['train']
    if train.get('target_scope') != 'selected_lineage' or config['recipe'].get('roi_role') != 'selection':
        raise ValueError('Recipe studies require train.target_scope=selected_lineage and recipe.roi_role=selection')
    if train.get('image_method') != 'attenuation_copy_proxy' or train.get('split_mode') not in ('scenario_only', 'unassigned'):
        raise ValueError('Recipe preparation requires attenuation_copy_proxy images and a supported split policy')
    maximum = train['max_variants']
    if isinstance(maximum, bool) or not isinstance(maximum, Integral) or not 1 <= maximum <= MAX_VARIANTS:
        raise ValueError(f'train.max_variants must be an integer between 1 and {MAX_VARIANTS}')
    if not isinstance(train['include_baseline'], bool):
        raise ValueError('train.include_baseline must be Boolean')
    fractions = []
    for name in ('validation_fraction', 'test_fraction'):
        value = train[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f'train.{name} must be finite and between 0 and 1')
        fractions.append(float(value))
    if sum(fractions) >= 1:
        raise ValueError('Validation/test fractions must sum to less than 1')
    seed = train['split_seed']
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**31-1:
        raise ValueError('train.split_seed must be an integer between 0 and 2147483647')
    _name(train['anatomy_family'], 'train.anatomy_family', _IDENTIFIER)
    names = tuple(config['recipe']['steps'])
    if not names or len(names) != len(set(names)) or set(names) != set(config['edits']):
        raise ValueError('recipe.steps must list every defined edit exactly once')
    sweeps = config.get('sweeps')
    if not isinstance(sweeps, dict) or not sweeps:
        raise ValueError('A recipe study requires at least one sweep')
    axes, count = [], 1
    for name, parameters in sorted(sweeps.items()):
        if name not in config['edits']:
            raise ValueError(f'sweep.{name} references an undefined edit')
        if not isinstance(parameters, dict) or not parameters or set(parameters) - SWEEP_FIELDS:
            raise ValueError(f'sweep.{name} contains unknown or empty sweep parameters')
        for key, values in sorted(parameters.items()):
            values = _checked_axis(values, key, f'sweep.{name}.{key}')
            if key == 'shape_k' and len(values) > 1 and config['edits'][name]['profile'] == 'uniform':
                raise ValueError(f'sweep.{name}: a uniform profile ignores shape_k; use one value')
            axes.append((name, key, tuple(sorted(values))))
            count *= len(values)
            if count + int(train['include_baseline']) > maximum:
                raise ValueError(f'Planned {count + int(train["include_baseline"])} variants exceeds train.max_variants={maximum}')
    base = {'case_id': config['input']['case_id'], 'frame': config['input']['frame'],
            'anatomy_family': train['anatomy_family']}
    # Include geometry, tissue selection and recipient policy in scenario IDs,
    # but never output paths, stage, display settings, model settings or seed.
    identity = {key: config[key] for key in ('selection', 'roi', 'rois', 'recipe', 'reassignment')}
    if 'centerline' in config:
        identity['centerline'] = config['centerline']
    variants = []
    if train['include_baseline']:
        variants.append({**base, 'variant_id': 'baseline', 'operation': 'none', 'split': 'train',
                         'edits': {name: _disabled(config['edits'][name]) for name in names}, 'parameters': {}})
    edited = []
    for values in product(*(axis[2] for axis in axes)):
        edits = deepcopy(config['edits'])
        parameters = {}
        for (name, key, _), value in zip(axes, values):
            edits[name][key] = value
            parameters.setdefault(name, {})[key] = value
        for name, edit in edits.items():
            if edit['distance_mm'] == 0:
                edits[name] = _disabled(edit)
        record = {**base, 'operation': 'recipe', 'edits': edits, 'parameters': parameters}
        digest = hashlib.sha256(_json({'scenario': record, 'recipe_settings': identity}).encode()).hexdigest()[:16]
        edited.append({**record, 'variant_id': 'recipe-' + digest, 'split': 'train'})
    if train['split_mode'] == 'unassigned':
        if any(fractions):
            raise ValueError('Unassigned cohorts cannot specify holdout fractions')
        return [{**row, 'split': 'unassigned'} for row in variants + edited]
    ranked = sorted(edited, key=lambda item: hashlib.sha256(f'{seed}:{item["variant_id"]}'.encode()).digest())
    counts = [max(1, math.floor(len(edited) * fraction)) if fraction else 0 for fraction in fractions]
    if sum(counts) > len(edited) or (sum(counts) >= len(edited) and not variants):
        raise ValueError('Too few variants to populate the requested holdouts and training set')
    for record in ranked[:counts[0]]:
        record['split'] = 'validation'
    for record in ranked[counts[0]:sum(counts)]:
        record['split'] = 'test'
    return variants + edited


def variant_recipe_config(config, variant):
    """Create an independent complete recipe, retaining its selection lineage contract."""
    result = deepcopy(config)
    result['edits'] = deepcopy(variant['edits'])
    return result


def validate_recipe_study_plan(config, resolved):
    """Check every variant's recipe passes and halo without opening voxel sources."""
    from fakect_recipe import validate_recipe
    variants = plan_recipe_variants(config)
    for variant in variants:
        try:
            for name, edit in variant['edits'].items():
                if edit['operation'] == 'erosion' and edit.get('assign_surrounding_tissue', True) is False:
                    raise ValueError(f'[edit.{name}] active erosion requires assign_surrounding_tissue=true '
                                     'for training; diagnostic released voxels cannot form training pairs')
            validate_recipe(variant_recipe_config(config, variant), resolved)
        except ValueError as error:
            raise ValueError(f'Variant {variant["variant_id"]} parameters={_json(variant["parameters"])}: {error}') from error
    return variants
