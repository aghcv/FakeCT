"""Metadata-only, origin-aware focused CoA rotation plans.

Age pairs are conservative provisional groups, not verified template ancestry.
No image, learned checkpoint or reserved-test payload is opened by this module.
"""
from copy import deepcopy
import configparser
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.coa-rotation-design/1'
SCORE = 'coa_local_target_volume_loss_fraction'
PROTOCOLS = ('family_holdout', 'same_age_other_sex', 'nearest_age_same_sex')
FIELDS = {
    'experiment': {'schema_version', 'output_directory'},
    'data': {'project', 'manifest', 'annotations'},
    'reference': {'geometry_config', 'experiment_directory'},
    'evaluation': {'protocols', 'seeds', 'reserved_case_ids', 'scenarios_per_case',
                   'label_threshold', 'prediction_threshold', 'nearest_age_tie_break'},
    'training': {'locator_epochs', 'classifier_epochs', 'locator_learning_rate',
                 'classifier_learning_rate', 'regression_weight', 'cache_limit_mb',
                 'host_memory_limit_mb', 'gpu_memory_limit_mb', 'require_gpu'},
}


def _integer(value, name, minimum=0, maximum=2**31-1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(name+' must be finite and strictly between zero and one')
    return float(value)


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                       empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(FIELDS):
        raise ValueError('Expected exactly [experiment], [data], [reference], [evaluation], [training]')
    c = {section: dict(parser[section]) for section in FIELDS}
    if any(set(c[section]) != fields for section, fields in FIELDS.items()):
        raise ValueError('Unsupported CoA rotation input fields')
    if any('\n' in value or '\r' in value for values in c.values() for value in values.values()):
        raise ValueError('Rotation input values must occupy one line')
    if c['experiment']['schema_version'] != SCHEMA:
        raise ValueError('Unsupported focused CoA rotation schema')
    for section in ('experiment', 'data', 'reference'):
        for key in FIELDS[section] - {'schema_version'}:
            p = Path(c[section][key]).expanduser()
            c[section][key] = str((p if p.is_absolute() else ROOT/p).resolve())
    e = c['evaluation']
    e['protocols'] = [x.strip() for x in e['protocols'].split(',')]
    e['reserved_case_ids'] = [x.strip() for x in e['reserved_case_ids'].split(',')]
    if not e['protocols'] or len(set(e['protocols'])) != len(e['protocols']) or not set(e['protocols']) <= set(PROTOCOLS):
        raise ValueError('evaluation.protocols requires distinct supported protocol names')
    if not e['reserved_case_ids'] or len(set(e['reserved_case_ids'])) != len(e['reserved_case_ids']):
        raise ValueError('evaluation.reserved_case_ids requires distinct nonblank case IDs')
    if e['nearest_age_tie_break'] != 'younger_then_case_id':
        raise ValueError('The declared nearest-age tie rule must be younger_then_case_id')
    e['seeds'] = [int(x.strip()) for x in e['seeds'].split(',')]
    if not e['seeds'] or len(e['seeds']) > 100 or len(set(e['seeds'])) != len(e['seeds']):
        raise ValueError('evaluation.seeds requires between 1 and 100 distinct seeds')
    for seed in e['seeds']:
        _integer(seed, 'evaluation.seed')
    e['scenarios_per_case'] = _integer(int(e['scenarios_per_case']), 'evaluation.scenarios_per_case', 2, 6)
    if e['scenarios_per_case'] % 2:
        raise ValueError('scenarios_per_case must contain complete negative/positive pair blocks')
    for key in ('label_threshold', 'prediction_threshold'):
        e[key] = _fraction(float(e[key]), 'evaluation.'+key)
    if e['label_threshold'] != .2:
        raise ValueError('This annotation revision requires label_threshold = 0.20')
    t = c['training']
    for key, low, high in (('locator_epochs', 1, 500), ('classifier_epochs', 1, 1000),
                           ('cache_limit_mb', 256, 12288), ('host_memory_limit_mb', 512, 32768),
                           ('gpu_memory_limit_mb', 512, 12288)):
        t[key] = _integer(int(t[key]), 'training.'+key, low, high)
    for key in ('locator_learning_rate', 'classifier_learning_rate', 'regression_weight'):
        t[key] = float(t[key])
        if not math.isfinite(t[key]) or t[key] < 0 or (key != 'regression_weight' and t[key] == 0):
            raise ValueError('training.'+key+' must be finite and positive (regression_weight may be zero)')
    t['require_gpu'] = parser.getboolean('training', 'require_gpu')
    if t['cache_limit_mb'] > t['host_memory_limit_mb']//2:
        raise ValueError('Image cache must leave at least half the host budget for the runtime')
    out = Path(c['experiment']['output_directory'])
    reference = Path(c['reference']['experiment_directory'])
    if out == reference or reference in out.parents:
        raise ValueError('Rotation output must be separate from its frozen reference')
    if any(out == Path(c['data'][key]).parent for key in ('manifest', 'annotations')):
        raise ValueError('Rotation output must differ from source artifact directories')
    return c


def build_inventory(project):
    """Parse explicit template names; never invent a numerical infant age."""
    if not isinstance(project, dict) or project.get('schema_version') != 'fakect.cohort-project/1':
        raise ValueError('Expected the CoA project metadata schema')
    cases = project.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Project needs a nonempty case inventory')
    result, seen = [], set()
    for row in cases:
        case_id, template, family = (row.get(k) for k in ('case_id', 'template', 'provisional_family'))
        if not all(isinstance(x, str) and x and x == x.strip() for x in (case_id, template, family)) or case_id in seen:
            raise ValueError('Every project case needs unique ID, template and provisional family')
        seen.add(case_id)
        adult = re.fullmatch(r'v(male|female)50\.nrb', template)
        child = re.fullmatch(r'(male|female)_(infant|[1-9][0-9]*yr)_ref\.nrb', template)
        if adult:
            sex, age_bin, age_years = adult[1], 'adult', 50
        elif child:
            sex, age_bin = child[1], child[2]
            age_years = None if age_bin == 'infant' else int(age_bin[:-2])
        else:
            raise ValueError('Template lacks an explicit supported age/sex convention: '+template)
        result.append({'case_id': case_id, 'template': template, 'sex': sex,
                       'age_bin': age_bin, 'age_years': age_years, 'anatomy_family': family,
                       'family_status': 'provisional', 'family_verified': False})
    by_age = {}
    for row in result:
        by_age.setdefault(row['age_bin'], []).append(row)
    for group in by_age.values():
        if len(group) != 2 or {r['sex'] for r in group} != {'male', 'female'} or len({r['anatomy_family'] for r in group}) != 1:
            raise ValueError('Each declared age group requires one male/female pair with the same provisional family')
    if len({r['anatomy_family'] for r in result}) != len(by_age):
        raise ValueError('Different age groups must not silently alias the same provisional family')
    return sorted(result, key=lambda r: r['case_id'])


def _reserved(inventory, reserved_case_ids):
    reserved = set(reserved_case_ids)
    infants = {r['case_id'] for r in inventory if r['age_bin'] == 'infant'}
    if not infants or reserved != infants:
        raise ValueError('The entire infant pair must remain reserved in this development protocol')
    return reserved


def select_matched_panel(rows, annotations, inventory, *, scenarios_per_case=6,
                         threshold=.2, reserved_case_ids=None):
    """Choose disjoint paired edit blocks using metadata, never model outcomes.

    Three sorted companion configurations cover low, middle and high achieved
    positive loss. Every negative is the closest below-threshold edit within
    that same companion configuration. Baselines remain locator references.
    """
    _integer(scenarios_per_case, 'scenarios_per_case', 2, 6)
    if scenarios_per_case % 2 or threshold != .2:
        raise ValueError('This panel requires complete pairs and the frozen 20% annotation threshold')
    reserved = _reserved(inventory, reserved_case_ids or [r['case_id'] for r in inventory if r['age_bin'] == 'infant'])
    if not isinstance(rows, list) or not isinstance(annotations, list):
        raise ValueError('Dataset and annotation samples must be lists')
    by_case = {r['case_id']: [] for r in inventory}
    known = {r['case_id']: r for r in inventory}
    annotation = {r.get('sample_id'): r for r in annotations}
    if len(annotation) != len(annotations) or None in annotation:
        raise ValueError('Annotation sample identities must be unique')
    seen = set()
    for row in rows:
        sample, case = row.get('sample_id'), row.get('case_id')
        if not isinstance(sample, str) or not sample or sample in seen or case not in known:
            raise ValueError('Dataset samples require unique IDs and inventoried cases')
        seen.add(sample)
        if row.get('anatomy_family') != known[case]['anatomy_family']:
            raise ValueError('Dataset/project anatomy-family mismatch: '+sample)
        by_case[case].append(row)
    if seen != set(annotation):
        raise ValueError('Dataset and annotation sample identities must match exactly')
    selected, pairs, all_ids = [], [], {}
    for case in sorted(set(by_case)-reserved):
        family = known[case]['anatomy_family']
        group = by_case[case]
        baseline = [r for r in group if r.get('baseline') is True]
        if len(baseline) != 1:
            raise ValueError('Each development case requires exactly one original locator reference')
        all_ids[case] = sorted(r['sample_id'] for r in group)
        blocks = {}
        for row in group:
            a = annotation[row['sample_id']]
            value = a.get(SCORE)
            if (a.get('case_id') != case or a.get('anatomy_family') != family or a.get('errors') or
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
                    not 0 <= value <= 1 or type(a.get('coa_benchmark_positive')) is not bool or
                    a['coa_benchmark_positive'] != (value >= threshold)):
                raise ValueError('Invalid achieved-loss annotation: '+row['sample_id'])
            if row.get('baseline') is True:
                if value != 0 or a.get('original_baseline') is not True:
                    raise ValueError('Original locator reference must have zero induced change')
                continue
            if row.get('baseline') is not False or a.get('original_baseline') is not False or a.get('coa_edit_applied') is not True:
                raise ValueError('Matched pairs must be explicitly edited siblings')
            parameters = row.get('parameters')
            if not isinstance(parameters, dict) or 'coa_narrow' not in parameters:
                raise ValueError('Edited sibling needs CoA and companion parameter metadata')
            companion = {k:v for k,v in parameters.items() if k != 'coa_narrow'}
            if not companion:
                raise ValueError('Matched companion parameter metadata is missing')
            key = json.dumps(companion, sort_keys=True, separators=(',', ':'), allow_nan=False)
            blocks.setdefault(key, {'companion': companion, 'negative': [], 'positive': []})[
                'positive' if a['coa_benchmark_positive'] else 'negative'].append((row, a))
        valid = [(key, block) for key, block in sorted(blocks.items()) if block['negative'] and block['positive']]
        pair_count = scenarios_per_case//2
        if len(valid) < pair_count:
            raise ValueError(f'Case {case} lacks {pair_count} distinct companion-matched class-complete blocks')
        for index, (key, block) in enumerate(valid[:pair_count]):
            negatives = sorted(block['negative'], key=lambda x: (-x[1][SCORE], x[0]['sample_id']))
            positives = sorted(block['positive'], key=lambda x: (x[1][SCORE], x[0]['sample_id']))
            quantile = index/(pair_count-1) if pair_count > 1 else 1
            positive = positives[int(math.floor(quantile*(len(positives)-1)+.5))]
            pair_id = case+':pair-'+hashlib.sha256(key.encode()).hexdigest()[:12]
            ids = []
            for role, (row, a) in (('edited_negative', negatives[0]), ('positive', positive)):
                selected.append({**deepcopy(row), 'annotation': deepcopy(a), 'pair_id': pair_id,
                                 'rotation_pair_id': pair_id,
                                 'pair_role': role, 'companion_parameters': deepcopy(block['companion'])})
                ids.append(row['sample_id'])
            pairs.append({'id': pair_id, 'case_id': case, 'sample_ids': ids,
                          'companion_parameters': deepcopy(block['companion'])})
    selected.sort(key=lambda r: (r['case_id'], r['pair_id'], r['pair_role']))
    return {'schema_version': SCHEMA, 'scenarios_per_case': scenarios_per_case,
            'label_threshold': threshold, 'samples': selected, 'pairs': pairs,
            'sample_ids_by_case': {case: sorted(r['sample_id'] for r in selected if r['case_id'] == case) for case in sorted(all_ids)},
            'all_sample_ids_by_case': all_ids,
            'baseline_sample_ids_by_case': {case: next(r['sample_id'] for r in by_case[case] if r.get('baseline') is True) for case in sorted(all_ids)},
            'baseline_samples': [{**deepcopy(next(r for r in by_case[case] if r.get('baseline') is True)),
                                  'annotation': deepcopy(annotation[next(r['sample_id'] for r in by_case[case] if r.get('baseline') is True)])}
                                 for case in sorted(all_ids)],
            'reserved_case_ids': sorted(reserved),
            'reserved_sample_ids': sorted(r['sample_id'] for case in reserved for r in by_case[case]),
            'selection_policy': 'Distinct companion-matched edited pairs; nearest-below-threshold negatives and low/middle/high positives across sorted companion blocks; metadata only.'}


def assert_isolation(stage, inventory, panel):
    """Validate every originating case and every sibling's single stage role."""
    known = {r['case_id'] for r in inventory}
    case_sets = {role: set(stage[role+'_case_ids']) for role in ('train', 'validation', 'final_test', 'reserved')}
    if any(len(case_sets[role]) != len(stage[role+'_case_ids']) for role in case_sets):
        raise ValueError('Case partitions may not contain duplicate originating IDs')
    if set.union(*case_sets.values()) != known or sum(map(len, case_sets.values())) != len(known):
        raise ValueError('Stage partitions must cover each originating case exactly once')
    expected_assignment = {case: ('excluded' if role == 'reserved' else role)
                           for role, cases in case_sets.items() for case in cases}
    if stage.get('case_assignment') != expected_assignment:
        raise ValueError('Case role lookup differs from its case partitions')
    if case_sets['reserved'] != set(panel['reserved_case_ids']) or not case_sets['train'] or not case_sets['final_test']:
        raise ValueError('Stage requires training/final-test cases and the unchanged reserved pair')
    if stage['stage'] == 'outer_refit' and case_sets['validation']:
        raise ValueError('Fresh outer refit must have no model-selection validation cases')
    if stage['stage'] != 'outer_refit' and not case_sets['validation']:
        raise ValueError('Selection stage requires model-selection validation')
    selected = panel['sample_ids_by_case']
    all_ids = panel['all_sample_ids_by_case']
    used = set()
    for role in ('train', 'validation', 'final_test'):
        for prefix, mapping in (('', selected), ('all_', all_ids)):
            expected = sorted(sample for case in case_sets[role] for sample in mapping[case])
            if stage[prefix+role+'_sample_ids'] != expected:
                raise ValueError('Stage sibling partition differs from its complete case assignment')
        current = set(stage[role+'_sample_ids'])
        if used & current or current & set(panel['reserved_sample_ids']):
            raise ValueError('Sibling leakage or reserved-test exposure')
        used |= current
    if stage['protocol'] == 'family_holdout':
        by_family = {}
        for row in inventory:
            role = next(role for role, cases in case_sets.items() if row['case_id'] in cases)
            previous = by_family.setdefault(row['anatomy_family'], role)
            if previous != role:
                raise ValueError('Primary protocol cannot split a provisional anatomy family')
    return True


def build_rotation_design(inventory, panel, *, protocols=PROTOCOLS, seeds=(42,),
                          reserved_case_ids=None, prediction_threshold=.5):
    """Plan five primary family folds and ten paired secondary case rotations."""
    reserved = _reserved(inventory, reserved_case_ids or panel['reserved_case_ids'])
    if not protocols or len(set(protocols)) != len(protocols) or not set(protocols) <= set(PROTOCOLS):
        raise ValueError('Rotation protocols must be distinct supported names')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Rotation seeds must be distinct')
    for seed in seeds:
        _integer(seed, 'seed')
    prediction_threshold = _fraction(prediction_threshold, 'prediction_threshold')
    development = [r for r in inventory if r['case_id'] not in reserved]
    if any(r['age_years'] is None for r in development):
        raise ValueError('Nearest-age evaluation requires known numerical development ages')
    if set(panel['sample_ids_by_case']) != {r['case_id'] for r in development}:
        raise ValueError('Panel must cover every development case and no reserved case')
    families = {}
    for row in development:
        families.setdefault(row['anatomy_family'], []).append(row['case_id'])
    if len(families) < 4:
        raise ValueError('Nested rotation requires at least four development families')
    stages = []

    def stage(protocol, seed, outer, kind, training, validation, test, **extra):
        suffix = extra.pop('suffix', kind)
        value = {'id': f'{protocol}/{outer}/seed-{seed}/{suffix}', 'protocol': protocol,
                 'seed': seed, 'stage': kind, 'outer_fold_id': outer,
                 'train_case_ids': sorted(training), 'validation_case_ids': sorted(validation),
                 'final_test_case_ids': sorted(test), 'reserved_case_ids': sorted(reserved),
                 'roles': {'train': 'weight_fitting', 'validation': 'model_selection_validation',
                           'final_test': 'final_prediction_only', 'reserved': 'unopened_final_family'},
                 'locator_policy': 'fresh_training_on_train_case_ids_only',
                 'classifier_policy': 'fresh_initialization', **extra}
        value['case_assignment'] = {case: ('excluded' if role == 'reserved' else role)
                                   for role in ('train', 'validation', 'final_test', 'reserved')
                                   for case in value[role+'_case_ids']}
        for role in ('train', 'validation', 'final_test'):
            value[role+'_sample_ids'] = sorted(sample for case in value[role+'_case_ids'] for sample in panel['sample_ids_by_case'][case])
            value['all_'+role+'_sample_ids'] = sorted(sample for case in value[role+'_case_ids'] for sample in panel['all_sample_ids_by_case'][case])
        value['counts'] = {role: {'cases': len(value[role+'_case_ids']),
                                'selected_samples': len(value.get(role+'_sample_ids', []))}
                           for role in ('train', 'validation', 'final_test', 'reserved')}
        assert_isolation(value, inventory, panel)
        stages.append(value)
        return value['id']

    for protocol in protocols:
        for seed in sorted(seeds):
            if protocol == 'family_holdout':
                for heldout in sorted(families):
                    other = sorted(set(families)-{heldout})
                    inner = []
                    for validation in other:
                        train = [case for family in other if family != validation for case in families[family]]
                        inner.append(stage(protocol, seed, heldout, 'inner_selection', train,
                                           families[validation], families[heldout],
                                           suffix='inner-'+validation, validation_family=validation,
                                           checkpoint_policy='minimum_model_selection_validation_bce',
                                           final_test_evaluation_permitted=False))
                    stage(protocol, seed, heldout, 'outer_refit',
                          [case for family in other for case in families[family]], [], families[heldout],
                          parent_inner_stage_ids=inner,
                          checkpoint_policy='inner_selected_fixed_budget_then_fresh_refit',
                          final_test_evaluation_permitted=True)
            else:
                for target in sorted(development, key=lambda r:r['case_id']):
                    if protocol == 'same_age_other_sex':
                        candidates = [r for r in development if r['age_bin'] == target['age_bin'] and r['sex'] != target['sex']]
                        if len(candidates) != 1:
                            raise ValueError('Same-age opposite-sex validation case is not unique')
                        validation = candidates[0]
                        distance = 0
                    else:
                        candidates = [r for r in development if r['sex'] == target['sex'] and r['case_id'] != target['case_id']]
                        if not candidates:
                            raise ValueError('Nearest-age same-sex validation case unavailable')
                        validation = min(candidates, key=lambda r:(abs(r['age_years']-target['age_years']), r['age_years'], r['case_id']))
                        distance = abs(validation['age_years']-target['age_years'])
                    train = [r['case_id'] for r in development if r['case_id'] not in (target['case_id'], validation['case_id'])]
                    stage(protocol, seed, 'case-'+target['case_id'], 'secondary_selection', train,
                          [validation['case_id']], [target['case_id']],
                          target_case_id=target['case_id'], validation_age_difference_years=distance,
                          endpoint='unseen_baseline_phantom_conditional_on_unverified_ancestry',
                          checkpoint_policy='minimum_model_selection_validation_bce',
                          final_test_evaluation_permitted=True)
    return {'schema_version': SCHEMA, 'task': 'focused_coa_classification',
            'inventory': deepcopy(inventory), 'family_status': 'provisional_not_ancestry_verified',
            'protocols': list(protocols), 'seeds': sorted(seeds),
            'reserved_case_ids': sorted(reserved), 'reserved_test_evaluated': False,
            'label_threshold': panel['label_threshold'], 'prediction_threshold': prediction_threshold,
            'nearest_age_rule': 'known_development_ages_only; absolute_year_difference_then_younger_then_case_id',
            'annotation_definition': 'Achieved local CoA target-volume loss >=20%; engineering proxy, not clinical diagnosis.',
            **{key: deepcopy(panel[key]) for key in ('samples', 'pairs', 'sample_ids_by_case',
               'all_sample_ids_by_case', 'baseline_sample_ids_by_case', 'baseline_samples',
               'reserved_sample_ids', 'selection_policy', 'scenarios_per_case')},
            'stages': stages,
            'scope': 'Infant pair reserved; family holdout primary; age/sex-based phantom rotations secondary. No existing trained locator is reused.'}


def _read_json(path, *, checksum=True):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if checksum:
        sidecar = Path(str(path)+'.sha256')
        if not sidecar.exists() or sidecar.read_text().split()[0] != digest:
            raise ValueError('Frozen metadata checksum missing or changed: '+str(path))
    return json.loads(raw), digest


def plan(c):
    """Join frozen JSON contracts and return a design without opening images."""
    project, project_hash = _read_json(c['data']['project'], checksum=False)
    manifest, manifest_hash = _read_json(c['data']['manifest'])
    annotations, annotations_hash = _read_json(c['data']['annotations'])
    if manifest.get('schema_version') != 'fakect.context-dataset/1' or manifest.get('complete') is not True:
        raise ValueError('Expected a complete context dataset manifest')
    if annotations.get('schema_version') != 'fakect.coa-label-review/1' or annotations.get('score_name') != SCORE:
        raise ValueError('Unsupported frozen CoA annotations')
    source_hashes = {r['path']: r['sha256'] for r in annotations.get('source_provenance', [])}
    if source_hashes.get(str(Path(c['data']['manifest']).resolve())) != manifest_hash:
        raise ValueError('Annotations are not pinned to this dataset manifest')
    inventory = build_inventory(project)
    e = c['evaluation']
    panel = select_matched_panel(manifest['samples'], annotations['samples'], inventory,
                                scenarios_per_case=e['scenarios_per_case'], threshold=e['label_threshold'],
                                reserved_case_ids=e['reserved_case_ids'])
    value = build_rotation_design(inventory, panel, protocols=e['protocols'], seeds=e['seeds'],
                                  reserved_case_ids=e['reserved_case_ids'], prediction_threshold=e['prediction_threshold'])
    value['source_metadata'] = {c['data']['project']: project_hash, c['data']['manifest']: manifest_hash,
                                c['data']['annotations']: annotations_hash}
    return value
