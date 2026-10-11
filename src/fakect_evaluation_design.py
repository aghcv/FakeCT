"""Deterministic, family-preserving evaluation plans over prepared datasets.

This module plans splits only. It does not open image data, select models, or
consume held-out outcomes. Model repeat seeds are separate from ``split_seed``
so that comparisons use the same independent test anatomies.
"""
import hashlib
import math
import re


SCHEMA = 'fakect.evaluation-design/1'
STRATEGIES = {'holdout', 'leave_one_out', 'nested_leave_one_out'}
FIELDS = {'strategy', 'case_counts', 'validation_families', 'seeds', 'threshold',
          'planning_project'}
SPLITS = ('train', 'validation', 'test')
MAX_FAMILIES = 128
MAX_DP_STATES = 100_000
MAX_SEEDS = 100
MAX_INTEGER = 2**31 - 1


def _integer(value, name, minimum=0):
    """Parse a bounded unsigned evaluation setting.

    Parameters
    ----------
    value : str
        Unsigned decimal field text.
    name : str
        Evaluation field name used in errors.
    minimum : int
        Inclusive lower bound.

    Returns
    -------
    int
        Integer between minimum and MAX_INTEGER.

    Raises
    ------
    ValueError
        The text or allowed numeric interval is invalid.
    """
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]+', value):
        raise ValueError(f'evaluation.{name} must be an integer >= {minimum}')
    result = int(value)
    if not minimum <= result <= MAX_INTEGER:
        raise ValueError(f'evaluation.{name} must be between {minimum} and {MAX_INTEGER}')
    return result


def parse_evaluation(values):
    """Parse explicit family-based holdout and repeat settings from INI text.

    Parameters
    ----------
    values : dict of str to str
        Single-line optional [evaluation] section fields.

    Returns
    -------
    dict
        Typed strategy, model seeds, threshold and applicable case-
        count/validation/planning settings.

    Raises
    ------
    ValueError
        Fields, strategy-specific options, repeat seeds or threshold are invalid.

    Notes
    -----
    Parse the optional model INI's explicit ``[evaluation]`` section.

    ``case_counts`` counts distinct base cases, never scenarios or dataset
    receipts. ``seeds`` are training repetitions; they do not redraw a split.
    Holdout requires explicit train, validation, test case counts. Both leave-
    one-out strategies hold out complete anatomy families, not individual IDs.
    """
    if not isinstance(values, dict):
        raise ValueError('[evaluation] must be a mapping of single-line strings')
    unknown = set(values) - FIELDS
    if unknown:
        raise ValueError(f'[evaluation] unknown fields: {sorted(unknown)}')
    if any(not isinstance(value, str) or '\n' in value or '\r' in value
           for value in values.values()):
        raise ValueError('[evaluation] values must each occupy one line')
    values = {key: value.strip() for key, value in values.items()}
    strategy = values.get('strategy')
    if strategy not in STRATEGIES:
        raise ValueError('evaluation.strategy must be holdout, leave_one_out, or nested_leave_one_out')
    if strategy != 'holdout' and 'case_counts' in values:
        raise ValueError('evaluation.case_counts is only used with strategy = holdout; remove it')
    if strategy != 'leave_one_out' and 'validation_families' in values:
        raise ValueError('evaluation.validation_families is only used with strategy = leave_one_out; remove it')
    if strategy == 'holdout' and 'case_counts' not in values:
        raise ValueError('evaluation.case_counts is required for holdout: train, validation, test')
    seeds = [_integer(value.strip(), 'seeds')
             for value in values.get('seeds', '42').split(',')]
    if len(seeds) > MAX_SEEDS or len(set(seeds)) != len(seeds):
        raise ValueError(f'evaluation.seeds must contain between 1 and {MAX_SEEDS} distinct integers')
    try:
        threshold = float(values.get('threshold', '0.5'))
    except (ValueError, OverflowError) as exc:
        raise ValueError('evaluation.threshold must be finite and strictly between 0 and 1') from exc
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('evaluation.threshold must be finite and strictly between 0 and 1')
    result = {'strategy': strategy, 'seeds': seeds, 'threshold': threshold}
    if 'planning_project' in values:
        if not values['planning_project'] or '\0' in values['planning_project']:
            raise ValueError('evaluation.planning_project must be a nonblank, single-line path')
        result['planning_project'] = values['planning_project']
    if strategy == 'holdout':
        counts = [_integer(value.strip(), 'case_counts', minimum=1)
                  for value in values['case_counts'].split(',')]
        if len(counts) != 3:
            raise ValueError('evaluation.case_counts requires three positive counts: train, validation, test')
        result['case_counts'] = counts
    elif strategy == 'leave_one_out':
        result['validation_families'] = _integer(values.get('validation_families', '1'),
                                                'validation_families', minimum=1)
    return result


def _typed_evaluation(evaluation):
    """Validate programmatic evaluation settings through the INI parser contract.

    Parameters
    ----------
    evaluation : dict
        Strategy settings with typed integer lists and numeric threshold.

    Returns
    -------
    dict
        Validated typed settings identical in semantics to parse_evaluation output.

    Raises
    ------
    ValueError
        Typed field values or the reconstructed configuration are invalid.

    Notes
    -----
    Check programmatic inputs with the same rules as the INI parser.
    """
    if not isinstance(evaluation, dict):
        raise ValueError('Evaluation settings must be the parsed [evaluation] mapping')
    raw = {}
    for key, value in evaluation.items():
        if key in ('seeds', 'case_counts'):
            if (not isinstance(value, (list, tuple)) or
                    any(isinstance(item, bool) or not isinstance(item, int) for item in value)):
                raise ValueError(f'evaluation.{key} must be a list of integers')
            raw[key] = ','.join(str(item) for item in value)
        elif key == 'validation_families':
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError('evaluation.validation_families must be an integer')
            raw[key] = str(value)
        elif key == 'threshold':
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError('evaluation.threshold must be a finite number')
            raw[key] = str(value)
        else:
            raw[key] = value
    return parse_evaluation(raw)


def _families(datasets):
    """Group distinct source cases into indivisible anatomy families.

    Parameters
    ----------
    datasets : list or tuple of dict
        Nonempty records identifying case_id and anatomy_family.

    Returns
    -------
    dict of str to list of str
        Sorted family-to-case mapping; repeated receipts for the same case do not add
        weight.

    Raises
    ------
    ValueError
        Case/family identity is invalid, a case has multiple families, or family
        capacity is exceeded.
    """
    if not isinstance(datasets, (list, tuple)) or not datasets:
        raise ValueError('Evaluation requires prepared datasets with case_id and anatomy_family')
    by_case = {}
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError('Each dataset must identify case_id and anatomy_family')
        case, family = dataset.get('case_id'), dataset.get('anatomy_family')
        if isinstance(case, bool) or not isinstance(case, (str, int)):
            raise ValueError('Each dataset needs a nonblank case_id')
        case = str(case)
        if not case or case != case.strip() or '\n' in case or '\r' in case:
            raise ValueError('Each dataset needs a nonblank, single-line case_id')
        if (not isinstance(family, str) or not family or family != family.strip()
                or '\n' in family or '\r' in family):
            raise ValueError('Each dataset needs a nonblank, single-line anatomy_family')
        previous = by_case.setdefault(case, family)
        if previous != family:
            raise ValueError(f'Case {case!r} belongs to multiple anatomy families: {previous!r}, {family!r}')
    result = {}
    for case, family in sorted(by_case.items()):
        result.setdefault(family, []).append(case)
    if len(result) > MAX_FAMILIES:
        raise ValueError(f'Evaluation design supports at most {MAX_FAMILIES} anatomy families')
    return {family: result[family] for family in sorted(result)}


def _ordered(families, seed, context):
    """Rank family names reproducibly within a split context.

    Parameters
    ----------
    families : iterable of str
        Candidate anatomy-family names.
    seed : int
        Split seed independent of model training repeat seeds.
    context : str
        Domain separator distinguishing holdout and validation selection.

    Returns
    -------
    list of str
        Names ordered by SHA256 rank, with name as a deterministic tie break.
    """
    def rank(family):
        value = f'{seed}\0{context}\0{family}'.encode('utf-8')
        return hashlib.sha256(value).hexdigest(), family
    return sorted(families, key=rank)


def validate_assignment(families, assignment, *, require_validation=True,
                        min_training_families=1):
    """Validate a complete indivisible family split and report its counts.

    Parameters
    ----------
    families : dict of str to list of str
        Each family mapped to distinct cases appearing nowhere else.
    assignment : dict of str to str
        One train/validation/test value for every family.
    require_validation : bool
        Require validation; false requires no validation for a final refit.
    min_training_families : int
        Minimum allowed number of training families.

    Returns
    -------
    dict
        Distinct case and family counts for train, validation and test.

    Raises
    ------
    ValueError
        Coverage, case uniqueness, split names or minimum nonempty partitions fail.

    Notes
    -----
    Require one complete split assignment per family; return split counts.

    A family is indivisible. ``families`` must map each family to its distinct
    case IDs; a case appearing in more than one family is invalid. Nested final
    refits may explicitly omit validation after inner model selection.
    """
    if not isinstance(families, dict) or not families:
        raise ValueError('An assignment requires a nonempty family-to-cases mapping')
    if not isinstance(assignment, dict) or set(assignment) != set(families):
        raise ValueError('An assignment must cover every anatomy family exactly once')
    cases_seen = set()
    for family, cases in families.items():
        if (not isinstance(cases, (list, tuple)) or not cases or
                any(not isinstance(case, str) or not case for case in cases) or
                len(set(cases)) != len(cases) or cases_seen.intersection(cases)):
            raise ValueError('Every case must appear exactly once in the family-to-cases mapping')
        cases_seen.update(cases)
        if assignment[family] not in SPLITS:
            raise ValueError('Family assignments must use train, validation, or test')
    family_counts = {split: sum(assignment[family] == split for family in families)
                     for split in SPLITS}
    case_counts = {split: sum(len(cases) for family, cases in families.items()
                              if assignment[family] == split)
                   for split in SPLITS}
    if family_counts['train'] < min_training_families or not family_counts['test']:
        raise ValueError(f'Assignment requires at least {min_training_families} training families and one test family')
    if require_validation and not family_counts['validation']:
        raise ValueError('Assignment requires at least one validation family')
    if not require_validation and family_counts['validation']:
        raise ValueError('Final refit must train on all development families without a validation split')
    return {'case_counts': case_counts, 'family_counts': family_counts}


def _holdout_assignment(families, counts, seed):
    """Solve an exact case-count partition without dividing anatomy families.

    Parameters
    ----------
    families : dict of str to list of str
        Distinct cases grouped by family.
    counts : sequence of int
        Required train, validation and test case counts, in that order.
    seed : int
        Split-ranking seed.

    Returns
    -------
    dict of str to str
        Deterministic family split assignment satisfying requested case counts.

    Raises
    ------
    ValueError
        Totals differ, exact partition is impossible or bounded dynamic-programming
        capacity is exceeded.

    Notes
    -----
    Exact group partition using a bounded two-dimensional dynamic program.
    """
    total = sum(map(len, families.values()))
    if sum(counts) != total:
        raise ValueError(f'evaluation.case_counts sums to {sum(counts)}, but prepared datasets contain '
                         f'{total} distinct cases; count each base case once, not each variant or receipt')
    targets = dict(zip(SPLITS, counts))
    # Track the two smallest bins. The third follows from the processed total.
    tracked = sorted(SPLITS, key=lambda split: (targets[split], SPLITS.index(split)))[:2]
    remaining_split = next(split for split in SPLITS if split not in tracked)
    order = _ordered(families, seed, 'holdout')
    layers = [{(0, 0): None}]
    processed, state_count = 0, 1
    for family in order:
        weight = len(families[family])
        processed += weight
        states = {}
        for state in layers[-1]:
            for split in SPLITS:
                next_state = tuple(value + (weight if split == tracked[index] else 0)
                                   for index, value in enumerate(state))
                derived = processed - sum(next_state)
                if (any(value > targets[tracked[index]] for index, value in enumerate(next_state))
                        or not 0 <= derived <= targets[remaining_split]):
                    continue
                if next_state not in states:
                    states[next_state] = state, split
                    state_count += 1
                    if state_count > MAX_DP_STATES:
                        raise ValueError(f'Exact holdout partition exceeds the {MAX_DP_STATES:,}-state planning budget; '
                                         'use fewer families or leave_one_out')
        layers.append(states)
        if not states:
            break
    state = tuple(targets[split] for split in tracked)
    if len(layers) != len(order) + 1 or state not in layers[-1]:
        sizes = ', '.join(f'{family}={len(cases)}' for family, cases in families.items())
        raise ValueError(f'Cannot achieve train/validation/test case_counts={counts} without splitting anatomy families. '
                         f'Family sizes: {sizes}. Choose compatible counts or leave_one_out.')
    assignment = {}
    for index in range(len(order), 0, -1):
        previous, split = layers[index][state]
        assignment[order[index-1]] = split
        state = previous
    return {family: assignment[family] for family in families}


def _fold(fold_id, families, assignment, *, inner_folds=None, final_refit=False,
          min_training_families=1):
    """Validate and describe one outer or inner family fold.

    Parameters
    ----------
    fold_id : str
        Stable outer/inner fold identifier.
    families : dict of str to list of str
        Distinct source cases grouped by family.
    assignment : dict of str to str
        Complete family split assignment.
    inner_folds : list of dict or None
        Nested development-fold records, when applicable.
    final_refit : bool
        Permit no validation for a final nested-CV refit.
    min_training_families : int
        Minimum training family count.

    Returns
    -------
    dict
        Fold ID, family membership, complete assignment, counts and optional nested
        folds.
    """
    counts = validate_assignment(families, assignment, require_validation=not final_refit,
                                 min_training_families=min_training_families)
    return {'id': fold_id, 'test_families': [family for family in families if assignment[family] == 'test'],
            'development_families': [family for family in families if assignment[family] != 'test'],
            'validation_families': [family for family in families if assignment[family] == 'validation'],
            'family_assignment': assignment, **counts, 'inner_folds': inner_folds or []}


def build_evaluation_design(datasets, evaluation, split_seed=42):
    """Plan deterministic outer and optional nested family partitions without image I/O.

    Parameters
    ----------
    datasets : list or tuple of dict
        Prepared or provisional case_id/anatomy_family records.
    evaluation : dict
        Typed holdout, leave_one_out or nested_leave_one_out settings.
    split_seed : int
        Nonnegative bounded split seed; training-repeat seeds do not redraw test
        partitions.

    Returns
    -------
    dict
        Versioned settings, family mapping, outer/inner folds and distinct-
        case/family/fit counts.

    Raises
    ------
    ValueError
        Settings, family independence/counts, requested partition or planning budget
        fail.

    Notes
    -----
    Build explicit outer and optional inner partitions without loading data.

    Repeated receipts for the same case do not change its weight. A holdout uses
    exactly the requested case counts, retaining complete anatomy families.
    Leave-one-out holds each family out once, selecting a deterministic subset
    of remaining families for validation. Nested leave-one-out holds each
    development family out in turn for inner validation, then refits on every
    development family; outer test data are never a validation partition.
    """
    evaluation = _typed_evaluation(evaluation)
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or not 0 <= split_seed <= MAX_INTEGER:
        raise ValueError(f'split_seed must be an integer between 0 and {MAX_INTEGER}')
    families = _families(datasets)
    strategy = evaluation['strategy']
    minimum = 4 if strategy == 'nested_leave_one_out' else 3
    if len(families) < minimum:
        raise ValueError(f'{strategy} requires at least {minimum} independent anatomy families')
    folds = []
    if strategy == 'holdout':
        assignment = _holdout_assignment(families, evaluation['case_counts'], split_seed)
        folds.append(_fold('outer-001', families, assignment))
    else:
        n_validation = evaluation.get('validation_families', 1)
        if n_validation >= len(families)-1:
            raise ValueError('evaluation.validation_families leaves no training family after holding one family out for testing')
        for index, test_family in enumerate(families, start=1):
            development = [family for family in families if family != test_family]
            inner_folds = []
            if strategy == 'nested_leave_one_out':
                for inner_index, validation_family in enumerate(development, start=1):
                    inner_assignment = {family: ('test' if family == test_family else
                                                 'validation' if family == validation_family else 'train')
                                        for family in families}
                    inner_folds.append(_fold(f'inner-{inner_index:03d}', families, inner_assignment,
                                             min_training_families=2))
                assignment = {family: 'test' if family == test_family else 'train' for family in families}
            else:
                validation = set(_ordered(development, split_seed, f'validation:{test_family}')[:n_validation])
                assignment = {family: ('test' if family == test_family else
                                       'validation' if family in validation else 'train')
                              for family in families}
            folds.append(_fold(f'outer-{index:03d}', families, assignment, inner_folds=inner_folds,
                               final_refit=strategy == 'nested_leave_one_out'))
    return {'schema_version': SCHEMA, **evaluation, 'split_seed': split_seed,
            'families': families, 'folds': folds,
            'counts': {'cases': sum(map(len, families.values())), 'families': len(families),
                       'outer_folds': len(folds),
                       'inner_folds': sum(len(fold['inner_folds']) for fold in folds),
                       'training_seeds': len(evaluation['seeds'])}}
