"""Training-only, class-present sampling for net aorta changes.

The completed comparison sampler remains frozen. This additive sampler uses
only rows explicitly assigned to training, balances case, present change class
and relative burden strata, and never disguises a missing target as a positive
patch. Burden is a native added/removed count divided by the baseline aorta
count. Its terciles are fitted separately inside each TRAIN case/class; they
are sampling strata, not disease severity thresholds.
"""
from collections import Counter
from copy import deepcopy

import numpy as np


CLASS_CODES = {'added': 1, 'removed': 2}
BURDEN_BINS = ('small', 'medium', 'large')


class EmptyTargetClassError(ValueError):
    """A requested positive class does not survive in the model-grid focus."""


class InvalidTargetPatchError(ValueError):
    """A supposedly positive patch does not actually contain its target."""


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


class BalancedChangeSchedule:
    """Return individual patch draws with exactly 1/7 original controls.

    Positive classes and populated burden bins are round-robin balanced per
    case. Each bin traverses a shuffled sibling queue without replacement
    before restarting. Single-type siblings can only enter their present class.
    Zero-net edited siblings are explicitly reported and excluded from positive
    queues; original unchanged controls preserve the declared 1/7 exposure.
    """
    def __init__(self, rows, case_ids, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError('Sampling seed must be a nonnegative integer')
        cases = list(case_ids)
        if not cases or any(not isinstance(c, str) or not c for c in cases) or len(set(cases)) != len(cases):
            raise ValueError('Training case IDs must be unique nonempty strings')
        self.cases = sorted(cases)
        self.rng = np.random.default_rng(seed)
        self.rows, self.controls, self.members = {}, {}, {}
        self.class_names, self.class_positions, self.bin_positions = {}, {}, {}
        self.orders, self.positions = {}, {}
        self.visits, self.target_visits = Counter(), Counter()
        self.bin_visits, self.case_visits = Counter(), Counter()
        self.edited_noop_ids, self.thresholds = [], {}
        self.blocks_completed = 0
        for source in rows:
            row = deepcopy(source)
            if row.get('split') != 'train':
                raise ValueError('Balanced sampling accepts explicitly TRAIN rows only')
            if row.get('case_id') not in self.cases:
                raise ValueError('A sampler row belongs to a case outside the TRAIN roster')
            sid = row.get('sample_id')
            if not isinstance(sid, str) or not sid or sid in self.rows:
                raise ValueError('Training sample IDs must be unique nonempty strings')
            if type(row.get('baseline')) is not bool:
                raise ValueError('Baseline status must be explicitly Boolean')
            counts = row.get('mask_counts', {})
            before = _count(counts.get('before'), 'Baseline mask count')
            if before == 0:
                raise ValueError('Burden normalization needs a nonempty baseline aorta')
            added = _count(counts.get('added'), 'Added mask count')
            removed = _count(counts.get('removed'), 'Removed mask count')
            if removed > before:
                raise ValueError('Removed count cannot exceed the baseline mask count')
            if 'after' in counts and _count(counts['after'], 'After mask count') != before + added - removed:
                raise ValueError('Mask counts do not describe a consistent net before/after difference')
            if row['baseline'] and (added or removed):
                raise ValueError('An original baseline control must have zero net change')
            self.rows[sid] = row
            if not row['baseline'] and not (added or removed):
                self.edited_noop_ids.append(sid)
        for case in self.cases:
            group = [r for r in self.rows.values() if r['case_id'] == case]
            controls = [r['sample_id'] for r in group if r['baseline']]
            if len(controls) != 1:
                raise ValueError('Each TRAIN case requires exactly one original unchanged control')
            self.controls[case] = controls[0]
            present = []
            for name in CLASS_CODES:
                positives = sorted((r['sample_id'], r['mask_counts'][name] / r['mask_counts']['before'])
                    for r in group if not r['baseline'] and r['mask_counts'][name] > 0)
                if not positives:
                    continue
                present.append(name)
                values = np.asarray([burden for _, burden in positives], np.float64)
                low, high = map(float, np.quantile(values, [1/3, 2/3]))
                equal_cutoffs = low == high
                strata = {label: [] for label in BURDEN_BINS}
                for sid, burden in positives:
                    if equal_cutoffs:
                        label = 'small' if burden < low else 'large' if burden > high else 'medium'
                    else:
                        label = 'small' if burden <= low else 'medium' if burden <= high else 'large'
                    strata[label].append(sid)
                self.thresholds[(case, name)] = {'lower_tercile': low, 'upper_tercile': high,
                    'positive_siblings': len(positives), 'equal_cutoffs': equal_cutoffs}
                for label in BURDEN_BINS:
                    if strata[label]:
                        key = (case, name, label)
                        self.members[key] = strata[label]
                        self.orders[key], self.positions[key] = [], 0
                self.bin_positions[(case, name)] = 0
            if not present:
                raise ValueError('Each TRAIN case needs an achieved positive change class')
            self.class_names[case] = tuple(present)
            self.class_positions[case] = 0

    def _positive(self, case):
        names = self.class_names[case]
        name = names[self.class_positions[case] % len(names)]
        self.class_positions[case] += 1
        labels = [b for b in BURDEN_BINS if (case, name, b) in self.members]
        label = labels[self.bin_positions[(case, name)] % len(labels)]
        self.bin_positions[(case, name)] += 1
        key = (case, name, label)
        if self.positions[key] == len(self.orders[key]):
            self.orders[key] = list(self.rng.permutation(self.members[key]))
            self.positions[key] = 0
        sid = str(self.orders[key][self.positions[key]])
        self.positions[key] += 1
        return {'sample_id': sid, 'case_id': case, 'mode': name,
            'class_id': CLASS_CODES[name], 'burden_bin': label,
            'burden_fraction': self.rows[sid]['mask_counts'][name] / self.rows[sid]['mask_counts']['before'],
            'control': False, 'control_kind': None}

    def block(self, updates_per_case=28):
        """Draw a reproducible case-balanced reporting block, not an epoch."""
        if type(updates_per_case) is not int or updates_per_case < 7 or updates_per_case % 7:
            raise ValueError('Updates per case must be a positive multiple of seven')
        result = []
        for case in self.rng.permutation(self.cases):
            case = str(case)
            healthy = updates_per_case // 7
            draws = [{'sample_id': self.controls[case], 'case_id': case,
                'mode': 'uniform', 'class_id': 0, 'burden_bin': None,
                'burden_fraction': 0., 'control': True, 'control_kind': 'original_baseline'}
                for _ in range(healthy)]
            draws.extend(self._positive(case) for _ in range(updates_per_case - healthy))
            for index in self.rng.permutation(len(draws)):
                record = draws[int(index)]
                self.visits[record['sample_id']] += 1
                self.case_visits[case] += 1
                if not record['control']:
                    self.target_visits[record['mode']] += 1
                    self.bin_visits[(case, record['mode'], record['burden_bin'])] += 1
                result.append(record)
        self.blocks_completed += 1
        return result

    def receipt(self):
        """Expose all exclusions and strata, including zero-visit siblings."""
        controls = set(self.controls.values())
        return {'training_only': True, 'training_case_ids': self.cases,
            'original_control_fraction': 1/7, 'blocks_completed': self.blocks_completed,
            'reporting_block_is_complete_population_epoch': False,
            'sample_patch_visits': {sid: self.visits[sid] for sid in sorted(self.rows)},
            'case_patch_visits': dict(sorted(self.case_visits.items())),
            'class_patch_visits': dict(sorted(self.target_visits.items())),
            'class_visit_semantics': 'scheduled class requests; actual targeted/uniform outcome is recorded by strict_training_patch',
            'control_updates': sum(self.visits[sid] for sid in controls),
            'edited_updates': sum(n for sid, n in self.visits.items() if sid not in controls),
            'edited_noop_ids': sorted(self.edited_noop_ids),
            'edited_noop_policy': 'explicitly excluded from positive queues; original controls retain 1/7 exposure',
            'burden_semantics': 'native net class count / baseline aorta count; TRAIN case/class terciles',
            'strata': [{'case_id': case, 'mode': name, 'burden_bin': label,
                'sample_ids': list(ids), 'patch_visits': self.bin_visits[key],
                **self.thresholds[(case, name)]}
                for key, ids in sorted(self.members.items())
                for case, name, label in [key]]}


def available_target_classes(reader):
    """Return TRAIN target classes that exist on the actual predicted grid."""
    result = set()
    for proposal in range(len(reader.origins_kji)):
        classes = reader.available_classes(proposal)
        if any(name not in CLASS_CODES for name in classes):
            raise ValueError('Reader returned an unknown net-change class')
        result.update(classes)
    return tuple(name for name in CLASS_CODES if name in result)


def _centered_training_tile(reader, patch_shape, rng, mode):
    """Center a TRAIN changed voxel for a diagnostic without moving the focus."""
    eligible = [i for i in range(len(reader.origins_kji)) if mode in reader.available_classes(i)]
    if not eligible:
        raise EmptyTargetClassError('Centered target absent from model-grid focus; no uniform fallback')
    index = eligible[int(rng.integers(len(eligible)))]
    # Use the existing bounded reservoir of points already verified against
    # nearest-neighbor model-grid sampling. No native mask enters the input.
    local = reader._change_points(CLASS_CODES[mode]) - reader.origins_kji[index]
    local = local[np.all((local >= 0) & (local < reader.shape_kji), axis=1)]
    if not len(local):
        raise EmptyTargetClassError('Centered target has no verified model-grid points; no uniform fallback')
    reader._patch_geometry(index, [0, 0, 0], patch_shape)
    shape = np.asarray(patch_shape, np.int64)
    center = local[int(rng.integers(len(local)))]
    offset = np.clip(center - shape // 2, 0, reader.shape_kji - shape)
    return {'proposal_index': index, 'offset_kji': tuple(map(int, offset)),
        'shape_kji': tuple(map(int, shape))}


def strict_training_patch(reader, patch_shape, rng, mode, changed_fraction=.75, *, centered=False):
    """Read one audited patch, with no missing-class positive fallback.

    Explicit uniform draws do not inspect training targets to choose a crop.
    For a class request, a declared background fraction may choose uniform;
    class availability is checked first so an invalid positive request cannot
    hide behind that random background draw. A targeted draw must contain the
    requested class after real nearest-neighbor target resampling. ``centered``
    is an explicit small-training-diagnostic option: it puts a verified changed
    voxel near the patch center instead of jittering its position. It changes
    neither locator origins nor the target-independent inference schedule.
    """
    if mode not in ('uniform', *CLASS_CODES):
        raise ValueError('Training sampling mode must be uniform, added or removed')
    if (isinstance(changed_fraction, (bool, np.bool_)) or not np.isscalar(changed_fraction)
            or not np.isfinite(changed_fraction) or not 0 <= changed_fraction <= 1):
        raise ValueError('Changed fraction must be a finite number in [0,1]')
    if type(centered) is not bool:
        raise ValueError('Centered diagnostic status must be explicitly Boolean')
    requested_code = CLASS_CODES.get(mode, 0)
    available = None
    if requested_code:
        available = available_target_classes(reader)
        if mode not in available:
            raise EmptyTargetClassError(f'Requested {mode} class is absent from the model-grid focus; no uniform fallback')
        realized = mode if rng.random() < changed_fraction else 'uniform'
    else:
        realized = 'uniform'
    tile = (_centered_training_tile(reader, patch_shape, rng, realized)
        if centered and realized != 'uniform' else reader.training_patch(patch_shape, rng, mode=realized))
    if realized != 'uniform' and realized not in reader.available_classes(tile['proposal_index']):
        raise InvalidTargetPatchError('Class-targeted draw selected an ineligible proposal')
    data = reader.read_patch(**tile)
    labels = np.asarray(data['labels'])
    if labels.ndim != 3 or not np.isin(labels, (0, 1, 2)).all():
        raise InvalidTargetPatchError('Patch contains invalid net-change label codes')
    counts = {name: int(np.count_nonzero(labels == code)) for name, code in CLASS_CODES.items()}
    targeted = realized != 'uniform'
    if targeted and not counts[realized]:
        raise InvalidTargetPatchError(f'Targeted {realized} class disappeared in the actual resampled patch; no uniform fallback')
    if 'valid' in data:
        valid = np.asarray(data['valid'])
        if valid.shape != labels.shape or not np.isin(valid, (0, 1)).all():
            raise InvalidTargetPatchError('Patch validity must be an aligned binary array')
        if np.any((labels > 0) & (valid == 0)):
            raise InvalidTargetPatchError('Positive labels occur outside acquisition validity')
    reason = None if targeted else 'declared_uniform' if mode == 'uniform' else 'declared_background_fraction'
    metadata = data.get('metadata', {})
    return {'tile': tile, 'data': data, 'audit': {
        'training_only': True, 'sample_id': metadata.get('sample_id'),
        'requested_mode': mode, 'realized_mode': realized,
        'requested_class_id': requested_code, 'targeted': targeted,
        'centered_diagnostic_requested': centered,
        'centering_policy': 'verified_changed_voxel_at_center' if centered and targeted
            else 'randomized_changed_voxel_position' if targeted else 'uniform',
        'changed_fraction': float(changed_fraction), 'uniform_reason': reason,
        'available_target_classes': list(available) if available is not None else None,
        'patch_class_counts': counts, 'positive_class_verified_after_resampling': targeted,
        'missing_class_fallback': False}}
