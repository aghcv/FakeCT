"""Local retained-volume and component-preservation checks for erosion trials.

The reference is made once before a recipe step, not once per retry or iteration.
Connectivity is a six-face voxel criterion within the loaded crop, not a minimum
lumen caliber, a vascular-tree inference, or a complete topology guarantee.
"""
import numpy as np
from scipy import ndimage


_STRUCTURE = ndimage.generate_binary_structure(3, 1)
_FIELDS = {'min_volume_ratio', 'preserve_connectivity', 'backoff_factor', 'max_backoff_steps'}
SCOPE_SEMANTICS = (
    'Pre-step selected target within the positive longitudinal profile support. '
    'For a ranged tube this uses its selected arc interval and Gaussian window, '
    'including the full selected cross-section before angular weighting, tissue '
    'resistance or distance-threshold proposals. The rest of the parent ROI is excluded.')
REFERENCE_SEMANTICS = (
    'Fixed immediately before this edit step, after earlier recipe steps; reused '
    'unchanged across its iterations and retries. Different named edits have separate references.')


def erosion_guard_spec(config):
    """Validate and normalize per-edit erosion acceptance and retry settings.

    Parameters
    ----------
    config : dict
        Normalized configuration sections; paths are resolved Path objects.

    Returns
    -------
    dict
        Volume ratio floor, connectivity flag, backoff factor/count and enabled status.
        Omitted checks are disabled.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    edit = config.get('edit', {})
    if _FIELDS.intersection(edit) and edit.get('operation', 'none') not in ('erosion', 'none'):
        raise ValueError('Erosion safeguard fields require operation=erosion')
    numbers = {}
    for key, default in (('min_volume_ratio', 0.), ('backoff_factor', .5)):
        value = edit.get(key, default)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f'edit.{key} must be a finite number')
        numbers[key] = float(value)
    floor, factor = numbers['min_volume_ratio'], numbers['backoff_factor']
    if not np.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('edit.min_volume_ratio must be finite and between 0 and 1')
    if not np.isfinite(factor) or not 0 < factor < 1:
        raise ValueError('edit.backoff_factor must be finite and strictly between 0 and 1')
    connectivity = edit.get('preserve_connectivity', False)
    if not isinstance(connectivity, (bool, np.bool_)):
        raise ValueError('edit.preserve_connectivity must be Boolean')
    limit = edit.get('max_backoff_steps', 8)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or not 0 <= limit <= 20:
        raise ValueError('edit.max_backoff_steps must be an integer between 0 and 20')
    return {**numbers, 'preserve_connectivity': bool(connectivity), 'max_backoff_steps': int(limit),
            'enabled': bool(floor > 0 or connectivity)}


def create_erosion_guard(arrays, resolved, config):
    """Capture the fixed pre-step local target and affected connectivity reference.

    Parameters
    ----------
    arrays : dict of ndarray
        Matching native k,j,i label (act), attenuation (atn), tissue, candidates, selected
        and ROI arrays. Inputs are not mutated.
    resolved : dict
        Resolved catalog, source identities and native crop geometry; coordinates use i,j,k
        indices and spacing uses millimetres.
    config : dict
        Normalized configuration sections; paths are resolved Path objects.

    Returns
    -------
    dict or None
        Independent scope mask, original local count, voxel volume in mm3 and optional full-
        target component reference; None when safeguards are disabled or the operation is
        not erosion.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    options = erosion_guard_spec(config)
    if not options['enabled'] or config.get('edit', {}).get('operation') != 'erosion':
        return None
    from fakect_morphology import strength_field_mm
    selected = np.asarray(arrays['selected'])
    candidates = np.asarray(arrays['candidates'])
    if selected.dtype.kind != 'b' or candidates.dtype.kind != 'b' or selected.shape != candidates.shape:
        raise ValueError('Erosion safeguard requires matching Boolean selected/candidates masks')
    scope = selected & (strength_field_mm(selected.shape, resolved, config) > 0)
    reference = {'scope_mask': scope, 'baseline_target_voxels': int(scope.sum()), 'options': options,
                 'voxel_volume_mm3': float(np.prod(resolved['spacing_ijk_mm'])),
                 'scope_semantics': SCOPE_SEMANTICS, 'reference_semantics': REFERENCE_SEMANTICS}
    if options['preserve_connectivity']:
        components, _ = ndimage.label(candidates, structure=_STRUCTURE)
        ids = np.unique(components[scope])
        ids = ids[ids != 0]
        reference.update(component_labels=components, affected_component_ids=ids,
                         component_sizes=np.bincount(components.ravel()))
    return reference


def evaluate_erosion_guard(reference, target_after):
    """Check local retained volume and affected component survival against a fixed reference.

    Parameters
    ----------
    reference : dict
        Pre-step safeguard measurement captured by create_erosion_guard.
    target_after : ndarray of bool
        Full current target membership on the same native k,j,i crop.

    Returns
    -------
    dict
        Retained count/ratio, optional component details, rejection reasons and accepted
        flag. No target voxels are changed.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    target = np.asarray(target_after)
    scope = reference['scope_mask']
    if target.dtype.kind != 'b' or target.shape != scope.shape:
        raise ValueError('Erosion safeguard requires a Boolean target matching its reference')
    baseline = reference['baseline_target_voxels']
    retained = int(np.count_nonzero(scope & target))
    ratio = retained / baseline if baseline else 1.
    options = reference['options']
    reasons, details = [], []
    if ratio + 1e-12 < options['min_volume_ratio']:
        reasons.append('local_volume_below_floor')
    connectivity_ok = None
    if options['preserve_connectivity']:
        after_components, after_count = ndimage.label(target, structure=_STRUCTURE)
        before_components = reference['component_labels']
        affected = reference['affected_component_ids']
        surviving_sizes = np.bincount(before_components[target],
                                      minlength=len(reference['component_sizes']))
        # Count distinct (before component, surviving component) pairs in one
        # volume traversal, avoiding a full-crop scan per tiny artery component.
        active = target & np.isin(before_components, affected)
        radix = int(after_count) + 1
        pairs = (before_components[active].astype(np.int64) * radix + after_components[active])
        pair_counts = np.bincount(np.unique(pairs) // radix,
                                  minlength=len(reference['component_sizes']))
        # Compare each affected component separately. A vanished component must
        # not cancel another component's split in an aggregate component count.
        for component_id in affected:
            count = int(pair_counts[component_id])
            details.append({'component_id': int(component_id),
                'before_voxels': int(reference['component_sizes'][component_id]),
                'surviving_voxels': int(surviving_sizes[component_id]),
                'surviving_components': count, 'ok': count == 1})
        connectivity_ok = all(row['ok'] for row in details)
        if not connectivity_ok:
            reasons.append('affected_component_split_or_lost')
    return {'retained_target_voxels': retained, 'retained_volume_ratio': ratio,
            'connectivity_ok': connectivity_ok, 'affected_components': len(details),
            'component_details': details, 'reasons': reasons, 'accepted': not reasons}


def apply_guarded_erosion(arrays, resolved, config, reference=None):
    """Retry erosion from identical pass inputs until safeguards accept a trial.

    Parameters
    ----------
    arrays : dict of ndarray
        Matching native k,j,i label (act), attenuation (atn), tissue, candidates, selected
        and ROI arrays. Inputs are not mutated.
    resolved : dict
        Resolved catalog, source identities and native crop geometry; coordinates use i,j,k
        indices and spacing uses millimetres.
    config : dict
        Normalized configuration sections; paths are resolved Path objects.
    reference : dict, optional
        Fixed pre-step safeguard reference; None captures a new reference when enabled.

    Returns
    -------
    dict
        Edited original/tissue labels, before/after target masks,
        proposal/accepted/blocked/unresolved masks, millimetre strength field, copied
        attenuation proxy and audit summary. Optional stiffness, directional, lineage and
        released-marker fields follow the configured mode. Source arrays are unchanged.
        Guarded results additionally contain scope/retained masks and accepted-
        distance/backoff evidence in summary. Failed trials never become retry inputs.

    Raises
    ------
    ValueError
        If guard settings or the input are invalid, settings changed relative to the
        reference, or even the no-op state violates it.
    RuntimeError
        If morphology invariants fail.
    """
    from fakect_morphology import _apply_morphology_once
    options = erosion_guard_spec(config)
    if not options['enabled'] or config.get('edit', {}).get('operation') != 'erosion':
        return _apply_morphology_once(arrays, resolved, config)
    if reference is None:
        reference = create_erosion_guard(arrays, resolved, config)
    if reference['options'] != options:
        raise ValueError('Erosion safeguard settings changed within a referenced step')
    requested = float(config['edit']['distance_mm'])
    source_ids = resolved.get('source_ids') or config['selection'].get('source_ids') or tuple(
        row['original_id'] for row in resolved['catalog']['records']
        if row['classification']['tissue_name'] == config['selection']['tissue'])
    attempts = []
    accepted_distance, result, evaluation = 0., None, None
    if reference['baseline_target_voxels']:
        for reduction in range(options['max_backoff_steps'] + 1):
            distance = requested * options['backoff_factor'] ** reduction
            if distance <= 0:
                break  # Finite positive factors can underflow; use the no-op fallback.
            trial_config = {**config, 'edit': {**config['edit'], 'distance_mm': distance}}
            trial = _apply_morphology_once(arrays, resolved, trial_config)
            evaluation = evaluate_erosion_guard(reference, np.isin(trial['edited_labels'], source_ids))
            attempts.append({'distance_mm': distance, **evaluation})
            if evaluation['accepted']:
                result, accepted_distance = trial, distance
                break
            del trial
    if result is None:
        # A rejected trial never becomes the input for the next attempt, and
        # none of its markers, proxy values, ancestry or audit masks escape.
        noop_config = {**config, 'edit': {**config['edit'], 'operation': 'none', 'distance_mm': 0.}}
        result = _apply_morphology_once(arrays, resolved, noop_config)
        evaluation = evaluate_erosion_guard(reference, np.isin(result['edited_labels'], source_ids))
        if not evaluation['accepted']:
            raise ValueError('Current pass input already violates its fixed erosion safeguard reference')
        attempts.append({'distance_mm': 0., **evaluation})
    status = 'skipped' if accepted_distance == 0 else ('accepted' if accepted_distance == requested else 'reduced')
    metadata = {**options, **evaluation, 'status': status,
                'requested_distance_mm': requested, 'accepted_distance_mm': accepted_distance,
                'baseline_target_voxels': reference['baseline_target_voxels'],
                'baseline_volume_mm3': reference['baseline_target_voxels'] * reference['voxel_volume_mm3'],
                'retained_volume_mm3': evaluation['retained_target_voxels'] * reference['voxel_volume_mm3'],
                'scope_semantics': reference['scope_semantics'],
                'reference_semantics': reference['reference_semantics'],
                'connectivity_semantics': 'Each affected pre-step full-crop target component must retain exactly one six-face-connected survivor component. Pre-existing disconnected components are checked separately; no repair or minimum caliber is implied.',
                'attempts': attempts}
    result.update(erosion_guard_scope_mask=reference['scope_mask'].copy(),
                  erosion_guard_retained_mask=reference['scope_mask'] & np.isin(result['edited_labels'], source_ids))
    result['summary']['erosion_safeguard'] = metadata
    result['summary']['requested_operation'] = 'erosion'
    if status != 'accepted':
        result['summary']['warnings'].append(
            f'Erosion safeguard {status}: requested {requested:g} mm; accepted {accepted_distance:g} mm; '
            f'local retained volume {evaluation["retained_volume_ratio"]:.1%}.')
    return result
