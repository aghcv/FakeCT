"""Ordered, bounded morphology recipes on a single immutable source crop.

Named ROIs are fixed spatial masks clipped by the study's outer ROI. Each pass
uses the preceding pass's labels and attenuation-copy proxy. Recipe order is
therefore part of the experiment; no convergence loop or topology guarantee is
implied by an iteration count.
"""
import hashlib

import numpy as np
from scipy import ndimage

from fakect_morphology import (ENGINE, MAX_MORPHOLOGY_VOXELS, apply_morphology,
                               validate_edit_geometry)
from fakect_roi import _tube_nodes, sphere_mask, tube_mask
from fakect_tissues import _validated_labels, coarse_labels
from fakect_reassignment import reassignment_masks, stiffness_field


RECIPE_ENGINE = 'ordered_named_roi_v1'
MAX_RECIPE_PASSES = 20
_STRUCTURE = ndimage.generate_binary_structure(3, 1)


def _roi_geometry(roi, resolved):
    kind = roi.get('shape')
    if kind == 'sphere':
        center = np.asarray(roi['center_ijk'], dtype=float)
        radius = np.asarray(roi['radius_mm'], dtype=float)
        if (center.shape != (3,) or not np.all(np.isfinite(center)) or np.any(center < 0)
                or radius.ndim != 0 or not np.isfinite(radius) or radius <= 0):
            raise ValueError('Named sphere requires finite nonnegative center_ijk and positive radius_mm')
        nodes, radii = center[None, :], radius.reshape(1)
    elif kind == 'tube':
        nodes, radii = _tube_nodes(roi['center_ijk'], roi['radius_mm'])
    else:
        raise ValueError('Named ROI shape must be sphere or tube')
    if 'shape_kji' in resolved and np.any(nodes > np.asarray(resolved['shape_kji'])[::-1] - 1):
        raise ValueError('Named ROI control point lies outside source dimensions')
    focus = tuple(map(int, np.rint(nodes[len(nodes) // 2])))
    return {**resolved, 'roi_kind': kind,
            'roi_nodes_ijk': tuple(tuple(map(float, n)) for n in nodes),
            'roi_radii_mm': tuple(map(float, radii)), 'focus_ijk': focus,
            'roi_node_low_ijk': tuple(map(float, nodes.min(axis=0))),
            'roi_node_high_ijk': tuple(map(float, nodes.max(axis=0)))}


def _step_config(config, name):
    edit = config['edits'][name]
    return {**config, 'roi': {**config['roi'], **config['rois'][edit['roi']]},
            'edit': {key: value for key, value in edit.items() if key not in ('roi', 'iterations')}}


def validate_recipe(config, resolved):
    """Return a serializable metadata-only plan; no voxel payload is read.

    Every active step's entire ROI envelope plus its edit/recipient halo must fit
    the common crop, even where the outer ROI later clips its editable mask.
    Overlap and nonempty-mask checks require native voxels and occur before the
    first pass in :func:`apply_recipe`.
    """
    recipe = config['recipe']
    names = tuple(recipe.get('steps', ()))
    if not names or len(names) != len(set(names)):
        raise ValueError('recipe.steps must contain distinct ordered edit names; use iterations for repetition')
    if recipe.get('overlap') not in ('sequential', 'error'):
        raise ValueError('recipe.overlap must be sequential or error')
    low = np.asarray(resolved['crop_low_ijk'])
    high = np.asarray(resolved['crop_high_ijk_exclusive'])
    if (low.shape != (3,) or high.shape != (3,) or np.any(high <= low)
            or np.any(low < 0) or np.any(low != np.floor(low)) or np.any(high != np.floor(high))):
        raise ValueError('Recipe requires valid integral common crop bounds')
    if int(np.prod(high - low)) > MAX_MORPHOLOGY_VOXELS:
        raise ValueError(f'Recipe crop exceeds {MAX_MORPHOLOGY_VOXELS:,} voxels')
    geometries = {name: _roi_geometry(roi, resolved) for name, roi in config['rois'].items()}
    steps, passes = [], 0
    for name in names:
        if name not in config['edits']:
            raise ValueError(f'Recipe refers to missing edit: {name}')
        edit = config['edits'][name]
        roi_name = edit.get('roi')
        if roi_name not in geometries:
            raise ValueError(f'Edit {name} refers to missing ROI: {roi_name}')
        iterations = edit.get('iterations', 1)
        if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or not 1 <= iterations <= 10:
            raise ValueError('Edit iterations must be an integer between 1 and 10')
        passes += int(iterations)
        if passes > MAX_RECIPE_PASSES:
            raise ValueError(f'Recipe requires more than {MAX_RECIPE_PASSES} total passes')
        geometry = geometries[roi_name]
        halo = validate_edit_geometry(geometry, _step_config(config, name))
        steps.append({'name': name, 'roi': roi_name, 'iterations': int(iterations),
                      'edit': {k: v for k, v in edit.items() if k not in ('roi', 'iterations')},
                      'geometry': {k: geometry[k] for k in ('roi_kind', 'roi_nodes_ijk', 'roi_radii_mm', 'focus_ijk')},
                      'halo': halo})
    return {'engine': RECIPE_ENGINE, 'morphology_engine': ENGINE,
            'steps': steps, 'total_passes': passes, 'overlap': recipe['overlap'],
            'outer_boundary': 'Every named ROI is intersected with the fixed study ROI.',
            'order_semantics': 'Each pass consumes the labels and scalar proxy produced by the preceding pass.'}


def _native_masks(arrays, resolved, config):
    shape = np.shape(arrays['act'])
    if len(shape) != 3 or not np.prod(shape) or np.prod(shape) > MAX_MORPHOLOGY_VOXELS:
        raise ValueError('Recipe requires a nonempty 3D crop within the morphology size limit')
    origin, spacing = resolved['crop_low_ijk'], resolved['spacing_ijk_mm']
    result = {}
    for name, roi in config['rois'].items():
        geometry = _roi_geometry(roi, resolved)
        if geometry['roi_kind'] == 'sphere':
            result[name] = sphere_mask(shape, origin, geometry['roi_nodes_ijk'][0], spacing,
                                       geometry['roi_radii_mm'][0])
        else:
            result[name] = tube_mask(shape, origin, geometry['roi_nodes_ijk'], spacing,
                                     geometry['roi_radii_mm'])
    return result


def recipe_masks(arrays, resolved, config):
    """Return fixed native-grid named masks intersected with the outer ROI."""
    outer = np.asarray(arrays['roi'])
    if outer.dtype.kind != 'b' or outer.shape != np.shape(arrays['act']):
        raise ValueError('Outer ROI must be a Boolean mask matching the crop')
    return {name: mask & outer for name, mask in _native_masks(arrays, resolved, config).items()}


def _digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _state(labels, atn, catalog, source_ids, roi, spacing):
    candidates = np.isin(labels, source_ids)
    return {'act': labels, 'atn': atn, 'tissue': coarse_labels(labels, catalog),
            'candidates': candidates, 'selected': candidates & roi, 'roi': roi,
            'attenuation_cm_inverse': atn / (spacing[0] / 10)}


def _components(mask):
    return int(ndimage.label(mask, structure=_STRUCTURE)[1])


def apply_recipe(arrays, resolved, config, on_step=None):
    """Apply a finite ordered recipe; source arrays are never mutated.

    ``on_step(event)`` is called synchronously after each pass and must treat its
    array references as read-only. Events contain ``before_arrays``, ``result``,
    ``resolved``, ``config``, ``step_name``, ``roi_name``, ``iteration`` and
    ``index`` (both 1-based). Retaining them is optional; the engine retains only
    compact per-pass summaries, not a history of full crops.

    Final ``changed_mask`` means original-versus-final label inequality.
    ``ever_changed_mask`` records activity across all passes, including changes
    later reversed. ``scalar_changed_mask`` separately captures proxy changes
    even where the final original label has been restored.
    """
    plan = validate_recipe(config, resolved)
    labels = _validated_labels(arrays['act'])
    atn = np.asarray(arrays['atn'])
    outer = np.asarray(arrays['roi'])
    if outer.dtype.kind != 'b' or outer.shape != labels.shape:
        raise ValueError('Outer ROI must be a Boolean mask matching the crop')
    if (atn.shape != labels.shape or atn.dtype.kind not in 'iuf' or not np.all(np.isfinite(atn))):
        raise ValueError('Source attenuation must be a finite numeric crop matching labels')
    if not np.array_equal(np.asarray(resolved['crop_high_ijk_exclusive']) - resolved['crop_low_ijk'], labels.shape[::-1]):
        raise ValueError('Resolved common crop bounds disagree with the label crop')
    source_ids = resolved.get('source_ids') or config['selection'].get('source_ids')
    if not source_ids:
        source_ids = tuple(r['original_id'] for r in resolved['catalog']['records']
                           if r['classification']['tissue_name'] == config['selection']['tissue'])
    source_ids = tuple(source_ids)
    original = _state(labels, atn, resolved['catalog'], source_ids, outer, resolved['spacing_ijk_mm'])
    for name in ('tissue', 'candidates', 'selected'):
        if not np.array_equal(arrays[name], original[name]):
            raise ValueError(f'Input {name} disagrees with original labels/selection/outer ROI')
    native = _native_masks(arrays, resolved, config)
    masks = {name: value & outer for name, value in native.items()}
    active_names = list(dict.fromkeys(step['roi'] for step in plan['steps']
                                      if step['edit'].get('operation', 'none') != 'none'))
    for name in active_names:
        if not masks[name].any():
            raise ValueError(f'Active ROI {name} has no voxels inside the outer ROI')
    coverage = {name: {'native_roi_voxels_in_crop': int(native[name].sum()),
                       'effective_roi_voxels': int(mask.sum()),
                       'clipped_by_outer_roi_voxels': int((native[name] & ~outer).sum()),
                       'original_target_voxels': int((original['candidates'] & mask).sum())}
                for name, mask in masks.items()}
    overlaps = []
    for index, name in enumerate(active_names):
        for other in active_names[index + 1:]:
            count = int((masks[name] & masks[other]).sum())
            overlaps.append({'roi_a': name, 'roi_b': other, 'effective_overlap_voxels': count,
                             'native_overlap_voxels_in_crop': int((native[name] & native[other]).sum())})
            if count and config['recipe']['overlap'] == 'error':
                raise ValueError(f'Active ROIs {name} and {other} overlap at {count} native voxels inside the outer ROI')
    union = np.zeros_like(outer)
    for name in active_names:
        union |= masks[name]
    cumulative = {key: np.zeros_like(outer) for key in ('ever_changed_mask', 'proposed_added_mask',
                  'proposed_removed_mask', 'blocked_mask', 'unresolved_mask')}
    strength = np.zeros(labels.shape, dtype=np.float64)
    current_labels, current_atn = labels.copy(), atn.copy()
    catalog_records = {record['original_id']: record for record in resolved['catalog']['records']}
    steps, warnings, activity, event_index = [], [], {'added': 0, 'removed': 0, 'changed': 0}, 0
    for step in plan['steps']:
        name, roi_name = step['name'], step['roi']
        step_resolved = _roi_geometry(config['rois'][roi_name], resolved)
        step_config = _step_config(config, name)
        for iteration in range(1, step['iterations'] + 1):
            event_index += 1
            before = _state(current_labels, current_atn, resolved['catalog'], source_ids,
                            masks[roi_name], resolved['spacing_ijk_mm'])
            requested = step_config['edit'].get('operation', 'none')
            empty_target = requested != 'none' and not before['selected'].any()
            execution_config = step_config
            if empty_target:
                execution_config = {**step_config, 'edit': {**step_config['edit'], 'operation': 'none'}}
            result = apply_morphology(before, step_resolved, execution_config)
            result['summary']['scalar_status'] = (
                'Copies winning INPUT-STATE target/recipient scalars. Earlier recipe passes may already '
                'have copied these values. Pass hashes record the state chain; this is a display proxy, '
                'not fresh AI recovery or a physical CT reconstruction. Original source arrays remain unchanged.')
            if config.get('reassignment', {}).get('mode') != 'stiffness':
                result['summary']['ownership_tie_break'] = (
                    'Shortest accepted six-edge physical path, then lowest signed anatomical source ID, '
                    'then earliest input-state seed k,j,i coordinate.')
            pass_target_after = np.isin(result['edited_labels'], source_ids)
            result['summary']['full_target_components_before'] = _components(before['candidates'])
            result['summary']['full_target_components_after'] = _components(pass_target_after)
            blocked_ids, blocked_counts = np.unique(current_labels[result['blocked_mask']], return_counts=True)
            result['summary']['blocked_input_labels'] = [
                {'original_id': int(key),
                 'original_name': catalog_records.get(int(key), {}).get('original_name', f'Unmapped original ID {int(key)}'),
                 'tissue_name': catalog_records.get(int(key), {}).get('classification', {}).get('tissue_name', 'unknown'),
                 'count': int(count)}
                for key, count in zip(blocked_ids, blocked_counts)]
            if empty_target:
                reason = 'No current target voxel remains inside this effective ROI; pass skipped.'
                result['summary']['requested_operation'] = requested
                result['summary']['no_op_reason'] = reason
                result['summary']['warnings'].append(reason)
            entry = {'index': event_index, 'step_name': name, 'roi_name': roi_name,
                     'iteration': iteration, 'status': 'empty_target' if empty_target else ('applied' if result['changed_mask'].any() else 'no_change'),
                     'requested_operation': requested, **result['summary'],
                     'labels_before_sha256': _digest(current_labels),
                     'labels_after_sha256': _digest(result['edited_labels']),
                     'attenuation_before_sha256': _digest(current_atn),
                     'attenuation_after_sha256': _digest(result['attenuation_proxy_per_pixel']),
                     'full_target_before': int(before['candidates'].sum()),
                     'full_target_after': int(pass_target_after.sum())}
            steps.append(entry)
            for warning in result['summary']['warnings']:
                warnings.append(f'{name} iteration {iteration}: {warning}')
            for key in activity:
                activity[key] += result['summary']['counts'][key]
            cumulative['ever_changed_mask'] |= result['changed_mask']
            for key in ('proposed_added_mask', 'proposed_removed_mask', 'blocked_mask', 'unresolved_mask'):
                cumulative[key] |= result[key]
            strength = np.maximum(strength, result['strength_mm'])
            if on_step is not None:
                on_step({'before_arrays': before, 'result': result, 'resolved': step_resolved,
                         'config': step_config, 'step_name': name, 'roi_name': roi_name,
                         'iteration': iteration, 'index': event_index, 'pass_summary': entry})
            current_labels, current_atn = result['edited_labels'], result['attenuation_proxy_per_pixel']
    target_before = original['selected']
    full_after = np.isin(current_labels, source_ids)
    target_after = full_after & outer
    changed, scalar_changed = current_labels != labels, current_atn != atn
    if np.any((changed | scalar_changed) & ~union):
        raise RuntimeError('Recipe changed values outside its active effective ROI union')
    _, protected, reassignment_policy = reassignment_masks(labels, original['candidates'], resolved['catalog'], config['reassignment'])
    if np.any((changed | scalar_changed) & protected):
        raise RuntimeError('Recipe changed an originally protected non-target voxel')
    added, removed = target_after & ~target_before, target_before & ~target_after
    counts = {name: int(mask.sum()) for name, mask in (
        ('before', target_before), ('after', target_after), ('added', added), ('removed', removed),
        ('changed', changed), ('scalar_changed', scalar_changed),
        ('ever_changed', cumulative['ever_changed_mask']),
        ('proposed_added', cumulative['proposed_added_mask']), ('proposed_removed', cumulative['proposed_removed_mask']),
        ('blocked', cumulative['blocked_mask']), ('unresolved', cumulative['unresolved_mask']))}
    transitions = []
    if changed.any():
        pairs, pair_counts = np.unique(np.column_stack((labels[changed], current_labels[changed])), axis=0, return_counts=True)
        transitions = [{'original_id': int(a), 'new_id': int(b), 'count': int(n)}
                       for (a, b), n in zip(pairs, pair_counts)]
    voxel_volume = float(np.prod(resolved['spacing_ijk_mm']))
    summary = {**plan, 'operation': 'recipe', 'counts': counts, 'steps': steps,
               'planned_steps': plan['steps'], 'activity_counts': activity,
               'volume_mm3': {key: counts[key] * voxel_volume for key in ('before', 'after', 'added', 'removed')},
               'components_before': _components(target_before), 'components_after': _components(target_after),
               'full_target_components_before': _components(original['candidates']),
               'full_target_components_after': _components(full_after),
               'full_target_before': int(original['candidates'].sum()), 'full_target_after': int(full_after.sum()),
               'roi_coverage': coverage, 'roi_overlaps': overlaps, 'transitions': transitions, 'warnings': warnings,
               'reassignment_policy': reassignment_policy,
               'changed_semantics': 'changed is original-versus-final label inequality; added/removed are net target-membership changes. ever_changed records unique voxels touched by any pass. activity_counts sum pass changes and can count a voxel repeatedly. Proposal/blocked/unresolved masks are unions of pass events.',
               'scalar_status': 'Each pass copies a winning CURRENT target/recipient scalar. The final attenuation-copy proxy may differ even where the original label is restored; scalar_changed_mask records this. This is not AI recovery or a physical CT reconstruction.',
               'topology_status': 'Connected-component counts are diagnostics. Repeated erosion/dilation can disconnect or merge regions; topology is not preserved by contract.',
               'original_labels_sha256': _digest(labels), 'final_labels_sha256': _digest(current_labels),
               'original_attenuation_sha256': _digest(atn), 'final_attenuation_sha256': _digest(current_atn)}
    result = {'edited_labels': current_labels, 'edited_tissue_labels': coarse_labels(current_labels, resolved['catalog']),
            'attenuation_proxy_per_pixel': current_atn, 'target_mask_before': target_before.copy(),
            'target_mask_after': target_after, 'added_mask': added, 'removed_mask': removed,
            'changed_mask': changed, 'scalar_changed_mask': scalar_changed,
            **cumulative, 'strength_mm': strength, 'summary': summary}
    if config.get('reassignment', {}).get('mode') == 'stiffness':
        result['stiffness_field'] = stiffness_field(labels, resolved['catalog'], config['reassignment'])
        result['final_stiffness_field'] = stiffness_field(current_labels, resolved['catalog'], config['reassignment'])
    return result
