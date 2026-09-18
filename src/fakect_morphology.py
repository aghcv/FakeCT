"""Bounded label edits with physical six-neighbor distances and explicit repair.

The six-neighbor topology and normalized Gaussian curve come from the phantom
branch's ``fakect.py``. This is a new engine: distance is a weighted grid-path
length in millimetres, not the legacy ROI scale, Euclidean radius, vessel
diameter ratio, mechanical model, or clinical stenosis percentage.
"""
import heapq

import numpy as np
from scipy import ndimage

from fakect_tissues import _validated_labels, coarse_labels
from fakect_reassignment import reassignment_masks, stiffness_field, validate_reassignment_policy
from fakect_direction import directional_spec, direction_weight_field


ENGINE = 'weighted_6_neighbor_mm_v1'
MAX_MORPHOLOGY_VOXELS = 2_000_000
MAX_DISTANCE_MM = 50.0
_TOL = 1e-9
_STRUCTURE = ndimage.generate_binary_structure(3, 1)


def _edit_spec(resolved, config):
    edit = config.get('edit', {})
    directional_spec(resolved, config)
    policy = config.get('reassignment', {})
    operation = edit.get('operation', 'none')
    if operation not in ('none', 'erosion', 'dilation'):
        raise ValueError('edit.operation must be none, erosion, or dilation')
    distance = float(edit.get('distance_mm', 0))
    if not np.isfinite(distance) or not 0 <= distance <= MAX_DISTANCE_MM:
        raise ValueError(f'edit.distance_mm must be finite and between 0 and {MAX_DISTANCE_MM:g}')
    if operation != 'none' and distance == 0:
        raise ValueError('Enabled morphology requires positive edit.distance_mm; use operation=none for identity')
    search = float(policy.get('max_distance_mm', 3))
    if not np.isfinite(search) or not 0 < search <= MAX_DISTANCE_MM:
        raise ValueError(f'reassignment.max_distance_mm must be positive and <= {MAX_DISTANCE_MM:g}')
    profile = edit.get('profile', 'uniform')
    axis = edit.get('profile_axis', 'k')
    if profile not in ('uniform', 'gaussian') or axis not in ('tube', 'i', 'j', 'k'):
        raise ValueError('Invalid edit.profile or edit.profile_axis')
    if profile == 'gaussian' and axis == 'tube' and resolved['roi_kind'] != 'tube':
        raise ValueError('edit.profile_axis=tube requires a tube ROI')
    shape_k = float(edit.get('shape_k', 10))
    window = np.asarray(edit.get('shape_window', (0, 1)), dtype=float)
    if not np.isfinite(shape_k) or shape_k <= 0:
        raise ValueError('edit.shape_k must be positive and finite')
    if window.shape != (2,) or not np.all(np.isfinite(window)) or not 0 <= window[0] < window[1] <= 1:
        raise ValueError('edit.shape_window must satisfy 0 <= start < end <= 1')
    unresolved = policy.get('unresolved', 'preserve')
    if unresolved not in ('preserve', 'error'):
        raise ValueError('reassignment.unresolved must be preserve or error')
    categories = {c['name']: c['id'] for c in resolved['catalog']['categories']}
    target = config['selection']['tissue']
    allowed = tuple(policy.get('allowed_tissues', ()))
    target_ids = resolved.get('source_ids') or config['selection'].get('source_ids') or tuple(
        record['original_id'] for record in resolved['catalog']['records']
        if record['classification']['tissue_name'] == target)
    validate_reassignment_policy(resolved['catalog'], policy, target, target_ids)
    if target not in categories or (operation != 'none' and target in ('unknown', 'background')):
        raise ValueError('The edit target must be a known anatomical tissue, not unknown/background')
    return {'operation': operation, 'distance': distance, 'search': search, 'profile': profile,
            'axis': axis, 'shape_k': shape_k, 'window': window, 'unresolved': unresolved,
            'target_code': categories[target], 'allowed_codes': tuple(categories[n] for n in allowed),
            'allowed_tissues': allowed}


def validate_edit_geometry(resolved, config):
    """Validate policy and conservative edit/search halo without reading voxels.

    The entire continuous ROI envelope, enlarged by edit distance + recipient
    search distance + one largest-spacing step, must fit within loaded voxel
    centers. This deliberately rejects boundary-truncated trials rather than
    treating a missing outside crop/source as background.
    """
    spec = _edit_spec(resolved, config)
    spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Morphology requires finite positive i,j,k spacing')
    if spec['operation'] == 'none':
        return {'required_halo_mm': 0.0, 'checked': False}
    low = np.asarray(resolved['crop_low_ijk'], dtype=float)
    high = np.asarray(resolved['crop_high_ijk_exclusive'], dtype=float)
    if (low.shape != (3,) or high.shape != (3,) or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high)) or np.any(low < 0) or np.any(high <= low)
            or np.any(low != np.floor(low)) or np.any(high != np.floor(high))):
        raise ValueError('Morphology requires valid crop bounds')
    if 'shape_kji' in resolved and np.any(high > np.asarray(resolved['shape_kji'])[::-1]):
        raise ValueError('Morphology crop exceeds source dimensions')
    if int(np.prod(high - low)) > MAX_MORPHOLOGY_VOXELS:
        raise ValueError(f'Morphology crop exceeds {MAX_MORPHOLOGY_VOXELS:,} voxels; reduce crop/path length')
    nodes = np.asarray(resolved['roi_nodes_ijk'], dtype=float)
    radii = np.asarray(resolved['roi_radii_mm'], dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1:] != (3,) or radii.shape != (len(nodes),)
            or not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(radii)) or np.any(radii <= 0)):
        raise ValueError('Invalid resolved ROI geometry')
    envelope_low = (nodes * spacing - radii[:, None]).min(axis=0)
    envelope_high = (nodes * spacing + radii[:, None]).max(axis=0)
    halo = spec['distance'] + spec['search'] + float(spacing.max())
    if (np.any(envelope_low - halo < low * spacing - _TOL)
            or np.any(envelope_high + halo > (high - 1) * spacing + _TOL)):
        message = (f'Insufficient crop/source halo: ROI needs {halo:g} mm of context beyond its envelope '
                   f'({spec["distance"]:g} mm edit + {spec["search"]:g} mm reassignment + '
                   f'{spacing.max():g} mm spacing). ')
        if 'shape_kji' in resolved and (
                np.any(envelope_low - halo < -_TOL) or
                np.any(envelope_high + halo > (np.asarray(resolved['shape_kji'])[::-1] - 1) * spacing + _TOL)):
            message += ('The required context extends beyond the source volume; increasing '
                        'crop_half_width_mm cannot supply it. Relocate/shorten the ROI or reduce '
                        'the edit/search distances.')
        else:
            margins = np.stack((envelope_low - low * spacing,
                                (high - 1) * spacing - envelope_high))
            side, axis = np.unravel_index(np.argmin(margins), margins.shape)
            message += (f'Only {margins[side, axis]:g} mm is available at the '
                        f'{("lower", "upper")[side]} {"ijk"[axis]} crop face. '
                        'Increase [roi] crop_half_width_mm; a larger crop may also need a larger '
                        '[preview] volume_stride. Recheck both with --validate-only.')
        raise ValueError(message)
    return {'required_halo_mm': halo, 'checked': True,
            'roi_envelope_ijk_mm': [envelope_low.tolist(), envelope_high.tolist()]}


def tube_position_mm(shape_kji, origin_ijk, nodes_ijk, spacing_ijk_mm):
    """Physical arc coordinate of the closest point on an ordered centerline.

    At equal distances the first supplied segment wins, matching the existing
    tube profile convention. Endcap positions clamp to the first/last point.
    """
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    nodes = np.asarray(nodes_ijk, dtype=float) * spacing
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    if len(nodes) < 2 or np.any(lengths <= 0):
        raise ValueError('Tube coordinates require distinct consecutive centers')
    cumulative = np.concatenate(([0.], np.cumsum(lengths)))
    k, j, i = np.ogrid[:shape_kji[0], :shape_kji[1], :shape_kji[2]]
    coordinates = [(v + origin_ijk[a]) * spacing[a] for a, v in enumerate((i, j, k))]
    closest = np.full(shape_kji, np.inf)
    position = np.zeros(shape_kji, dtype=np.float64)
    for index, (start, end) in enumerate(zip(nodes[:-1], nodes[1:])):
        delta = end - start
        t = np.clip(sum((coordinates[a] - start[a]) * delta[a] for a in range(3))
                    / (lengths[index] ** 2), 0, 1)
        squared = sum((coordinates[a] - start[a] - t*delta[a]) ** 2 for a in range(3))
        nearer = squared < closest - _TOL
        closest[nearer] = squared[nearer]
        arc = cumulative[index] + t*lengths[index]
        position[nearer] = arc[nearer]
    return position


def strength_field_mm(shape_kji, resolved, config):
    """Spatial edit budget, including zero-ended Gaussian tube arc-length taper."""
    spec = _edit_spec(resolved, config)
    if spec['operation'] == 'none':
        return np.zeros(shape_kji, dtype=np.float64)
    if spec['profile'] == 'uniform':
        return np.full(shape_kji, spec['distance'], dtype=np.float64)
    spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
    origin = np.asarray(resolved['crop_low_ijk'], dtype=float)
    nodes = np.asarray(resolved['roi_nodes_ijk'], dtype=float) * spacing
    radii = np.asarray(resolved['roi_radii_mm'], dtype=float)
    k, j, i = np.ogrid[:shape_kji[0], :shape_kji[1], :shape_kji[2]]
    coordinates = [(v + origin[a]) * spacing[a] for a, v in enumerate((i, j, k))]
    if spec['axis'] == 'tube':
        parent = resolved.get('range_parent_nodes_ijk', resolved['roi_nodes_ijk'])
        full_length = np.linalg.norm(np.diff(np.asarray(parent) * spacing, axis=0), axis=1).sum()
        start, end = resolved.get('range_interval_mm', (0., full_length))
        position = (tube_position_mm(shape_kji, origin, parent, spacing) - start) / (end - start)
    else:
        axis = ('i', 'j', 'k').index(spec['axis'])
        lo = float((nodes[:, axis] - radii).min())
        hi = float((nodes[:, axis] + radii).max())
        position = np.broadcast_to((coordinates[axis] - lo) / (hi - lo), shape_kji)
    low, high = spec['window']
    x = (position - low) / (high - low)
    steepness = max(1e-6, spec['shape_k'])
    base = np.exp(-steepness * .25)
    gaussian = (np.exp(-steepness * (x - .5) ** 2) - base) / max(1e-8, 1 - base)
    gaussian = np.where((position > low) & (position < high), np.clip(gaussian, 0, 1), 0)
    return np.asarray(gaussian * spec['distance'], dtype=np.float64)


def _neighbors(index, shape, spacing):
    nz, ny, nx = shape
    z, plane = divmod(index, ny * nx)
    y, x = divmod(plane, nx)
    if x:
        yield index - 1, spacing[0]
    if x + 1 < nx:
        yield index + 1, spacing[0]
    if y:
        yield index - nx, spacing[1]
    if y + 1 < ny:
        yield index + nx, spacing[1]
    if z:
        yield index - ny*nx, spacing[2]
    if z + 1 < nz:
        yield index + ny*nx, spacing[2]


def _distances(domain, seeds, spacing, maximum, initial=None):
    """Bounded weighted six-edge shortest paths, without ownership overhead."""
    flat_domain = domain.ravel()
    distances = np.full(domain.size, np.inf)
    ids = np.flatnonzero(seeds)
    values = np.zeros(len(ids)) if initial is None else initial.ravel()[ids]
    heap = [(float(d), int(p)) for p, d in zip(ids, values) if d <= maximum + _TOL]
    for distance, index in heap:
        distances[index] = distance
    heapq.heapify(heap)
    while heap:
        distance, index = heapq.heappop(heap)
        if distance > distances[index] + _TOL:
            continue
        for neighbor, step in _neighbors(index, domain.shape, spacing):
            candidate = distance + step
            if flat_domain[neighbor] and candidate <= maximum + _TOL and candidate < distances[neighbor] - _TOL:
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return distances.reshape(domain.shape)


def _owned_paths(domain, seeds, spacing, maximum, labels, budgets=None, initial=None):
    """Accepted frontier propagation; ties use original signed ID then seed index."""
    flat_domain, flat_labels, flat_seeds = domain.ravel(), labels.ravel(), seeds.ravel()
    limit = None if budgets is None else budgets.ravel()
    distances = np.full(domain.size, np.inf)
    owners = np.full(domain.size, -1, dtype=np.int32)
    ids = np.flatnonzero(seeds)
    values = np.zeros(len(ids)) if initial is None else np.asarray(initial).ravel()[ids]
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Recipient seed costs must be finite and nonnegative')
    accepted = values <= maximum + _TOL
    ids, values = ids[accepted], values[accepted]
    heap = [(float(cost), int(flat_labels[p]), int(p), int(p)) for p, cost in zip(ids, values)]
    distances[ids], owners[ids] = values, ids
    heapq.heapify(heap)
    while heap:
        distance, owner_label, owner, index = heapq.heappop(heap)
        if owners[index] != owner or distance > distances[index] + _TOL:
            continue
        for neighbor, step in _neighbors(index, domain.shape, spacing):
            candidate = distance + step
            if not flat_domain[neighbor] or candidate > maximum + _TOL:
                continue
            if limit is not None and not flat_seeds[neighbor] and candidate > limit[neighbor] + _TOL:
                continue
            previous = int(owners[neighbor])
            closer = candidate < distances[neighbor] - _TOL
            tie = abs(candidate - distances[neighbor]) <= _TOL
            preferred = previous < 0 or (owner_label, owner) < (int(flat_labels[previous]), previous)
            if closer or (tie and preferred):
                distances[neighbor], owners[neighbor] = candidate, owner
                heapq.heappush(heap, (candidate, owner_label, owner, neighbor))
    return distances.reshape(domain.shape), owners.reshape(domain.shape)


def _erosion_boundary(candidates, spacing):
    """Inside voxels adjacent to full candidate complement, with correct edge cost."""
    initial = np.full(candidates.shape, np.inf)
    for array_axis, edge_cost in enumerate(spacing[::-1]):
        for shift in (-1, 1):
            neighbor = np.roll(candidates, shift, axis=array_axis)
            edge = [slice(None)] * 3
            edge[array_axis] = 0 if shift == 1 else -1
            neighbor[tuple(edge)] = False
            border = candidates & ~neighbor
            initial[border] = np.minimum(initial[border], edge_cost)
    return np.isfinite(initial), initial


def apply_morphology(arrays, resolved, config):
    """Return edited copies, exact audit masks, and a scalar COPY proxy.

    The configured policy determines eligible current-state donors/recipients.
    Release propagation traverses only requested release voxels;
    dilation traverses only accepted target/proposed eligible donor voxels.
    """
    labels = _validated_labels(arrays['act'])
    if labels.ndim != 3 or labels.size == 0:
        raise ValueError('Morphology requires a nonempty 3D label crop')
    attenuation = np.asarray(arrays['atn'])
    if attenuation.shape != labels.shape or attenuation.dtype.kind not in 'iuf' or not np.all(np.isfinite(attenuation)):
        raise ValueError('Source attenuation must be a finite numeric crop matching the labels')
    masks = {}
    for name in ('candidates', 'selected', 'roi'):
        value = np.asarray(arrays[name])
        if value.shape != labels.shape or value.dtype.kind != 'b':
            raise ValueError(f'{name} must be a Boolean mask matching the labels')
        masks[name] = value
    candidates, before, roi = (masks[n] for n in ('candidates', 'selected', 'roi'))
    if not np.array_equal(before, candidates & roi):
        raise ValueError('selected must equal candidates AND roi')
    tissue = coarse_labels(labels, resolved['catalog'])
    if not np.array_equal(tissue, arrays['tissue']):
        raise ValueError('Input tissue labels disagree with the source catalog')
    spec = _edit_spec(resolved, config)
    if np.any(candidates & (tissue != spec['target_code'])):
        raise ValueError('Candidate mask contains labels outside the selected target tissue')
    if spec['operation'] != 'none' and not np.any(before):
        raise ValueError('Enabled morphology requires a nonempty selected target within the ROI')
    geometry = dict(resolved)
    actual_high = tuple(np.asarray(resolved['crop_low_ijk']) + labels.shape[::-1])
    geometry.setdefault('crop_high_ijk_exclusive', actual_high)
    if not np.array_equal(geometry['crop_high_ijk_exclusive'], actual_high):
        raise ValueError('Resolved crop bounds disagree with actual label array shape')
    halo = validate_edit_geometry(geometry, config)
    spacing = tuple(map(float, resolved['spacing_ijk_mm']))
    strength = strength_field_mm(labels.shape, resolved, config)
    strength[~roi] = 0
    directional = None
    if config.get('edit', {}).get('direction', 'all') != 'all':
        directional = direction_weight_field(labels.shape, resolved, config, roi)
        strength *= directional['direction_weight']
    resistance = (stiffness_field(labels, resolved['catalog'], config['reassignment'])
                  if config.get('reassignment', {}).get('mode') == 'stiffness' else None)
    effective_strength = strength if resistance is None else strength * (1 - resistance)
    proposed_added = np.zeros_like(before)
    proposed_removed = np.zeros_like(before)
    requested_removed = np.zeros_like(before)
    added, removed, blocked, unresolved = (np.zeros_like(before) for _ in range(4))
    edited = labels.copy()
    proxy = attenuation.copy()
    eligible, protected, reassignment_policy = reassignment_masks(labels, candidates, resolved['catalog'], config.get('reassignment', {}))
    warnings = []
    if directional is not None:
        skipped = directional['summary']['unreliable_roi_voxels']
        if skipped:
            warnings.append(f'Curvature direction is unreliable at {skipped} ROI voxels; their directional edit budget is zero.')
        if not directional['summary']['angular_supported_roi_voxels']:
            warnings.append('No ROI voxel has a reliable direction inside the selected angular sector; this pass cannot change labels.')
    if spec['operation'] == 'dilation' and spec['distance'] > 0:
        distance = _distances(roi, before, spacing, spec['distance'])
        proposed_added = roi & ~candidates & (distance <= strength + _TOL) & (strength > 0)
        domain = before | (proposed_added & eligible)
        _, owners = _owned_paths(domain, before, spacing, spec['distance'], labels, effective_strength)
        added = proposed_added & (owners >= 0)
        blocked = proposed_added & ~added
        edited[added] = labels.ravel()[owners[added]]
        proxy[added] = attenuation.ravel()[owners[added]]
    elif spec['operation'] == 'erosion' and spec['distance'] > 0:
        seeds, boundary_distance = _erosion_boundary(candidates, spacing)
        distance = _distances(candidates, seeds, spacing, spec['distance'], boundary_distance)
        requested_removed = before & (distance <= strength + _TOL) & (strength > 0)
        proposed_removed = before & (distance <= effective_strength + _TOL) & (effective_strength > 0)
        recipient_seeds = eligible & ndimage.binary_dilation(proposed_removed, structure=_STRUCTURE)
        domain = proposed_removed | recipient_seeds
        initial = None if resistance is None else resistance * spec['search']
        _, owners = _owned_paths(domain, recipient_seeds, spacing, spec['search'], labels, initial=initial)
        removed = proposed_removed & (owners >= 0)
        unresolved = proposed_removed & ~removed
        if np.any(unresolved) and spec['unresolved'] == 'error':
            raise ValueError(f'{int(unresolved.sum())} proposed released voxels have no eligible recipient within reassignment.max_distance_mm')
        edited[removed] = labels.ravel()[owners[removed]]
        proxy[removed] = attenuation.ravel()[owners[removed]]
    changed = added | removed
    after = (before | added) & ~removed
    if np.any(blocked):
        warnings.append('Some dilation proposals were blocked by tissue eligibility, protected barriers, or the accepted-path distance budget.')
    if np.any(unresolved):
        warnings.append('Unresolved proposed releases retain their original target label and scalar; requested erosion was only partly achieved.')
    if spec['operation'] != 'none' and not np.any(changed):
        warnings.append('No label changed; inspect profile/grid spacing, allowed tissues, and blocked/unresolved counts.')
    transitions = []
    if np.any(changed):
        pairs, counts = np.unique(np.column_stack((labels[changed], edited[changed])), axis=0, return_counts=True)
        transitions = [{'original_id': int(a), 'new_id': int(b), 'count': int(n)} for (a, b), n in zip(pairs, counts)]
    counts = {name: int(mask.sum()) for name, mask in (
        ('before', before), ('after', after), ('proposed_added', proposed_added), ('proposed_removed', proposed_removed),
        ('added', added), ('removed', removed), ('blocked', blocked), ('unresolved', unresolved), ('changed', changed))}
    voxel_volume = float(np.prod(spacing))
    summary = {'engine': ENGINE, 'operation': spec['operation'], 'distance_mm': spec['distance'],
               'strength_semantics': 'Maximum weighted six-neighbor path length in mm; edge costs equal axis spacing. Axis-biased grid metric, not Euclidean radius, diameter ratio, or clinical stenosis.',
               'profile': spec['profile'], 'profile_axis': spec['axis'], 'shape_k': spec['shape_k'],
               'shape_window': spec['window'].tolist(), 'counts': counts,
               'volume_mm3': {name: counts[name] * voxel_volume for name in ('before', 'after', 'added', 'removed')},
               'transitions': transitions, 'warnings': warnings,
               'components_before': int(ndimage.label(before, structure=_STRUCTURE)[1]),
               'components_after': int(ndimage.label(after, structure=_STRUCTURE)[1]),
               'scalar_status': 'attenuation_proxy_per_pixel copies the winning original target/recipient source voxel value on accepted changes. This is a display proxy, not AI recovery or a physical CT reconstruction. Source atn remains unchanged.',
               'ownership_tie_break': 'Shortest accepted six-edge physical path, then lowest signed original ID, then earliest original source k,j,i coordinate.',
               'allowed_tissues': list(spec['allowed_tissues']), 'recipient_max_distance_mm': spec['search'],
               'unresolved_policy': spec['unresolved'], 'halo': halo}
    summary['reassignment_policy'] = reassignment_policy
    if directional is not None:
        summary['direction'] = directional['summary']
        if np.any(changed & ((directional['direction_weight'] <= 0) | ~directional['direction_reliable_mask'])):
            raise RuntimeError('Morphology changed a voxel outside reliable directional support')
    if resistance is not None:
        summary['counts']['requested_removed_before_stiffness'] = int(requested_removed.sum())
        summary['counts']['release_suppressed_by_target_stiffness'] = int((requested_removed & ~proposed_removed).sum())
        summary['stiffness_count_semantics'] = ('Dilation proposed_added uses the original requested distance; blocked includes resistance and other eligibility/path limits. Erosion proposed_removed is after target resistance; requested_removed_before_stiffness records the unscaled request. Unresolved counts failed recipient assignments after that release selection.')
        summary['effective_distance_mm_range_in_roi'] = ([float(effective_strength[roi].min()), float(effective_strength[roi].max())]
                                                        if roi.any() else [0., 0.])
        summary['ownership_tie_break'] = ('Recipient: minimum seed stiffness penalty plus physical six-edge path distance; '
                                         'then lowest signed original ID and earliest seed coordinate. Dilation: physical path with local donor budgets.')
    if np.any(changed & protected) or np.any((proxy != attenuation) & protected):
        raise RuntimeError('Morphology changed a protected input label/scalar')
    result = {'edited_labels': edited, 'edited_tissue_labels': coarse_labels(edited, resolved['catalog']),
            'target_mask_before': before.copy(), 'target_mask_after': after,
            'proposed_added_mask': proposed_added, 'proposed_removed_mask': proposed_removed,
            'added_mask': added, 'removed_mask': removed, 'blocked_mask': blocked,
            'unresolved_mask': unresolved, 'changed_mask': changed,
            'attenuation_proxy_per_pixel': proxy, 'strength_mm': strength, 'summary': summary}
    if resistance is not None:
        result.update(stiffness_field=resistance, effective_strength_mm=effective_strength,
                      requested_removed_mask=requested_removed)
    if directional is not None:
        result.update({key: value for key, value in directional.items() if isinstance(value, np.ndarray)})
    return result
