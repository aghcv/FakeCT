"""Curvature-relative angular budgets inside an existing native tube ROI."""
import numpy as np


DEFAULT_CENTERLINE = {'smoothing_mm': 1.0, 'sample_step_mm': 1.0,
                      'min_curvature_per_mm': .002}
MAX_DIRECTION_VOXELS = 2_000_000


def directional_spec(resolved, config):
    edit = config.get('edit', {})
    direction = edit.get('direction', 'all')
    if direction not in ('all', 'inner', 'outer'):
        raise ValueError('edit.direction must be all, inner, or outer')
    if direction == 'all':
        if 'angular_width_deg' in edit:
            raise ValueError('angular_width_deg requires direction=inner or outer')
        return None
    width = float(edit.get('angular_width_deg', 180.))
    if not np.isfinite(width) or not 0 < width <= 180:
        raise ValueError('angular_width_deg must be finite, positive, and at most 180')
    nodes = resolved.get('range_parent_nodes_ijk', resolved['roi_nodes_ijk'])
    if resolved['roi_kind'] != 'tube' or len(nodes) < 4:
        raise ValueError('Curvature-relative editing requires a parent tube with at least four points')
    return {'direction': direction, 'angular_width_deg': width}


def direction_weight_field(shape_kji, resolved, config, roi_mask):
    """Compute a raised-cosine angular factor, leaving source geometry fixed.

    Original parent arc position selects the spline frame via its retained
    parent parameter. Reliable bracketing normals are interpolated, projected
    perpendicular to the interpolated tangent, and normalized. Intervals with
    opposing normals are unusable: true curvature direction is never sign-flipped
    to manufacture continuity through an inflection.
    """
    from fakect_centerline_frame import build_centerline_frame
    from fakect_morphology import tube_position_mm

    spec = directional_spec(resolved, config)
    if spec is None:
        raise ValueError('direction_weight_field requires an inner or outer edit')
    roi = np.asarray(roi_mask)
    if (roi.dtype != np.bool_ or roi.shape != tuple(shape_kji) or roi.ndim != 3
            or not roi.size or roi.size > MAX_DIRECTION_VOXELS):
        raise ValueError('Directional ROI must be a bounded native Boolean crop')
    spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
    parent = resolved.get('range_parent_nodes_ijk', resolved['roi_nodes_ijk'])
    settings = {**DEFAULT_CENTERLINE, **config.get('centerline', {})}
    frame = build_centerline_frame(parent, spacing, **settings)
    parent_length = np.linalg.norm(np.diff(np.asarray(parent) * spacing, axis=0), axis=1).sum()
    position = tube_position_mm(shape_kji, resolved['crop_low_ijk'], parent, spacing)[roi] / parent_length
    fractions = np.asarray(frame['parent_fraction'])
    upper = np.clip(np.searchsorted(fractions, position, side='right'), 1, len(fractions) - 1)
    lower = upper - 1
    blend = np.clip((position - fractions[lower]) / (fractions[upper] - fractions[lower]), 0, 1)

    def interpolate(field):
        values = np.asarray(frame[field])
        return values[lower] * (1 - blend[:, None]) + values[upper] * blend[:, None]

    centers = interpolate('xyz_mm')
    tangent = interpolate('tangent')
    tangent_norm = np.linalg.norm(tangent, axis=1)
    tangent /= np.maximum(tangent_norm[:, None], 1e-15)
    normal = interpolate('normal')
    normal -= np.sum(normal * tangent, axis=1)[:, None] * tangent
    normal_norm = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(normal_norm[:, None], 1e-15)
    orientation_agreement = np.sum(np.asarray(frame['normal'])[lower] * np.asarray(frame['normal'])[upper], axis=1)
    usable = (np.asarray(frame['reliable'])[lower] & np.asarray(frame['reliable'])[upper]
              & (tangent_norm > 1e-10) & (normal_norm > 1e-10) & (orientation_agreement > 0))
    # A voxel that maps exactly onto a sample uses that sample's reliability.
    # This does not permit interpolation through an invalid interval.
    exact_lower = blend <= 1e-12
    exact_upper = blend >= 1 - 1e-12
    usable[exact_lower] = np.asarray(frame['reliable'])[lower[exact_lower]]
    usable[exact_upper] = np.asarray(frame['reliable'])[upper[exact_upper]]
    coordinates = (np.argwhere(roi)[:, ::-1] + resolved['crop_low_ijk']) * spacing
    radial = coordinates - centers
    radial -= np.sum(radial * tangent, axis=1)[:, None] * tangent
    radial_length = np.linalg.norm(radial, axis=1)
    axis_undefined = radial_length <= 1e-9
    side = 1. if spec['direction'] == 'inner' else -1.
    cosine = np.clip(side * np.sum(radial * normal, axis=1) / np.maximum(radial_length, 1e-15), -1, 1)
    angle = np.arccos(cosine)
    half_width = np.deg2rad(spec['angular_width_deg'] / 2)
    supported = usable & ~axis_undefined & (angle < half_width)
    weights = np.zeros(len(position), dtype=np.float64)
    weights[supported] = .5 * (1 + np.cos(np.pi * angle[supported] / half_width))
    weight_field = np.zeros(roi.shape, dtype=np.float64)
    reliable_field = np.zeros(roi.shape, dtype=bool)
    cosine_field = np.zeros(roi.shape, dtype=np.float32)
    weight_field[roi], reliable_field[roi], cosine_field[roi] = weights, usable & ~axis_undefined, cosine
    summary = {**spec, 'settings': settings, 'frame_metadata': frame['metadata'],
               'roi_voxels': int(roi.sum()), 'unreliable_roi_voxels': int((~usable).sum()),
               'axis_undefined_roi_voxels': int((usable & axis_undefined).sum()),
               'angular_supported_roi_voxels': int(supported.sum()),
               'weight_range_in_roi': [float(weights.min()), float(weights.max())] if len(weights) else [0., 0.],
               'weight_semantics': '0.5*(1+cos(pi*angle/half_width)) inside the angular sector, zero elsewhere; multiplied by longitudinal edit distance before tissue resistance',
               'direction_semantics': 'Inner is principal curvature normal N; outer is -N; binormal B=T cross N',
               'mapping': 'Original parent closest-arc fraction maps to bracketing arc-resampled spline frames; reliable vectors interpolated and re-orthogonalized',
               'undefined_policy': 'Skip low-curvature, endpoint, opposing-normal interpolation intervals and points on the transverse axis',
               'roi_geometry_changed': False}
    return {'direction_weight': weight_field, 'direction_reliable_mask': reliable_field,
            'direction_cosine': cosine_field, 'summary': summary}


def recipe_centerline_frames(config, resolved):
    """Frames for the original parent paths, independent of selected intervals."""
    from fakect_centerline_frame import build_centerline_frame
    if 'centerline' not in config and not any(edit.get('direction', 'all') != 'all'
                                            for edit in config.get('edits', {}).values()):
        return {}
    settings = {**DEFAULT_CENTERLINE, **config.get('centerline', {})}
    parents = {}
    for edit in config['edits'].values():
        name = edit['roi']
        parent = config['roi'] if name == 'main' else config['rois'][name]
        if parent['shape'] == 'tube':
            parents[name] = parent
    if not parents:
        raise ValueError('[centerline] requires a referenced parent tube')
    return {name: build_centerline_frame(parent['center_ijk'], resolved['spacing_ijk_mm'], **settings)
            for name, parent in parents.items()}
