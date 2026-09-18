"""Resolve an ordered tube subpath by node number or physical path percentage.

The result retains the ordinary tube geometry: linearly interpolated centers
and radii with round end caps. This helper does not construct a voxel mask or
introduce flat cuts across the parent tube.
"""
import numpy as np


def _number_pair(value, name):
    try:
        pair = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain two finite numbers') from exc
    if pair.shape != (2,) or not np.all(np.isfinite(pair)):
        raise ValueError(f'{name} must contain two finite numbers')
    return pair


def _endpoint(nodes, radii, distances, distance):
    """Interpolate one endpoint, snapping only floating-point node coincidences."""
    insertion = int(np.searchsorted(distances, distance))
    nearest = min((max(0, insertion - 1), min(len(distances) - 1, insertion)),
                  key=lambda index: abs(float(distances[index]) - distance))
    # Percent-to-distance arithmetic can move an exact node endpoint by a few
    # representable float values. Snap those coincidences so the same control
    # does not appear twice as an interpolated endpoint plus an interior node.
    tolerance = 8 * abs(float(np.spacing(max(abs(distance), abs(float(distances[nearest]))))))
    if abs(float(distances[nearest]) - distance) <= tolerance:
        return nodes[nearest], float(radii[nearest]), float(distances[nearest]), nearest + 1
    segment = min(len(nodes) - 2, max(0, insertion - 1))
    fraction = (distance - distances[segment]) / (distances[segment + 1] - distances[segment])
    center = nodes[segment] + fraction * (nodes[segment + 1] - nodes[segment])
    radius = float(radii[segment] + fraction * (radii[segment + 1] - radii[segment]))
    return center, radius, distance, None


def resolve_tube_range(nodes_ijk, radii_mm, spacing_ijk_mm, *, point_range=None, path_percent=None):
    """Return a tube subpath without sorting or changing caller-owned inputs.

    ``point_range=(2, 8)`` retains original controls 2 through 8, inclusive,
    using one-based numbering. ``path_percent=(30, 75)`` takes 30% through 75%
    of the ordered parent's physical arc length, accounting for anisotropic
    spacing. Its endpoint centers and radii are interpolated on their parent
    segments; original controls strictly between the endpoints are retained.
    Supply at most one selector. Omitting both returns the complete path.

    Ranges must have a strictly increasing start and end. Reversing the parent
    path reverses the direction in which percentages and point numbers run.
    """
    if point_range is not None and path_percent is not None:
        raise ValueError('point_range and path_percent are mutually exclusive')
    try:
        nodes = np.asarray(nodes_ijk, dtype=float)
        radii = np.asarray(radii_mm, dtype=float)
        spacing = np.asarray(spacing_ijk_mm, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Tube nodes, radii, and spacing must contain finite numeric values') from exc
    if (nodes.ndim != 2 or nodes.shape[1:] != (3,) or len(nodes) < 2
            or not np.all(np.isfinite(nodes)) or np.any(nodes < 0)):
        raise ValueError('Tube nodes must contain at least two finite nonnegative i,j,k triples')
    if np.any(np.all(np.diff(nodes, axis=0) == 0, axis=1)):
        raise ValueError('Adjacent tube nodes must be distinct; preserve their path order')
    if radii.shape != (len(nodes),) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError('Tube radii must contain one positive finite value per node')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Spacing must contain three positive finite i,j,k values')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        physical_segments = np.diff(nodes, axis=0) * spacing
        # Chained hypot avoids unnecessary overflow when squaring large values.
        lengths = np.hypot(np.hypot(physical_segments[:, 0], physical_segments[:, 1]),
                           physical_segments[:, 2])
        distances = np.r_[0., np.cumsum(lengths)]
    if (not np.all(np.isfinite(distances)) or np.any(lengths <= 0)
            or np.any(np.diff(distances) <= 0)):
        raise ValueError('Tube segment lengths must remain finite and distinct at physical coordinate precision')
    total_length = float(distances[-1])
    selector, selector_values = 'full', None
    if point_range is not None:
        pair = _number_pair(point_range, 'point_range')
        raw = np.asarray(point_range)
        if raw.dtype.kind not in 'iu' or np.any(pair != np.trunc(pair)):
            raise ValueError('point_range must contain two integer, one-based node numbers')
        first, last = map(int, pair)
        if not 1 <= first < last <= len(nodes):
            raise ValueError('point_range must satisfy 1 <= start < end <= the parent node count')
        selector, selector_values = 'point_range', [first, last]
        selected_nodes = nodes[first - 1:last]
        selected_radii = radii[first - 1:last]
        start, end = float(distances[first - 1]), float(distances[last - 1])
        retained = list(range(first, last + 1))
    elif path_percent is not None:
        first, last = _number_pair(path_percent, 'path_percent')
        if not 0 <= first < last <= 100:
            raise ValueError('path_percent must satisfy 0 <= start < end <= 100')
        selector, selector_values = 'path_percent', [float(first), float(last)]
        first_node, first_radius, start, first_number = _endpoint(
            nodes, radii, distances, total_length * (float(first) / 100))
        last_node, last_radius, end, last_number = _endpoint(
            nodes, radii, distances, total_length * (float(last) / 100))
        if end <= start:
            raise ValueError('path_percent endpoints are indistinguishable at physical coordinate precision')
        interior = np.flatnonzero((distances > start) & (distances < end))
        selected_nodes = np.vstack((first_node, nodes[interior], last_node))
        selected_radii = np.r_[first_radius, radii[interior], last_radius]
        if np.any(np.all(np.diff(selected_nodes, axis=0) == 0, axis=1)):
            raise ValueError('path_percent endpoints are indistinguishable at native coordinate precision')
        retained = ([first_number] if first_number is not None else []) + (interior + 1).tolist()
        if last_number is not None:
            retained.append(last_number)
    else:
        selected_nodes, selected_radii = nodes, radii
        start, end = 0., total_length
        retained = list(range(1, len(nodes) + 1))
    return {
        'nodes_ijk': tuple(tuple(map(float, node)) for node in selected_nodes),
        'radii_mm': tuple(map(float, selected_radii)),
        'metadata': {
            'selector': selector, 'selector_values': selector_values,
            'point_numbering': 'one-based inclusive', 'parent_node_count': len(nodes),
            'selected_node_count': len(selected_nodes), 'parent_total_length_mm': total_length,
            'start_distance_mm': start, 'end_distance_mm': end,
            'start_fraction': start / total_length, 'end_fraction': end / total_length,
            'selected_length_mm': end - start, 'original_node_numbers_retained': retained,
            'tube_end_caps': 'round',
            'semantics': 'Ordered tube subpath; endpoint centers and radii interpolate on parent segments',
        },
    }
