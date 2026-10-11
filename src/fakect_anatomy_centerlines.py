"""Conservative two-pole/ring recovery from bounded XCAT triangle soups.

Following DPI's structural proof, accept a closed triangular sphere only when
two unique high-valence poles bound an annulus and every non-pole belongs to a
simple circumferential ring. No triangle ordering or anatomical direction is
assumed. Failed proofs return unavailable evidence, never a guessed centerline.
"""
from __future__ import annotations

from collections import Counter, deque
import numpy as np

MAX_TRIANGLES = 20000
MAX_WELD_PAIRS = 200000
WELD_TOLERANCE_MM = 0.001


def _topology(faces, count):
    """Build undirected mesh topology from indexed triangles.

    Parameters
    ----------
    faces : array_like of shape (n, 3)
        Triangle vertex indices.
    count : int
        Number of vertices used to allocate adjacency sets.

    Returns
    -------
    tuple of collections.Counter, list of set and set
        Edge multiplicities, vertex adjacency sets and the set of used vertex IDs.
    """
    edges = Counter()
    adjacency = [set() for _ in range(count)]
    used = set()
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        used.update((a, b, c))
        for x, y in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((x, y)))] += 1
            adjacency[x].add(y)
            adjacency[y].add(x)
    return edges, adjacency, used


def _connected(vertices, adjacency):
    """Check connectivity of an induced vertex subgraph.

    Parameters
    ----------
    vertices : set of int
        Vertex IDs whose induced subgraph is tested.
    adjacency : list of set
        Undirected neighbors indexed by vertex ID.

    Returns
    -------
    bool
        True when every requested vertex is reached; an empty vertex set returns False.
    """
    if not vertices:
        return False
    seen, pending = set(), [min(vertices)]
    while pending:
        value = pending.pop()
        if value not in seen:
            seen.add(value)
            pending.extend((adjacency[value] & vertices) - seen)
    return seen == vertices


def _cycle(vertices, adjacency):
    """Check whether an induced vertex subgraph is a single simple cycle.

    Parameters
    ----------
    vertices : set of int
        Candidate circumferential ring vertices.
    adjacency : list of set
        Mesh vertex adjacency.

    Returns
    -------
    bool
        True for a connected set of at least three vertices with degree two inside the
        set.
    """
    return (len(vertices) >= 3 and all(len(adjacency[i] & vertices) == 2 for i in vertices)
            and _connected(vertices, adjacency))


def extract_centerline(triangles):
    """Recover bounded two-pole/ring centerline evidence from a triangle soup.

    Parameters
    ----------
    triangles : array_like of shape (n, 3, 3)
        Finite triangle vertex coordinates in raw x,y,z millimetres; at most
        ``MAX_TRIANGLES`` are processed.

    Returns
    -------
    dict
        Serializable proof and centerline/radius coordinates in raw millimetres when
        verified; otherwise unavailable status and reason.

    Raises
    ------
    ValueError
        Input is not finite triangle coordinates with the required shape.

    Notes
    -----
    Topology failures return unavailable evidence rather than a guessed curve. Structural order does not imply proximal/distal anatomy.

    Return serializable structural evidence; peak input is MAX_TRIANGLES.
    """
    result = {'status': 'unavailable', 'method': 'closed_sphere_two_poles_verified_ring_cycles',
              'weld_tolerance_mm': WELD_TOLERANCE_MM,
              'direction_semantics': 'Deterministic structural order; not proximal/distal anatomy.'}

    def fail(reason):
        result['reason'] = reason
        return result

    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.all(np.isfinite(triangles)):
        raise ValueError('Centerline triangles must be finite (n,3,3) coordinates')
    if len(triangles) == 0 or len(triangles) > MAX_TRIANGLES:
        return fail('surface_exceeds_bounded_centerline_budget_or_is_empty')
    from scipy.spatial import cKDTree
    points, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    tree = cKDTree(points)
    pair_count = (tree.count_neighbors(tree, WELD_TOLERANCE_MM) - len(points)) // 2
    if pair_count > MAX_WELD_PAIRS:
        return fail('weld_pair_budget_exceeded')
    parent = np.arange(len(points))

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in tree.query_pairs(WELD_TOLERANCE_MM, output_type='ndarray'):
        ra, rb = root(int(a)), root(int(b))
        parent[max(ra, rb)] = min(ra, rb)
    representatives = np.array([root(i) for i in range(len(points))])
    if np.any(np.linalg.norm(points - points[representatives], axis=1) > WELD_TOLERANCE_MM * (1 + 1e-12)):
        return fail('transitive_weld_exceeds_tolerance')
    unique, remap = np.unique(representatives, return_inverse=True)
    points = points[unique]
    faces = remap[inverse].reshape(-1, 3)
    nondegenerate = np.logical_and.reduce((faces[:, 0] != faces[:, 1], faces[:, 0] != faces[:, 2], faces[:, 1] != faces[:, 2]))
    result['collapsed_degenerate_triangles'] = int(np.count_nonzero(~nondegenerate))
    faces = faces[nondegenerate]
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        return fail('duplicate_triangles_after_welding')
    edges, adjacency, used = _topology(faces, len(points))
    euler = len(used) - len(edges) + len(faces)
    result['proof'] = {'welded_vertex_count': len(points), 'retained_triangle_count': len(faces),
                       'euler_characteristic': euler}
    if len(used) != len(points) or not _connected(used, adjacency) or euler != 2 or any(n != 2 for n in edges.values()):
        return fail('not_a_single_closed_manifold_sphere')
    degrees = [len(a) for a in adjacency]
    histogram = Counter(degrees)
    modal = min(histogram, key=lambda degree: (-histogram[degree], degree))
    maximum = max(degrees)
    poles = [i for i, degree in enumerate(degrees) if degree == maximum]
    if len(poles) != 2 or maximum <= modal:
        return fail('no_unique_high_valence_pole_pair')
    annular_faces = faces[~np.isin(faces, poles).any(axis=1)]
    ann_edges, ann_adj, ann_used = _topology(annular_faces, len(points))
    nonpoles = used - set(poles)
    ann_euler = len(ann_used) - len(ann_edges) + len(annular_faces)
    first, last = adjacency[poles[0]], adjacency[poles[1]]
    boundary_edges = {edge for edge, n in ann_edges.items() if n == 1}
    expected_boundary = {edge for ring in (first, last) for edge in edges if edge[0] in ring and edge[1] in ring}
    if (ann_used != nonpoles or ann_euler != 0 or not _connected(ann_used, ann_adj) or
            any(n not in (1, 2) for n in ann_edges.values()) or
            not _cycle(first, adjacency) or not _cycle(last, adjacency) or
            first & last or boundary_edges != expected_boundary):
        return fail('poles_do_not_bound_a_verified_annulus')
    # Every cross-ring edge spans exactly one layer in a structured tube. BFS
    # layers from a pole recover complete rings without trusting RAW row order.
    levels = np.full(len(points), -1, dtype=int)
    levels[poles[0]] = 0
    queue = deque([poles[0]])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if levels[neighbor] == -1:
                levels[neighbor] = levels[current] + 1
                queue.append(neighbor)
    rings = [{i for i in nonpoles if levels[i] == level} for level in range(1, int(levels[poles[1]]))]
    if (len(rings) < 2 or rings[0] != first or rings[-1] != last or
            set().union(*rings) != nonpoles or not all(_cycle(ring, adjacency) for ring in rings)):
        return fail('nonpole_vertices_do_not_form_exhaustive_ordered_rings')
    face_levels = levels[annular_faces]
    if np.any(face_levels.max(axis=1) - face_levels.min(axis=1) != 1):
        return fail('triangles_do_not_join_adjacent_rings')
    ring_points = [points[sorted(ring)] for ring in rings]
    centroids = np.asarray([ring.mean(axis=0) for ring in ring_points])
    radii = [np.linalg.norm(ring - center, axis=1) for ring, center in zip(ring_points, centroids)]
    if any(float(radius.max()) <= WELD_TOLERANCE_MM for radius in radii):
        return fail('degenerate_geometric_ring')
    centerline = np.vstack((points[poles[0]], centroids, points[poles[1]]))
    maximum_radii = [float(radius.max()) for radius in radii]
    result.update(status='verified_two_pole_rings', centerline_raw_mm=centerline.tolist(),
                  ring_mean_radii_mm=[float(radius.mean()) for radius in radii],
                  ring_max_radii_mm=maximum_radii,
                  centerline_envelope_radii_mm=[WELD_TOLERANCE_MM] +
                    [radius + WELD_TOLERANCE_MM for radius in maximum_radii] + [WELD_TOLERANCE_MM],
                  ring_vertex_counts=[len(ring) for ring in rings])
    result['proof'].update(annulus_euler_characteristic=ann_euler, boundary_cycles=2,
                           pole_valence=maximum, modal_valence=modal, ring_count=len(rings),
                           all_nonpole_vertices_assigned=True, all_ring_links_simple_cycles=True,
                           all_triangles_connect_adjacent_rings=True)
    return result


def centerline_envelope_overlap(centerline, coordinate_contract):
    """Test conservative tube-envelope overlap using segment bounding boxes.

    Parameters
    ----------
    centerline : dict
        Verified ring centerline with raw-mm nodes and envelope radii.
    coordinate_contract : dict
        Validated raw-grid contract containing ROI-envelope bounds in millimetres.

    Returns
    -------
    bool or None
        Whether expanded segment boxes overlap the ROI envelope; None when the
        centerline or coordinate contract is unavailable.

    Notes
    -----
    Conservative union-of-segment bounding boxes, expanded by ring radius.

    Endpoint balls enclose ring vertices (plus welding error). Their segment
    boxes enclose every intervening triangle. This is still only broad phase.
    """
    if centerline['status'] != 'verified_two_pole_rings' or coordinate_contract['status'] != 'bounded_contract_validated':
        return None
    nodes = np.asarray(centerline['centerline_raw_mm'])
    radii = np.asarray(centerline['centerline_envelope_radii_mm'])[:, None]
    lower, upper = nodes - radii, nodes + radii
    envelope = coordinate_contract['roi_envelope_raw_mm']
    low = np.minimum(lower[:-1], lower[1:])
    high = np.maximum(upper[:-1], upper[1:])
    return bool(np.any(np.all(high >= envelope['min_xyz'], axis=1) &
                       np.all(low <= envelope['max_xyz'], axis=1)))
