import numpy as np
import trimesh
from typing import Tuple
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_fill_holes
import igl

EPS = 1e-9

def world_to_index_axis(x, o, s, N):
    # subtract a tiny epsilon so values exactly on a boundary map to the lower cell
    idx = np.floor((x - o)/s - EPS).astype(int)
    return np.clip(idx, 0, N-1)

def sweep_along_axis(on_mask: np.ndarray, axis: str) -> np.ndarray:
    """Return inside mask computed by parity flips along one axis."""
    N = on_mask.shape[0]
    inside = np.zeros_like(on_mask, dtype=bool)

    if axis == 'z':
        a = range(on_mask.shape[0])       # sweep along k (Z)
        slicer = lambda k: (k, slice(None), slice(None))
    elif axis == 'y':
        a = range(on_mask.shape[1])       # sweep along j (Y)
        slicer = lambda j: (slice(None), j, slice(None))
    elif axis == 'x':
        a = range(on_mask.shape[2])       # sweep along i (X)
        slicer = lambda i: (slice(None), slice(None), i)
    else:
        raise ValueError("axis must be one of 'x','y','z'")

    # iterate along chosen axis
    if axis == 'z':
        for j in range(on_mask.shape[1]):
            for i in range(on_mask.shape[2]):
                col = on_mask[:, j, i]
                state = False
                prev_on = False
                for k in range(len(col)):
                    curr_on = col[k]
                    if prev_on and not curr_on:
                        state = not state
                    if not curr_on and state:
                        inside[k, j, i] = True
                    prev_on = curr_on
    elif axis == 'y':
        for k in range(on_mask.shape[0]):
            for i in range(on_mask.shape[2]):
                col = on_mask[k, :, i]
                state = False
                prev_on = False
                for j in range(len(col)):
                    curr_on = col[j]
                    if prev_on and not curr_on:
                        state = not state
                    if not curr_on and state:
                        inside[k, j, i] = True
                    prev_on = curr_on
    elif axis == 'x':
        for k in range(on_mask.shape[0]):
            for j in range(on_mask.shape[1]):
                col = on_mask[k, j, :]
                state = False
                prev_on = False
                for i in range(len(col)):
                    curr_on = col[i]
                    if prev_on and not curr_on:
                        state = not state
                    if not curr_on and state:
                        inside[k, j, i] = True
                    prev_on = curr_on

    return inside

def densify_on_mask(mesh, grid, on_mask, radius_vox=1):
    N = grid["shape"][0]; s = grid["spacing"][0]; ox, oy, oz = grid["origin"]
    verts = mesh.vertices
    edges = mesh.edges_unique

    # ROI around current on voxels
    if radius_vox > 0:
        from scipy.ndimage import binary_dilation
        roi_mask = binary_dilation(on_mask, iterations=radius_vox)
    else:
        roi_mask = np.ones_like(on_mask, dtype=bool)

    new_on = on_mask.copy()

    for e in edges:
        v1, v2 = mesh.vertices[e]
        # convert to index-space floats (i,j,k)
        p1 = ((v1[0]-ox)/s, (v1[1]-oy)/s, (v1[2]-oz)/s)
        p2 = ((v2[0]-ox)/s, (v2[1]-oy)/s, (v2[2]-oz)/s)

        voxels = rasterize_segment_voxels(p1, p2, N)
        if not voxels: continue
        ks, js, is_ = zip(*voxels)
        ks = np.asarray(ks); js = np.asarray(js); is_ = np.asarray(is_)
        hits = roi_mask[ks, js, is_]
        if np.any(hits):
            new_on[ks[hits], js[hits], is_[hits]] = True

    return new_on
'''
def densify_on_mask(mesh, grid, on_mask, radius_vox=1):
    """
    Densify the 'on' mask by sampling mesh edges in the neighborhood
    of existing on-voxels to capture large faces and missing edge voxels.
    """
    N = grid["shape"][0]
    s = grid["spacing"][0]
    ox, oy, oz = grid["origin"]

    verts = mesh.vertices
    edges = mesh.edges_unique  # (E, 2) indices

    # convert vertex coordinates to voxel indices
    vi = np.floor((verts[:, 0] - ox) / s).astype(int)
    vj = np.floor((verts[:, 1] - oy) / s).astype(int)
    vk = np.floor((verts[:, 2] - oz) / s).astype(int)
    valid = (
        (vi >= 0) & (vi < N) &
        (vj >= 0) & (vj < N) &
        (vk >= 0) & (vk < N)
    )
    vi, vj, vk = vi[valid], vj[valid], vk[valid]

    new_on = on_mask.copy()

    # region of interest (small window around current on voxels)
    if radius_vox > 0:
        from scipy.ndimage import binary_dilation
        roi_mask = binary_dilation(on_mask, iterations=radius_vox)
    else:
        roi_mask = np.ones_like(on_mask, dtype=bool)

    # traverse edges
    for e in edges:
        v1, v2 = mesh.vertices[e]
        p1 = np.array([(v1[0]-ox)/s, (v1[1]-oy)/s, (v1[2]-oz)/s])
        p2 = np.array([(v2[0]-ox)/s, (v2[1]-oy)/s, (v2[2]-oz)/s])
        diff = p2 - p1
        steps = int(np.ceil(np.linalg.norm(diff)))
        if steps < 2:
            continue
        t = np.linspace(0, 1, steps)
        pts = p1[None,:] + t[:,None] * diff[None,:]
        i = np.clip(np.round(pts[:,0]).astype(int), 0, N-1)
        j = np.clip(np.round(pts[:,1]).astype(int), 0, N-1)
        k = np.clip(np.round(pts[:,2]).astype(int), 0, N-1)
        # only add those within the ROI to limit cost
        mask_hits = roi_mask[k, j, i]
        new_on[k[mask_hits], j[mask_hits], i[mask_hits]] = True

    return new_on
'''

def close_surface_holes(on_mask: np.ndarray, m: int = 3) -> np.ndarray:
    """
    Morphologically close small holes in the 'on' surface mask.

    Parameters
    ----------
    on_mask : np.ndarray
        Boolean 3D array (Z,Y,X) marking surface voxels.
    m : int
        Neighborhood cube size. 3 → immediate 26-neighbor closure,
        5 → thicker connection radius.

    Returns
    -------
    np.ndarray : boolean closed surface mask
    """
    if m < 3 or m % 2 == 0:
        raise ValueError("m must be odd and ≥3 (e.g., 3, 5, 7).")

    # Create a fully connected 3D neighborhood
    structure = generate_binary_structure(3, 3)  # 26-connectivity
    # Iterations = radius in voxels
    radius = (m - 1) // 2

    # Dilation followed by erosion (closing)
    closed = binary_dilation(on_mask, structure=structure, iterations=radius)
    closed = binary_erosion(closed, structure=structure, iterations=radius)

    return closed

def rasterize_segment_voxels(p1_xyz, p2_xyz, N):
    """
    3D Amanatides–Woo traversal from p1 to p2 in INDEX space (i,j,k floats).
    Returns a list of (k,j,i) voxels visited by the segment, clipped to [0,N-1].
    p1_xyz, p2_xyz are 3-vectors in (i,j,k) index space.
    """
    p1 = np.array(p1_xyz, dtype=float)
    p2 = np.array(p2_xyz, dtype=float)
    d  = p2 - p1
    # start voxel (use floor with epsilon)
    EPS = 1e-9
    i = int(np.clip(np.floor(p1[0] - EPS), 0, N-1))
    j = int(np.clip(np.floor(p1[1] - EPS), 0, N-1))
    k = int(np.clip(np.floor(p1[2] - EPS), 0, N-1))

    # step per axis
    step_i = 1 if d[0] > 0 else (-1 if d[0] < 0 else 0)
    step_j = 1 if d[1] > 0 else (-1 if d[1] < 0 else 0)
    step_k = 1 if d[2] > 0 else (-1 if d[2] < 0 else 0)

    # compute tMax/tDelta per axis
    def axis_init(p, dp, c):
        if dp > 0:
            next_boundary = (np.floor(p) + 1.0)
            tMax  = (next_boundary - p) / dp
            tDelta = 1.0 / dp
        elif dp < 0:
            next_boundary = np.floor(p)
            tMax  = (p - next_boundary) / (-dp)
            tDelta = 1.0 / (-dp)
        else:
            tMax  = np.inf
            tDelta = np.inf
        return tMax, tDelta

    tMax_i, tDelta_i = axis_init(p1[0], d[0], i)
    tMax_j, tDelta_j = axis_init(p1[1], d[1], j)
    tMax_k, tDelta_k = axis_init(p1[2], d[2], k)

    visited = []
    # clip end param
    t_end = 1.0 + 1e-12

    while 0 <= i < N and 0 <= j < N and 0 <= k < N:
        visited.append((k, j, i))
        # choose smallest tMax
        if tMax_i <= tMax_j and tMax_i <= tMax_k:
            if tMax_i > t_end: break
            i += step_i
            tMax_i += tDelta_i
        elif tMax_j <= tMax_i and tMax_j <= tMax_k:
            if tMax_j > t_end: break
            j += step_j
            tMax_j += tDelta_j
        else:
            if tMax_k > t_end: break
            k += step_k
            tMax_k += tDelta_k

    return visited

def classify_by_winding(mesh: trimesh.Trimesh, grid: dict, band: float = 0.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use libigl’s fast winding number to classify inside / on / out.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Triangulated surface (should be closed / watertight).
    grid : dict
        As from make_cube_grid_from_mesh, with spacing, origin, shape.
    band : float
        Multiple of spacing to define “on” band: |phi| ≤ band * s.

    Returns
    -------
    inside_u8, on_u8, out_u8 : np.ndarray of dtype uint8, shape (N,N,N)
    """
    N = grid["shape"][0]
    s = grid["spacing"][0]
    ox, oy, oz = grid["origin"]

    # 1. Build query points = centers of all voxels
    # Note: we assume each voxel center is at (ox + (i + 0.5)*s, oy + (j + 0.5)*s, oz + (k + 0.5)*s)
    # But marching-cubes logic might expect different alignments; you may adjust by 0.0 or 0.5 accordingly.

    # Generate coordinates for centers
    coords = np.indices((N, N, N), dtype=float)  # shape (3, N, N, N), indices in [0…N-1]
    # coords[0] is k-index, coords[1] is j-index, coords[2] is i-index

    # Convert to world coordinates
    # world_x = ox + (i + 0.5) * s, etc.
    xs = ox + (coords[2] + 0.5) * s
    ys = oy + (coords[1] + 0.5) * s
    zs = oz + (coords[0] + 0.5) * s

    Q = np.stack((xs.ravel(), ys.ravel(), zs.ravel()), axis=1)  # shape (N³, 3)

    # 2. Mesh vertices and faces
    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    # 3. Compute winding numbers
    WN = igl.fast_winding_number_for_meshes(V, F, Q)  # type: ignore
    WN = np.array(WN).reshape((N, N, N))  # shape in (k, j, i) order

    # 4. Inside mask: winding > 0.5 indicates inside
    inside = (WN > 0.5)

    # 5. “On” mask: narrow band around surface
    # But we don’t directly have signed distance phi; we can approximate band by 
    # checking neighbors: define on = voxels that are inside but have a neighbor outside (or vice versa).
    # Another simpler approximate: on = (|WN - 0.5| <= band * ???)
    # But better: morphological boundary extraction

    # Extract boundary voxels of inside (6-connected)
    from scipy.ndimage import binary_dilation
    dil = binary_dilation(inside, structure=np.ones((3,3,3), dtype=bool))
    boundary = dil ^ inside  # voxels adjacent to inside

    on = boundary & inside  # or boundary & ~inside, depending on convention

    # 6. Out mask
    out = ~ (inside | on)

    return inside.astype(np.uint8), on.astype(np.uint8), out.astype(np.uint8)

def classify_by_parity_multi(mesh: trimesh.Trimesh, grid: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-axis parity-based in/on/out classification.
    Combines sweeps along X, Y, and Z for robust inside detection.
    """
    N = grid["shape"][0]
    s = grid["spacing"][0]
    ox, oy, oz = grid["origin"]

    # --- Map vertices to voxel indices ---
    verts = mesh.vertices
    i_idx = world_to_index_axis(verts[:, 0], ox, s, N)  # X -> i
    j_idx = world_to_index_axis(verts[:, 1], oy, s, N)  # Y -> j
    k_idx = world_to_index_axis(verts[:, 2], oz, s, N)  # Z -> k

    on_mask = np.zeros((N, N, N), dtype=bool)
    on_mask[k_idx, j_idx, i_idx] = True
    on_mask = densify_on_mask(mesh, grid, on_mask, radius_vox=1)
    on_mask = close_surface_holes(on_mask, m=3)

    # --- Perform sweeps along all three axes ---
    inside_z = sweep_along_axis(on_mask, axis='z')
    inside_y = sweep_along_axis(on_mask, axis='y')
    inside_x = sweep_along_axis(on_mask, axis='x')

    # --- Intersection of inside volumes ---
    inside_final = inside_x & inside_y & inside_z
    # Combine inside + on
    union_mask = inside_final | on_mask

    # Fill any enclosed cavities
    filled_union = binary_fill_holes(union_mask)

    # Subtract on to recover pure interior
    inside_filled = filled_union & (~on_mask)
    out_final = ~(inside_filled | on_mask)

    return inside_final.astype(np.uint8), on_mask.astype(np.uint8), out_final.astype(np.uint8)
