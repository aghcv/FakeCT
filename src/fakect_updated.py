#!/usr/bin/env python3
# ---------------------------------------------------------------------
# fakect.py
# Produces in/on/out masks (winding via python-igl when available) and optionally launches a Dash viewer.
#
# Usage examples:
#   python src/fakect.py --in data/cube.stl --n 8 --out outputs/cube_masks.npz
#   python src/fakect.py --in data/carotid.stl --n 9 --margin 0.10 --out outputs/carotid_masks.npz
#
# Requirements & installation
#  Recommended (conda - easiest, python-igl optional):
#    conda create -n fakect python=3.10 -y
#    conda activate fakect
#    conda install -c conda-forge trimesh scipy scikit-image plotly dash -y
#    # Optional, if available on your platform:
#    conda install -c conda-forge python-igl -y
#
#  Pip-only (virtualenv): python-igl often requires conda; use pip for the pure-Python deps
#    python -m venv .venv
#    source .venv/bin/activate
#    pip install --upgrade pip
#    pip install trimesh scipy scikit-image plotly dash
#    # If your platform supports it, you can try:
#    pip install igl
#    # But python-igl is best installed from conda-forge on many systems.
#
#  Quick note: to run without opening the Dash viewer (faster / headless), pass --no-show.
# ---------------------------------------------------------------------

import sys
import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple
from scipy.ndimage import binary_dilation
from skimage import measure
import plotly.graph_objects as go
import dash
from dash import dcc, html, Output, Input, callback_context

# libigl for fast winding number
try:
    import igl  # pip/conda: python-igl
except Exception as e:
    igl = None

# ---------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------
def info(msg): print(f"\033[94m[INFO]\033[0m {msg}")
def ok(msg):   print(f"\033[92m[DONE]\033[0m {msg}")
def warn(msg): print(f"\033[93m[WARN]\033[0m {msg}")
def err(msg):  print(f"\033[91m[ERROR]\033[0m {msg}")

# ---------------------------------------------------------------------
# Grid + (optional) voxelization
# ---------------------------------------------------------------------
def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def _grid_extents_display(grid):
    """Return (extent_x, extent_y, extent_z) in mm from grid shape × spacing."""
    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    return (Nx * sx, Ny * sy, Nz * sz)

def _aspectratio_from_extents(extents_xyz):
    """Normalize extents so that the largest side = 1 (realistic proportions)."""
    mx = max(extents_xyz)
    if mx <= 0:
        return dict(x=1, y=1, z=1)
    x, y, z = (e / mx for e in extents_xyz)
    return dict(x=x, y=y, z=z)

def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy of the mesh centered at the origin."""
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    return m

def make_cube_grid_from_mesh(
    mesh: trimesh.Trimesh,
    spacing: float | None = None,
    n: int | None = None,
    margin_frac: float = 0.10,
    min_margin_voxels: int = 3
) -> dict:
    """
    Build a cubic (or near-cubic) grid safely larger than the mesh AABB.
    Guarantees at least `min_margin_voxels` empty voxels between geometry
    and grid boundaries even when spacing is inferred from n.

    This preserves the physical aspect ratios for visualization.
    """
    bounds = mesh.bounds
    Lx, Ly, Lz = (bounds[1] - bounds[0]).astype(float)
    Lmax = float(max(Lx, Ly, Lz))

    # --- explicit voxel pitch (compute N as next pow2 of required voxels) ---
    if spacing is not None:
        margin_abs = max(Lmax * margin_frac, min_margin_voxels * spacing)
        raw = int(np.ceil((Lmax + 2 * margin_abs) / spacing))
        N = next_pow2(raw)
        s = float(spacing)

    else:
        if n is None:
            n = 7
        N = 2 ** int(n)

        # initial spacing estimate using fractional margin only
        s = (Lmax * (1.0 + 2.0 * margin_frac)) / N

        # ensure at least min_margin_voxels safety
        min_margin_mm = min_margin_voxels * s
        margin_abs = max(Lmax * margin_frac, min_margin_mm)

        # adjust spacing so total extent matches new margin
        s = (Lmax + 2 * margin_abs) / N

    extent = N * s
    origin = (-extent / 2.0, -extent / 2.0, -extent / 2.0)

    grid = {
        "shape": (N, N, N),
        "spacing": (s, s, s),
        "origin": origin,
        "extent_mm": (extent, extent, extent),
        "aabb_mm": (Lx, Ly, Lz),
        "margin_mm": margin_abs,
    }

    info(f"Grid spacing={s:.4f} mm | N={N} | extent={extent:.3f} mm | safe margin={margin_abs:.3f} mm")
    return grid

def voxelize_mesh(mesh: trimesh.Trimesh, grid: dict) -> np.ndarray:
    """
    OPTIONAL: Voxelize mesh with trimesh.voxelized(pitch=s) and embed it in the cube.
    (Not required for winding classification, but kept for parity / debugging.)
    Returns a boolean (Z,Y,X) volume aligned with the grid.
    """
    s = grid["spacing"][0]
    N = grid["shape"][0]

    vg = mesh.voxelized(pitch=s)
    vol = vg.matrix.astype(bool)  # native (z,y,x) order from trimesh

    # center-embed into NxNxN cube
    cube = np.zeros(grid["shape"], dtype=bool)
    sz, sy, sx = vol.shape
    oz = max((N - sz) // 2, 0); oy = max((N - sy) // 2, 0); ox = max((N - sx) // 2, 0)
    z1, y1, x1 = oz + sz, oy + sy, ox + sx
    cube[oz:z1, oy:y1, ox:x1] = vol
    info(f"Voxelized (optional) → vol={vol.shape}, cube={cube.shape}, filled={cube.mean():.5f}")
    return cube

# ---------------------------------------------------------------------
# Winding-based in/on/out (original approach)
# ---------------------------------------------------------------------
def classify_by_winding(mesh: trimesh.Trimesh, grid: dict, band: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use libigl’s fast winding number to classify inside / on / out.
    Returns uint8 masks of shape (N,N,N) in (Z,Y,X) order.
    """
    if igl is None:
        raise RuntimeError(
            "python-igl is required for winding classification. "
            "Install with: conda install -c conda-forge python-igl  (or pip install igl)"
        )

    N = grid["shape"][0]
    s = grid["spacing"][0]
    ox, oy, oz = grid["origin"]

    # Query points = centers of all voxels (Z,Y,X → world X,Y,Z)
    kji = np.indices((N, N, N), dtype=float)  # (3, N, N, N)
    xs = ox + (kji[2] + 0.5) * s  # i
    ys = oy + (kji[1] + 0.5) * s  # j
    zs = oz + (kji[0] + 0.5) * s  # k
    Q = np.stack((xs.ravel(), ys.ravel(), zs.ravel()), axis=1)  # (N^3, 3)

    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    WN = igl.fast_winding_number_for_meshes(V, F, Q)  # type:ignore
    WN = np.asarray(WN).reshape((N, N, N))  # (k, j, i) == (Z,Y,X)

    # Inside = WN > 0.5
    inside = (WN > 0.5)

    # On = narrow band around surface: boundary voxels of 'inside'
    dil = binary_dilation(inside, structure=np.ones((3, 3, 3), dtype=bool))
    boundary = dil ^ inside
    on = boundary & ~inside

    out = ~(inside | on)

    return inside.astype(np.uint8), on.astype(np.uint8), out.astype(np.uint8)

def classify_by_trimesh_contains(
    mesh: trimesh.Trimesh,
    grid: dict,
    batch_size: int = 200_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use trimesh.contains (ray parity) to classify inside / on / out.
    Slower than winding but avoids python-igl.
    """
    N = grid["shape"][0]
    s = grid["spacing"][0]
    ox, oy, oz = grid["origin"]

    total = N * N * N
    inside_flat = np.zeros(total, dtype=bool)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idx = np.arange(start, end, dtype=np.int64)

        k = idx // (N * N)
        rem = idx % (N * N)
        j = rem // N
        i = rem % N

        xs = ox + (i + 0.5) * s
        ys = oy + (j + 0.5) * s
        zs = oz + (k + 0.5) * s
        Q = np.stack((xs, ys, zs), axis=1)

        try:
            inside_flat[start:end] = mesh.contains(Q)
        except Exception as exc:
            raise RuntimeError(
                "trimesh.contains failed; consider installing python-igl or using a different mesh"
            ) from exc

    inside = inside_flat.reshape((N, N, N))

    dil = binary_dilation(inside, structure=np.ones((3, 3, 3), dtype=bool))
    boundary = dil ^ inside
    on = boundary & ~inside
    out = ~(inside | on)

    return inside.astype(np.uint8), on.astype(np.uint8), out.astype(np.uint8)

# ---------------------------------------------------------------------
# Viewer utilities
# ---------------------------------------------------------------------
def _slice_frames(grid, i_idx, j_idx, k_idx, color_map=None, line_width=4):
    """
    Draw wireframe rectangles showing slice locations (X/Y/Z).
    Colors correspond to sliders: X=blue, Y=green, Z=orange.
    """
    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]
    color_map = color_map or {"x": "#3B82F6", "y": "#22C55E", "z": "#F59E0B"}

    extent_x = Nx * sx
    extent_y = Ny * sy
    extent_z = Nz * sz

    Xr = [ox, ox + extent_x]
    Yr = [oy, oy + extent_y]
    Zr = [oz, oz + extent_z]

    # convert index → position
    Xpos = ox + i_idx * sx
    Ypos = oy + j_idx * sy
    Zpos = oz + k_idx * sz

    def rect(x, y, z, color, name):
        return go.Scatter3d(
            x=x + [x[0]], y=y + [y[0]], z=z + [z[0]],
            mode="lines", line=dict(color=color, width=line_width),
            name=name, showlegend=False
        )

    frames = []
    # X-slice (YZ plane)
    frames.append(rect(
        x=[Xpos]*4,
        y=[Yr[0], Yr[0], Yr[1], Yr[1]],
        z=[Zr[0], Zr[1], Zr[1], Zr[0]],
        color=color_map["x"], name="x-slice"
    ))
    # Y-slice (XZ plane)
    frames.append(rect(
        x=[Xr[0], Xr[1], Xr[1], Xr[0]],
        y=[Ypos]*4,
        z=[Zr[0], Zr[0], Zr[1], Zr[1]],
        color=color_map["y"], name="y-slice"
    ))
    # Z-slice (XY plane)
    frames.append(rect(
        x=[Xr[0], Xr[1], Xr[1], Xr[0]],
        y=[Yr[0], Yr[0], Yr[1], Yr[1]],
        z=[Zpos]*4,
        color=color_map["z"], name="z-slice"
    ))
    return frames

def _roi_frame(
    grid,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    r_vox: int,
    *,
    n_lat: int = 12,
    n_lon: int = 12,
    color: str = "#E11D48",
    line_width: int = 1,
):
    """
    Build a spherical ROI wireframe centered at voxel indices (i,j,k)
    with radius specified in voxels (r_vox). Returns a list of Scatter3d traces
    similar to _slice_frames(), intended to be directly added to the 3D figure.

    Notes:
    - Indices (i,j,k) are interpreted consistently with _slice_frames: position
      = origin + index * spacing (voxel-aligned). This mirrors the slice planes.
    - Radius is converted to millimeters using grid spacing.
    - The wireframe consists of latitude and longitude rings for clarity and speed.
    """
    if r_vox is None or r_vox <= 0:
        return []

    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]

    # Center in world coordinates (voxel-aligned like slice frames)
    Xc = ox + i_idx * sx
    Yc = oy + j_idx * sy
    Zc = oz + k_idx * sz

    # Radius in mm (use isotropic spacing assumption in this grid)
    s = sx  # == sy == sz
    r = float(r_vox) * s
    if r <= 0:
        return []

    # Sampling resolution for wireframe rings
    n_lat = max(3, int(n_lat))   # latitude (phi in [0, pi])
    n_lon = max(4, int(n_lon))   # longitude (theta in [0, 2pi))
    n_circle_pts = 60            # points per ring

    traces = []

    # Latitude rings (parallel to XY planes): phi from epsilon..pi-epsilon to avoid degeneracy
    phis = np.linspace(1e-6, np.pi - 1e-6, n_lat)
    thetas_ring = np.linspace(0, 2 * np.pi, n_circle_pts)
    for phi in phis:
        sinp = np.sin(phi)
        cosp = np.cos(phi)
        xs = Xc + r * sinp * np.cos(thetas_ring)
        ys = Yc + r * sinp * np.sin(thetas_ring)
        zs = Zc + r * np.full_like(thetas_ring, cosp)
        traces.append(
            go.Scatter3d(
                x=np.r_[xs, xs[0]], y=np.r_[ys, ys[0]], z=np.r_[zs, zs[0]],
                mode="lines",
                line=dict(color=color, width=line_width),
                name="roi-sphere-lat",
                showlegend=False,
            )
        )

    # Longitude rings (half-meridians closing into lines): theta grid
    thetas = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    phis_mer = np.linspace(0, np.pi, n_circle_pts)
    for th in thetas:
        cost = np.cos(th)
        sint = np.sin(th)
        sinp = np.sin(phis_mer)
        cosp = np.cos(phis_mer)
        xs = Xc + r * sinp * cost
        ys = Yc + r * sinp * sint
        zs = Zc + r * cosp
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=line_width),
                name="roi-sphere-lon",
                showlegend=False,
            )
        )

    return traces

def mask_to_trace(mask_u8, grid, color, name, opacity=0.4):
    total = int(np.sum(mask_u8))
    info(f"[{name}] mask sum={total}")
    if total == 0:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(mask_u8, level=0.5)
        info(f"[{name}] marching cubes → verts={len(verts)}, faces={len(faces)}")
    except Exception as e:
        err(f"[{name}] marching cubes failed: {e}")
        return None

    s = grid["spacing"][0]
    origin = np.array(grid["origin"])
    # marching_cubes returns verts in (z,y,x) index space → map to world (x,y,z)
    coords = origin + s * verts[:, [2, 1, 0]]

    i, j, k = faces.T
    return go.Mesh3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        i=i, j=j, k=k,
        name=name, color=color, opacity=opacity,
        flatshading=True, showscale=False
    )

def _binary_colorscale():
    return [[0.0, "black"], [1.0, "white"]]

def _slice_fig(z2d, title, border_color="#ffffff"):
    fig = go.Figure(
        data=[go.Heatmap(z=z2d, zmin=0, zmax=1, colorscale=_binary_colorscale(), showscale=False)],
        layout=go.Layout(
            title=title, margin=dict(l=2, r=2, t=28, b=2),
            xaxis=dict(showticklabels=False), yaxis=dict(autorange="reversed", showticklabels=False),
            plot_bgcolor="black", paper_bgcolor="black"
        )
    )
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=1, y1=1,
        xref="paper", yref="paper",
        line=dict(color=border_color, width=3),
        fillcolor="rgba(0,0,0,0)"
    )
    return fig

def compose_slice(masks, axis, idx):
    """Combine active masks along a slice (arrays are (Z,Y,X))."""
    if len(masks) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if axis == 'x':   # i: keep Z,Y
        slices = [m[:, :, idx] for m in masks]
    elif axis == 'y': # j: keep Z,X
        slices = [m[:, idx, :] for m in masks]
    elif axis == 'z': # k: keep Y,X
        slices = [m[idx, :, :] for m in masks]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    out = np.zeros_like(slices[0], dtype=np.uint8)
    for s in slices:
        out |= (s > 0).astype(np.uint8)
    return out

# ---------------------------------------------------------------------
# ROI-based masking utilities
# ---------------------------------------------------------------------
def filter_inside_by_roi(
        inside_u8: np.ndarray,
        i_idx: int,
        j_idx: int,
        k_idx: int,
        r_vox: int,
) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intersect the inside_u8 mask with a spherical ROI centered at voxel
        indices (i_idx, j_idx, k_idx) with radius r_vox (in voxels).

        Inputs
        - inside_u8: uint8 mask array of shape (Z, Y, X); nonzero denotes inside
        - i_idx, j_idx, k_idx: ROI center in voxel indices matching the UI sliders
          and _slice_frames convention (i=X, j=Y, k=Z)
        - r_vox: radius of the sphere in voxels

        Returns (masked_in_roi, masked_outside_roi)
        - masked_in_roi:  uint8 mask (Z, Y, X) of inside_u8 limited to the ROI
        - masked_outside_roi: uint8 mask (Z, Y, X) of inside_u8 outside the ROI

        Notes
        - Computation happens in index space (voxel units). Grid spacing is not
          required since r_vox is already in voxels.
        """
        if inside_u8 is None:
                raise ValueError("inside_u8 must not be None")
        if r_vox is None or r_vox <= 0:
                # Nothing selected → all inside remains outside-ROI part
                outside_only = (inside_u8 > 0).astype(np.uint8)
                return np.zeros_like(inside_u8, dtype=np.uint8), outside_only

        Nz, Ny, Nx = inside_u8.shape
        # Distance in voxel units from (i_idx, j_idx, k_idx)
        ii = np.arange(Nx) - int(i_idx)
        jj = np.arange(Ny) - int(j_idx)
        kk = np.arange(Nz) - int(k_idx)

        rr2 = (ii * ii)[None, None, :] + (jj * jj)[None, :, None] + (kk * kk)[:, None, None]
        roi = rr2 <= (int(r_vox) * int(r_vox))

        inside_bool = (inside_u8 > 0)
        masked_in_roi = (inside_bool & roi).astype(np.uint8)
        masked_outside_roi = (inside_bool & (~roi)).astype(np.uint8)
        return masked_in_roi, masked_outside_roi

def _bitwise_dilate6(mask_bool: np.ndarray) -> np.ndarray:
    """Single-iteration 3D dilation using 6-neighborhood via bitwise shifts.
    mask_bool is shape (Z, Y, X), dtype=bool.
    """
    zpos = np.pad(mask_bool[:-1, :, :], ((1,0),(0,0),(0,0)), mode='constant', constant_values=False)
    zneg = np.pad(mask_bool[1:,  :, :], ((0,1),(0,0),(0,0)), mode='constant', constant_values=False)
    ypos = np.pad(mask_bool[:, :-1, :], ((0,0),(1,0),(0,0)), mode='constant', constant_values=False)
    yneg = np.pad(mask_bool[:, 1:,  :], ((0,0),(0,1),(0,0)), mode='constant', constant_values=False)
    xpos = np.pad(mask_bool[:, :, :-1], ((0,0),(0,0),(1,0)), mode='constant', constant_values=False)
    xneg = np.pad(mask_bool[:, :, 1: ], ((0,0),(0,0),(0,1)), mode='constant', constant_values=False)
    out = mask_bool | zpos | zneg | ypos | yneg | xpos | xneg
    return out

def _bitwise_erode6(mask_bool: np.ndarray) -> np.ndarray:
    """Single-iteration 3D erosion using 6-neighborhood via bitwise shifts.
    mask_bool is shape (Z, Y, X), dtype=bool.
    """
    zpos = np.pad(mask_bool[:-1, :, :], ((1,0),(0,0),(0,0)), mode='constant', constant_values=False)
    zneg = np.pad(mask_bool[1:,  :, :], ((0,1),(0,0),(0,0)), mode='constant', constant_values=False)
    ypos = np.pad(mask_bool[:, :-1, :], ((0,0),(1,0),(0,0)), mode='constant', constant_values=False)
    yneg = np.pad(mask_bool[:, 1:,  :], ((0,0),(0,1),(0,0)), mode='constant', constant_values=False)
    xpos = np.pad(mask_bool[:, :, :-1], ((0,0),(0,0),(1,0)), mode='constant', constant_values=False)
    xneg = np.pad(mask_bool[:, :, 1: ], ((0,0),(0,0),(0,1)), mode='constant', constant_values=False)
    out = mask_bool & zpos & zneg & ypos & yneg & xpos & xneg
    return out

def proximity_bridge(mask_u8: np.ndarray, max_dist: int = 2) -> np.ndarray:
    """
    Bridge disconnected components that are within `max_dist` voxels of each other.
    - mask_u8: uint8 mask (Z,Y,X)
    - max_dist: Chebyshev distance threshold (voxels)

    Strategy:
        1. Dilate mask by max_dist
        2. Compute potential bridging voxels that lie between disconnected components
        3. Activate these voxels in the output mask
    """
    mask = (mask_u8 > 0)
    if max_dist <= 0:
        return mask_u8

    # Step 1: Chebyshev dilation (cube of size (2*max_dist+1))
    size = 2 * max_dist + 1
    kernel = np.ones((size, size, size), dtype=bool)
    dil = binary_dilation(mask, structure=kernel)

    # Step 2: bridging candidates = dilated minus original
    bridge = dil & (~mask)

    # Step 3: activate only voxels that lie between ≥2 different components
    # Identify connected components
    labeled, num = measure.label(mask.astype(np.uint8), return_num=True, connectivity=1)

    if num <= 1:
        # nothing to connect
        out = mask.copy()
        return out.astype(np.uint8)

    # For each bridging voxel, check if neighboring labels belong to >=2 components
    Z, Y, X = np.where(bridge)
    out = mask.copy()

    for z, y, x in zip(Z, Y, X):
        # Neighborhood (6-connectivity)
        neigh = []
        if z > 0:     neigh.append(labeled[z-1, y, x])
        if z < labeled.shape[0]-1: neigh.append(labeled[z+1, y, x])
        if y > 0:     neigh.append(labeled[z, y-1, x])
        if y < labeled.shape[1]-1: neigh.append(labeled[z, y+1, x])
        if x > 0:     neigh.append(labeled[z, y, x-1])
        if x < labeled.shape[2]-1: neigh.append(labeled[z, y, x+1])

        neigh = [n for n in neigh if n != 0]

        if len(set(neigh)) >= 2:
            out[z, y, x] = True

    return out.astype(np.uint8)

def apply_roi_morphology(
    inside_u8: np.ndarray,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    r_vox: int,
    iterations: int = 1,
    mode: str = "dilate",
) -> np.ndarray:
    """
    Apply iterative 3D dilation or shrink (erosion) confined to a spherical ROI.

    Steps
    - Use filter_inside_by_roi to split the global inside mask into:
      inside_in_roi, inside_outside_roi.
    - Apply `iterations` rounds of bitwise morphology (6-neighborhood) to
      inside_in_roi only, clamping to the ROI so changes don't leak outside.
    - Recombine with inside_outside_roi to form the new global inside mask.

    Parameters
    - inside_u8: uint8 global inside mask (Z,Y,X)
    - i_idx, j_idx, k_idx: center of ROI in voxel indices (i=X, j=Y, k=Z)
    - r_vox: ROI radius in voxels
    - iterations: number of iterative morphology steps
    - mode: 'dilate' to expand, 'shrink' to erode

    Returns
    - uint8 global inside mask after local morphology within ROI.
    """
    if iterations is None or iterations <= 0:
        return (inside_u8 > 0).astype(np.uint8)

    roi_inside_u8, outside_roi_inside_u8 = filter_inside_by_roi(inside_u8, i_idx, j_idx, k_idx, r_vox)

    # Build ROI boolean from geometry (always clamp to the full spherical ROI)
    Nz, Ny, Nx = inside_u8.shape
    ii = np.arange(Nx) - int(i_idx)
    jj = np.arange(Ny) - int(j_idx)
    kk = np.arange(Nz) - int(k_idx)
    rr2 = (ii * ii)[None, None, :] + (jj * jj)[None, :, None] + (kk * kk)[:, None, None]
    roi_bool = rr2 <= (int(r_vox) * int(r_vox))

    work = roi_inside_u8.astype(bool)
    for _ in range(int(iterations)):
        if mode == "dilate":
            work = _bitwise_dilate6(work)
        elif mode in ("shrink", "erode", "erosion"):
            work = _bitwise_erode6(work)
        else:
            raise ValueError("mode must be 'dilate' or 'shrink' (alias: 'erode','erosion')")
        # Clamp to ROI to prevent leaking outside
        work = work & roi_bool

    # Recompose: updated ROI portion + untouched outside-ROI portion
    updated_roi_u8 = work.astype(np.uint8)
    new_inside = ((outside_roi_inside_u8 > 0) | (updated_roi_u8 > 0)).astype(np.uint8)
    # Bridge any ROI→non-ROI pairs within 2 voxels
    new_inside = proximity_bridge(new_inside, max_dist=2)
    
    return new_inside

def _sigmoid_iteration_schedule(num_steps: int, total_iters: int) -> np.ndarray:
    """
    Distribute `total_iters` across `num_steps` with a slow-fast-slow profile.
    Uses a Hann window to produce low weights at the ends and high in the middle.

    Returns an int array of length `num_steps` with at least 1 per step
    and approximately summing to `total_iters`.
    """
    if num_steps <= 0 or total_iters <= 0:
        return np.array([], dtype=int)

    t = np.linspace(0.0, 1.0, num_steps)
    weights = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)  # Hann window: 0 at ends, peak mid
    weights = weights + 1e-6  # avoid exact zeros
    weights = weights / np.sum(weights) * float(total_iters)

    iters = np.maximum(1, np.round(weights).astype(int))
    diff = int(total_iters - int(np.sum(iters)))
    if diff != 0:
        # Adjust to match total_iters by tweaking largest/smallest weights
        order = np.argsort(-weights) if diff > 0 else np.argsort(weights)
        for idx in order[:abs(diff)]:
            iters[idx] += 1 if diff > 0 else -1
            if iters[idx] < 1:
                iters[idx] = 1
    return iters

def _sigmoid_sparse_apply_schedule(total_ticks: int, num_applies: int) -> np.ndarray:
    """
    Build a boolean schedule of length `total_ticks` marking which ticks should
    apply a single-iteration morphology step. The placement follows a
    slow–fast–slow (sigmoidal) density using a Hann window: denser in the
    middle, sparser at the ends. Ensures exactly `num_applies` True entries
    (clamped to `total_ticks`).

    This keeps each morphology change to a single-voxel iteration so
    `proximity_bridge` can reliably handle gaps.
    """
    if total_ticks <= 0 or num_applies <= 0:
        return np.zeros(max(0, total_ticks), dtype=bool)
    num_applies = min(num_applies, total_ticks)

    # Hann window weights across ticks (exclude endpoint to avoid duplicate zero)
    t = np.linspace(0.0, 1.0, total_ticks, endpoint=False)
    weights = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)

    order = np.argsort(-weights)  # descending by weight (middle first)
    chosen = order[:num_applies]
    schedule = np.zeros(total_ticks, dtype=bool)
    schedule[chosen] = True
    return schedule

def show_viewer_dash(mesh, inside_u8, on_u8, out_u8, grid, port=8050):
    """Interactive viewer: X/Y/Z slices + 3-D, with mask toggles."""
    Nz, Ny, Nx = inside_u8.shape
    x_mid, y_mid, z_mid = Nx // 2, Ny // 2, Nz // 2

    # Mesh (as loaded, centered at origin)
    mesh_trace = go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        name="mesh", color="#AB1616", opacity=0.15
    )
    # We will recompute the inside trace dynamically to reflect morphology changes
    # inside_trace = mask_to_trace(inside_u8, grid, "#3B82F6", "inside", 0.35)
    on_trace     = mask_to_trace(on_u8,     grid, "#22C55E", "on",     0.55)
    out_trace    = mask_to_trace(out_u8,    grid, "#F59E0B", "out",    0.08)

    # Mutable state for inside mask so we can apply morphology and reset
    inside_state = {"inside": inside_u8.copy(), "original": inside_u8.copy()}

    app = dash.Dash(__name__)
    app.layout = html.Div(
        style={"display": "grid", "gridTemplateColumns": "300px 1fr", "gap": "10px",
               "height": "100vh", "backgroundColor": "#0f1115", "color": "#e6e6e6",
               "padding": "10px"},
        children=[
            html.Div(
                style={"backgroundColor": "#151922", "borderRadius": "10px",
                       "padding": "12px", "display": "flex", "flexDirection": "column", "gap": "10px"},
                children=[
                    html.H3("Tools", style={"margin": "0 0 6px 0"}),
                    dcc.Checklist(
                        id="mask-check",
                        options=[
                            {"label": " Inside", "value": "inside"},
                            {"label": " On", "value": "on"},
                            {"label": " Out", "value": "out"},
                        ],
                        value=["inside", "on", "out"],
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "block", "marginBottom": "4px"}
                    ),
                    html.Hr(),
                    dcc.Tabs(id="tools-tabs", value="roi", children=[
                        dcc.Tab(label="ROI", value="roi", 
                                style={'backgroundColor': 'black', 'color': 'white'},
                                selected_style={'backgroundColor': 'black', 'color': 'white', 'border': '2px solid blue', 'fontWeight': 'bold'},
                                children=[
                            html.Div([
                                html.Label("X-slice (i)"),
                                dcc.Slider(id="x-slider", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag"),
                                html.Label("Y-slice (j)"),
                                dcc.Slider(id="y-slider", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                html.Label("Z-slice (k)"),
                                dcc.Slider(id="z-slider", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                html.Hr(),
                                html.Div("ROI (sphere)", style={"fontWeight": 600, "opacity": 0.9}),
                                dcc.Tabs(id="roi-mode-tabs", value="center", children=[
                                    dcc.Tab(label="Center + Radius", value="center", 
                                            style={'backgroundColor': 'black', 'color': 'white'},
                                            selected_style={'backgroundColor': 'black', 'color': 'white', 'border': '2px solid blue', 'fontWeight': 'bold'},
                                            children=[
                                        html.Label("ROI X (mm)"),
                                        dcc.Input(id="roi-x-mm", type="text", placeholder="mm", value="",
                                            style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                        html.Label("ROI Y (mm)"),
                                        dcc.Input(id="roi-y-mm", type="text", placeholder="mm", value="",
                                            style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                        html.Label("ROI Z (mm)"),
                                        dcc.Input(id="roi-z-mm", type="text", placeholder="mm", value="",
                                            style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                        html.Label("ROI X (i)"),
                                        dcc.Slider(id="roi-x", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag"),
                                        html.Label("ROI Y (j)"),
                                        dcc.Slider(id="roi-y", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                        html.Label("ROI Z (k)"),
                                        dcc.Slider(id="roi-z", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                        html.Label("ROI Radius (voxels)"),
                                        dcc.Slider(id="roi-r", min=0, max=int(max(Nx, Ny, Nz)//2), step=1, value=1, updatemode="drag",
                                                   tooltip={"always_visible": False, "placement": "bottom"}),
                                        html.Label("ROI Radius (mm)"),
                                        dcc.Input(id="roi-r-mm", type="text", placeholder="mm", value="",
                                            style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                    ]),
                                    dcc.Tab(label="Two Points", value="twopoints", 
                                            style={'backgroundColor': 'black', 'color': 'white'},
                                            selected_style={'backgroundColor': 'black', 'color': 'white', 'border': '2px solid blue', 'fontWeight': 'bold'},
                                            children=[
                                        html.Div("Point A", style={"fontWeight": 600, "opacity": 0.9, "marginTop": "6px"}),
                                        html.Label("A: i"),
                                        dcc.Slider(id="pA-i", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag"),
                                        html.Label("A: j"),
                                        dcc.Slider(id="pA-j", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                        html.Label("A: k"),
                                        dcc.Slider(id="pA-k", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                        html.Div("Point B", style={"fontWeight": 600, "opacity": 0.9, "marginTop": "6px"}),
                                        html.Label("B: i"),
                                        dcc.Slider(id="pB-i", min=0, max=Nx-1, step=1, value=x_mid+1 if x_mid+1 < Nx else x_mid, updatemode="drag"),
                                        html.Label("B: j"),
                                        dcc.Slider(id="pB-j", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                        html.Label("B: k"),
                                        dcc.Slider(id="pB-k", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                        html.Div("Note: Sphere center = midpoint(A,B); radius = half distance(A,B)",
                                                 style={"fontSize": "12px", "opacity": 0.8, "marginTop": "6px"}),
                                    ])
                                ])
                            ], style={"padding": "6px"})
                        ]),
                        dcc.Tab(label="Operations", value="ops", 
                                style={'backgroundColor': 'black', 'color': 'white'},
                                selected_style={'backgroundColor': 'black', 'color': 'white', 'border': '2px solid blue', 'fontWeight': 'bold'},
                                children=[
                            html.Div([
                                html.Div("Morphology Mode", style={"fontWeight": 600, "opacity": 0.9}),
                                dcc.RadioItems(
                                    id="morph-ui-mode",
                                    options=[
                                        {"label": " Manual", "value": "manual"},
                                        {"label": " Automated (percent diameter stenosis)", "value": "auto"},
                                    ],
                                    value="manual",
                                    labelStyle={"display": "block", "marginBottom": "4px"}
                                ),
                                html.Div(id="ops-manual", children=[
                                    dcc.RadioItems(
                                        id="morph-mode",
                                        options=[
                                            {"label": " Dilate", "value": "dilate"},
                                            {"label": " Erode (shrink)", "value": "shrink"},
                                        ],
                                        value="dilate",
                                        labelStyle={"display": "block", "marginBottom": "4px"}
                                    ),
                                    html.Label("Iterations"),
                                    dcc.Input(id="morph-iters", type="number", min=1, step=1, value=1,
                                              style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                     "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                    html.Button("Apply Morphology", id="apply-morph",
                                                style={"background": "#16A34A", "color": "white",
                                                       "border": "none", "padding": "6px 10px",
                                                       "borderRadius": "6px", "marginTop": "6px"}),
                                ], style={"marginTop": "8px"}),
                                html.Div(id="ops-auto", children=[
                                    html.Label("Percent Diameter Stenosis (%)"),
                                    dcc.Input(id="stenosis-pct", type="number", min=1, max=99, step=1, value=50,
                                              style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                     "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                    html.Button("Run Automated Stenosis", id="apply-auto",
                                                style={"background": "#F59E0B", "color": "black",
                                                       "border": "none", "padding": "6px 10px",
                                                       "borderRadius": "6px", "marginTop": "6px"}),
                                ], style={"marginTop": "8px"}),
                            ], style={"padding": "6px"})
                        ]),
                    ]),
                    html.Button("Reset", id="reset-btn",
                                style={"background": "#2563EB", "color": "white",
                                       "border": "none", "padding": "6px 10px",
                                       "borderRadius": "6px", "marginTop": "8px"}),
                    html.Div(id="status", style={"fontSize": "12px", "marginTop": "8px"})
                ]
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                       "gridTemplateRows": "1fr 1fr", "gap": "10px"},
                children=[
                    dcc.Graph(id="x-view"), dcc.Graph(id="y-view"),
                    dcc.Graph(id="z-view"), dcc.Graph(id="threeD-view")
                ]
            )
        ]
    )

    # Helper function to convert mm to voxel index
    def mm_to_idx(val_mm, origin, spacing, limit):
        if val_mm is None:
            return None
        if isinstance(val_mm, str) and val_mm.strip() == "":
            return None
        try:
            v = float(val_mm)
        except Exception:
            return None
        idx = int(round((v - origin) / spacing))
        idx = max(0, min(limit - 1, idx))
        return idx

    # Callback to update ROI sliders when mm coordinates are entered
    @app.callback(
        Output("roi-x", "value"),
        Output("roi-y", "value"),
        Output("roi-z", "value"),
        Input("roi-x-mm", "value"),
        Input("roi-y-mm", "value"),
        Input("roi-z-mm", "value"),
        prevent_initial_call=True
    )
    def update_roi_from_mm(roi_x_mm, roi_y_mm, roi_z_mm):
        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        s = grid["spacing"][0]
        ox, oy, oz = grid["origin"]
        Nz, Ny, Nx = grid["shape"]
        
        # Get current slider values (from State would be better, but using defaults)
        current_roi_x = x_mid
        current_roi_y = y_mid
        current_roi_z = z_mid
        
        new_roi_x = current_roi_x
        new_roi_y = current_roi_y
        new_roi_z = current_roi_z
        
        # Only update if a mm input was triggered and has a valid value
        if "roi-x-mm.value" in triggered:
            ix = mm_to_idx(roi_x_mm, ox, s, Nx)
            if ix is not None:
                new_roi_x = ix
        
        if "roi-y-mm.value" in triggered:
            iy = mm_to_idx(roi_y_mm, oy, s, Ny)
            if iy is not None:
                new_roi_y = iy
        
        if "roi-z-mm.value" in triggered:
            iz = mm_to_idx(roi_z_mm, oz, s, Nz)
            if iz is not None:
                new_roi_z = iz
        
        return new_roi_x, new_roi_y, new_roi_z

    # Callback to update mm coordinates when ROI sliders are moved
    @app.callback(
        Output("roi-x-mm", "value"),
        Output("roi-y-mm", "value"),
        Output("roi-z-mm", "value"),
        Input("roi-x", "value"),
        Input("roi-y", "value"),
        Input("roi-z", "value"),
        prevent_initial_call=True
    )
    def update_mm_from_roi(roi_x, roi_y, roi_z):
        s = grid["spacing"][0]
        ox, oy, oz = grid["origin"]
        
        # Convert voxel indices to mm coordinates
        roi_x_mm_val = ox + roi_x * s
        roi_y_mm_val = oy + roi_y * s
        roi_z_mm_val = oz + roi_z * s
        
        return f"{roi_x_mm_val:.2f}", f"{roi_y_mm_val:.2f}", f"{roi_z_mm_val:.2f}"

    # Callback to update ROI radius slider from mm input
    @app.callback(
        Output("roi-r", "value"),
        Input("roi-r-mm", "value"),
        prevent_initial_call=True
    )
    def update_radius_from_mm(roi_r_mm):
        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        s = grid["spacing"][0]
        
        # If mm input was triggered and has a valid value
        if "roi-r-mm.value" in triggered and roi_r_mm is not None and isinstance(roi_r_mm, str) and roi_r_mm.strip():
            try:
                r_mm = float(roi_r_mm)
                r_vox = int(round(r_mm / s))
                r_vox = max(0, min(int(max(Nx, Ny, Nz)//2), r_vox))
                return r_vox
            except Exception:
                pass
        
        # Return current slider value if no valid input
        return 1

    # Callback to update mm radius display when slider changes
    @app.callback(
        Output("roi-r-mm", "value"),
        Input("roi-r", "value"),
        prevent_initial_call=True
    )
    def update_radius_mm_display(roi_r):
        s = grid["spacing"][0]
        
        if roi_r is not None:
            r_mm = roi_r * s
            return f"{r_mm:.2f}"
        
        return ""

    @app.callback(
        Output("x-view", "figure"),
        Output("y-view", "figure"),
        Output("z-view", "figure"),
        Output("threeD-view", "figure"),
        Output("status", "children"),
        Input("mask-check", "value"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("roi-mode-tabs", "value"),
        Input("roi-x", "value"),
        Input("roi-y", "value"),
        Input("roi-z", "value"),
        Input("roi-r", "value"),
        Input("pA-i", "value"),
        Input("pA-j", "value"),
        Input("pA-k", "value"),
        Input("pB-i", "value"),
        Input("pB-j", "value"),
        Input("pB-k", "value"),
        Input("tools-tabs", "value"),
        Input("morph-ui-mode", "value"),
        Input("morph-mode", "value"),
        Input("morph-iters", "value"),
        Input("apply-morph", "n_clicks"),
        Input("stenosis-pct", "value"),
        Input("apply-auto", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=False
    )
    def update(mask_values, x_idx, y_idx, z_idx, roi_mode,
               roi_x, roi_y, roi_z, roi_r,
               pA_i, pA_j, pA_k, pB_i, pB_j, pB_k,
               tabs_value, ui_mode, morph_mode, morph_iters, apply_clicks,
               stenosis_pct, apply_auto_clicks, n_clicks):
        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        if "reset-btn.n_clicks" in triggered:
            mask_values = ["inside", "on", "out"]
            x_idx, y_idx, z_idx = x_mid, y_mid, z_mid
            roi_x, roi_y, roi_z, roi_r = x_mid, y_mid, z_mid, 1
            pA_i, pA_j, pA_k = x_mid, y_mid, z_mid
            pB_i, pB_j, pB_k = min(x_mid+1, Nx-1), y_mid, z_mid
            # Reset the working inside mask
            inside_state["inside"] = inside_state["original"].copy()

        # Apply morphology when requested
        if "apply-morph.n_clicks" in triggered and (ui_mode == "manual"):
            try:
                iters = int(morph_iters) if morph_iters is not None else 1
                iters = max(1, iters)
            except Exception:
                iters = 1
            mode = morph_mode if morph_mode in ("dilate", "shrink") else "dilate"
            # Update only the inside portion within ROI, keep others untouched, then recombine
            inside_state["inside"] = apply_roi_morphology(
                inside_state["inside"], roi_x, roi_y, roi_z, roi_r, iterations=iters, mode=mode
            )

        # Determine effective ROI center+radius depending on ROI mode
        if roi_mode == "twopoints":
            # Midpoint center and half-distance radius (in voxels)
            cx = int(np.round((int(pA_i) + int(pB_i)) / 2))
            cy = int(np.round((int(pA_j) + int(pB_j)) / 2))
            cz = int(np.round((int(pA_k) + int(pB_k)) / 2))
            dx = int(pA_i) - int(pB_i)
            dy = int(pA_j) - int(pB_j)
            dz = int(pA_k) - int(pB_k)
            dist = int(np.round(np.sqrt(dx*dx + dy*dy + dz*dz)))
            eff_roi_x, eff_roi_y, eff_roi_z = cx, cy, cz
            eff_roi_r = max(1, dist // 2)
        else:
            eff_roi_x, eff_roi_y, eff_roi_z, eff_roi_r = int(roi_x), int(roi_y), int(roi_z), int(roi_r)

        # Automated stenosis: shrink ROI radius using a sparse sigmoidal tick schedule
        if "apply-auto.n_clicks" in triggered and (ui_mode == "auto"):
            try:
                pct = float(stenosis_pct)
            except Exception:
                pct = 50.0
            pct = max(1.0, min(99.0, pct))

            target_r = int(max(0, np.floor(eff_roi_r * (pct / 100.0))))
            start_r  = int(eff_roi_r)
            steps = max(0, start_r - target_r)  # number of single-iteration erosions needed
            auto_status = ""
            # Safety threshold: do not erode below 10% of the initial in-ROI voxels
            safety_frac = 0.10
            base_roi_u8, _ = filter_inside_by_roi(inside_state["inside"], eff_roi_x, eff_roi_y, eff_roi_z, start_r)
            base_count = int(np.sum(base_roi_u8))
            safe_min = max(1, int(np.floor(safety_frac * base_count)))

            if steps > 0:
                # Total time ticks: default to 2×steps (keeps overall duration similar)
                total_ticks = 2 * steps
                if total_ticks < steps:
                    total_ticks = steps
                apply_ticks = _sigmoid_sparse_apply_schedule(total_ticks, steps)

                current_r = start_r
                safety_triggered = False
                applied = 0
                for tick in range(total_ticks):
                    if current_r <= target_r:
                        break
                    if not apply_ticks[tick]:
                        continue  # skip this tick to throttle rate (no morphology this tick)

                    # Apply a single-iteration shrink confined to current ROI radius
                    inside_state["inside"] = apply_roi_morphology(
                        inside_state["inside"], eff_roi_x, eff_roi_y, eff_roi_z, current_r,
                        iterations=1, mode="shrink"
                    )
                    applied += 1

                    # Safety check: count current in-ROI voxels at this radius
                    cur_roi_u8, _ = filter_inside_by_roi(inside_state["inside"], eff_roi_x, eff_roi_y, eff_roi_z, current_r)
                    cur_count = int(np.sum(cur_roi_u8))
                    if cur_count < safe_min:
                        safety_triggered = True
                        auto_status = (
                            f" | Auto stopped early: r={current_r}, vox={cur_count} < {safe_min}, "
                            f"applied={applied}/{steps}, ticks={total_ticks}"
                        )
                        break

                    current_r -= 1

                if not safety_triggered:
                    auto_status = f" | Auto done: steps={applied}/{steps}, ticks={total_ticks}"
            # UI slider for roi-r remains user-controlled; internal loop uses shrinking radii

        active = []
        show_inside = "inside" in mask_values
        show_on     = "on"     in mask_values
        show_out    = "out"    in mask_values
        if show_inside:
            active.append(inside_state["inside"])
        if show_on:
            active.append(on_u8)
        if show_out:
            active.append(out_u8)

        x_fig = _slice_fig(compose_slice(active, "x", x_idx), f"X-slice (i={x_idx})", "#3B82F6")
        y_fig = _slice_fig(compose_slice(active, "y", y_idx), f"Y-slice (j={y_idx})", "#22C55E")
        z_fig = _slice_fig(compose_slice(active, "z", z_idx), f"Z-slice (k={z_idx})", "#F59E0B")

        # Rebuild inside surface dynamically from current inside state
        dyn_inside_trace = mask_to_trace(inside_state["inside"], grid, "#3B82F6", "inside", 0.35) if show_inside else None

        valid_traces = [t for t in [mesh_trace,
                                    dyn_inside_trace,
                                    on_trace     if show_on     else None,
                                    out_trace    if show_out    else None] if t is not None]
        extent_x, extent_y, extent_z = grid["extent_mm"]
        aspectratio = _aspectratio_from_extents((extent_x, extent_y, extent_z))

        fig3d = go.Figure(data=valid_traces)
        for frame in _slice_frames(grid, x_idx, y_idx, z_idx):
            fig3d.add_trace(frame)

        # ROI sphere (wireframe)
        roi_traces = _roi_frame(grid, eff_roi_x, eff_roi_y, eff_roi_z, eff_roi_r,
                                n_lat=6, n_lon=8, color="#E11D48", line_width=3)
        for t in roi_traces:
            fig3d.add_trace(t)
        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                bgcolor="#000000",
                aspectmode="manual",
                aspectratio=aspectratio,
                xaxis=dict(title="X (mm)", range=[-extent_x/2, extent_x/2], showgrid=False),
                yaxis=dict(title="Y (mm)", range=[-extent_y/2, extent_y/2], showgrid=False),
                zaxis=dict(title="Z (mm)", range=[-extent_z/2, extent_z/2], showgrid=False),
            )
        )

        status = (
            f"Active masks: {', '.join(mask_values)} | X={x_idx}, Y={y_idx}, Z={z_idx} "
            f"| ROI: (i,j,k)=({eff_roi_x},{eff_roi_y},{eff_roi_z}), r={eff_roi_r} vox"
        )
        try:
            if auto_status:
                status += auto_status
        except Exception:
            pass
        return x_fig, y_fig, z_fig, fig3d, status

    # Open browser on a fresh port each run to avoid caching
    import threading, webbrowser, random
    port = random.randint(8050, 9000)
    threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(debug=False, port=port)

# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
def run_pipeline(
    in_mesh_path: str,
    spacing: float | None = None,
    n: int | None = None,
    margin_frac: float = 0.10,
    out_npz: str = "outputs/masks_demo.npz",
    show: bool = True,
    mc_map: str = "xyz",
    viewer: str = "dash",
    port: int = 8050,
    method: str = "auto"
):
    mesh_path = Path(in_mesh_path).resolve()
    if not mesh_path.exists():
        err(f"Input mesh not found: {in_mesh_path}")
        raise FileNotFoundError(in_mesh_path)

    mesh = trimesh.load(mesh_path, force="mesh")
    info(f"Loaded mesh: {mesh_path.name} | watertight={getattr(mesh, 'is_watertight', 'unknown')}")

    # Center mesh at origin
    mesh = center_mesh(mesh) # type: ignore

    # Grid (power-of-two cubic grid)
    grid = make_cube_grid_from_mesh(mesh, spacing=spacing, n=n, margin_frac=margin_frac)

    # In/on/out masks
    if method == "auto":
        if igl is not None:
            inside_u8, on_u8, out_u8 = classify_by_winding(mesh, grid, band=0.6)
        else:
            warn("python-igl not available; falling back to trimesh.contains (slower).")
            inside_u8, on_u8, out_u8 = classify_by_trimesh_contains(mesh, grid)
    elif method == "winding":
        if igl is None:
            raise RuntimeError("python-igl is required for method='winding'.")
        inside_u8, on_u8, out_u8 = classify_by_winding(mesh, grid, band=0.6)
    elif method == "trimesh":
        inside_u8, on_u8, out_u8 = classify_by_trimesh_contains(mesh, grid)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save masks in a single compressed NPZ with spacing/origin like core.run_pipeline
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, inside=inside_u8, on=on_u8, out=out_u8,
                        spacing=np.array(grid["spacing"]), origin=np.array(grid["origin"]))
    ok(f"Masks saved (uint8) → {out_npz}")

    # Viewer
    if show:
        if viewer == "dash":
            show_viewer_dash(mesh, inside_u8, on_u8, out_u8, grid, port=port)
        else:
            # fallback: use the 3-D html viewer (not implemented separately here)
            show_viewer_dash(mesh, inside_u8, on_u8, out_u8, grid, port=port)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Standalone FakeCT pipeline (winding or trimesh)")
    ap.add_argument("--in", dest="in_mesh", required=True,
                    help="Input mesh (.stl/.obj/.ply)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="Voxel edge length in mm (if provided, N will be computed as next power-of-two)")
    ap.add_argument("--n", type=int, default=7,
                    help="Grid exponent (2^n per side). Used if --spacing is None.")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="Extra margin fraction around the mesh AABB (default 0.10 = 10%)")
    ap.add_argument("--mc-map", type=str, default="zyx",
                    choices=["zyx", "xyz", "xzy", "yxz", "yzx", "zxy"],
                    help="Axis mapping from marching-cubes (z,y,x) → (X,Y,Z).")
    ap.add_argument("--out", default="outputs/masks_demo.npz",
                    help="Output compressed npz file")
    ap.add_argument("--method", choices=["auto", "winding", "trimesh"], default="auto",
                    help="Inside/outside method: auto (default), winding (python-igl), or trimesh")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open viewer")
    ap.add_argument("--viewer", choices=["dash","html"], default="dash",
                    help="Viewer type: 'dash' or 'html'.")
    ap.add_argument("--port", type=int, default=8050,
                    help="Port for Dash viewer (default 8050).")

    args = ap.parse_args()

    if args.method == "winding" and igl is None:
        err("python-igl is required for --method winding. Install via conda-forge:\n"
            "  conda install -c conda-forge python-igl\n"
            "or pip:\n"
            "  pip install igl")
        sys.exit(1)

    spacing = None if (args.spacing is None or args.spacing <= 0) else float(args.spacing)

    run_pipeline(
        in_mesh_path=args.in_mesh,
        spacing=spacing,
        n=args.n,
        margin_frac=args.margin,
        out_npz=args.out,
        show=not args.no_show,
        mc_map=args.mc_map,
        viewer=args.viewer,
        port=args.port,
        method=args.method
    )

if __name__ == "__main__":
    main()
