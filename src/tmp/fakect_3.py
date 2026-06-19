#!/usr/bin/env python3
# ---------------------------------------------------------------------
# fakect.py
# Produces winding-based in/on/out masks and (optionally) launches a Dash viewer.
#
# Usage examples:
#   python src/fakect.py --in data/cube.stl --n 8 --out outputs/cube_masks.npz
#   python src/fakect.py --in data/carotid.stl --n 9 --margin 0.10 --out outputs/carotid_masks.npz
#
# Requirements & installation
#  Recommended (conda - easiest, includes python-igl):
#    conda create -n fakect python=3.10 -y
#    conda activate fakect
#    conda install -c conda-forge python-igl trimesh scipy scikit-image plotly dash -y
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

def apply_iterative_stenosis(
    inside_u8: np.ndarray,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    r_vox: int,
    stenosis_percent: float = 50.0,
) -> np.ndarray:
    """
    Iteratively apply stenosis (narrowing) within a spherical ROI by eroding
    the cross-sectional area progressively until reaching the target stenosis percentage.

    The stenosis percentage represents how much of the ROI's cross-sectional area
    is occupied by the ROI sphere itself:
    - 0%:   No stenosis (full lumen preserved)
    - 50%:  Half the lumen area is occupied
    - 100%: Complete occlusion (entire cross-section occupied by ROI sphere)

    Strategy
    1. Extract the inside mask within the ROI sphere
    2. Compute the initial cross-sectional area (count of inside voxels in ROI)
    3. Iteratively erode the ROI portion until the remaining area matches target
    4. Recombine with outside-ROI portion to form the stenosed global mask

    Parameters
    - inside_u8: uint8 global inside mask (Z,Y,X)
    - i_idx, j_idx, k_idx: center of ROI in voxel indices (i=X, j=Y, k=Z)
    - r_vox: radius of the stenosis ROI sphere in voxels
    - stenosis_percent: stenosis severity (0-100%)
      * 0 = no stenosis (preserve full lumen)
      * 100 = complete occlusion (remove all lumen in ROI)

    Returns
    - uint8 global inside mask with stenosis applied locally within ROI.

    Notes
    - Stenosis is applied via iterative erosion (shrinking) of the lumen in the ROI
    - The erosion continues until the remaining voxel count reaches approximately
      the target percentage of the original ROI area
    - Changes are confined to the spherical ROI; outside geometry remains unchanged
    """
    if r_vox is None or r_vox <= 0:
        warn("ROI radius <= 0; no stenosis applied")
        return inside_u8.copy()

    # Clamp stenosis_percent to valid range [0, 100]
    stenosis_percent = max(0.0, min(100.0, float(stenosis_percent)))

    # Extract inside voxels within the ROI
    roi_inside_u8, outside_roi_inside_u8 = filter_inside_by_roi(inside_u8, i_idx, j_idx, k_idx, r_vox)

    # Build the spherical ROI boundary for clamping
    Nz, Ny, Nx = inside_u8.shape
    ii = np.arange(Nx) - int(i_idx)
    jj = np.arange(Ny) - int(j_idx)
    kk = np.arange(Nz) - int(k_idx)
    rr2 = (ii * ii)[None, None, :] + (jj * jj)[None, :, None] + (kk * kk)[:, None, None]
    roi_bool = rr2 <= (int(r_vox) * int(r_vox))

    # Calculate initial lumen area (voxel count) within ROI
    roi_lumen_bool = (roi_inside_u8 > 0)
    initial_area = int(np.sum(roi_lumen_bool))

    # Calculate target area based on stenosis percentage
    # stenosis_percent = 0 → preserve 100% of lumen
    # stenosis_percent = 100 → preserve 0% of lumen (complete occlusion)
    target_area = int(initial_area * (1.0 - stenosis_percent / 100.0))

    if initial_area == 0:
        warn("No lumen detected within ROI; stenosis not applied")
        return inside_u8.copy()

    if target_area < 0:
        target_area = 0

    info(f"[STENOSIS] Initial lumen area in ROI: {initial_area} voxels")
    info(f"[STENOSIS] Target area ({stenosis_percent:.1f}% stenosis): {target_area} voxels")

    # Iteratively erode until reaching target area
    work = roi_lumen_bool.copy()
    iteration = 0
    max_iterations = max(initial_area, 100)  # safety limit

    while np.sum(work) > target_area and iteration < max_iterations:
        # Single erosion step confined to ROI
        work = _bitwise_erode6(work)
        work = work & roi_bool

        current_area = int(np.sum(work))
        iteration += 1

        # Log progress every 5 iterations or when near target
        if iteration % 5 == 0 or current_area <= target_area + 5:
            info(f"[STENOSIS] Iteration {iteration}: {current_area} voxels (target: {target_area})")

        if current_area == 0:
            break

    final_area = int(np.sum(work))
    actual_stenosis = (1.0 - final_area / initial_area) * 100.0 if initial_area > 0 else 100.0
    ok(f"[STENOSIS] Applied {actual_stenosis:.1f}% stenosis in {iteration} iterations "
       f"({initial_area} → {final_area} voxels)")

    # Recompose: updated ROI portion + untouched outside-ROI portion
    updated_roi_u8 = work.astype(np.uint8)
    new_inside = ((outside_roi_inside_u8 > 0) | (updated_roi_u8 > 0)).astype(np.uint8)

    # Bridge any disconnected components near the ROI boundary
    new_inside = proximity_bridge(new_inside, max_dist=2)

    return new_inside

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
                       "padding": "12px", "display": "flex", "flexDirection": "column", "gap": "10px", "overflowY": "auto"},
                children=[
                    html.H3("Tools", style={"margin": "0 0 6px 0"}),
                    dcc.Tabs(
                        id="control-tabs",
                        value="view-tab",
                        style={
                            "borderBottom": "2px solid #2a2f3a"
                        },
                        parent_style={},
                        children=[
                            dcc.Tab(
                                label="View & ROI",
                                value="view-tab",
                                style={
                                    "padding": "10px",
                                    "backgroundColor": "#0f1115",
                                    "color": "#a0aec0",
                                    "fontWeight": "normal",
                                    "borderRadius": "6px 6px 0 0",
                                    "border": "1px solid transparent"
                                },
                                selected_style={
                                    "padding": "10px",
                                    "backgroundColor": "#151922",
                                    "color": "#e6e6e6",
                                    "fontWeight": "bold",
                                    "borderRadius": "6px 6px 0 0",
                                    "border": "2px solid #3b82f6",
                                    "borderBottom": "none"
                                },
                                children=[
                                    html.Div(style={"display": "flex", "flexDirection": "column", "gap": "10px"}, children=[
                                        html.Div("Mask Display", style={"fontWeight": 600, "opacity": 0.9}),
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
                                        html.Div("Slice Planes", style={"fontWeight": 600, "opacity": 0.9}),
                                        html.Label("X-slice (i)"),
                                        dcc.Slider(id="x-slider", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag"),
                                        html.Label("Y-slice (j)"),
                                        dcc.Slider(id="y-slider", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                        html.Label("Z-slice (k)"),
                                        dcc.Slider(id="z-slider", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                        html.Hr(),
                                        html.Div("ROI (sphere)", style={"fontWeight": 600, "opacity": 0.9}),
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
                                        html.Label("ROI Radius (voxels)"),
                                        dcc.Input(id="roi-r-vox", type="text", placeholder="voxels", value="",
                                                  style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                         "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                        html.Hr(),
                                        html.Button("Reset", id="reset-btn-view",
                                                    style={"background": "#2563EB", "color": "white",
                                                           "border": "none", "padding": "6px 10px",
                                                           "borderRadius": "6px", "width": "100%"}),
                                    ])
                                ]
                            ),
                            dcc.Tab(
                                label="Operations",
                                value="ops-tab",
                                style={
                                    "padding": "10px",
                                    "backgroundColor": "#0f1115",
                                    "color": "#a0aec0",
                                    "fontWeight": "normal",
                                    "borderRadius": "6px 6px 0 0",
                                    "border": "1px solid transparent"
                                },
                                selected_style={
                                    "padding": "10px",
                                    "backgroundColor": "#151922",
                                    "color": "#e6e6e6",
                                    "fontWeight": "bold",
                                    "borderRadius": "6px 6px 0 0",
                                    "border": "2px solid #3b82f6",
                                    "borderBottom": "none"
                                },
                                children=[
                                    html.Div(style={"display": "flex", "flexDirection": "column", "gap": "10px"}, children=[
                                        html.Div("Mask Display (for reference)", style={"fontWeight": 600, "opacity": 0.9}),
                                        dcc.Checklist(
                                            id="mask-check-ops",
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
                                        html.Div("Morphology", style={"fontWeight": 600, "opacity": 0.9}),
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
                                                           "borderRadius": "6px", "marginTop": "6px", "width": "100%"}),
                                        html.Hr(),
                                        html.Div("Stenosis", style={"fontWeight": 600, "opacity": 0.9}),
                                        html.Label("Stenosis Percentage (0-100%)"),
                                        dcc.Input(id="stenosis-percent", type="number", min=0, max=100, step=0.5, value=0,
                                                  style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                         "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                        html.Small("0% = no stenosis | 100% = complete occlusion",
                                                  style={"fontSize": "11px", "opacity": 0.7, "marginTop": "4px", "display": "block"}),
                                        html.Button("Apply Stenosis", id="apply-stenosis",
                                                    style={"background": "#DC2626", "color": "white",
                                                           "border": "none", "padding": "6px 10px",
                                                           "borderRadius": "6px", "marginTop": "6px", "width": "100%"}),
                                        html.Hr(),
                                        html.Button("Reset", id="reset-btn",
                                                    style={"background": "#2563EB", "color": "white",
                                                           "border": "none", "padding": "6px 10px",
                                                           "borderRadius": "6px", "marginTop": "8px", "width": "100%"}),
                                        html.Div(id="status", style={"fontSize": "12px", "marginTop": "8px"})
                                    ])
                                ]
                            ),
                        ]
                    )
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

    # Callback to update ROI radius slider from mm or voxel inputs
    @app.callback(
        Output("roi-r", "value"),
        Input("roi-r-mm", "value"),
        Input("roi-r-vox", "value"),
        prevent_initial_call=True
    )
    def update_radius_from_inputs(roi_r_mm, roi_r_vox):
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
        
        # If voxel input was triggered and has a valid value
        if "roi-r-vox.value" in triggered and roi_r_vox is not None and isinstance(roi_r_vox, str) and roi_r_vox.strip():
            try:
                r_vox = int(roi_r_vox)
                r_vox = max(0, min(int(max(Nx, Ny, Nz)//2), r_vox))
                return r_vox
            except Exception:
                pass
        
        # Return current slider value if no valid input
        return 1

    # Callback to update radius display values when slider changes
    @app.callback(
        Output("roi-r-mm", "value"),
        Output("roi-r-vox", "value"),
        Input("roi-r", "value"),
        prevent_initial_call=True
    )
    def update_radius_displays(roi_r):
        s = grid["spacing"][0]
        
        if roi_r is not None:
            r_mm = roi_r * s
            return f"{r_mm:.2f}", f"{roi_r}"
        
        return "", ""

    # Callback to sync mask-check to mask-check-ops
    @app.callback(
        Output("mask-check-ops", "value"),
        Input("mask-check", "value"),
        prevent_initial_call=True
    )
    def sync_masks_to_ops(mask_check_value):
        return mask_check_value

    # Callback to sync mask-check-ops to mask-check
    @app.callback(
        Output("mask-check", "value"),
        Input("mask-check-ops", "value"),
        prevent_initial_call=True
    )
    def sync_masks_from_ops(mask_check_ops_value):
        return mask_check_ops_value

    @app.callback(
        Output("x-view", "figure"),
        Output("y-view", "figure"),
        Output("z-view", "figure"),
        Output("threeD-view", "figure"),
        Output("status", "children"),
        Input("mask-check", "value"),
        Input("mask-check-ops", "value"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("roi-x", "value"),
        Input("roi-y", "value"),
        Input("roi-z", "value"),
        Input("roi-r", "value"),
        Input("morph-mode", "value"),
        Input("morph-iters", "value"),
        Input("apply-morph", "n_clicks"),
        Input("stenosis-percent", "value"),
        Input("apply-stenosis", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("reset-btn-view", "n_clicks"),
        prevent_initial_call=False
    )
    def update(mask_values, mask_values_ops, x_idx, y_idx, z_idx,
               roi_x, roi_y, roi_z, roi_r,
               morph_mode, morph_iters, apply_clicks, 
               stenosis_percent, apply_stenosis_clicks, n_clicks, n_clicks_view):
        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        
        # Since mask-check and mask-check-ops are synced, use mask_values
        # (they will always be the same value)
        active_masks = mask_values if mask_values is not None else ["inside", "on", "out"]
        
        if "reset-btn.n_clicks" in triggered or "reset-btn-view.n_clicks" in triggered:
            active_masks = ["inside", "on", "out"]
            x_idx, y_idx, z_idx = x_mid, y_mid, z_mid
            roi_x, roi_y, roi_z, roi_r = x_mid, y_mid, z_mid, 1
            stenosis_percent = 0
            # clear mm inputs on reset (visual clearing would require Outputs; keep placeholders)
            # Reset the working inside mask
            inside_state["inside"] = inside_state["original"].copy()

        # Apply morphology when requested
        if "apply-morph.n_clicks" in triggered:
            try:
                iters = int(morph_iters) if morph_iters is not None else 1
                iters = max(1, iters)
            except Exception:
                iters = 1
            mode = morph_mode if morph_mode in ("dilate", "shrink") else "dilate"

            inside_state["inside"] = apply_roi_morphology(
                inside_state["inside"], roi_x, roi_y, roi_z, roi_r, iterations=iters, mode=mode
            )

        # Apply stenosis when requested
        if "apply-stenosis.n_clicks" in triggered:
            try:
                stenosis_pct = float(stenosis_percent) if stenosis_percent is not None else 0
                stenosis_pct = max(0.0, min(100.0, stenosis_pct))
            except Exception:
                stenosis_pct = 0

            inside_state["inside"] = apply_iterative_stenosis(
                inside_state["inside"], roi_x, roi_y, roi_z, roi_r, stenosis_percent=stenosis_pct
            )

        active = []
        show_inside = "inside" in active_masks
        show_on     = "on"     in active_masks
        show_out    = "out"    in active_masks
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
        roi_traces = _roi_frame(grid, roi_x, roi_y, roi_z, roi_r,
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

        # Compute world-space ROI center (mm) from indices for display
        try:
            s = grid["spacing"][0]
            ox, oy, oz = grid["origin"]
            roi_x_mm_display = ox + roi_x * s
            roi_y_mm_display = oy + roi_y * s
            roi_z_mm_display = oz + roi_z * s
            roi_mm_str = f"({roi_x_mm_display:.2f} mm, {roi_y_mm_display:.2f} mm, {roi_z_mm_display:.2f} mm)"
        except Exception:
            roi_mm_str = "(n/a)"

        status = (
            f"Active masks: {', '.join(active_masks)} | X={x_idx}, Y={y_idx}, Z={z_idx} "
            f"| ROI idx=(i,j,k)=({roi_x},{roi_y},{roi_z}), r={roi_r} vox | ROI mm={roi_mm_str}"
        )
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
    port: int = 8050
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

    # Winding-based masks
    inside_u8, on_u8, out_u8 = classify_by_winding(mesh, grid, band=0.6)

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

    ap = argparse.ArgumentParser(description="Standalone FakeCT pipeline (winding-based)")
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
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open viewer")
    ap.add_argument("--viewer", choices=["dash","html"], default="dash",
                    help="Viewer type: 'dash' or 'html'.")
    ap.add_argument("--port", type=int, default=8050,
                    help="Port for Dash viewer (default 8050).")

    args = ap.parse_args()

    if igl is None:
        err("python-igl is required (fast winding number). Install via conda-forge:\n"
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
        port=args.port
    )

if __name__ == "__main__":
    main()
