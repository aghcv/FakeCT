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
import time
import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple
from scipy.ndimage import binary_dilation
from skimage import measure
import plotly.graph_objects as go
import dash
from dash import dcc, html, Output, Input, State, callback_context
from dash.exceptions import PreventUpdate

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
# Visualization defaults
# ---------------------------------------------------------------------
OPACITY_LEVELS = {
    "low": 0.01,
    "mid": 0.12,
    "high": 0.24
}

SLICE_BORDER_WIDTH = 6

COLOR_SCHEME = {
    "slice_x": "#3B82F6",
    "slice_y": "#22C55E",
    "slice_z": "#F59E0B",
    "inside": "#E11D48",
    "on": "#22C55E",
    "out": "#F59E0B",
    "mesh": "#E11D48",
    "roi_p1": "#60A5FA",
    "roi_p2": "#F472B6",
    "roi_line": "#E5E7EB",
    "roi_sphere": "#A855F7"
}

ROI_DEFAULTS = {
    "p1_opacity": 0.9,
    "p2_opacity": 0.9,
    "line_opacity": 0.8,
    "sphere_opacity": 0.05
}

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
def _slice_frames(grid, i_idx, j_idx, k_idx, color_map=None, line_width=SLICE_BORDER_WIDTH):
    """
    Draw wireframe rectangles showing slice locations (X/Y/Z).
    Colors correspond to sliders: X=blue, Y=green, Z=orange.
    """
    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]
    color_map = color_map or {
        "x": COLOR_SCHEME["slice_x"],
        "y": COLOR_SCHEME["slice_y"],
        "z": COLOR_SCHEME["slice_z"]
    }

    extent_x = Nx * sx
    extent_y = Ny * sy
    extent_z = Nz * sz

    Xr = [ox, ox + extent_x]
    Yr = [oy, oy + extent_y]
    Zr = [oz, oz + extent_z]

    # convert index → voxel center position
    Xpos = ox + (i_idx + 0.5) * sx
    Ypos = oy + (j_idx + 0.5) * sy
    Zpos = oz + (k_idx + 0.5) * sz

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

def mask_to_trimesh(mask_u8: np.ndarray, grid: dict, name: str | None = None) -> trimesh.Trimesh | None:
    """Convert a binary mask to a surface mesh using marching cubes."""
    if mask_u8 is None or int(np.sum(mask_u8)) == 0:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(mask_u8, level=0.5)
    except Exception as e:
        err(f"[{name or 'mask'}] marching cubes failed: {e}")
        return None

    s = grid["spacing"][0]
    origin = np.array(grid["origin"])
    coords = origin + s * verts[:, [2, 1, 0]]
    return trimesh.Trimesh(vertices=coords, faces=faces, process=False)

def save_mask_stl(mask_u8: np.ndarray, grid: dict, out_path: Path, name: str) -> bool:
    """Save a binary mask as STL (returns True when written)."""
    mesh = mask_to_trimesh(mask_u8, grid, name=name)
    if mesh is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh.export(out_path)
    except Exception as e:
        err(f"[{name}] STL export failed: {e}")
        return False
    return True

def _binary_colorscale(on_color="#ffffff"):
    return [[0.0, "black"], [1.0, on_color]]

def _slice_fig(z2d, title, border_color="#ffffff", colorscale=None, zmax=1):
    if colorscale is None:
        colorscale = _binary_colorscale(border_color)
    x_vals = np.arange(z2d.shape[1])
    y_vals = np.arange(z2d.shape[0])
    fig = go.Figure(
        data=[go.Heatmap(
            z=z2d,
            x=x_vals,
            y=y_vals,
            zmin=0,
            zmax=zmax,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z}<extra></extra>"
        )],
        layout=go.Layout(
            title=dict(text=title, font=dict(color=border_color, size=14)),
            margin=dict(l=2, r=2, t=28, b=2),
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

def _empty_slice(grid, axis):
    Nz, Ny, Nx = grid["shape"]
    if axis == "x":
        return np.zeros((Nz, Ny), dtype=np.uint8)
    if axis == "y":
        return np.zeros((Nz, Nx), dtype=np.uint8)
    if axis == "z":
        return np.zeros((Ny, Nx), dtype=np.uint8)
    raise ValueError("axis must be 'x', 'y', or 'z'")

def _roi_slice_mask(grid, axis, idx, center, radius):
    if radius <= 0:
        return _empty_slice(grid, axis)

    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]

    x_coords = ox + (np.arange(Nx) + 0.5) * sx
    y_coords = oy + (np.arange(Ny) + 0.5) * sy
    z_coords = oz + (np.arange(Nz) + 0.5) * sz
    cx, cy, cz = center
    r2 = radius * radius

    if axis == "x":
        x = ox + (idx + 0.5) * sx
        Y, Z = np.meshgrid(y_coords, z_coords)
        dist2 = (x - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        return (dist2 <= r2).astype(np.uint8)
    if axis == "y":
        y = oy + (idx + 0.5) * sy
        X, Z = np.meshgrid(x_coords, z_coords)
        dist2 = (X - cx) ** 2 + (y - cy) ** 2 + (Z - cz) ** 2
        return (dist2 <= r2).astype(np.uint8)
    if axis == "z":
        z = oz + (idx + 0.5) * sz
        X, Y = np.meshgrid(x_coords, y_coords)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (z - cz) ** 2
        return (dist2 <= r2).astype(np.uint8)
    raise ValueError("axis must be 'x', 'y', or 'z'")


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
        name="mesh", color=COLOR_SCHEME["mesh"], opacity=OPACITY_LEVELS["low"]
    )
    # We will recompute the inside trace dynamically to reflect morphology changes
    # inside_trace = mask_to_trace(inside_u8, grid, "#3B82F6", "inside", 0.35)
    on_trace     = mask_to_trace(on_u8,     grid, "#22C55E", "on",     0.55)
    out_trace    = mask_to_trace(out_u8,    grid, "#F59E0B", "out",    0.08)

    # Mutable state for inside mask so we can apply morphology and reset
    inside_state = {"inside": inside_u8.copy(), "original": inside_u8.copy()}

    app = dash.Dash(__name__)
    panel_card = {
        "backgroundColor": "#0f131a",
        "border": "1px solid #232838",
        "borderRadius": "8px",
        "padding": "10px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "8px"
    }
    control_row = {
        "display": "flex",
        "alignItems": "center",
        "gap": "8px"
    }
    color_chip = {
        "width": "32px",
        "height": "24px",
        "padding": "0",
        "border": "1px solid #2b3347",
        "borderRadius": "4px",
        "background": "#111522"
    }
    slider_compact = {
        "flex": "1 1 auto",
        "minWidth": "140px"
    }
    dropdown_compact = {
        "minHeight": "34px",
        "fontSize": "12px"
    }
    button_compact = {
        "height": "30px",
        "padding": "4px 10px",
        "fontSize": "12px"
    }
    slider_css = f"""
    .slider-x .rc-slider-rail {{ background-color: #1f2937; }}
    .slider-y .rc-slider-rail {{ background-color: #1f2937; }}
    .slider-z .rc-slider-rail {{ background-color: #1f2937; }}

    .slider-x .rc-slider-track {{ background-color: {COLOR_SCHEME['slice_x']}; }}
    .slider-y .rc-slider-track {{ background-color: {COLOR_SCHEME['slice_y']}; }}
    .slider-z .rc-slider-track {{ background-color: {COLOR_SCHEME['slice_z']}; }}

    .slider-x .rc-slider-handle {{ border-color: {COLOR_SCHEME['slice_x']}; background-color: {COLOR_SCHEME['slice_x']}; }}
    .slider-y .rc-slider-handle {{ border-color: {COLOR_SCHEME['slice_y']}; background-color: {COLOR_SCHEME['slice_y']}; }}
    .slider-z .rc-slider-handle {{ border-color: {COLOR_SCHEME['slice_z']}; background-color: {COLOR_SCHEME['slice_z']}; }}

    .slider-x .rc-slider-handle:focus,
    .slider-x .rc-slider-handle:hover,
    .slider-x .rc-slider-handle:active {{ box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.35); }}

    .slider-y .rc-slider-handle:focus,
    .slider-y .rc-slider-handle:hover,
    .slider-y .rc-slider-handle:active {{ box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.35); }}

    .slider-z .rc-slider-handle:focus,
    .slider-z .rc-slider-handle:hover,
    .slider-z .rc-slider-handle:active {{ box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.35); }}
    """
    wheel_js = """
    (function() {
        function attachWheel(id, axis) {
            var el = document.getElementById(id);
            if (!el || el.__wheelBound) return;
            el.__wheelBound = true;
            el.addEventListener("wheel", function(e) {
                if (!window.dash_clientside || !window.dash_clientside.set_props) return;
                e.preventDefault();
                var delta = e.deltaY > 0 ? 1 : -1;
                window.dash_clientside.set_props("wheel-store", {data: {axis: axis, delta: delta, ts: Date.now()}});
            }, {passive: false});
        }
        function init() {
            attachWheel("x-view", "x");
            attachWheel("y-view", "y");
            attachWheel("z-view", "z");
        }
        document.addEventListener("DOMContentLoaded", init);
        setInterval(init, 1000);
    })();
    """
    app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
            {slider_css}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
                <script>
                {wheel_js}
                </script>
            </footer>
        </body>
    </html>
    """

    app.layout = html.Div(
        style={"display": "grid", "gridTemplateColumns": "300px 1fr", "gap": "10px",
               "height": "100vh", "backgroundColor": "#0f1115", "color": "#e6e6e6",
               "padding": "10px"},
        children=[
            html.Div(
                style={"backgroundColor": "#151922", "borderRadius": "10px",
                       "padding": "12px", "display": "flex", "flexDirection": "column", "gap": "10px"},
                children=[
                    html.H3("Selection Panel", style={"margin": "0 0 6px 0"}),
                    html.Div(
                        style=panel_card,
                        children=[
                            html.Div("Mask", style={"fontWeight": "600"}),
                            dcc.Dropdown(
                                id="mask-check",
                                options=[
                                    {"label": "Inside", "value": "inside"},
                                    {"label": "On", "value": "on"},
                                    {"label": "Out", "value": "out"},
                                ],
                                value=["inside"],
                                multi=True,
                                placeholder="Select masks",
                                clearable=False,
                                style=dropdown_compact
                            ),
                            html.Button("Reset", id="reset-btn",
                                        style={"background": "#2563EB", "color": "white",
                                               "border": "none", "borderRadius": "6px",
                                               **button_compact}),
                        ]
                    ),
                    html.Div(
                        style=panel_card,
                        children=[
                            html.Div("Slices", style={"fontWeight": "600"}),
                            html.Label("X-slice (i)"),
                            dcc.Slider(id="x-slider", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag", marks=None, className="slider-x"),
                            html.Label("Y-slice (j)"),
                            dcc.Slider(id="y-slider", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag", marks=None, className="slider-y"),
                            html.Label("Z-slice (k)"),
                            dcc.Slider(id="z-slider", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag", marks=None, className="slider-z"),
                        ]
                    ),
                    html.Hr(),
                    dcc.Tabs(id="tools-tabs", value="roi", children=[
                        dcc.Tab(label="ROI", value="roi", children=[
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
                                    dcc.Tab(label="Center + Radius", value="center", children=[
                                        html.Label("ROI X (i)"),
                                        dcc.Slider(id="roi-x", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag"),
                                        html.Label("ROI Y (j)"),
                                        dcc.Slider(id="roi-y", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                        html.Label("ROI Z (k)"),
                                        dcc.Slider(id="roi-z", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                        html.Label("ROI Radius (voxels)"),
                                        dcc.Slider(id="roi-r", min=0, max=int(max(Nx, Ny, Nz)//2), step=1, value=1, updatemode="drag",
                                                   tooltip={"always_visible": False, "placement": "bottom"}),
                                    ]),
                                    dcc.Tab(label="Two Points", value="twopoints", children=[
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
                        dcc.Tab(label="Operations", value="ops", children=[
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
                    dcc.Graph(id="z-view"), dcc.Graph(id="threeD-view"),
                    dcc.Store(id="camera-store"),
                    dcc.Store(id="wheel-store"),
                    dcc.Store(id="hover-store"),
                    dcc.Store(id="catch-store"),
                    dcc.Store(id="fakect-store")
                ]
            )
        ]
    )

    @app.callback(
        Output("x-slider", "value"),
        Output("y-slider", "value"),
        Output("z-slider", "value"),
        Input("wheel-store", "data"),
        State("x-slider", "value"),
        State("y-slider", "value"),
        State("z-slider", "value"),
        prevent_initial_call=True
    )
    def update_sliders_from_wheel(data, x_idx, y_idx, z_idx):
        if not data:
            raise PreventUpdate
        axis = data.get("axis")
        try:
            delta = int(data.get("delta", 0))
        except (TypeError, ValueError):
            raise PreventUpdate
        if delta == 0:
            raise PreventUpdate

        if axis == "x":
            x_idx = int(np.clip(x_idx + delta, 0, Nx - 1))
        elif axis == "y":
            y_idx = int(np.clip(y_idx + delta, 0, Ny - 1))
        elif axis == "z":
            z_idx = int(np.clip(z_idx + delta, 0, Nz - 1))
        else:
            raise PreventUpdate

        return x_idx, y_idx, z_idx

    @app.callback(
        Output("roi-scale-pos", "value"),
        Output("roi-scale-input", "value"),
        Output("roi-scale-display", "children"),
        Input("roi-scale-pos", "value"),
        Input("roi-scale-input", "value"),
        prevent_initial_call=False
    )
    def sync_roi_scale(scale_pos, scale_input):
        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        src = triggered[0]["prop_id"].split(".")[0]

        if src == "roi-scale-input":
            try:
                val = float(scale_input)
            except (TypeError, ValueError):
                val = _pos_to_scale(scale_pos if scale_pos is not None else 0.5)
            val = max(0.001, min(99.999, val))
            display_val = val
            return float(_scale_to_pos(val)), float(val), f"Current: {display_val:.3f}"

        pos = scale_pos if scale_pos is not None else 0.5
        display_val = _pos_to_scale(pos)
        return float(pos), float(min(99.999, max(0.001, display_val))), f"Current: {display_val:.3f}"

    @app.callback(
        Output("shape-window", "value"),
        Input("shape-window", "value"),
        Input("shape-symmetric", "value"),
        prevent_initial_call=True
    )
    def enforce_symmetric_window(window_vals, symmetric_vals):
        if not window_vals or len(window_vals) != 2:
            raise PreventUpdate
        symmetric = "on" in (symmetric_vals or [])
        if not symmetric:
            return window_vals
        left, right = window_vals
        left = max(0.0, min(1.0, float(left)))
        right = max(0.0, min(1.0, float(right)))
        half_width = abs(0.5 - left)
        new_left = max(0.0, 0.5 - half_width)
        new_right = min(1.0, 0.5 + half_width)
        return [new_left, new_right]

    @app.callback(
        Output("roi-batch-status", "children"),
        Input("roi-batch-export", "n_clicks"),
        State("p1-x", "value"),
        State("p1-y", "value"),
        State("p1-z", "value"),
        State("p2-x", "value"),
        State("p2-y", "value"),
        State("p2-z", "value"),
        State("roi-batch-scale-min", "value"),
        State("roi-batch-scale-max", "value"),
        State("roi-batch-scale-count", "value"),
        State("roi-batch-k-min", "value"),
        State("roi-batch-k-max", "value"),
        State("roi-batch-k-count", "value"),
        State("shape-window", "value"),
        State("roi-batch-out-dir", "value"),
        State("roi-batch-stl", "value"),
        prevent_initial_call=True
    )
    def export_batch_masks(n_clicks,
                           p1_x, p1_y, p1_z,
                           p2_x, p2_y, p2_z,
                           scale_min, scale_max, scale_count,
                           k_min, k_max, k_count,
                           shape_window, out_dir, export_stl_flags):
        if not n_clicks:
            raise PreventUpdate

        center_idx = (
            int(np.clip(round((p1_x + p2_x) / 2.0), 0, Nx - 1)),
            int(np.clip(round((p1_y + p2_y) / 2.0), 0, Ny - 1)),
            int(np.clip(round((p1_z + p2_z) / 2.0), 0, Nz - 1))
        )
        radius_vox = int(round(0.5 * float(np.linalg.norm(
            np.array([p2_x - p1_x, p2_y - p1_y, p2_z - p1_z], dtype=float)
        ))))

        scales = _linspace_safe(scale_min, scale_max, scale_count)
        ks = _linspace_safe(k_min, k_max, k_count)
        w0, w1 = (0.25, 0.75)
        if shape_window and len(shape_window) == 2:
            try:
                w0, w1 = float(shape_window[0]), float(shape_window[1])
            except (TypeError, ValueError):
                w0, w1 = (0.25, 0.75)
        w0 = max(0.0, min(1.0, w0))
        w1 = max(0.0, min(1.0, w1))
        if w1 <= w0:
            w0, w1 = (0.25, 0.75)

        out_dir = out_dir or "outputs/stenosis_batch"
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        total = 0
        stl_total = 0
        base_inside = inside_state["original"]
        for sf in scales:
            sf = max(0.001, min(99.999, float(sf)))
            for k in ks:
                k = max(0.1, float(k))
                mask = apply_roi_scale(
                    base_inside,
                    center_idx,
                    radius_vox,
                    sf,
                    shape_k=k,
                    shape_window=(w0, w1)
                )
                base_name = (
                    f"inside_sf{_fmt_param(sf)}_k{_fmt_param(k)}"
                    f"_w{_fmt_param(w0, 2)}-{_fmt_param(w1, 2)}.npz"
                )
                np.savez_compressed(
                    out_path / base_name,
                    inside=mask.astype(np.uint8),
                    spacing=np.array(grid["spacing"]),
                    origin=np.array(grid["origin"]),
                    scale_factor=np.array(sf),
                    shape_k=np.array(k),
                    shape_window=np.array([w0, w1]),
                    roi_center_ijk=np.array(center_idx),
                    roi_radius_vox=np.array(radius_vox)
                )
                export_stl = "on" in (export_stl_flags or [])
                if export_stl:
                    stl_name = base_name.replace(".npz", ".stl")
                    if save_mask_stl(mask, grid, out_path / stl_name, name="inside"):
                        stl_total += 1
                total += 1

        if stl_total > 0:
            return (
                f"Export done. Wrote {total} masks and {stl_total} STL files to {out_path} "
                f"(scales={len(scales)}, k={len(ks)})."
            )
        return (
            f"Export done. Wrote {total} masks to {out_path} "
            f"(scales={len(scales)}, k={len(ks)})."
        )

    @app.callback(
        Output("fakect-status", "children"),
        Output("fakect-store", "data"),
        Input("fakect-apply", "n_clicks"),
        State("fakect-model", "value"),
        State("fakect-model-path", "value"),
        State("fakect-mask-source", "value"),
        State("fakect-axis", "value"),
        State("fakect-context-step", "value"),
        prevent_initial_call=True
    )
    def apply_fakect(n_clicks, model_dropdown, model_path, mask_source, axis, context_step):
        if not n_clicks:
            raise PreventUpdate

        model_path = (model_path or "").strip() or (model_dropdown or "").strip()
        if not model_path:
            return "Select a fakenoise model path.", None

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            return f"Model not found: {path}", None

        try:
            import tensorflow as tf
        except Exception:
            return "TensorFlow is not available. Install tensorflow to run FakeCT.", None

        try:
            if fakect_state["model_path"] != str(path):
                fakect_state["model"] = tf.keras.models.load_model(str(path))
                fakect_state["model_path"] = str(path)
            model = fakect_state["model"]
            th, tw, c = _infer_keras_input(model)
        except Exception as e:
            return f"Failed to load model: {e}", None

        if mask_source == "on":
            mask_vol = on_u8
        elif mask_source == "out":
            mask_vol = out_u8
        else:
            mask_vol = inside_state["current"]

        try:
            fakect_state["target_hw"] = (th, tw)
            fakect_state["input_channels"] = c
            fakect_state["context_step"] = int(context_step or 1)
            fakect_state["mask_source"] = mask_source or "inside"
            fakect_state["volume"] = _predict_fake_volume(
                mask_vol,
                model,
                (th, tw),
                c,
                axis=axis or "i",
                context_step=fakect_state["context_step"]
            )
        except Exception as e:
            return f"FakeCT inference failed: {e}", None

        channel_note = ""
        if c > 1 and (c - 1) % 2 != 0:
            channel_note = " (non-symmetric channel count)"
        axis_label = (axis or "i").lower()
        msg = f"FakeCT ready: {path.name} | axis={axis_label} | input={th}x{tw}x{c}{channel_note}"
        return msg, {"ts": time.time()}

    @app.callback(
        Output("x-view", "figure"),
        Output("y-view", "figure"),
        Output("z-view", "figure"),
        Output("threeD-view", "figure"),
        Output("camera-store", "data"),
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

        p1 = _voxel_center(grid, p1_x, p1_y, p1_z)
        p2 = _voxel_center(grid, p2_x, p2_y, p2_z)
        center = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)
        radius = 0.5 * float(np.linalg.norm(np.array(p2) - np.array(p1)))

        if len(active) > 0:
            x_slice = compose_slice(active, "x", x_idx)
            y_slice = compose_slice(active, "y", y_idx)
            z_slice = compose_slice(active, "z", z_idx)
        else:
            x_slice = _empty_slice(grid, "x")
            y_slice = _empty_slice(grid, "y")
            z_slice = _empty_slice(grid, "z")

        roi_x = _roi_slice_mask(grid, "x", x_idx, center, radius)
        roi_y = _roi_slice_mask(grid, "y", y_idx, center, radius)
        roi_z = _roi_slice_mask(grid, "z", z_idx, center, radius)

        x_fig = _slice_fig(
            x_slice,
            f"X-slice (i={x_idx})",
            COLOR_SCHEME["slice_x"],
            colorscale=_slice_colorscale(COLOR_SCHEME["slice_x"]),
            zmax=1
        )
        y_fig = _slice_fig(
            y_slice,
            f"Y-slice (j={y_idx})",
            COLOR_SCHEME["slice_y"],
            colorscale=_slice_colorscale(COLOR_SCHEME["slice_y"]),
            zmax=1
        )
        z_fig = _slice_fig(
            z_slice,
            f"Z-slice (k={z_idx})",
            COLOR_SCHEME["slice_z"],
            colorscale=_slice_colorscale(COLOR_SCHEME["slice_z"]),
            zmax=1
        )

        show_fakect = "on" in (fakect_show or [])
        fake_vol = fakect_state.get("volume")
        if show_fakect and isinstance(fake_vol, np.ndarray) and fake_vol.shape == inside_state["current"].shape:
            alpha = max(0.0, min(1.0, float(fakect_alpha if fakect_alpha is not None else 0.6)))
            fake_x = fake_vol[:, :, x_idx]
            fake_y = fake_vol[:, y_idx, :]
            fake_z = fake_vol[z_idx, :, :]
            overlay = dict(colorscale=_binary_colorscale("#ffffff"), showscale=False, opacity=alpha, zmin=0, zmax=1, hoverinfo="skip")
            x_fig.add_trace(go.Heatmap(z=fake_x, **overlay))
            y_fig.add_trace(go.Heatmap(z=fake_y, **overlay))
            z_fig.add_trace(go.Heatmap(z=fake_z, **overlay))

        x_fig.add_trace(_roi_overlay_trace(roi_x, sphere_color, sphere_opacity))
        y_fig.add_trace(_roi_overlay_trace(roi_y, sphere_color, sphere_opacity))
        z_fig.add_trace(_roi_overlay_trace(roi_z, sphere_color, sphere_opacity))

        # Rebuild inside surface dynamically from current inside state
        dyn_inside_trace = mask_to_trace(inside_state["inside"], grid, "#3B82F6", "inside", 0.35) if show_inside else None

        valid_traces = [t for t in [mesh_trace,
                                    dyn_inside_trace,
                                    on_trace     if show_on     else None,
                                    out_trace    if show_out    else None] if t is not None]
        extent_x, extent_y, extent_z = grid["extent_mm"]
        aspectratio = _aspectratio_from_extents((extent_x, extent_y, extent_z))

        fig3d = go.Figure(data=valid_traces)
        plane_traces = [
            _slice_plane_surface(
                x_slice,
                grid,
                "x",
                x_idx,
                COLOR_SCHEME["slice_x"],
                colorscale=_slice_plane_colorscale(COLOR_SCHEME["slice_x"], sphere_color, OPACITY_LEVELS["high"]),
                cmax=2
            ),
            _slice_plane_surface(
                y_slice,
                grid,
                "y",
                y_idx,
                COLOR_SCHEME["slice_y"],
                colorscale=_slice_plane_colorscale(COLOR_SCHEME["slice_y"], sphere_color, OPACITY_LEVELS["high"]),
                cmax=2
            ),
            _slice_plane_surface(
                z_slice,
                grid,
                "z",
                z_idx,
                COLOR_SCHEME["slice_z"],
                colorscale=_slice_plane_colorscale(COLOR_SCHEME["slice_z"], sphere_color, OPACITY_LEVELS["high"]),
                cmax=2
            )
        ]
        for plane in plane_traces:
            if plane is not None:
                fig3d.add_trace(plane)
        for frame in _slice_frames(grid, x_idx, y_idx, z_idx):
            fig3d.add_trace(frame)

        # ROI sphere (wireframe)
        roi_traces = _roi_frame(grid, eff_roi_x, eff_roi_y, eff_roi_z, eff_roi_r,
                                n_lat=6, n_lon=8, color="#E11D48", line_width=3)
        for t in roi_traces:
            fig3d.add_trace(t)
        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            uirevision=uirev,
            scene=dict(
                uirevision=uirev,
                bgcolor="#000000",
                aspectmode="manual",
                aspectratio=aspectratio,
                xaxis=dict(title="X (mm)", range=[-extent_x/2, extent_x/2], showgrid=False, showspikes=False),
                yaxis=dict(title="Y (mm)", range=[-extent_y/2, extent_y/2], showgrid=False, showspikes=False),
                zaxis=dict(title="Z (mm)", range=[-extent_z/2, extent_z/2], showgrid=False, showspikes=False),
            )
        )
        if camera_state:
            fig3d.update_layout(scene_camera=camera_state)

        return x_fig, y_fig, z_fig, fig3d, camera_state

    @app.callback(
        Output("hover-store", "data"),
        Input("x-view", "hoverData"),
        Input("y-view", "hoverData"),
        Input("z-view", "hoverData"),
        State("x-slider", "value"),
        State("y-slider", "value"),
        State("z-slider", "value"),
        prevent_initial_call=True
    )
    def update_hover_store(x_hover, y_hover, z_hover, x_idx, y_idx, z_idx):
        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        prop = triggered[0]["prop_id"]
        data = None
        if prop.startswith("x-view"):
            data = x_hover
            axis = "x"
        elif prop.startswith("y-view"):
            data = y_hover
            axis = "y"
        elif prop.startswith("z-view"):
            data = z_hover
            axis = "z"
        else:
            raise PreventUpdate

        if not data or "points" not in data or not data["points"]:
            raise PreventUpdate

        pt = data["points"][0]
        try:
            xv = int(round(pt.get("x", 0)))
            yv = int(round(pt.get("y", 0)))
        except (TypeError, ValueError):
            raise PreventUpdate

        if axis == "x":
            i_idx = x_idx
            j_idx = int(np.clip(xv, 0, Ny - 1))
            k_idx = int(np.clip(yv, 0, Nz - 1))
        elif axis == "y":
            i_idx = int(np.clip(xv, 0, Nx - 1))
            j_idx = y_idx
            k_idx = int(np.clip(yv, 0, Nz - 1))
        else:
            i_idx = int(np.clip(xv, 0, Nx - 1))
            j_idx = int(np.clip(yv, 0, Ny - 1))
            k_idx = z_idx

        return {"axis": axis, "i": i_idx, "j": j_idx, "k": k_idx, "ts": data.get("event", {}).get("timeStamp", None)}

    @app.callback(
        Output("catch-store", "data"),
        Input("x-view", "clickData"),
        Input("y-view", "clickData"),
        Input("z-view", "clickData"),
        State("x-slider", "value"),
        State("y-slider", "value"),
        State("z-slider", "value"),
        prevent_initial_call=True
    )
    def update_catch_store(x_click, y_click, z_click, x_idx, y_idx, z_idx):
        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        prop = triggered[0]["prop_id"]
        data = None
        if prop.startswith("x-view"):
            data = x_click
            axis = "x"
        elif prop.startswith("y-view"):
            data = y_click
            axis = "y"
        elif prop.startswith("z-view"):
            data = z_click
            axis = "z"
        else:
            raise PreventUpdate

        if not data or "points" not in data or not data["points"]:
            raise PreventUpdate

        pt = data["points"][0]
        try:
            xv = int(round(pt.get("x", 0)))
            yv = int(round(pt.get("y", 0)))
        except (TypeError, ValueError):
            raise PreventUpdate

        if axis == "x":
            i_idx = x_idx
            j_idx = int(np.clip(xv, 0, Ny - 1))
            k_idx = int(np.clip(yv, 0, Nz - 1))
        elif axis == "y":
            i_idx = int(np.clip(xv, 0, Nx - 1))
            j_idx = y_idx
            k_idx = int(np.clip(yv, 0, Nz - 1))
        else:
            i_idx = int(np.clip(xv, 0, Nx - 1))
            j_idx = int(np.clip(yv, 0, Ny - 1))
            k_idx = z_idx

        return {"axis": axis, "i": i_idx, "j": j_idx, "k": k_idx}

    @app.callback(
        Output("p1-x", "value"),
        Output("p1-y", "value"),
        Output("p1-z", "value"),
        Input("p1-set-btn", "n_clicks"),
        State("catch-store", "data"),
        State("p1-x", "value"),
        State("p1-y", "value"),
        State("p1-z", "value"),
        prevent_initial_call=True
    )
    def set_p1_from_catch(n_clicks, catch_data, p1_x, p1_y, p1_z):
        if not n_clicks or not catch_data:
            raise PreventUpdate
        return catch_data.get("i", p1_x), catch_data.get("j", p1_y), catch_data.get("k", p1_z)

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

    if export_stl:
        stl_root = Path(stl_dir) if stl_dir else Path(out_npz).parent
        stl_root.mkdir(parents=True, exist_ok=True)
        save_mask_stl(inside_u8, grid, stl_root / "inside.stl", name="inside")
        save_mask_stl(on_u8, grid, stl_root / "on.stl", name="on")
        save_mask_stl(out_u8, grid, stl_root / "out.stl", name="out")
        ok(f"STL export complete → {stl_root}")

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
    ap.add_argument("--export-stl", action="store_true",
                    help="Export inside/on/out masks as STL files.")
    ap.add_argument("--stl-dir", default=None,
                    help="Directory for STL export (defaults to --out folder).")

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
