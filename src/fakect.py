#!/usr/bin/env python3
# ---------------------------------------------------------------------
# fakect.py
# Produces winding-based in/on/out masks and (optionally) launches a Dash viewer.
#
# Usage examples:
#   python fakect.py --in cube.stl --n 8 --out cube_masks.npz
#   python fakect.py --in carotid.stl --n 9 --margin 0.10 --out examples/outputs/carotid_masks.npz
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

    WN = igl.fast_winding_number_for_meshes(V, F, Q)  # (N^3,)
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

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(255,255,255,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _slice_colorscale(mask_color: str) -> list:
    return [[0.0, "black"], [1.0, mask_color]]

def _slice_plane_colorscale(mask_color: str, roi_color: str, alpha: float) -> list:
    return [
        [0.0, "rgba(0,0,0,0)"],
        [0.5, _hex_to_rgba(mask_color, alpha)],
        [1.0, _hex_to_rgba(roi_color, alpha)]
    ]

def _roi_overlay_trace(roi_mask: np.ndarray, color: str, opacity: float) -> go.Heatmap:
    z = np.where(roi_mask > 0, 1.0, np.nan)
    return go.Heatmap(
        z=z,
        zmin=0,
        zmax=1,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        hoverinfo="skip",
        opacity=max(0.0, min(1.0, float(opacity)))
    )

def _voxel_center(grid, i_idx, j_idx, k_idx):
    sx, sy, sz = grid["spacing"]
    ox, oy, oz = grid["origin"]
    x = ox + (i_idx + 0.5) * sx
    y = oy + (j_idx + 0.5) * sy
    z = oz + (k_idx + 0.5) * sz
    return x, y, z

def _sphere_surface(center, radius, color, opacity, resolution=24):
    if radius <= 0:
        return None
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros_like(x),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        name="ROI sphere",
        hoverinfo="skip"
    )

def _slice_plane_surface(
    slice2d,
    grid,
    axis,
    idx,
    color,
    alpha=OPACITY_LEVELS["high"],
    colorscale=None,
    cmax=1
):
    """
    Add a semi-transparent slice plane colored by the same mask as the 2D view.
    slice2d shapes (Z,Y) for x, (Z,X) for y, (Y,X) for z.
    """
    if slice2d.size == 0:
        return None

    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]

    x_coords = ox + (np.arange(Nx) + 0.5) * sx
    y_coords = oy + (np.arange(Ny) + 0.5) * sy
    z_coords = oz + (np.arange(Nz) + 0.5) * sz

    if axis == "x":
        Y, Z = np.meshgrid(y_coords, z_coords)
        X = np.full_like(Y, ox + (idx + 0.5) * sx)
    elif axis == "y":
        X, Z = np.meshgrid(x_coords, z_coords)
        Y = np.full_like(X, oy + (idx + 0.5) * sy)
    elif axis == "z":
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = np.full_like(X, oz + (idx + 0.5) * sz)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    if colorscale is None:
        colorscale = [
            [0.0, "rgba(0,0,0,0)"],
            [1.0, _hex_to_rgba(color, alpha)]
        ]

    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=slice2d,
        cmin=0, cmax=cmax,
        colorscale=colorscale,
        showscale=False,
        name=f"{axis}-slice-plane",
        hoverinfo="skip",
        opacity=1.0
    )

def mask_to_trace(mask_u8, grid, color, name, opacity=OPACITY_LEVELS["high"]):
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

def _scale_to_pos(scale_val: float) -> float:
    """Map scale value in [0.001, 99.999] to slider position in [0, 1] with 1.0 at 0.5."""
    v = max(0.001, min(99.999, float(scale_val)))
    if v <= 1.0:
        return 0.5 * (np.log10(v) - np.log10(0.001)) / (np.log10(1.0) - np.log10(0.001))
    return 0.5 + 0.5 * (np.log10(v) - np.log10(1.0)) / (np.log10(99.999) - np.log10(1.0))

def _pos_to_scale(pos: float) -> float:
    """Map slider position in [0, 1] to scale value in [0.001, 99.999] with 1.0 at 0.5."""
    p = max(0.0, min(1.0, float(pos)))
    if p <= 0.5:
        t = p / 0.5
        return float(10 ** (np.log10(0.001) + t * (np.log10(1.0) - np.log10(0.001))))
    t = (p - 0.5) / 0.5
    return float(10 ** (np.log10(1.0) + t * (np.log10(99.999) - np.log10(1.0))))

def _linspace_safe(vmin: float, vmax: float, count: int) -> list:
    """Generate a safe list of values for batch export (count >= 1)."""
    try:
        vmin = float(vmin)
        vmax = float(vmax)
    except (TypeError, ValueError):
        return [1.0]
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 1
    count = max(1, count)
    if count == 1:
        return [vmin]
    return list(np.linspace(vmin, vmax, count))

def _fmt_param(val: float, decimals: int = 3) -> str:
    """Format a float for filenames (avoid dots)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        v = 0.0
    return f"{v:.{decimals}f}".replace(".", "p")

def _gaussian_shape(x: float, k: float) -> float:
    """Normalized Gaussian shape with S(0)=0, S(0.5)=1, S(1)=0."""
    k = max(1e-6, float(k))
    s = np.exp(-k * (x - 0.5) ** 2)
    s0 = np.exp(-k * 0.25)
    denom = max(1e-8, 1.0 - s0)
    val = (s - s0) / denom
    return float(max(0.0, min(1.0, val)))

def _roi_mask_vox(shape_zyx, center_ijk, r_vox):
    """Return a boolean spherical ROI mask in voxel index space."""
    if r_vox is None or r_vox <= 0:
        return np.zeros(shape_zyx, dtype=bool)
    Nz, Ny, Nx = shape_zyx
    i0, j0, k0 = center_ijk
    ii = np.arange(Nx) - int(i0)
    jj = np.arange(Ny) - int(j0)
    kk = np.arange(Nz) - int(k0)
    rr2 = (ii * ii)[None, None, :] + (jj * jj)[None, :, None] + (kk * kk)[:, None, None]
    return rr2 <= (int(r_vox) * int(r_vox))

def _bitwise_dilate6(mask_bool: np.ndarray) -> np.ndarray:
    """Single-iteration 3D dilation using 6-neighborhood via bitwise shifts."""
    zpos = np.pad(mask_bool[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=False)
    zneg = np.pad(mask_bool[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=False)
    ypos = np.pad(mask_bool[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=False)
    yneg = np.pad(mask_bool[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=False)
    xpos = np.pad(mask_bool[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=False)
    xneg = np.pad(mask_bool[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=False)
    return mask_bool | zpos | zneg | ypos | yneg | xpos | xneg

def _bitwise_erode6(mask_bool: np.ndarray) -> np.ndarray:
    """Single-iteration 3D erosion using 6-neighborhood via bitwise shifts."""
    zpos = np.pad(mask_bool[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=False)
    zneg = np.pad(mask_bool[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=False)
    ypos = np.pad(mask_bool[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=False)
    yneg = np.pad(mask_bool[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=False)
    xpos = np.pad(mask_bool[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=False)
    xneg = np.pad(mask_bool[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=False)
    return mask_bool & zpos & zneg & ypos & yneg & xpos & xneg

def proximity_bridge(mask_u8: np.ndarray, max_dist: int = 8) -> np.ndarray:
    """Bridge disconnected components within a small Chebyshev distance."""
    mask = (mask_u8 > 0)
    if max_dist <= 0:
        return mask_u8

    size = 2 * max_dist + 1
    kernel = np.ones((size, size, size), dtype=bool)
    dil = binary_dilation(mask, structure=kernel)
    bridge = dil & (~mask)

    labeled, num = measure.label(mask.astype(np.uint8), return_num=True, connectivity=1)
    if num <= 1:
        return mask.astype(np.uint8)

    Z, Y, X = np.where(bridge)
    out = mask.copy()
    for z, y, x in zip(Z, Y, X):
        neigh = []
        if z > 0:
            neigh.append(labeled[z - 1, y, x])
        if z < labeled.shape[0] - 1:
            neigh.append(labeled[z + 1, y, x])
        if y > 0:
            neigh.append(labeled[z, y - 1, x])
        if y < labeled.shape[1] - 1:
            neigh.append(labeled[z, y + 1, x])
        if x > 0:
            neigh.append(labeled[z, y, x - 1])
        if x < labeled.shape[2] - 1:
            neigh.append(labeled[z, y, x + 1])

        neigh = [n for n in neigh if n != 0]
        if len(set(neigh)) >= 2:
            out[z, y, x] = True

    return out.astype(np.uint8)

def apply_roi_scale(
    inside_u8: np.ndarray,
    center_ijk: tuple[int, int, int],
    r_vox: int,
    scale_factor: float,
    bridge_dist: int = 2,
    shape_k: float = 10.0,
    shape_window: tuple[float, float] = (0.25, 0.75),
) -> np.ndarray:
    """Scale vessel diameter within a spherical ROI using bitwise morphology."""
    if inside_u8 is None:
        return inside_u8
    try:
        sf = float(scale_factor)
    except (TypeError, ValueError):
        return inside_u8

    r0 = int(round(r_vox))
    if r0 <= 0 or sf == 1.0:
        return inside_u8

    # Convert scale factor to voxel diameter space for 1-voxel iterations.
    base_diam = max(1, int(round(2 * r0)))
    target_diam = max(0, int(round(base_diam * sf)))
    r1 = int(round(target_diam / 2.0))
    if r1 == r0:
        return inside_u8

    inside_bool = (inside_u8 > 0)
    if r1 <= 0:
        r1 = 1

    # Ensure target ROI contains at least one voxel of volume.
    roi_target = _roi_mask_vox(inside_u8.shape, center_ijk, r1)
    if np.sum(roi_target) < 1:
        r1 = 1
        roi_target = _roi_mask_vox(inside_u8.shape, center_ijk, r1)

    # Step ROI size toward target, applying one morphology iteration per step.
    mode = "dilate" if r1 > r0 else "erode"
    step = 1 if r1 > r0 else -1
    r_current = r0
    current_inside = inside_bool.copy()
    last_valid_inside = current_inside.copy()
    min_roi_fraction = 0.05
    initial_roi = _roi_mask_vox(current_inside.shape, center_ijk, r_current)
    initial_count = int(np.sum(current_inside & initial_roi))
    min_inside = max(1, int(round(initial_count * min_roi_fraction)))
    total_steps = abs(r1 - r0)
    step_idx = 0
    accum = 0.0

    while r_current != r1:
        roi_current = _roi_mask_vox(current_inside.shape, center_ijk, r_current)
        inside_in_roi = current_inside & roi_current
        outside = current_inside & (~roi_current)

        x = 0.5
        if total_steps > 1:
            x = step_idx / float(total_steps - 1)

        w0, w1 = shape_window
        w0 = max(0.0, min(1.0, float(w0)))
        w1 = max(0.0, min(1.0, float(w1)))
        if w1 <= w0:
            w0, w1 = 0.25, 0.75
        if x < w0 or x > w1:
            s = 0.0
        else:
            x = (x - w0) / (w1 - w0)
            s = _gaussian_shape(x, shape_k)

        accum += s
        apply_step = accum >= 1.0
        if apply_step:
            accum -= 1.0

        work = inside_in_roi
        if apply_step:
            if mode == "dilate":
                work = _bitwise_dilate6(inside_in_roi)
            else:
                work = _bitwise_erode6(inside_in_roi)

        # Clamp to current ROI for this iteration.
        work = work & roi_current
        merged = (outside | work).astype(np.uint8)
        if bridge_dist and bridge_dist > 0:
            merged = proximity_bridge(merged, max_dist=bridge_dist)
        current_inside = (merged > 0)

        # Safety: never erase the ROI completely during erosion.
        if mode == "erode":
            roi_after = _roi_mask_vox(current_inside.shape, center_ijk, r_current)
            if int(np.sum(current_inside & roi_after)) < min_inside:
                current_inside = last_valid_inside
                break
            last_valid_inside = current_inside.copy()

        r_current += step
        step_idx += 1

    return current_inside.astype(np.uint8)

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
    inside_trace = mask_to_trace(inside_u8, grid, COLOR_SCHEME["inside"], "inside", OPACITY_LEVELS["mid"])
    on_trace     = mask_to_trace(on_u8,     grid, COLOR_SCHEME["on"],     "on",     OPACITY_LEVELS["low"])
    out_trace    = mask_to_trace(out_u8,    grid, COLOR_SCHEME["out"],    "out",    OPACITY_LEVELS["low"])

    inside_state = {
        "current": inside_u8.copy(),
        "original": inside_u8.copy(),
        "trace": inside_trace
    }

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
                    html.Div(
                        style=panel_card,
                        children=[
                            dcc.Tabs(
                                id="tool-tabs",
                                value="roi",
                                children=[
                                    dcc.Tab(
                                        label="ROI",
                                        value="roi",
                                        children=[
                                            html.Div(
                                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "padding": "4px 0"},
                                                children=[
                                                    html.Div("ROI", style={"fontWeight": "600"}),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            html.Label("P1 (i, j, k)", style={"flex": "1 1 auto"}),
                                                            html.Button("Set from view", id="p1-set-btn",
                                                                        style={"background": "#1f2937", "color": "#e5e7eb",
                                                                               "border": "1px solid #2b3347", "borderRadius": "6px",
                                                                               **button_compact})
                                                        ]
                                                    ),
                                                    dcc.Slider(id="p1-x", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag", marks=None, className="slider-x"),
                                                    dcc.Slider(id="p1-y", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag", marks=None, className="slider-y"),
                                                    dcc.Slider(id="p1-z", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag", marks=None, className="slider-z"),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            dcc.Input(id="p1-color", type="color", value=COLOR_SCHEME["roi_p1"], style=color_chip),
                                                            html.Div("Opacity", style={"fontSize": "11px", "minWidth": "54px"}),
                                                            html.Div(
                                                                dcc.Slider(id="p1-opacity", min=0.05, max=1.0, step=0.05, value=ROI_DEFAULTS["p1_opacity"], updatemode="drag", marks=None),
                                                                style=slider_compact
                                                            )
                                                        ]
                                                    ),
                                                    html.Div(
                                                        style={**control_row, "marginTop": "6px"},
                                                        children=[
                                                            html.Label("P2 (i, j, k)", style={"flex": "1 1 auto"}),
                                                            html.Button("Set from view", id="p2-set-btn",
                                                                        style={"background": "#1f2937", "color": "#e5e7eb",
                                                                               "border": "1px solid #2b3347", "borderRadius": "6px",
                                                                               **button_compact})
                                                        ]
                                                    ),
                                                    dcc.Slider(id="p2-x", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag", marks=None, className="slider-x"),
                                                    dcc.Slider(id="p2-y", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag", marks=None, className="slider-y"),
                                                    dcc.Slider(id="p2-z", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag", marks=None, className="slider-z"),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            dcc.Input(id="p2-color", type="color", value=COLOR_SCHEME["roi_p2"], style=color_chip),
                                                            html.Div("Opacity", style={"fontSize": "11px", "minWidth": "54px"}),
                                                            html.Div(
                                                                dcc.Slider(id="p2-opacity", min=0.05, max=1.0, step=0.05, value=ROI_DEFAULTS["p2_opacity"], updatemode="drag", marks=None),
                                                                style=slider_compact
                                                            )
                                                        ]
                                                    ),
                                                    html.Label("ROI line", style={"marginTop": "6px"}),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            dcc.Input(id="roi-line-color", type="color", value=COLOR_SCHEME["roi_line"], style=color_chip),
                                                            html.Div("Opacity", style={"fontSize": "11px", "minWidth": "54px"}),
                                                            html.Div(
                                                                dcc.Slider(id="roi-line-opacity", min=0.05, max=1.0, step=0.05, value=ROI_DEFAULTS["line_opacity"], updatemode="drag", marks=None),
                                                                style=slider_compact
                                                            )
                                                        ]
                                                    ),
                                                    html.Label("ROI sphere", style={"marginTop": "6px"}),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            dcc.Input(id="roi-sphere-color", type="color", value=COLOR_SCHEME["roi_sphere"], style=color_chip),
                                                            html.Div("Opacity", style={"fontSize": "11px", "minWidth": "54px"}),
                                                            html.Div(
                                                                dcc.Slider(id="roi-sphere-opacity", min=0.05, max=1.0, step=0.05, value=ROI_DEFAULTS["sphere_opacity"], updatemode="drag", marks=None),
                                                                style=slider_compact
                                                            )
                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    ),
                                    dcc.Tab(
                                        label="Stenosis",
                                        value="stenosis",
                                        children=[
                                            html.Div(
                                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "padding": "4px 0"},
                                                children=[
                                                    html.Div("ROI Morphology", style={"fontWeight": "600"}),
                                                    html.Div("Scale factor (diameter)", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    html.Div(id="roi-scale-display", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    dcc.Slider(
                                                        id="roi-scale-pos",
                                                        min=0.0,
                                                        max=1.0,
                                                        step=0.001,
                                                        value=0.5,
                                                        updatemode="drag",
                                                        marks=None
                                                    ),
                                                    dcc.Input(
                                                        id="roi-scale-input",
                                                        type="number",
                                                        min=0.001,
                                                        max=99.999,
                                                        step=0.001,
                                                        value=1.000,
                                                        style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                               "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                               "padding": "4px 6px"}
                                                    ),
                                                    html.Div("Gaussian profile", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    html.Div(
                                                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                                                        children=[
                                                            html.Div(
                                                                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                                                                children=[
                                                                    html.Div("Steepness k", style={"fontSize": "12px", "opacity": "0.85"}),
                                                                    html.Span("i", title="Steepness for Gaussian profile (larger k = sharper).",
                                                                              style={"fontSize": "11px", "opacity": "0.7", "cursor": "help"})
                                                                ]
                                                            ),
                                                            dcc.Input(
                                                                id="shape-k",
                                                                type="number",
                                                                min=0.1,
                                                                max=100.0,
                                                                step=0.1,
                                                                value=10.0,
                                                                style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                       "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                       "padding": "4px 6px"}
                                                            )
                                                        ]
                                                    ),
                                                    html.Div("Transition window", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    dcc.RangeSlider(
                                                        id="shape-window",
                                                        min=0.0,
                                                        max=1.0,
                                                        step=0.01,
                                                        value=[0.25, 0.75],
                                                        allowCross=False,
                                                        marks=None
                                                    ),
                                                    dcc.Checklist(
                                                        id="shape-symmetric",
                                                        options=[{"label": " Symmetric window", "value": "on"}],
                                                        value=["on"],
                                                        inputStyle={"marginRight": "6px"}
                                                    ),
                                                    html.Div("< 1 erosion, > 1 dilation", style={"fontSize": "11px", "opacity": "0.7"}),
                                                    html.Button("Apply", id="roi-scale-apply",
                                                                style={"background": "#16A34A", "color": "white",
                                                                       "border": "none", "borderRadius": "6px",
                                                                       **button_compact})
                                                    ,
                                                    html.Div("Batch export", style={"fontWeight": "600", "marginTop": "8px"}),
                                                    html.Div("Output folder", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    dcc.Input(
                                                        id="roi-batch-out-dir",
                                                        type="text",
                                                        value="outputs/stenosis_batch",
                                                        style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                               "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                               "padding": "4px 6px"}
                                                    ),
                                                    html.Div("Scale factor range", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    html.Div(
                                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "6px",
                                                               "fontSize": "11px", "opacity": "0.7"},
                                                        children=["min", "max", "count"]
                                                    ),
                                                    html.Div(
                                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "6px"},
                                                        children=[
                                                            dcc.Input(id="roi-batch-scale-min", type="number", value=0.8, step=0.01,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"}),
                                                            dcc.Input(id="roi-batch-scale-max", type="number", value=1.2, step=0.01,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"}),
                                                            dcc.Input(id="roi-batch-scale-count", type="number", value=5, step=1,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"})
                                                        ]
                                                    ),
                                                    html.Div("Gaussian k range", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    html.Div(
                                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "6px",
                                                               "fontSize": "11px", "opacity": "0.7"},
                                                        children=["min", "max", "count"]
                                                    ),
                                                    html.Div(
                                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "6px"},
                                                        children=[
                                                            dcc.Input(id="roi-batch-k-min", type="number", value=6.0, step=0.1,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"}),
                                                            dcc.Input(id="roi-batch-k-max", type="number", value=14.0, step=0.1,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"}),
                                                            dcc.Input(id="roi-batch-k-count", type="number", value=5, step=1,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"})
                                                        ]
                                                    ),
                                                    html.Button("Export batch", id="roi-batch-export",
                                                                style={"background": "#0EA5E9", "color": "white",
                                                                       "border": "none", "borderRadius": "6px",
                                                                       **button_compact}),
                                                    dcc.Checklist(
                                                        id="roi-batch-stl",
                                                        options=[{"label": " Export STL", "value": "on"}],
                                                        value=["on"],
                                                        inputStyle={"marginRight": "6px"}
                                                    ),
                                                    html.Div(id="roi-batch-status", style={"fontSize": "12px", "opacity": "0.85"})
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(id="status", style={"fontSize": "12px", "marginTop": "4px"})
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
                    dcc.Store(id="catch-store")
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
        Output("x-view", "figure"),
        Output("y-view", "figure"),
        Output("z-view", "figure"),
        Output("threeD-view", "figure"),
        Output("camera-store", "data"),
        Input("mask-check", "value"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("p1-x", "value"),
        Input("p1-y", "value"),
        Input("p1-z", "value"),
        Input("p2-x", "value"),
        Input("p2-y", "value"),
        Input("p2-z", "value"),
        Input("p1-color", "value"),
        Input("p2-color", "value"),
        Input("p1-opacity", "value"),
        Input("p2-opacity", "value"),
        Input("roi-line-color", "value"),
        Input("roi-line-opacity", "value"),
         Input("roi-sphere-color", "value"),
         Input("roi-sphere-opacity", "value"),
        Input("roi-scale-pos", "value"),
        Input("roi-scale-apply", "n_clicks"),
        Input("shape-k", "value"),
        Input("shape-window", "value"),
        Input("threeD-view", "relayoutData"),
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=False
    )
    def update(mask_values, x_idx, y_idx, z_idx,
               p1_x, p1_y, p1_z,
               p2_x, p2_y, p2_z,
               p1_color, p2_color,
               p1_opacity, p2_opacity,
               line_color, line_opacity,
             sphere_color, sphere_opacity,
             scale_pos, scale_apply_clicks,
             shape_k,
             shape_window,
             relayout_data, n_clicks):
        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        reset_triggered = "reset-btn.n_clicks" in triggered
        if reset_triggered:
            inside_state["current"] = inside_state["original"].copy()
            inside_state["trace"] = mask_to_trace(
                inside_state["current"],
                grid,
                COLOR_SCHEME["inside"],
                "inside",
                OPACITY_LEVELS["mid"]
            )

        center_idx = (
            int(np.clip(round((p1_x + p2_x) / 2.0), 0, Nx - 1)),
            int(np.clip(round((p1_y + p2_y) / 2.0), 0, Ny - 1)),
            int(np.clip(round((p1_z + p2_z) / 2.0), 0, Nz - 1))
        )
        radius_vox = int(round(0.5 * float(np.linalg.norm(
            np.array([p2_x - p1_x, p2_y - p1_y, p2_z - p1_z], dtype=float)
        ))))

        if (not reset_triggered) and ("roi-scale-apply.n_clicks" in triggered):
            sf = _pos_to_scale(scale_pos if scale_pos is not None else 0.5)
            sf = max(0.001, min(99.999, sf))
            inside_state["current"] = apply_roi_scale(
                inside_state["current"],
                center_idx,
                radius_vox,
                sf,
                shape_k=10.0 if shape_k is None else float(shape_k),
                shape_window=(shape_window[0], shape_window[1]) if shape_window else (0.25, 0.75)
            )
            inside_state["trace"] = mask_to_trace(
                inside_state["current"],
                grid,
                COLOR_SCHEME["inside"],
                "inside",
                OPACITY_LEVELS["mid"]
            )

        active = []
        show_inside = "inside" in mask_values
        show_on     = "on"     in mask_values
        show_out    = "out"    in mask_values
        if show_inside: active.append(inside_state["current"])
        if show_on:     active.append(on_u8)
        if show_out:    active.append(out_u8)

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

        x_fig.add_trace(_roi_overlay_trace(roi_x, sphere_color, sphere_opacity))
        y_fig.add_trace(_roi_overlay_trace(roi_y, sphere_color, sphere_opacity))
        z_fig.add_trace(_roi_overlay_trace(roi_z, sphere_color, sphere_opacity))

        valid_traces = [t for t in [mesh_trace,
                        inside_state["trace"] if show_inside else None,
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
        fig3d.add_trace(go.Scatter3d(
            x=[p1[0]], y=[p1[1]], z=[p1[2]],
            mode="markers",
            marker=dict(size=6, color=p1_color, opacity=p1_opacity),
            name="P1",
            showlegend=False,
            hoverinfo="skip"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[p2[0]], y=[p2[1]], z=[p2[2]],
            mode="markers",
            marker=dict(size=6, color=p2_color, opacity=p2_opacity),
            name="P2",
            showlegend=False,
            hoverinfo="skip"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color=line_color, width=4),
            opacity=line_opacity,
            name="ROI line",
            showlegend=False,
            hoverinfo="skip"
        ))
        sphere = _sphere_surface(center, radius, sphere_color, sphere_opacity)
        if sphere is not None:
            fig3d.add_trace(sphere)
        uirev = 0 if n_clicks is None else n_clicks
        camera_state = None
        if isinstance(relayout_data, dict):
            if "scene.camera" in relayout_data:
                camera_state = relayout_data.get("scene.camera")
            elif any(k.startswith("scene.camera") for k in relayout_data.keys()):
                camera_state = {
                    k.split("scene.camera.", 1)[-1]: v
                    for k, v in relayout_data.items()
                    if k.startswith("scene.camera.")
                }
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

    @app.callback(
        Output("p2-x", "value"),
        Output("p2-y", "value"),
        Output("p2-z", "value"),
        Input("p2-set-btn", "n_clicks"),
        State("catch-store", "data"),
        State("p2-x", "value"),
        State("p2-y", "value"),
        State("p2-z", "value"),
        prevent_initial_call=True
    )
    def set_p2_from_catch(n_clicks, catch_data, p2_x, p2_y, p2_z):
        if not n_clicks or not catch_data:
            raise PreventUpdate
        return catch_data.get("i", p2_x), catch_data.get("j", p2_y), catch_data.get("k", p2_z)

    @app.callback(
        Output("status", "children"),
        Input("mask-check", "value"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("p1-x", "value"),
        Input("p1-y", "value"),
        Input("p1-z", "value"),
        Input("p2-x", "value"),
        Input("p2-y", "value"),
        Input("p2-z", "value"),
        Input("hover-store", "data"),
        Input("catch-store", "data"),
        prevent_initial_call=False
    )
    def update_status(mask_values, x_idx, y_idx, z_idx,
                      p1_x, p1_y, p1_z, p2_x, p2_y, p2_z,
                      hover_data, catch_data):
        hover_txt = "Hover: -"
        if isinstance(hover_data, dict):
            hover_txt = f"Hover: ({hover_data.get('i')},{hover_data.get('j')},{hover_data.get('k')})"
        catch_txt = "Catch: -"
        if isinstance(catch_data, dict):
            catch_txt = f"Catch: ({catch_data.get('i')},{catch_data.get('j')},{catch_data.get('k')})"

        return (
            f"Active masks: {', '.join(mask_values)} | X={x_idx}, Y={y_idx}, Z={z_idx}"
            f" | P1=({p1_x},{p1_y},{p1_z}) P2=({p2_x},{p2_y},{p2_z})"
            f" | {hover_txt} | {catch_txt}"
        )

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
    export_stl: bool = False,
    stl_dir: str | None = None
):
    mesh_path = Path(in_mesh_path).resolve()
    if not mesh_path.exists():
        err(f"Input mesh not found: {in_mesh_path}")
        raise FileNotFoundError(in_mesh_path)

    mesh = trimesh.load(mesh_path, force="mesh")
    info(f"Loaded mesh: {mesh_path.name} | watertight={getattr(mesh, 'is_watertight', 'unknown')}")

    # Center mesh at origin
    mesh = center_mesh(mesh)

    # Grid (power-of-two cubic grid)
    grid = make_cube_grid_from_mesh(mesh, spacing=spacing, n=n, margin_frac=margin_frac)

    # Winding-based masks
    inside_u8, on_u8, out_u8 = classify_by_winding(mesh, grid, band=0.6)

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
    ap.add_argument("--export-stl", action="store_true",
                    help="Export inside/on/out masks as STL files.")
    ap.add_argument("--stl-dir", default=None,
                    help="Directory for STL export (defaults to --out folder).")

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
        port=args.port,
        export_stl=bool(args.export_stl),
        stl_dir=args.stl_dir
    )

if __name__ == "__main__":
    main()
