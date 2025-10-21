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
    inside_trace = mask_to_trace(inside_u8, grid, "#3B82F6", "inside", 0.35)
    on_trace     = mask_to_trace(on_u8,     grid, "#22C55E", "on",     0.55)
    out_trace    = mask_to_trace(out_u8,    grid, "#F59E0B", "out",    0.08)

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
                    html.Label("X-slice (i)"),
                    dcc.Slider(id="x-slider", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag"),
                    html.Label("Y-slice (j)"),
                    dcc.Slider(id="y-slider", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                    html.Label("Z-slice (k)"),
                    dcc.Slider(id="z-slider", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
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
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=False
    )
    def update(mask_values, x_idx, y_idx, z_idx, n_clicks):
        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        if "reset-btn.n_clicks" in triggered:
            mask_values = ["inside", "on", "out"]
            x_idx, y_idx, z_idx = x_mid, y_mid, z_mid

        active = []
        show_inside = "inside" in mask_values
        show_on     = "on"     in mask_values
        show_out    = "out"    in mask_values
        if show_inside: active.append(inside_u8)
        if show_on:     active.append(on_u8)
        if show_out:    active.append(out_u8)

        x_fig = _slice_fig(compose_slice(active, "x", x_idx), f"X-slice (i={x_idx})", "#3B82F6")
        y_fig = _slice_fig(compose_slice(active, "y", y_idx), f"Y-slice (j={y_idx})", "#22C55E")
        z_fig = _slice_fig(compose_slice(active, "z", z_idx), f"Z-slice (k={z_idx})", "#F59E0B")

        valid_traces = [t for t in [mesh_trace,
                                    inside_trace if show_inside else None,
                                    on_trace     if show_on     else None,
                                    out_trace    if show_out    else None] if t is not None]
        extent_x, extent_y, extent_z = grid["extent_mm"]
        aspectratio = _aspectratio_from_extents((extent_x, extent_y, extent_z))

        fig3d = go.Figure(data=valid_traces)
        for frame in _slice_frames(grid, x_idx, y_idx, z_idx):
            fig3d.add_trace(frame)
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

        status = f"Active masks: {', '.join(mask_values)} | X={x_idx}, Y={y_idx}, Z={z_idx}"
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
