# src/fakect/core.py
# Mesh → power-of-two cubic grid derived from mesh AABB → inside/on/out masks → Viewer
# Examples:
#   python -m fakect.core --in data/carotid.stl --spacing 1.0 --mc-map xyz --viewer dash
#   python -m fakect.core --in data/carotid.stl --n 7 --margin 0.10 --mc-map xyz --viewer html

from __future__ import annotations
import os
import argparse
import numpy as np
import trimesh
from scipy.ndimage import binary_erosion
from skimage import measure
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
from typing import Tuple, List
from pathlib import Path
from fakect.fill_parity import classify_by_parity_multi, classify_by_winding

# ---------------------------
# Utilities
# ---------------------------

def _grid_extents_display(grid, mc_map: str):
    """
    Return (extent_X, extent_Y, extent_Z) in mm in the *displayed* axes,
    consistent with apply_axis_map() and your mc_map string.
    """
    Nz, Ny, Nx = grid["shape"]    # mask index order is (z, y, x)
    sz, sy, sx = grid["spacing"]  # voxel size per index axis
    ext = {"z": Nz * sz, "y": Ny * sy, "x": Nx * sx}

    # apply_axis_map for mc_map:
    #   "xyz" : X<-z, Y<-y, Z<-x
    #   "zyx" : X<-x, Y<-y, Z<-z
    #   "xzy" : X<-z, Y<-x, Z<-y
    #   "yxz" : X<-y, Y<-z, Z<-x
    #   "yzx" : X<-y, Y<-x, Z<-z
    #   "zxy" : X<-x, Y<-z, Z<-y
    mapping = {
        "zyx": ("x", "y", "z"),
        "xyz": ("z", "y", "x"),
        "xzy": ("z", "x", "y"),
        "yxz": ("y", "z", "x"),
        "yzx": ("y", "x", "z"),
        "zxy": ("x", "z", "y"),
    }
    a, b, c = mapping.get(mc_map, ("z", "y", "x"))
    return (ext[a], ext[b], ext[c])

def _aspectratio_from_extents(extents_xyz):
    """Normalize to avoid giant numbers; largest dimension → 1."""
    mx = max(extents_xyz)
    if mx <= 0:
        return dict(x=1, y=1, z=1)
    x, y, z = (e / mx for e in extents_xyz)
    return dict(x=x, y=y, z=z)

def next_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()

def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh or Scene and concatenate parts if necessary."""
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    parts = getattr(mesh, "geometry", {}).values()
    if not parts:
        raise ValueError("Scene has no geometry parts")
    return trimesh.util.concatenate(tuple(parts))

def is_closed(mesh: trimesh.Trimesh) -> bool:
    return bool(mesh.is_watertight)

def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    return m

# NOTE: demo mesh generation helpers were intentionally removed.
# Inputs must be provided by the user or placed in the repository `data/` directory.

# ---------------------------
# Grid construction (derived from mesh AABB)
# ---------------------------
def make_cube_grid_from_mesh(
    mesh: trimesh.Trimesh,
    spacing: float | None,
    n: int | None,
    margin_frac: float = 0.10,
    min_margin_voxels: int = 3
) -> dict:
    """
    Build a cubic grid safely larger than the mesh AABB.
    Guarantees at least `min_margin_voxels` empty voxels between geometry and grid boundaries,
    even when spacing is inferred from n.
    """
    bounds = mesh.bounds
    Lx, Ly, Lz = (bounds[1] - bounds[0]).astype(float)
    Lmax = float(max(Lx, Ly, Lz))

    if spacing is not None:
        # --- explicit voxel pitch ---
        margin_abs = max(Lmax * margin_frac, min_margin_voxels * spacing)
        raw = int(np.ceil((Lmax + 2 * margin_abs) / spacing))
        N = next_pow2(raw)
        s = float(spacing)
    else:
        # --- implicit voxel pitch determined from n ---
        if n is None:
            n = 7
        N = 2 ** int(n)

        # initial estimate for spacing (using fractional margin only)
        s = (Lmax * (1.0 + 2.0 * margin_frac)) / N

        # recompute margin to ensure min voxel safety
        min_margin_mm = min_margin_voxels * s
        margin_abs = max(Lmax * margin_frac, min_margin_mm)

        # adjust spacing again so total extent matches new margin
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

    print(f"[grid] spacing={s:.4f} mm | N={N} | safe margin={margin_abs:.3f} mm "
          f"≈ {margin_abs/s:.1f} voxels per side")

    return grid

# ---------------------------
# Voxelization and masks
# ---------------------------
def fit_to_shape_centered(vol: np.ndarray, target_shape: Tuple[int, int, int]):
    """
    Center-pad or crop a 3D volume to match target shape (Z,Y,X).
    Returns (out, (oz, oy, ox)) where (oz,oy,ox) are the index offsets the original 'vol' is placed at.
    """
    out = np.zeros(target_shape, dtype=vol.dtype)
    sz, sy, sx = vol.shape
    tz, ty, tx = target_shape

    oz = max((tz - sz) // 2, 0)
    oy = max((ty - sy) // 2, 0)
    ox = max((tx - sx) // 2, 0)

    z0 = max((sz - tz) // 2, 0)
    y0 = max((sy - ty) // 2, 0)
    x0 = max((sx - tx) // 2, 0)

    z1, y1, x1 = z0 + min(sz, tz), y0 + min(sy, ty), x0 + min(sx, tx)
    out[oz:oz + (z1 - z0), oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = vol[z0:z1, y0:y1, x0:x1]
    return out, (oz, oy, ox)

def voxelize_mesh(mesh: trimesh.Trimesh, grid: dict):
    """
    Voxelize mesh into boolean volume (True = inside) with isotropic pitch equal to spacing.
    Returns:
      inside_bool  : centered boolean volume (Z,Y,X)
      voxel_xform  : (4x4) affine mapping from *padded array* index (z,y,x,1) to world (X,Y,Z,1)
    """
    s = grid["spacing"][0]
    vg = mesh.voxelized(pitch=s)             # VoxelGrid with its own transform (index -> world)
    vol = vg.matrix.astype(bool)             # shape (sz, sy, sx) in original voxel frame

    # Center into our target cubic grid and record where we placed it
    inside_bool, (oz, oy, ox) = fit_to_shape_centered(vol, grid["shape"])

    s = grid["spacing"][0]
    origin = np.array(grid["origin"])
    voxel_xform = np.eye(4)
    voxel_xform[:3, :3] = np.diag([s, s, s])
    voxel_xform[:3, 3] = origin

    return inside_bool, voxel_xform

def classify_in_on_out(inside_bool: np.ndarray):
    """
    Compute inside, on (1-voxel shell), and out masks with perfect complementarity.
    Returns uint8 arrays where inside + on + out == 1 everywhere.
    """
    eroded = binary_erosion(inside_bool, structure=np.ones((3, 3, 3)), iterations=1)
    on = inside_bool ^ eroded

    inside_u8 = inside_bool.astype(np.uint8)
    on_u8 = on.astype(np.uint8)

    # Out = 1 - (inside ∪ on), implemented explicitly for robustness
    out_u8 = np.ones_like(inside_u8, dtype=np.uint8)
    out_u8[(inside_u8 == 1) | (on_u8 == 1)] = 0

    return inside_u8, on_u8, out_u8

# ---------------------------
# Axis mapping (world z,y,x -> Plotly X,Y,Z)
# ---------------------------
def apply_axis_map(verts: np.ndarray, origin: Tuple[float,float,float], mc_map: str):
    z, y, x = verts[:, 0], verts[:, 1], verts[:, 2]
    ox, oy, oz = origin
    if mc_map == "zyx":
        X, Y, Z = x + ox, y + oy, z + oz
    elif mc_map == "xyz":
        X, Y, Z = z + ox, y + oy, x + oz
    elif mc_map == "xzy":
        X, Y, Z = z + ox, x + oy, y + oz
    elif mc_map == "yxz":
        X, Y, Z = y + ox, z + oy, x + oz
    elif mc_map == "yzx":
        X, Y, Z = y + ox, x + oy, z + oz
    elif mc_map == "zxy":
        X, Y, Z = x + ox, z + oy, y + oz
    else:
        X, Y, Z = z + ox, y + oy, x + oz
    return X, Y, Z

# ---------------------------
# Plotly 3-D traces (used by both viewers)
# ---------------------------
def mask_to_trace(mask_u8, grid, name, color, opacity, mc_map, transform=None):
    """
    Convert binary mask (uint8) to Plotly Mesh3d trace with axis remap.
    Marching cubes is done WITHOUT spacing (index units), then mapped via transform.
    """
    if mask_u8 is None or not np.any(mask_u8):
        return None

    # marching_cubes returns vertices in (z, y, x)
    verts_idx, faces, _, _ = measure.marching_cubes(mask_u8, level=0.5)

    # Reorder to (x, y, z) before applying the transform
    verts_idx = verts_idx[:, [2, 1, 0]]

    if transform is not None:
        homog = np.c_[verts_idx, np.ones(len(verts_idx))]
        verts_world = (transform @ homog.T).T[:, :3]
    else:
        s = grid["spacing"][0]
        origin = np.array(grid["origin"])
        verts_world = origin + s * verts_idx

    X, Y, Z = apply_axis_map(verts_world, (0, 0, 0), mc_map)
    i, j, k = faces.T
    return go.Mesh3d(
        x=X, y=Y, z=Z, i=i, j=j, k=k,
        name=name, color=color, opacity=opacity,
        flatshading=True, showscale=False
    )


def mesh_to_trace(mesh: trimesh.Trimesh, name="mesh", color="#AB1616", opacity=0.15, mc_map="xyz"):
    if mesh is None or mesh.vertices.size == 0:
        return None
    v = mesh.vertices
    # Apply the same axis mapping used for masks so mesh & masks align 1:1
    X, Y, Z = apply_axis_map(v, (0, 0, 0), mc_map)
    f = mesh.faces
    return go.Mesh3d(
        x=X, y=Y, z=Z,
        i=f[:,0], j=f[:,1], k=f[:,2],
        name=name, color=color, opacity=opacity, showscale=False
    )

# ---------------------------
# HTML viewer (legacy)
# ---------------------------
def show_viewer_html(*, mesh, inside_u8, on_u8, out_u8, grid, voxel_transform,
                     html_path="outputs/viewer.html", mc_map="xyz"):
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    color_map = {"mesh":"#AB1616", "inside":"#3B82F6", "on":"#22C55E", "out":"#F59E0B"}

    traces = [
        mesh_to_trace(mesh, name="mesh", color=color_map["mesh"], opacity=0.15, mc_map=mc_map),
        mask_to_trace(inside_u8, grid, "inside", color_map["inside"], 0.35, mc_map, transform=voxel_transform),
        mask_to_trace(on_u8, grid, "on", color_map["on"], 0.55, mc_map, transform=voxel_transform),
        mask_to_trace(out_u8, grid, "out", color_map["out"], 0.08, mc_map, transform=voxel_transform),
    ]
    traces = [t for t in traces if t]

    fig = go.Figure(data=traces)
    extents_xyz = _grid_extents_display(grid, mc_map)
    aspectratio = _aspectratio_from_extents(extents_xyz)
    fig.update_layout(
        title="Mesh + in/on/out masks (3-D)",
        scene=dict(
            xaxis=dict(title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            zaxis=dict(title="Z (mm)"),
            aspectmode="data",
            #aspectratio=aspectratio
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="All",    method="update", args=[{"visible": [True]*len(list(fig.data))}]),
                dict(label="Inside", method="update", args=[{"visible": [getattr(t, "name", None) in ('mesh','inside') for t in fig.data]}]),
                dict(label="On",     method="update", args=[{"visible": [getattr(t, "name", None) in ('mesh','on') for t in fig.data]}]),
                dict(label="Out",    method="update", args=[{"visible": [getattr(t, "name", None) in ('mesh','out') for t in fig.data]}]),
                dict(label="None",   method="update", args=[{"visible": [getattr(t, "name", None) == 'mesh' for t in fig.data]}]),
            ],
            direction="right", x=0.0, y=1.05, xanchor="left", yanchor="bottom", pad={"r": 4, "t": 2}
        )]
    )

    pio.write_html(fig, html_path, auto_open=False, include_plotlyjs=True, full_html=True)
    try:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception:
        pass
    print(f"Viewer saved to {html_path}")

# ---------------------------
# Dash viewer (orthogonal slices + 3-D, synced)
# ---------------------------
def _binary_colorscale():
    # 0 -> black, 1 -> white
    return [[0.0, "black"], [1.0, "white"]]

def _slice_fig(z2d: np.ndarray, title: str,
               border_color: str = "#ffffff", border_width: int = 4):
    """
    Draw a 2-D binary slice with a colored border to indicate its axis.
    """
    fig = go.Figure(
        data=[go.Heatmap(
            z=z2d, zmin=0, zmax=1,
            colorscale=_binary_colorscale(),
            showscale=False
        )],
        layout=go.Layout(
            title=title,
            margin=dict(l=border_width, r=border_width,
                        t=30, b=border_width),
            xaxis=dict(showticklabels=False),
            yaxis=dict(autorange="reversed", showticklabels=False),
            plot_bgcolor="black",
            paper_bgcolor="black"
        )
    )

    # add border rectangle with colored lines
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        xref="paper", yref="paper",
        line=dict(color=border_color, width=border_width),
        fillcolor="rgba(0,0,0,0)"
    )

    return fig

def _compose_slice(masks: List[np.ndarray], axis: str, idx: int) -> np.ndarray:
    """
    Compose multiple binary masks into a single binary slice (logical OR),
    so the panel remains black/white while mask toggles still reflect what's shown.
    axis in {'i','j','k'}
    """
    if len(masks) == 0:
        # Return a zeros array with shape matching a typical mask slice
        # Assuming masks are expected to be 3D arrays, use a default shape (e.g., (1, 1))
        return np.zeros((1, 1), dtype=np.uint8)
    if axis == 'i':
        # YZ view at x=idx
        slices = [m[:, :, idx] for m in masks]
    elif axis == 'j':
        # XZ view at y=idx
        slices = [m[:, idx, :] for m in masks]
    elif axis == 'k':
        # XY view at z=idx
        slices = [m[idx, :, :] for m in masks]
    else:
        raise ValueError("axis must be one of {'i','j','k'}")
    out = np.zeros_like(slices[0], dtype=np.uint8)
    for s in slices:
        out |= (s > 0).astype(np.uint8)
    return out

def show_viewer_dash(*, mesh, inside_u8, on_u8, out_u8, grid, voxel_transform, mc_map="xyz", port=8050):
    """
    Launch a Dash app with a left tool bar (mask toggles + 3 sliders + reset)
    and a 2x2 panel (I,J,K, and 3-D). All panels are synced to the same state.
    """
    import dash
    from dash import dcc, html, Output, Input, State, callback_context

    Nz, Ny, Nx = inside_u8.shape
    i_mid, j_mid, k_mid = Nx // 2, Ny // 2, Nz // 2

    # Precompute static 3-D traces (we'll toggle visibility via callbacks)
    mesh_trace = mesh_to_trace(mesh, name="mesh", color="#AB1616", opacity=0.15, mc_map=mc_map)
    inside_3d = mask_to_trace(inside_u8, grid, "inside", "#3B82F6", 0.35, mc_map, transform=voxel_transform)
    on_3d     = mask_to_trace(on_u8,     grid, "on",     "#22C55E", 0.55, mc_map, transform=voxel_transform)
    out_3d    = mask_to_trace(out_u8,    grid, "out",    "#F59E0B", 0.08, mc_map, transform=voxel_transform)

    def make_3d_fig(show_inside=True, show_on=True, show_out=True):
        traces = [t for t in [mesh_trace,
                            inside_3d if show_inside else None,
                            on_3d     if show_on else None,
                            out_3d    if show_out else None] if t is not None]
        fig = go.Figure(data=traces)

        extents_xyz = _grid_extents_display(grid, mc_map)      # displayed-axis extents
        aspectratio = _aspectratio_from_extents(extents_xyz)

        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=10),
            scene=dict(
                bgcolor="#000000",  # full black background
                xaxis=dict(
                    title="X (mm)",
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    backgroundcolor="#000000"
                ),
                yaxis=dict(
                    title="Y (mm)",
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    backgroundcolor="#000000"
                ),
                zaxis=dict(
                    title="Z (mm)",
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    backgroundcolor="#000000"
                ),
                aspectmode="data"
            )
,
            showlegend=False
        )
        return fig
    
    app = dash.Dash(__name__)

    app.layout = html.Div(
        style={"display": "grid", "gridTemplateColumns": "320px 1fr", "gap": "12px",
               "height": "100vh", "padding": "10px", "boxSizing": "border-box",
               "backgroundColor": "#0f1115", "color": "#e6e6e6"},
        children=[
            # --- Tool bar ---
            html.Div(
                style={"backgroundColor": "#151922", "borderRadius": "12px", "padding": "14px",
                       "display": "flex", "flexDirection": "column", "gap": "12px",
                       "boxShadow": "0 2px 10px rgba(0,0,0,0.35)"},
                children=[
                    html.H3("Tools", style={"margin":"0 0 8px 0"}),
                    html.Div([
                        html.Label("Masks", style={"fontWeight":"600"}),
                        dcc.Checklist(
                            id="mask-check",
                            options=[
                                {"label":" Inside", "value":"inside"},
                                {"label":" On", "value":"on"},
                                {"label":" Out", "value":"out"},
                            ],
                            value=["inside","on","out"],
                            inputStyle={"marginRight":"6px"},
                            labelStyle={"display":"block","marginBottom":"4px"}
                        ),
                    ]),
                    html.Hr(style={"borderColor":"#22293a"}),
                    html.Label("I-slice (x index)", style={"fontWeight":"600"}),
                    dcc.Slider(id="i-slider", min=0, max=Nx-1, step=1, value=i_mid,
                               tooltip={"always_visible": False}, updatemode="drag"),
                    html.Label("J-slice (y index)", style={"fontWeight":"600", "marginTop":"8px"}),
                    dcc.Slider(id="j-slider", min=0, max=Ny-1, step=1, value=j_mid,
                               tooltip={"always_visible": False}, updatemode="drag"),
                    html.Label("K-slice (z index)", style={"fontWeight":"600", "marginTop":"8px"}),
                    dcc.Slider(id="k-slider", min=0, max=Nz-1, step=1, value=k_mid,
                               tooltip={"always_visible": False}, updatemode="drag"),
                    html.Div(style={"display":"flex","gap":"8px","marginTop":"10px"}, children=[
                        html.Button("Reset view", id="reset-btn",
                                    style={"background":"#2d6cdf","border":"none","color":"white",
                                           "padding":"8px 10px","borderRadius":"8px","cursor":"pointer"}),
                    ]),
                    html.Div(id="status", style={"fontSize":"12px","opacity":0.8})
                ]
            ),

            # --- 4-panels area ---
            html.Div(
                style={"display":"grid", "gridTemplateColumns":"1fr 1fr",
                       "gridTemplateRows":"1fr 1fr", "gap":"10px", "height":"100%"},
                children=[
                    dcc.Graph(id="i-graph", style={"height":"100%"}),
                    dcc.Graph(id="j-graph", style={"height":"100%"}),
                    dcc.Graph(id="k-graph", style={"height":"100%"}),
                    dcc.Graph(id="threeD-graph", style={"height":"100%"}),
                ]
            )
        ]
    )

    # --- Callbacks ---
    def _slice_frames(grid, i_idx, j_idx, k_idx, mc_map,
                  color_map=None, line_width=4):
        """
        Draw wireframe rectangles showing slice locations
        (i: X-slice/Z direction, j: Y-slice, k: Z-slice/X direction).
        Colors correspond to i/j/k sliders.
        """
        Nz, Ny, Nx = grid["shape"]
        sz, sy, sx = grid["spacing"]
        ox, oy, oz = grid["origin"]
        color_map = color_map or {"i": "#3B82F6", "j": "#22C55E", "k": "#F59E0B"}

        # Convert voxel indices → physical coordinates (remember: array order z,y,x)
        Xpos = oz + k_idx * sz   # k-slider → X
        Ypos = oy + j_idx * sy   # j-slider → Y
        Zpos = ox + i_idx * sx   # i-slider → Z

        frames = []

        def rect(x, y, z, color, name):
            return go.Scatter3d(
                x=x + [x[0]],
                y=y + [y[0]],
                z=z + [z[0]],
                mode="lines",
                line=dict(color=color, width=line_width),
                name=name,
                showlegend=False
            )

        # Physical extents of each axis
        Xr = [oz, oz + Nz * sz]
        Yr = [oy, oy + Ny * sy]
        Zr = [ox, ox + Nx * sx]

        # --- I-slice (constant Zpos) spans X–Y plane ---
        frames.append(rect(
            x=[Xr[0], Xr[1], Xr[1], Xr[0]],
            y=[Yr[0], Yr[0], Yr[1], Yr[1]],
            z=[Zpos]*4,
            color=color_map["i"], name="i-frame"
        ))

        # --- J-slice (constant Ypos) spans X–Z plane ---
        frames.append(rect(
            x=[Xr[0], Xr[1], Xr[1], Xr[0]],
            y=[Ypos]*4,
            z=[Zr[0], Zr[0], Zr[1], Zr[1]],
            color=color_map["j"], name="j-frame"
        ))

        # --- K-slice (constant Xpos) spans Y–Z plane (✅ fixed) ---
        frames.append(rect(
            x=[Xpos]*4,
            y=[Yr[0], Yr[0], Yr[1], Yr[1]],
            z=[Zr[0], Zr[1], Zr[1], Zr[0]],
            color=color_map["k"], name="k-frame"
        ))

        return frames

    @app.callback(
        Output("i-graph","figure"),
        Output("j-graph","figure"),
        Output("k-graph","figure"),
        Output("threeD-graph","figure"),
        Output("status","children"),
        Input("mask-check","value"),
        Input("i-slider","value"),
        Input("j-slider","value"),
        Input("k-slider","value"),
        Input("reset-btn","n_clicks"),
        prevent_initial_call=False
    )
    def update_all(mask_values, i_idx, j_idx, k_idx, n_clicks):
        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        reset = ("reset-btn.n_clicks" in triggered)

        # Handle Reset: mid slices + all masks on
        if reset:
            mask_values = ["inside","on","out"]
            i_idx, j_idx, k_idx = i_mid, j_mid, k_mid

        # Which masks are active?
        active_masks = []
        show_inside = "inside" in mask_values
        show_on     = "on"     in mask_values
        show_out    = "out"    in mask_values
        if show_inside: active_masks.append(inside_u8)
        if show_on:     active_masks.append(on_u8)
        if show_out:    active_masks.append(out_u8)

        # Compose binary slices (logical OR across active masks)
        i_slice = _compose_slice(active_masks, 'i', i_idx) if active_masks else np.zeros((inside_u8.shape[0], inside_u8.shape[1]), dtype=np.uint8)
        j_slice = _compose_slice(active_masks, 'j', j_idx) if active_masks else np.zeros((inside_u8.shape[0], inside_u8.shape[2]), dtype=np.uint8)
        k_slice = _compose_slice(active_masks, 'k', k_idx) if active_masks else np.zeros((inside_u8.shape[1], inside_u8.shape[2]), dtype=np.uint8)

        i_fig = _slice_fig(i_slice, f"I-slice (x={i_idx})", border_color="#3B82F6")
        j_fig = _slice_fig(j_slice, f"J-slice (y={j_idx})", border_color="#22C55E")
        k_fig = _slice_fig(k_slice, f"K-slice (z={k_idx})", border_color="#F59E0B")


        # 3-D fig with visibility tied to mask toggles
        threeD_fig = make_3d_fig(show_inside=show_inside, show_on=show_on, show_out=show_out)

        # Add dynamic slice planes
        for frame in _slice_frames(grid, i_idx, j_idx, k_idx, mc_map):
            threeD_fig.add_trace(frame)

        status = f"Active masks: {', '.join(mask_values) or 'none'} | i={i_idx}, j={j_idx}, k={k_idx}"
        return i_fig, j_fig, k_fig, threeD_fig, status

    # Open browser automatically
    import threading, webbrowser
    def _open():
        webbrowser.open_new(f"http://127.0.0.1:{port}")
    threading.Timer(0.8, _open).start()

    app.run(debug=False, port=port)


# ---------------------------
# Pipeline
# ---------------------------
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
    # Require explicit input meshes from the repo-global `data/` directory.
    # Do NOT auto-generate input meshes or create new directories here.
    if not os.path.exists(in_mesh_path):
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data"
        raise FileNotFoundError(
            f"Input mesh not found: {in_mesh_path}\n"
            f"Place demo meshes in the repository data directory: {data_dir}\n"
            "Run `python scripts/generate_demo_meshes.py` to create small demo meshes."
        )

    mesh = load_mesh(in_mesh_path)
    print(f"Loaded mesh: {in_mesh_path} | closed={is_closed(mesh)}")

    # Center mesh so its AABB is symmetric around the origin
    mesh = center_mesh(mesh)

    # Build grid from mesh AABB (no mesh scaling)
    grid = make_cube_grid_from_mesh(mesh, spacing=spacing, n=n, margin_frac=margin_frac)

    # Log grid decisions
    N = grid["shape"][0]
    s = grid["spacing"][0]
    Lx, Ly, Lz = grid["aabb_mm"]
    extent = grid["extent_mm"][0]
    print(f"[grid] spacing={s:.6f} mm | N={N} (2^n) | cube extent={extent:.3f} mm")
    print(f"[grid] mesh AABB (mm): Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f} | margin={margin_frac*100:.1f}%")

    # Voxelize and classify (returns padded mask + effective transform)
    inside_bool, voxel_transform = voxelize_mesh(mesh, grid)
    #inside_u8, on_u8, out_u8 = classify_in_on_out(inside_bool)
    #inside_u8, on_u8, out_u8 = classify_by_parity_multi(mesh, grid)
    inside_u8, on_u8, out_u8 = classify_by_winding(mesh, grid)

    # Save masks
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, inside=inside_u8, on=on_u8, out=out_u8,
                        spacing=np.array(grid["spacing"]), origin=np.array(grid["origin"]))
    print(f"Masks saved (uint8) → {out_npz}")

    # Viewer
    if show:
        if viewer == "dash":
            show_viewer_dash(
                mesh=mesh,
                inside_u8=inside_u8, on_u8=on_u8, out_u8=out_u8,
                grid=grid, voxel_transform=voxel_transform,
                mc_map=mc_map, port=port
            )
        else:
            show_viewer_html(
                mesh=mesh,
                inside_u8=inside_u8, on_u8=on_u8, out_u8=out_u8,
                grid=grid, voxel_transform=voxel_transform,
                html_path="outputs/viewer.html", mc_map=mc_map
            )

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Voxelize a mesh onto a power-of-two cubic grid derived from mesh AABB")
    ap.add_argument("--in", dest="in_mesh", default="inputs/example_cube.stl",
                    help="Input mesh (.stl/.obj/.ply)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="Voxel edge length in mm (if provided, N will be computed as next power-of-two)")
    ap.add_argument("--n", type=int, default=8,
                    help="Grid exponent (2^n per side). Used if --spacing is None. If both are given, spacing takes precedence.")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="Extra margin fraction around the mesh AABB (default 0.10 = 10%)")
    ap.add_argument("--mc-map", type=str, default="zyx",
                    choices=["zyx", "xyz", "xzy", "yxz", "yzx", "zxy"],
                    help="Axis mapping from marching-cubes (z,y,x) → (X,Y,Z).")
    ap.add_argument("--out", default="outputs/masks_demo.npz",
                    help="Output compressed npz file")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open viewer")
    ap.add_argument("--viewer", choices=["dash","html"], default="dash",
                    help="Viewer type: 'dash' (orthogonal slices + 3D) or 'html' (3D only).")
    ap.add_argument("--port", type=int, default=8050,
                    help="Port for Dash viewer (default 8050).")
    args = ap.parse_args()

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
