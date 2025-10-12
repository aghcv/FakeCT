# src/fakect/core.py
# Mesh → power-of-two cubic grid derived from mesh AABB → inside/on/out masks → Plotly viewer
# Examples:
#   python -m fakect.core --in data/carotid.stl --spacing 1.0 --mc-map xyz
#   python -m fakect.core --in data/carotid.stl --n 7 --margin 0.10 --mc-map xyz

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
from typing import Tuple

# ---------------------------
# Utilities
# ---------------------------
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

def generate_cube_stl(path: str, side_mm: float = 60.0):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cube = trimesh.creation.box(extents=(side_mm, side_mm, side_mm))
    cube.export(path)
    print(f"Generated cube mesh: {path}")

# ---------------------------
# Grid construction (derived from mesh AABB)
# ---------------------------
def make_cube_grid_from_mesh(
    mesh: trimesh.Trimesh,
    spacing: float | None,
    n: int | None,
    margin_frac: float = 0.10
) -> dict:
    """
    Build an isotropic cubic grid whose side is the next power-of-two large enough
    to contain the mesh AABB (after adding a margin). Either spacing or n can be given:
      - If spacing is given: N = next_pow2(ceil(Lmax_with_margin / spacing))
      - If n is given: spacing = Lmax_with_margin / (2^n)
      - If both are given: spacing takes precedence and N computed from it.
    The grid is centered at (0,0,0); mesh should be centered beforehand.
    """
    bounds = mesh.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    Lx, Ly, Lz = (bounds[1] - bounds[0]).astype(float)
    Lmax = float(max(Lx, Ly, Lz))
    Lmax_with_margin = Lmax * (1.0 + margin_frac)
    if spacing is not None:
        raw = int(np.ceil(Lmax_with_margin / spacing))
        N = next_pow2(raw)
        s = float(spacing)
    else:
        if n is None:
            n = 7
        N = 2 ** int(n)
        s = float(Lmax_with_margin / N)

    extent = N * s
    origin = (-extent / 2.0, -extent / 2.0, -extent / 2.0)  # world origin at cube low corner
    grid = {
        "shape": (N, N, N),                 # index order: (Z, Y, X)
        "spacing": (s, s, s),               # voxel edge length (mm)
        "origin": origin,                   # world origin (mm) at low corner
        "extent_mm": (extent, extent, extent),
        "aabb_mm": (Lx, Ly, Lz),
        "aabb_margin_mm": Lmax_with_margin,
    }
    return grid

# ---------------------------
# Voxelization and masks
# ---------------------------
def fit_to_shape_centered(vol: np.ndarray, target_shape: Tuple[int, int, int]):
    """
    Center-pad or crop a 3D volume to match target shape (Z,Y,X).
    Returns (out, (oz, oy, ox)) where (oz,oy,ox) are the index offsets at which
    the original 'vol' was placed into 'out'.
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

    # Build effective transform that maps *padded* array indices -> world
    # marching_cubes gives vertices in the padded array's index space (z,y,x),
    # but vg.transform expects original indices; compensate by shifting indices by (-oz,-oy,-ox).
    voxel_xform = vg.transform.copy()
    index_shift = np.eye(4, dtype=float)
    index_shift[:3, 3] = [-oz, -oy, -ox]     # shift indices before applying vg.transform
    voxel_xform = voxel_xform @ index_shift

    return inside_bool, voxel_xform

def classify_in_on_out(inside_bool: np.ndarray):
    """
    Compute inside, on (1-voxel shell), and out masks.
    Ensures perfect complementarity: inside + on + out = 1 everywhere.

    Returns
    -------
    inside_u8, on_u8, out_u8 : np.ndarray (uint8)
        Mutually exclusive binary volumes whose sum equals 1.
    """
    eroded = binary_erosion(inside_bool, structure=np.ones((3, 3, 3)), iterations=1)
    on = inside_bool ^ eroded

    inside_u8 = inside_bool.astype(np.uint8)
    on_u8 = on.astype(np.uint8)

    # Start from all ones and zero out voxels occupied by inside or on
    out_u8 = np.ones_like(inside_u8, dtype=np.uint8)
    out_u8[(inside_u8 == 1) | (on_u8 == 1)] = 0

    # Optional safety check
    # assert np.all((inside_u8 + on_u8 + out_u8) == 1), "In+On+Out not complementary"

    return inside_u8, on_u8, out_u8


# ---------------------------
# Marching-cubes → Plotly axis mapping
# ---------------------------
def apply_axis_map(verts: np.ndarray, origin: Tuple[float,float,float], mc_map: str):
    """
    Input verts are (N,3) in world coords but ordered as (z,y,x) because marching_cubes
    operates in (z,y,x). Remap to (X,Y,Z) for Plotly.
    """
    z, y, x = verts[:, 0], verts[:, 1], verts[:, 2]
    ox, oy, oz = origin

    if mc_map == "zyx":      # (X=x,Y=y,Z=z)
        X, Y, Z = x + ox, y + oy, z + oz
    elif mc_map == "xyz":    # (X=z,Y=y,Z=x)
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
# Plotly viewer
# ---------------------------
def mask_to_trace(mask_u8, grid, name, color, opacity, mc_map, transform=None):
    """
    Convert binary mask (uint8) to Plotly Mesh3d trace with axis remap.
    IMPORTANT:
      - We call marching_cubes WITHOUT spacing so vertices are in raw index units (z,y,x).
      - Then we map indices to world via 'transform' (voxel index -> world).
    """
    if mask_u8 is None or not np.any(mask_u8):
        return None

    # 1) Extract iso-surface in index space (no spacing!) to avoid double-scaling
    verts_idx, faces, _, _ = measure.marching_cubes(mask_u8, level=0.5)

    # 2) Map (z,y,x) index coords -> world coords using the effective transform
    if transform is not None:
        homog = np.c_[verts_idx, np.ones(len(verts_idx))]
        verts_world = (transform @ homog.T).T[:, :3]
    else:
        # Fallback: simple origin + spacing (not recommended; transform is preferred)
        s = grid["spacing"][0]
        origin = np.array(grid["origin"])
        verts_world = origin + s * verts_idx[:, ::-1]  # crude fallback; reverse order to (x,y,z)

    # 3) Remap (world z,y,x) -> Plotly (X,Y,Z)
    X, Y, Z = apply_axis_map(verts_world, (0, 0, 0), mc_map)

    i, j, k = faces.T
    return go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k,
                     name=name, color=color, opacity=opacity,
                     flatshading=True, showscale=False)

def mesh_to_trace(mesh: trimesh.Trimesh, name="mesh", color="#FFFFFF", opacity=0.15):
    """Convert trimesh to Plotly Mesh3d trace (mesh already in (x,y,z))."""
    if mesh is None or mesh.vertices.size == 0:
        return None
    v, f = mesh.vertices, mesh.faces
    return go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2],
                     i=f[:,0], j=f[:,1], k=f[:,2],
                     name=name, color=color, opacity=opacity, showscale=False)

def show_viewer(*, mesh, inside_u8, on_u8, out_u8, grid, voxel_transform,
                html_path="outputs/viewer.html", mc_map="xyz"):
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    color_map = {"mesh":"#AB1616", "inside":"#3B82F6", "on":"#22C55E", "out":"#F59E0B"}

    traces = [
        mesh_to_trace(mesh, name="mesh", color=color_map["mesh"], opacity=0.15),
        mask_to_trace(inside_u8, grid, "inside", color_map["inside"], 0.08, mc_map, transform=voxel_transform),
        mask_to_trace(on_u8,     grid, "on",     color_map["on"],     0.08, mc_map, transform=voxel_transform),
        mask_to_trace(out_u8,    grid, "out",    color_map["out"],    0.08, mc_map, transform=voxel_transform),
    ]
    traces = [t for t in traces if t]

    fig = go.Figure(data=traces)

    # Enforce physical aspect (cube) since grid is cubic and isotropic
    fig.update_layout(
        title="Mesh + in/on/out masks (click legend or buttons to toggle)",
        scene=dict(
            xaxis=dict(title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            zaxis=dict(title="Z (mm)"),
            aspectmode="cube"   # equal scaling on x,y,z visually
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Toggle buttons (mesh always visible; toggle masks)
    names = [t.name for t in fig.data] # type: ignore
    def vis(active):
        return [True if n == "mesh" or n in active else False for n in names]
    buttons = [
        dict(label="All",    method="update", args=[{"visible": [True]*len(names)}]),
        dict(label="Inside", method="update", args=[{"visible": vis({'inside'})}]),
        dict(label="On",     method="update", args=[{"visible": vis({'on'})}]),
        dict(label="Out",    method="update", args=[{"visible": vis({'out'})}]),
        dict(label="None",   method="update", args=[{"visible": vis(set())}]),
    ]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=buttons,
            direction="right",
            x=0.0, y=1.05,
            xanchor="left", yanchor="bottom",
            pad={"r": 4, "t": 2}
        )]
    )

    pio.write_html(fig, html_path, auto_open=False, include_plotlyjs="cdn", full_html=True) #type: ignore
    try:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception:
        pass
    print(f"Viewer saved to {html_path}")

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
                mc_map: str = "xyz"
            ):
    if not os.path.exists(in_mesh_path):
        generate_cube_stl(in_mesh_path)

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
    inside_u8, on_u8, out_u8 = classify_in_on_out(inside_bool)

    # Save masks
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, inside=inside_u8, on=on_u8, out=out_u8,
                        spacing=np.array(grid["spacing"]),
                        origin=np.array(grid["origin"]))
    print(f"Masks saved (uint8) → {out_npz}")

    # Viewer
    if show:
        show_viewer(
            mesh=mesh,
            inside_u8=inside_u8,
            on_u8=on_u8,
            out_u8=out_u8,
            grid=grid,
            voxel_transform=voxel_transform,
            html_path="outputs/viewer.html",
            mc_map=mc_map,
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
    ap.add_argument("--margin", type=float, default=0.10,
                    help="Extra margin fraction around the mesh AABB (default 0.10 = 10%)")
    ap.add_argument("--mc-map", type=str, default="xyz",
                    choices=["zyx", "xyz", "xzy", "yxz", "yzx", "zxy"],
                    help="Axis mapping from marching-cubes (z,y,x) → (X,Y,Z). Use 'xyz' if your STL looks correct with X=verts[:,0],Y=verts[:,1],Z=verts[:,2].")
    ap.add_argument("--out", default="outputs/masks_demo.npz",
                    help="Output compressed npz file")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open viewer")
    args = ap.parse_args()

    spacing = None if (args.spacing is None or args.spacing <= 0) else float(args.spacing)
    run_pipeline(
        in_mesh_path=args.in_mesh,
        spacing=spacing,
        n=args.n,
        margin_frac=args.margin,
        out_npz=args.out,
        show=not args.no_show,
        mc_map=args.mc_map
    )

if __name__ == "__main__":
    main()
