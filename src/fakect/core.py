# src/fakect/simple_pipeline.py
# Minimal mesh → CT-like grid → in/on/out masks → interactive viewer
# Deps: numpy, trimesh, matplotlib, scipy (for binary erosion)
# Run:  python -m fakect.simple_pipeline --in inputs/example_cube.stl

from __future__ import annotations
import os
import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import binary_erosion

# ------------------------------
# 1) Mesh I/O + basic checks
# ------------------------------
def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")

    # If it's already a single Trimesh, return it
    if isinstance(mesh, trimesh.Trimesh):
        return mesh

    # If it's a Scene (multiple parts), merge geometries robustly:
    # Try direct Scene.geometry first, then fall back to dump() serialized dict.
    # Use an explicit Scene type check so type-checkers know what we're doing.
    try:
        Scene = trimesh.Scene
    except Exception:
        Scene = None

    parts = ()
    if Scene is not None and isinstance(mesh, Scene):
        geoms = getattr(mesh, "geometry", None)
        if geoms:
            parts = tuple(geoms.values())

    if not parts:
        raise ValueError("Loaded mesh is a Scene with no geometry parts")

    return trimesh.util.concatenate(parts)

def is_closed(mesh: trimesh.Trimesh) -> bool:
    return bool(mesh.is_watertight)

# ------------------------------
# 2) Define clinical-like grid
# ------------------------------
def make_grid(
    rows: int = 256, cols: int = 256, slices: int = 256,
    spacing_xyz=(0.8, 0.8, 1.5),  # (dy, dx, dz) in mm; dz ~ slice thickness
    origin_xyz=None
):
    """
    Returns grid spec and the world coords of the volume center.
    Order convention: vol[z, y, x] with spacing (dy, dx, dz) noted separately.
    """
    dy, dx, dz = spacing_xyz
    # indices
    yy = np.arange(rows) * dy
    xx = np.arange(cols) * dx
    zz = np.arange(slices) * dz
    if origin_xyz is None:
        origin_xyz = (-(xx[-1] / 2.0), -(yy[-1] / 2.0), -(zz[-1] / 2.0))
    # Center in world coords:
    center_world = (0.0, 0.0, 0.0)  # by construction (origin centered)
    grid = {
        "rows": rows, "cols": cols, "slices": slices,
        "spacing": (dy, dx, dz),
        "origin": origin_xyz,
        "extent_mm": (xx[-1], yy[-1], zz[-1]),
        "center_world": center_world
    }
    return grid

# ------------------------------
# 3) Center mesh inside grid
# ------------------------------
def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # move mesh centroid to (0,0,0)
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    return mesh

def scale_mesh_to_fit(mesh: trimesh.Trimesh, grid, margin_frac=0.1) -> trimesh.Trimesh:
    """Uniformly scale mesh to fit within grid extents with a small margin."""
    mesh = mesh.copy()
    ext_x, ext_y, ext_z = grid["extent_mm"]
    bbox = mesh.extents  # (sx, sy, sz)
    if np.any(bbox == 0):
        return mesh
    # allowed max size inside grid (leave a margin)
    scale_lim = np.array([ext_x, ext_y, ext_z]) * (1.0 - margin_frac)
    s = float(np.min(scale_lim / bbox))
    mesh.apply_scale(s)
    return mesh

# ------------------------------
# 4) Voxelization + in/on/out masks
# ------------------------------
def voxelize_mesh(
    mesh: trimesh.Trimesh,
    grid,
    engine="trimesh"
):
    """
    Create boolean volume 'inside' using trimesh voxelization.
    Note: trimesh.voxelized uses an isotropic 'pitch'. We choose pitch = min(dy, dx, dz).
    """
    dy, dx, dz = grid["spacing"]
    pitch = float(min(dy, dx, dz))

    # Workaround for anisotropy: scale mesh into isotropic space, voxelize, then keep indices.
    sx, sy, sz = 1.0/dx, 1.0/dy, 1.0/dz
    iso_mesh = mesh.copy()
    iso_mesh.apply_scale((sx, sy, sz))

    vg = iso_mesh.voxelized(pitch=1.0)  # isotropic grid in scaled space
    inside_iso = vg.matrix.astype(bool)  # [z, y, x] boolean

    # Fit to requested grid shape by simple pad/crop to (slices, rows, cols)
    target = (grid["slices"], grid["rows"], grid["cols"])
    vol = fit_to_shape(inside_iso, target)
    return vol

def fit_to_shape(vol: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Center-pad or crop a 3D array to target shape."""
    out = np.zeros(target_shape, dtype=bool)
    sz, sy, sx = vol.shape
    tz, ty, tx = target_shape
    # start indices to center
    oz = max((tz - sz) // 2, 0); oy = max((ty - sy) // 2, 0); ox = max((tx - sx) // 2, 0)
    z0 = max((sz - tz) // 2, 0); y0 = max((sy - ty) // 2, 0); x0 = max((sx - tx) // 2, 0)
    z1 = z0 + min(sz, tz); y1 = y0 + min(sy, ty); x1 = x0 + min(sx, tx)
    out[oz:oz+(y1-y0)+(z1-z0)-(z1-z0), oy:oy+(y1-y0), ox:ox+(x1-x0)] = vol[z0:z1, y0:y1, x0:x1]
    # ^ slightly verbose to make the index math explicit for students
    out[oz:oz+(z1-z0), oy:oy+(y1-y0), ox:ox+(x1-x0)] = vol[z0:z1, y0:y1, x0:x1]
    return out

def classify_in_on_out(inside: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    inside: boolean 3D mask (True=interior)
    on:     boundary voxels by XOR between mask and its eroded version
    out:    complement of inside
    """
    eroded = binary_erosion(inside, structure=np.ones((3,3,3)), iterations=1, border_value=0)
    on = inside ^ eroded
    out = ~inside
    return inside, on, out

# ------------------------------
# 5) Visualization (3 slices + 3D)
# ------------------------------
class Viewer:
    def __init__(self, inside, on, out, grid):
        self.inside = inside
        self.on = on
        self.out = out
        self.grid = grid

        self.nz, self.ny, self.nx = inside.shape
        self.iz = self.nz // 2
        self.iy = self.ny // 2
        self.ix = self.nx // 2

        self.fig = plt.figure(figsize=(12, 6))
        self.ax_axial   = self.fig.add_subplot(2, 2, 1)
        self.ax_coronal = self.fig.add_subplot(2, 2, 2)
        self.ax_sag     = self.fig.add_subplot(2, 2, 3)
        self.ax_3d      = self.fig.add_subplot(2, 2, 4, projection="3d")

        # draw initial
        self._draw_slices()
        self._draw_3d()

        # scroll to move slices: axial=Z, coronal=Y, sagittal=X
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

    def _composite_slice(self, slc):
        """
        Map out/inside/on to 0/1/2 for a simple colored visualization.
        """
        img = np.zeros_like(slc, dtype=np.uint8)
        img[self.out[slc]] = 0
        img[self.inside[slc]] = 1
        img[self.on[slc]] = 2
        return img

    def _draw_slices(self):
        # axial (Z): show YX
        img_ax = self._composite_slice((self.iz, slice(None), slice(None)))
        self.ax_axial.imshow(img_ax, origin="lower", interpolation="nearest")
        self.ax_axial.set_title(f"Axial (Z={self.iz})"); self.ax_axial.set_xlabel("X"); self.ax_axial.set_ylabel("Y")
        self.ax_axial.spines[:].set_color("r")  # red border

        # coronal (Y): show ZX
        img_co = self._composite_slice((slice(None), self.iy, slice(None)))
        self.ax_coronal.imshow(img_co, origin="lower", interpolation="nearest")
        self.ax_coronal.set_title(f"Coronal (Y={self.iy})"); self.ax_coronal.set_xlabel("X"); self.ax_coronal.set_ylabel("Z")
        self.ax_coronal.spines[:].set_color("g")  # green border

        # sagittal (X): show ZY
        img_sa = self._composite_slice((slice(None), slice(None), self.ix))
        self.ax_sag.imshow(img_sa, origin="lower", interpolation="nearest")
        self.ax_sag.set_title(f"Sagittal (X={self.ix})"); self.ax_sag.set_xlabel("Y"); self.ax_sag.set_ylabel("Z")
        self.ax_sag.spines[:].set_color("b")  # blue border

    def _draw_3d(self):
        self.ax_3d.clear()
        self.ax_3d.set_title("3D: mesh proxy + slice planes")
        # draw 3D proxy from ON-mask voxels (sparse)
        zz, yy, xx = np.where(self.on)
        if zz.size > 0:
            # plot small points for boundary voxels
            # Use keyword `zs=` to disambiguate the 3D scatter signature so
            # type-checkers (Pylance) don't think the third positional arg is
            # the 2D `s` (size) parameter.
            self.ax_3d.scatter(xx, yy, zs=zz.tolist(), s=1, alpha=0.2)

        # draw slice planes as transparent rectangles
        self._draw_plane_x(self.ix)
        self._draw_plane_y(self.iy)
        self._draw_plane_z(self.iz)

        self.ax_3d.set_xlabel("X"); self.ax_3d.set_ylabel("Y"); self.ax_3d.set_zlabel("Z")
        self.ax_3d.set_box_aspect((self.nx, self.ny, self.nz))
        self.ax_3d.view_init(elev=20, azim=35)
        plt.tight_layout()

    def _draw_plane_x(self, ix):
        # plane perpendicular to X at ix
        Y, Z = np.mgrid[0:self.ny, 0:self.nz]
        X = np.full_like(Y, ix)
        self.ax_3d.plot_surface(X, Y, Z, alpha=0.15, rstride=8, cstride=8)

    def _draw_plane_y(self, iy):
        X, Z = np.mgrid[0:self.nx, 0:self.nz]
        Y = np.full_like(X, iy)
        self.ax_3d.plot_surface(X, Y, Z, alpha=0.15, rstride=8, cstride=8)

    def _draw_plane_z(self, iz):
        X, Y = np.mgrid[0:self.nx, 0:self.ny]
        Z = np.full_like(X, iz)
        self.ax_3d.plot_surface(X, Y, Z, alpha=0.15, rstride=8, cstride=8)

    def on_scroll(self, event):
        # scroll over each axes adjusts that axis' slice index
        step = 1 if event.button == "up" else -1
        if event.inaxes == self.ax_axial:
            self.iz = np.clip(self.iz + step, 0, self.nz-1)
        elif event.inaxes == self.ax_coronal:
            self.iy = np.clip(self.iy + step, 0, self.ny-1)
        elif event.inaxes == self.ax_sag:
            self.ix = np.clip(self.ix + step, 0, self.nx-1)
        else:
            return
        # redraw updated slices and 3D planes
        self.ax_axial.clear(); self.ax_coronal.clear(); self.ax_sag.clear()
        self._draw_slices()
        self._draw_3d()
        self.fig.canvas.draw_idle()

# ------------------------------
# 6) Utilities: save masks, demo cube
# ------------------------------
def save_masks_npz(path: str, inside, on, out, grid):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, inside=inside.astype(np.uint8),
        on=on.astype(np.uint8), out=out.astype(np.uint8),
        spacing=np.array(grid["spacing"]), origin=np.array(grid["origin"])
    )
    print(f"Saved masks: {path}")

def generate_cube_stl(path: str, side_mm: float = 60.0):
    """Generate a triangulated cube and save to STL (units in mm)."""
    box = trimesh.creation.box(extents=(side_mm, side_mm, side_mm))
    box.export(path)
    print(f"Generated demo mesh: {path}")

# ------------------------------
# 7) End-to-end driver
# ------------------------------
def run_pipeline(
    in_mesh_path: str,
    grid_rows=256, grid_cols=256, grid_slices=256,
    spacing_xyz=(0.8, 0.8, 1.5),
    out_npz="outputs/masks_demo.npz",
    show=True
):
    # Load or create mesh
    if not os.path.exists(in_mesh_path):
        os.makedirs(os.path.dirname(in_mesh_path) or ".", exist_ok=True)
        generate_cube_stl(in_mesh_path)

    mesh = load_mesh(in_mesh_path)
    print(f"Loaded mesh: {in_mesh_path} | closed={is_closed(mesh)}")
    mesh = center_mesh(mesh)

    grid = make_grid(grid_rows, grid_cols, grid_slices, spacing_xyz)
    mesh = scale_mesh_to_fit(mesh, grid, margin_frac=0.1)

    inside = voxelize_mesh(mesh, grid)
    inside, on, out = classify_in_on_out(inside)
    save_masks_npz(out_npz, inside, on, out, grid)

    if show:
        viewer = Viewer(inside, on, out, grid)
        plt.show()

# ------------------------------
# 8) CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Minimal mesh→grid voxelization demo")
    ap.add_argument("--in", dest="in_mesh", default="inputs/example_cube.stl", help="Input mesh (.stl/.obj/.ply)")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--slices", type=int, default=128)
    ap.add_argument("--spacing", type=float, nargs=3, default=(0.8, 0.8, 1.5), help="(dy, dx, dz) mm; dz≈slice thickness")
    ap.add_argument("--out", default="outputs/masks_demo.npz")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    run_pipeline(
        in_mesh_path=args.in_mesh,
        grid_rows=args.rows, grid_cols=args.cols, grid_slices=args.slices,
        spacing_xyz=tuple(args.spacing),
        out_npz=args.out,
        show=not args.no_show
    )

if __name__ == "__main__":
    main()
