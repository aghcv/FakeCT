# tests/cube_test.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for CI servers

from fakect.core import (
    generate_cube_stl,
    load_mesh,
    is_closed,
    make_grid,
    center_mesh,
    scale_mesh_to_fit,
    voxelize_mesh,
    classify_in_on_out,
    save_masks_npz,
    Viewer,
)

def test_cube_pipeline(tmp_path):
    """End-to-end test: cube mesh → grid → in/on/out masks."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "outputs"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    cube_path = data_dir / "cube.stl"
    generate_cube_stl(str(cube_path))

    mesh = load_mesh(str(cube_path))
    assert is_closed(mesh), "Cube mesh must be watertight"

    grid = make_grid(rows=64, cols=64, slices=64, spacing_xyz=(1, 1, 1))
    mesh = center_mesh(mesh)
    mesh = scale_mesh_to_fit(mesh, grid)

    inside = voxelize_mesh(mesh, grid)
    inside, on, out = classify_in_on_out(inside)

    # shape + sanity checks
    assert inside.shape == (64, 64, 64)
    assert inside.dtype == bool
    assert not np.any(on & out), "Masks must not overlap"

    # ensure non-empty inside fraction
    frac_in = inside.mean()
    assert 0.01 < frac_in < 0.5, f"Unexpected inside fraction {frac_in:.3f}"

    # save results
    out_npz = out_dir / "cube_masks.npz"
    save_masks_npz(str(out_npz), inside, on, out, grid)
    assert out_npz.exists(), "Output mask file not created"

    # viewer instantiation (headless test)
    v = Viewer(inside, on, out, grid)
    assert hasattr(v, "fig")

def test_viewer_static_render(tmp_path):
    """Minimal viewer smoke test without showing the figure."""
    nz, ny, nx = 32, 32, 32
    inside = np.zeros((nz, ny, nx), bool)
    inside[8:24, 8:24, 8:24] = True
    inside, on, out = classify_in_on_out(inside)

    grid = make_grid(rows=nx, cols=ny, slices=nz)
    v = Viewer(inside, on, out, grid)

    # Save one static frame for CI verification
    out_png = tmp_path / "viewer_test.png"
    v.fig.savefig(out_png)
    assert out_png.exists(), "Viewer did not save PNG output"
