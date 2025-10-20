"""Cross-platform example runner for demo_sphere"""
from pathlib import Path
from fakect.core import run_pipeline

repo_root = Path(__file__).resolve().parents[1]
mesh = repo_root / "data" / "sphere.stl"
out = repo_root / "examples" / "outputs" / "sphere_masks.npz"
out.parent.mkdir(parents=True, exist_ok=True)

run_pipeline(str(mesh), n=7, margin_frac=0.1, out_npz=str(out), show=False)
print(f"Wrote: {out}")
