# src/fakect/cli.py
import click
from .core import run_pipeline

@click.group()
def cli():
    """FakeCT command-line tools"""
    pass

@cli.command("simple")
@click.option("--in-mesh", default="inputs/example_cube.stl", help="Input mesh path")
@click.option("--rows", default=128, type=int)
@click.option("--cols", default=128, type=int)
@click.option("--slices", default=128, type=int)
@click.option("--spacing", nargs=3, type=float, default=(0.8, 0.8, 1.5))
@click.option("--out", default="outputs/masks_demo.npz")
@click.option("--show/--no-show", default=True)
def simple(in_mesh, rows, cols, slices, spacing, out, show):
    """Run minimal mesh→grid voxelization with viewer."""
    run_pipeline(in_mesh, rows, cols, slices, spacing, out, show)
