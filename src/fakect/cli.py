# src/fakect/cli.py
import click
import subprocess
from pathlib import Path
from .core import run_pipeline

@click.group()
def cli():
    """FakeCT command-line tools"""
    pass

@cli.command("example")
@click.argument("name")
def example(name):
    """Run an example script located in the repository `examples/` directory.

    Pass the script basename (e.g. `demo_cube` or `demo_carotid`). The command
    will execute `examples/<name>.sh` using the system bash interpreter.
    """
    # Locate the project root relative to this file: src/fakect/cli.py -> project root is parents[2]
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if not examples_dir.exists():
        raise click.ClickException(f"Examples directory not found: {examples_dir}")

    script_name = name if name.endswith(".sh") else f"{name}.sh"
    script_path = examples_dir / script_name
    if not script_path.exists():
        # Provide a helpful list of available examples
        avail = sorted(p.stem for p in examples_dir.glob("*.sh"))
        raise click.ClickException(f"Example not found: {script_name}\nAvailable: {', '.join(avail)}")

    click.echo(f"Running example script: {script_path}")
    try:
        subprocess.run(["/bin/bash", str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Example script failed with exit code {e.returncode}")
