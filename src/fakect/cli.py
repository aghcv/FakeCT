# src/fakect/cli.py
import click
import subprocess
import sys
import shutil
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

    # Prefer Python example scripts for cross-platform compatibility
    py_script = script_path.with_suffix('.py')
    if py_script.exists():
        click.echo(f"Running Python example: {py_script}")
        try:
            subprocess.run([sys.executable, str(py_script)], check=True)
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Example script failed with exit code {e.returncode}")
        return

    # Fall back to shell script if present; locate a bash executable
    bash_exe = shutil.which("bash")
    if bash_exe and script_path.suffix == ".sh":
        click.echo(f"Running shell example with: {bash_exe} {script_path}")
        try:
            subprocess.run([bash_exe, str(script_path)], check=True)
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Example script failed with exit code {e.returncode}")
        return

    # No runnable example found for this platform
    if script_path.suffix == ".sh":
        raise click.ClickException(
            f"Shell example found ({script_path}) but no POSIX 'bash' is available on this system.\n"
            "On Windows you can run the script with Git Bash, WSL, or run the Python example if available (examples/<name>.py)."
        )
    else:
        raise click.ClickException(f"No runnable example found for: {script_path}")
