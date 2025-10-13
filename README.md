# Relevant Papers Douglass, M. J. J., et al. (2025). “An open-source tool for converting 3D mesh volumes into synthetic DICOM CT images for medical physics research.”

# FakeCT — Minimal synthetic CT / voxelization toolkit

Instructions shows how to load a mesh, voxelize it into a CT-like grid,
create in/on/out masks and inspect the result with a simple viewer.

<img width="1899" height="991" alt="Screenshot 2025-10-13 at 13 20 37" src="https://github.com/user-attachments/assets/5df4d975-da4d-40c0-a29c-85d6af4d81eb" />

This README includes instructions for installing prerequisites (VS Code, Git, Conda),
cloning the repo, creating an environment, installing the package (editable), and running the demo.

## Prerequisites

Before following the quick start, make sure you have these tools installed. The links go to official installers and the one-liners work on macOS (zsh).

- Visual Studio Code — editor and debugging UI
	- Website: https://code.visualstudio.com/
	- macOS (Homebrew):

		```bash
		brew install --cask visual-studio-code
		```

- Git — version control
	- Website: https://git-scm.com/
	- macOS one-liners (choose one):

		```bash
		# Install Xcode command-line tools (includes git)
		xcode-select --install

		# or via Homebrew
		brew install git
		```

- Conda (Miniconda recommended) — environment and package manager
	- Miniconda: https://docs.conda.io/en/latest/miniconda.html
	- macOS (Homebrew) one-liner:

		```bash
		brew install --cask miniconda
		# initialize conda for zsh and reload your shell
		conda init zsh
		exec $SHELL
		```

	If you prefer Anaconda, use the Anaconda installer instead. Follow the official installer pages for platform-specific guidance.

## Quick start (for students)

1. Clone the repository:

```bash
git clone https://github.com/aghcv/FakeCT.git
cd FakeCT
```

2. Create and activate the Conda environment (uses `environment.yml`):

```bash
conda env create -f environment.yml
conda activate fakect
```

3. Install the package in editable/development mode and the minimal runtime deps:

```bash
pip install -e .
```

This installs the project as the `fakect` package and the `fakect` command-line entrypoint.


4. Run the demo:

```bash
# run the demo_cube script (runs the pipeline with predefined args)
fakect example demo_cube

# list of example scripts is in the `examples/` folder (demo_cube, demo_carotid, demo_sphere)
```

If you prefer to run the script directly from shell:

```bash
bash examples/demo_cube.sh
```

Note on demo meshes:

Example scripts expect demo geometry to live in the repository-global `data/` folder
at the repository root. To populate that folder with small demo meshes, run:

```bash
# from repo root
python scripts/generate_demo_meshes.py
# This writes: data/cube.stl, data/sphere.stl, data/carotid.stl
```

By default the demo will pop up a small matplotlib-based viewer showing three orthogonal
slices and a sparse 3D proxy of boundary voxels.

## Developer instructions (make changes & run tests)

1. Make code changes in `src/fakect/` using your editor of choice.

2. Run the unit tests with pytest. The project includes a small placeholder test so you can
	 validate the test/CI pipeline:

```bash
# from the repo root, with the venv activated
pytest
```

3. If you change package metadata or dependencies, update `pyproject.toml`.

4. To try your changes interactively, install the package in editable mode (step 3 above)
	 so that imports pick up the local source without reinstalling.

5. Linting and formatting (recommended):

```bash
black src tests
flake8
isort src tests
```

## Continuous integration (notes for maintainers)

- The `tests/` directory contains the unit tests. The placeholder test ensures the CI pipeline
	can run; expand the tests as you add features.
- Recommended CI steps:
	- Set up a Python 3.10 runner
	- Create a virtual environment and install `pip install -e .[dev]`
	- Run `pytest` and optionally `pytest --cov` for coverage
	- Run linters (black/flake8/isort)

## Contact / contributing

Open an issue or submit a pull request. Keep changes small and add tests for new behavior.

---
Small, clear, and focused so students can follow the flow from clone → run → edit → test.
