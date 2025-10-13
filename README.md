# FakeCT — Minimal synthetic CT / voxelization toolkit

Instructions shows how to load a mesh, voxelize it into a CT-like grid,
create in/on/out masks and inspect the result with a simple viewer.

This README includes instruction for clone, create a virtual
environment, install the package (editable), and run the demo.

## Quick start (for students)

1. Clone the repository:

```bash
git clone https://github.com/aghcv/FakeCT.git
cd FakeCT
```

2. Create and activate a Python virtual environment (macOS / Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the package in editable/development mode and the minimal runtime deps:

```bash
pip install -e .[dev]
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
