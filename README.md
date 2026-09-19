# FakeCT — Minimal synthetic CT / voxelization toolkit

Instructions shows how to load a mesh, voxelize it into a CT-like grid,
create in/on/out masks and inspect the result with a simple viewer.

This README includes instructions for installing prerequisites (VS Code, Git, Conda),
cloning the repo, creating an environment, installing runtime dependencies, and running the demo.

## Main branch viewers

Both viewer implementations are retained after reconciling the local stenosis
work with the remote VTI work:

- `src/fakect.py` is the VTI/mesh viewer with scalar transfer functions and label layers. Its `--out` argument is an output **directory**.
- `src/fakect_stenosis.py` preserves the existing mesh stenosis viewer, including ROI erosion/dilation and iteration schedules. Its `--out` argument is an NPZ **filename**.

```bash
# VTI import without opening a browser
python3 src/fakect.py --in data/arm.vti --out outputs/vti --no-show

# Existing mesh stenosis workflow
python3 src/fakect_stenosis.py --in /path/to/mesh.stl --n 7 --out outputs/mesh_masks.npz
```

The INI-based XCAT cohort workflow remains on the separate `fakect.26.09.16`
branch. Synchronizing `main` does not merge that development branch.

## Current Version
<img width="1899" height="991" alt="Screenshot 2025-10-13 at 13 20 37" src="https://github.com/user-attachments/assets/5df4d975-da4d-40c0-a29c-85d6af4d81eb" />


## Next Version - Stenosis Tool
<img width="3095" height="1615" alt="image" src="https://github.com/user-attachments/assets/7513c9f2-93ba-4769-968e-10bfc146692f" />



## Your Tasks:
```bash
1- Install the prerequisites
2- Follow the quick start to try the demo examples: cube, sphere, and carotid
3- Identify user inputs you think is needed needed for the stenosis tool
```

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

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/aghcv/FakeCT.git
cd FakeCT
```

2. Create and activate a Conda environment:

```bash
conda create -n fakect python=3.10 -y
conda activate fakect
```

3. Install the runtime dependencies:

```bash
conda install -c conda-forge python-igl trimesh scipy scikit-image plotly dash -y
```

If you prefer a pip-based environment for the pure Python packages, use:

```bash
pip install --upgrade pip
pip install trimesh scipy scikit-image plotly dash
```

`igl` is typically easiest to install from `conda-forge`.


4. Run the demo:

```bash
# cube
python src/fakect.py --in data/cube.stl --n 8 --out outputs

# sphere
python src/fakect.py --in data/sphere.stl --n 8 --out outputs

# carotid
python src/fakect.py --in data/carotid.stl --n 9 --margin 0.10 --out outputs
```

To skip opening the Dash viewer, add `--no-show`.

`--out` now expects a directory path. The tool auto-generates an NPZ filename from the input name (for example, `cube_masks.npz` or `vti_masks.npz`) inside that directory.

You can inspect the available CLI options with:

```bash
python src/fakect.py --help
```

## VTI Import Test

You can start testing VTI import now. The CLI supports both a single `.vti` file and a directory of tiled `.vti` files.

The bundled `data/vti/activity_grid_000_000_000.vti` is `CellData` named `activity` with `Float32`
values from about `-1011` to `2213`, so the tool treats it as scalar/range data rather than
integer anatomy labels. Scalar VTI files keep their sampled value volume in the saved NPZ as
`scalar_values` and open in a volume-rendering workflow.

```bash
# Directory of VTI tiles (uses the sample folder in this repo)
python src/fakect.py --in data/vti --out outputs --no-show

# Optional: full-resolution sampling (can be much heavier)
python src/fakect.py --in data/vti --vti-max-dim 0 --out outputs --no-show

# Single VTI file example
python src/fakect.py --in data/vti/activity_grid_000_000_000.vti --out outputs --no-show
```

Useful VTI-specific flags:

```bash
--vti-array <name>          # choose a DataArray by name
--vti-background <value>    # set the background label/scalar value
--vti-background-eps <eps>  # tolerance for floating-point background matching
--vti-max-dim <int>         # browser-friendly downsampling cap (0 = full resolution)
--vti-max-labels <int>      # max number of discrete labels split into layers
```

In the 3D panel, discrete integer-label VTI files use per-label visibility and opacity controls.
Scalar/range VTI files use a ParaView-style transfer map: choose one color scheme for the full
range, add or remove opacity points, type exact opacity values, and drag points horizontally on the
map to reposition them.

By default the Dash viewer opens with three orthogonal slices and a linked 3D view. Use `--no-show`
for a headless import/export smoke test.

## Developer instructions (make changes & run tests)

1. Make code changes in `src/fakect.py` using your editor of choice.

2. There is currently no packaged test suite in this repository. Use the CLI directly to validate changes:

```bash
python src/fakect.py --in data/cube.stl --n 8 --out outputs --no-show
```

3. If you change runtime dependencies, update this README with the revised install command.

4. To try your changes interactively, run `python src/fakect.py --help` or one of the demo commands above.

5. Linting and formatting are not configured in this repository yet. If you add them later, document the commands here.

## Continuous integration (notes for maintainers)

- The `tests/` directory contains VTI input fixtures; an automated CI test suite is not configured.
- Recommended CI steps:
	- Set up a Python 3.10 runner
	- Install the runtime dependencies listed above
	- Run `python src/fakect.py --help` as a smoke test
	- Add targeted automated tests before relying on CI for behavior changes

## Contact / contributing

Open an issue or submit a pull request. Keep changes small and add tests for new behavior.

---
Small, clear, and focused so students can follow the flow from clone → run → edit → test.

# Relevant Papers
Douglass, M. J. J., et al. (2025). “An open-source tool for converting 3D mesh volumes into synthetic DICOM CT images for medical physics research.” (LINK:https://doi.org/10.1007/s13246-025-01599-x)


## Existing FakeNoise and XCAT cluster workflows

Open a web-based viewer for a single NRRD volume:

```bash
python src/fakenoise.py --mode viewer --in /path/to/volume.nrrd
```

Generate a CSV that pairs grayscale volumes with their `.seg.nrrd` masks
and uses the sagittal (X) slice index for training:

```bash
python src/fakenoise.py --mode pair --dataset-dir /path/to/dataset_root
```

This writes `paired_datasets/pairs.csv` under the dataset root and saves one
example PNG preview (gray left, mask right) in the same folder.

Train a mask-to-gray model using single-slice masks:

```bash
python src/fakenoise.py --mode train --csv /path/to/dataset_root/paired_datasets/pairs.csv
```

Train with context slices (stack neighbor masks as extra channels):

```bash
python src/fakenoise.py --mode train --csv /path/to/dataset_root/paired_datasets/pairs.csv --context 4
```

Train with context + stride (skip slices between neighbors):

```bash
python src/fakenoise.py --mode train --csv /path/to/dataset_root/paired_datasets/pairs.csv --context 4 --context-step 3
```

### XCAT phantom jobs (SLURM)

These scripts assume the XCAT binary and parameter template live under `./outputs/xcat` by default.
See [scripts/xcat_job.sh](scripts/xcat_job.sh) and [scripts/xcat_pool.sh](scripts/xcat_pool.sh).

Single job (runs XCAT and converts any `.raw` files to OBJ):

```bash
sbatch scripts/xcat_job.sh --phantom_id phantom_A \
	--set organ_file=vmale50.nrb \
	--set heart_base=vmale50_heart.nrb
```

Parameter sweep (creates the full Cartesian product):

```bash
scripts/xcat_pool.sh --phantom_id phantom_A \
	--set organ_file=vmale50.nrb,vfemale50.nrb \
	--set heart_base=vmale50_heart.nrb,vfemale50_heart.nrb
```

Disable OBJ conversion if you only want raw outputs:

```bash
sbatch scripts/xcat_job.sh --phantom_id phantom_A --convert_raw 0
```

