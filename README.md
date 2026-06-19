# FakeCT — Minimal synthetic CT / voxelization toolkit

Instructions shows how to load a mesh, voxelize it into a CT-like grid,
create in/on/out masks and inspect the result with a simple viewer.

This README includes instructions for installing prerequisites (VS Code, Git, Conda),
cloning the repo, creating an environment, installing runtime dependencies, and running the demo.

<!--
## Current Version
<img width="1899" height="991" alt="Screenshot 2025-10-13 at 13 20 37" src="https://github.com/user-attachments/assets/5df4d975-da4d-40c0-a29c-85d6af4d81eb" />

## Next Version - Stenosis Tool
<img width="3095" height="1615" alt="image" src="https://github.com/user-attachments/assets/7513c9f2-93ba-4769-968e-10bfc146692f" />
-->


## fakect.py ROI workflow (interactive masks)
These examples show how `fakect.py` supports ROI creation for bitwise morphology manipulation.

![ROI creation for interactive mask editing](images/ROI.png)

![Stenosis ROI example](images/stenosis.png)


## fakenoise.py training previews (context + context step)
These previews show how `fakenoise.py` trains across context settings to generate grayscale images from a slice neighborhood around the target mask.

Default context (single target mask slice):
![Training preview default context](images/train_preview_1_na.png)

`--context 4 --context_step 1`:
![Training preview context 4 step 1](images/train_preview_4_1.png)

`--context 4 --context_step 10`:
![Training preview context 4 step 10](images/train_preview_4_10.png)



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
	# FakeCT — Minimal synthetic CT / voxelization toolkit

	This repository provides a small educational pipeline to convert a surface mesh
	into voxelized "inside / on / out" masks and optionally inspect results with a
	Dash viewer.

	Quick start
	-----------
	The canonical quick start lives in the header of `fakect.py` —
	that file contains usage examples and platform-specific install notes (conda vs pip).

	Minimal example (conda recommended):

	```bash
	conda create -n fakect python=3.10 -y
	conda activate fakect
	conda install -c conda-forge trimesh scipy scikit-image plotly dash -y
	# Optional, if available on your platform:
	conda install -c conda-forge python-igl -y

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

## Script usage examples

### fakect.py (winding-based masks + viewer)

Run the pipeline directly (from repo root):

```bash
python src/fakect.py --in data/cube.stl --n 8 --out outputs/cube_masks.npz
```

Run without opening the viewer (headless):

```bash
python src/fakect.py --in data/carotid.stl --n 9 --margin 0.10 --out outputs/carotid_masks.npz --no-show
```

### fakenoise.py (NRRD viewer + paired dataset CSV)

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



	Notes
	-----
	- `python-igl` is recommended from `conda-forge` when available; pip installs of `igl`
	  often fail on many systems. On macOS x86_64, conda-forge does not provide a build,
	  so the CLI falls back to a slower `trimesh.contains` method by default.
	- To force a method, use `--method winding` (requires python-igl) or `--method trimesh`.

	Relevant paper
	--------------
	Douglass, M. J. J., et al. (2025). “An open-source tool for converting 3D mesh volumes into
	synthetic DICOM CT images for medical physics research.” https://doi.org/10.1007/s13246-025-01599-x


