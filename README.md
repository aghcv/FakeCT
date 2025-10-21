# FakeCT — Minimal synthetic CT / voxelization toolkit

Instructions shows how to load a mesh, voxelize it into a CT-like grid,
create in/on/out masks and inspect the result with a simple viewer.

This README includes instructions for installing prerequisites (VS Code, Git, Conda),
cloning the repo, creating an environment, installing the package (editable), and running the demo.

## Current Version
<img width="1899" height="991" alt="Screenshot 2025-10-13 at 13 20 37" src="https://github.com/user-attachments/assets/5df4d975-da4d-40c0-a29c-85d6af4d81eb" />


## Next Version - Stenosis Tool
<img width="3095" height="1615" alt="image" src="https://github.com/user-attachments/assets/7513c9f2-93ba-4769-968e-10bfc146692f" />



## Your Tasks:
```bash
1- Install the prerequisites
2- Follow the 4 steps of the quickstart guideline to try demo exmaples: demo_cube, demo_sphere, and demo_carotid 
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
	conda install -c conda-forge python-igl trimesh scipy scikit-image plotly dash -y

	python src/fakect.py --in data/carotid.stl --n 7 --out outputs/carotid_masks.npz
	```

	Notes
	-----
	- `python-igl` is recommended to be installed from `conda-forge` (best cross-platform)
	  — pip installs of `igl` often fail on many systems. If `igl` is not available the
	  stand-alone script will currently exit with an error; I can add an automatic
	  fallback to the parity-based classifier if you prefer.

	Relevant paper
	--------------
	Douglass, M. J. J., et al. (2025). “An open-source tool for converting 3D mesh volumes into
	synthetic DICOM CT images for medical physics research.” https://doi.org/10.1007/s13246-025-01599-x


