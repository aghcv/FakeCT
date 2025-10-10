# FakeCT
Generate realistic synthetic CT volumes from closed 3D surface meshes — voxelize, assign Hounsfield units, and simulate imaging blur/noise for AI and visualization demos.



# Relevant Papers
Douglass, M. J. J., et al. (2025). “An open-source tool for converting 3D mesh volumes into synthetic DICOM CT images for medical physics research.”
This work describes DICOMator, which voxelizes input meshes, assigns Hounsfield units, and simulates CT artefacts (noise, metal, partial volume), exporting synthetic CT datasets in DICOM format. It is very close in spirit to what your toolkit aims to do.


## ⚙️ Installation

### Option 1 — via Conda (recommended)
```bash
# Clone the repository
git clone https://github.com/<your-username>/FakeCT.git
cd FakeCT

# Create the virtual environment
conda env create -f environment.yml
conda activate fakect

# Install in editable mode
pip install -e .