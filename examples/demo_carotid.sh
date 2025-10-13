#!/usr/bin/env bash
# ============================================================
# Demo: Run the FakeCT core voxelization pipeline end-to-end
# ============================================================

set -e  # Exit on any error

# 1. Setup
FILE_NAME="carotid"
# Use the repository-global data directory (repo_root/data)
INPUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
OUTPUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/examples/outputs"
MESH_PATH="${INPUT_DIR}/${FILE_NAME}.stl"
NPZ_PATH="${OUTPUT_DIR}/${FILE_NAME}_masks.npz"

mkdir -p "${OUTPUT_DIR}"
if [ ! -f "${MESH_PATH}" ]; then
  echo "Input mesh not found: ${MESH_PATH}"
  echo "Create demo meshes with: python scripts/generate_demo_meshes.py"
  exit 2
fi

echo "=== FakeCT Demo Run ==="
echo "Input mesh:   ${MESH_PATH}"
echo "Output masks: ${NPZ_PATH}"
echo "-------------------------------------------"

# 2. Run voxelization (2^6 = 64^3 grid, 1 mm voxels)
echo "[INFO] Running voxelization pipeline..."
python -m fakect.core \
  --in "${MESH_PATH}" \
  --n 7 \
  --margin 0.10 \
  --mc-map zyx \
  --out "${NPZ_PATH}"
