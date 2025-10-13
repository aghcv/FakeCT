#!/usr/bin/env bash
# ============================================================
# Demo: Run the FakeCT core voxelization pipeline end-to-end
# ============================================================

set -e  # Exit on any error

# 1. Setup
FILE_NAME="cube"
INPUT_DIR="data"
OUTPUT_DIR="outputs"
MESH_PATH="${INPUT_DIR}/${FILE_NAME}.stl"
NPZ_PATH="${OUTPUT_DIR}/${FILE_NAME}_masks.npz"

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"

echo "=== FakeCT Demo Run ==="
echo "Input mesh:   ${MESH_PATH}"
echo "Output masks: ${NPZ_PATH}"
echo "-------------------------------------------"

# 2. Run voxelization (2^6 = 64^3 grid, 1 mm voxels)
echo "[INFO] Running voxelization pipeline..."
python -m fakect.core \
  --in "${MESH_PATH}" \
  --n 6 \
  --margin 0.1 \
  --out "${NPZ_PATH}"

