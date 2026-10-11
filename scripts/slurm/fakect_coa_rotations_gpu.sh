#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=coa_rotation
#SBATCH --output=logs/coa_rotation.%j.out
#SBATCH --error=logs/coa_rotation.%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_rotation_config=${1:?Pass the frozen rotating-evaluation INI}
fakect_rotation_stage=${2:-smoke}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-coa-rotation-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
if [[ "$fakect_rotation_stage" == smoke ]]; then
    crun -p "$HOME/envs/fakect" python scripts/train_coa_rotations.py --config "$fakect_rotation_config" --stage smoke
else
    if [[ "$fakect_rotation_stage" == primary_inner_remaining || "$fakect_rotation_stage" == outer_refit || "$fakect_rotation_stage" == secondary_eligible ]]; then
        fakect_rotation_list=$(crun -p "$HOME/envs/fakect" python scripts/train_coa_rotations.py --config "$fakect_rotation_config" --stage list --ids-only --group "$fakect_rotation_stage")
        mapfile -t fakect_rotation_ids <<< "$fakect_rotation_list"
        fakect_rotation_index=${SLURM_ARRAY_TASK_ID:?Array group requires SLURM_ARRAY_TASK_ID}
        fakect_rotation_stage=${fakect_rotation_ids[$fakect_rotation_index]:?Array index outside frozen eligible stages}
    fi
    crun -p "$HOME/envs/fakect" python scripts/train_coa_rotations.py --config "$fakect_rotation_config" --stage fit --run-id "$fakect_rotation_stage"
fi
