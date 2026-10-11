#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=aorta_change
#SBATCH --output=logs/aorta_change.%A_%a.out
#SBATCH --error=logs/aorta_change.%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_change_config=${1:?Pass focused-change INI}
fakect_change_action=${2:-smoke}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-change-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
if [[ "$fakect_change_action" == smoke ]]; then
    crun -p "$HOME/envs/fakect" python scripts/research/train_focused_change.py --config "$fakect_change_config" --stage smoke
else
    fakect_change_list=$(crun -p "$HOME/envs/fakect" python scripts/research/train_focused_change.py --config "$fakect_change_config" --stage list --ids-only)
    mapfile -t fakect_change_ids <<< "$fakect_change_list"
    fakect_change_index=${SLURM_ARRAY_TASK_ID:?Fold fit requires array index}
    fakect_change_run=${fakect_change_ids[$fakect_change_index]:?Invalid fold index}
    crun -p "$HOME/envs/fakect" python scripts/research/train_focused_change.py --config "$fakect_change_config" --stage fit --run-id "$fakect_change_run"
fi
