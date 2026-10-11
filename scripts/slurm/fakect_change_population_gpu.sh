#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=aorta_diverse_change
#SBATCH --output=logs/aorta_diverse_change.%A_%a.out
#SBATCH --error=logs/aorta_diverse_change.%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_diverse_config=${1:?Pass expanded model INI}
fakect_diverse_action=${2:-smoke}
fakect_diverse_guard=${3:?Pass the submitted code/input checksum file}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
sha256sum --check --status "$fakect_diverse_guard"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}" TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-diverse-change-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
if [[ "$fakect_diverse_action" == smoke ]]; then
    # Planning happens only after the entire larger population is complete.
    crun -p "$HOME/envs/fakect" python scripts/train_change_population.py --config "$fakect_diverse_config" --stage plan
    crun -p "$HOME/envs/fakect" python scripts/train_change_population.py --config "$fakect_diverse_config" --stage smoke
else
    fakect_diverse_list=$(crun -p "$HOME/envs/fakect" python scripts/train_change_population.py --config "$fakect_diverse_config" --stage list --ids-only)
    mapfile -t fakect_diverse_ids <<< "$fakect_diverse_list"
    fakect_diverse_index=${SLURM_ARRAY_TASK_ID:?Fold fitting requires array index}
    fakect_diverse_run=${fakect_diverse_ids[$fakect_diverse_index]:?Invalid fold index}
    crun -p "$HOME/envs/fakect" python scripts/train_change_population.py --config "$fakect_diverse_config" --stage fit --run-id "$fakect_diverse_run"
fi
