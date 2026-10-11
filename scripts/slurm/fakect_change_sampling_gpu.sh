#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=aorta_sampling
#SBATCH --output=logs/aorta_sampling.%j.out
#SBATCH --error=logs/aorta_sampling.%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_sampling_config=${1:?Pass TRAIN diagnostic INI}
fakect_sampling_action=${2:-smoke}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1 TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}" TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-sampling-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
if [[ "$fakect_sampling_action" == smoke ]]; then
    crun -p "$HOME/envs/fakect" python scripts/train_change_sampling.py --config "$fakect_sampling_config" --stage smoke
else
    fakect_sampling_id=$(crun -p "$HOME/envs/fakect" python scripts/train_change_sampling.py --config "$fakect_sampling_config" --stage list --ids-only)
    crun -p "$HOME/envs/fakect" python scripts/train_change_sampling.py --config "$fakect_sampling_config" --stage fit --run-id "$fakect_sampling_id"
fi
