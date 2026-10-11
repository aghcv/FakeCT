#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=coa_classonly
#SBATCH --output=logs/coa_classonly.%j.out
#SBATCH --error=logs/coa_classonly.%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_classonly_config=${1:?Pass the frozen classification-only INI}
fakect_classonly_action=${2:-smoke}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-coa-classonly-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
if [[ "$fakect_classonly_action" == smoke ]]; then
    crun -p "$HOME/envs/fakect" python scripts/research/train_coa_classification.py --config "$fakect_classonly_config" --stage smoke
else
    fakect_classonly_list=$(crun -p "$HOME/envs/fakect" python scripts/research/train_coa_classification.py --config "$fakect_classonly_config" --stage list --ids-only)
    mapfile -t fakect_classonly_ids <<< "$fakect_classonly_list"
    fakect_classonly_index=${SLURM_ARRAY_TASK_ID:?Fitting requires SLURM_ARRAY_TASK_ID}
    fakect_classonly_run=${fakect_classonly_ids[$fakect_classonly_index]:?Array index outside frozen folds}
    crun -p "$HOME/envs/fakect" python scripts/research/train_coa_classification.py --config "$fakect_classonly_config" --stage fit --run-id "$fakect_classonly_run"
fi
