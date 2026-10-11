#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=coa_budget200
#SBATCH --output=logs/coa_budget200.%j.out
#SBATCH --error=logs/coa_budget200.%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_budget_config=${1:?Pass the frozen classifier-budget INI}
fakect_budget_action=${2:-smoke}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-coa-budget-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
if [[ "$fakect_budget_action" == smoke ]]; then
    crun -p "$HOME/envs/fakect" python scripts/research/train_coa_budget.py --config "$fakect_budget_config" --stage smoke
else
    fakect_budget_list=$(crun -p "$HOME/envs/fakect" python scripts/research/train_coa_budget.py --config "$fakect_budget_config" --stage list --ids-only)
    mapfile -t fakect_budget_ids <<< "$fakect_budget_list"
    fakect_budget_index=${SLURM_ARRAY_TASK_ID:?Fitting requires SLURM_ARRAY_TASK_ID}
    fakect_budget_run=${fakect_budget_ids[$fakect_budget_index]:?Array index outside frozen folds}
    crun -p "$HOME/envs/fakect" python scripts/research/train_coa_budget.py --config "$fakect_budget_config" --stage fit --run-id "$fakect_budget_run"
fi
