#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=coa_lr_control
#SBATCH --output=logs/coa_lr_control.%j.out
#SBATCH --error=logs/coa_lr_control.%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_pilot_control_config=${1:?Pass the frozen focused-pilot learning-rate-control INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-coa-pilot-control-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
# Release smoke allocations before beginning the development fit.
crun -p "$HOME/envs/fakect" python scripts/research/train_coa_pilot_control.py --config "$fakect_pilot_control_config" --stage smoke
crun -p "$HOME/envs/fakect" python scripts/research/train_coa_pilot_control.py --config "$fakect_pilot_control_config" --stage fit
