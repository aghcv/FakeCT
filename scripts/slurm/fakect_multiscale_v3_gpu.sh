#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=coa_multiscale3
#SBATCH --output=logs/coa_multiscale3.%j.out
#SBATCH --error=logs/coa_multiscale3.%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_input=${1:?Pass the frozen v3 pilot INI}
fakect_checkout=${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd -- "$fakect_checkout"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-multiscale3-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
# Separate process so synthetic warmup models do not consume real-run memory.
crun -p "$HOME/envs/fakect" python scripts/train_multiscale_coa_v3.py --config "$fakect_input" --stage smoke
crun -p "$HOME/envs/fakect" python scripts/train_multiscale_coa_v3.py --config "$fakect_input" --stage all
