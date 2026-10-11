#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=aorta_pretrained_focus
#SBATCH --output=logs/aorta_pretrained_focus.%j.out
#SBATCH --error=logs/aorta_pretrained_focus.%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_focus_config=${1:?Pass focus diagnostic INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env pytorch-gpu/2.8.0
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export nnUNet_compile=false
export MPLCONFIGDIR="/tmp/fakect-focus-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/evaluation/probe_pretrained_focus.py --config "$fakect_focus_config"
