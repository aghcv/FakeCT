#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=aorta_pretrained
#SBATCH --output=logs/aorta_pretrained.%A_%a.out
#SBATCH --error=logs/aorta_pretrained.%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_pretrained_config=${1:?Pass benchmark INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env pytorch-gpu/2.8.0
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export nnUNet_compile=false
export MPLCONFIGDIR="/tmp/fakect-pretrained-${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="$PWD/outputs/environments/totalseg-r1/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
fakect_pretrained_list=$(python3 scripts/research/benchmark_pretrained_aorta.py --config "$fakect_pretrained_config" --stage list --ids-only)
mapfile -t fakect_pretrained_cases <<< "$fakect_pretrained_list"
fakect_pretrained_case=${fakect_pretrained_cases[${SLURM_ARRAY_TASK_ID:?Requires case array index}]:?Invalid case index}
crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/benchmark_pretrained_aorta.py --config "$fakect_pretrained_config" --stage predict --case-id "$fakect_pretrained_case"
