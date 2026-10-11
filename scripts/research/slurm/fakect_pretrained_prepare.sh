#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_pretrained_prepare
#SBATCH --output=logs/aorta_pretrained_prepare.%A_%a.out
#SBATCH --error=logs/aorta_pretrained_prepare.%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
fakect_pretrained_config=${1:?Pass benchmark INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
fakect_pretrained_list=$(python3 scripts/research/benchmark_pretrained_aorta.py --config "$fakect_pretrained_config" --stage list --ids-only)
mapfile -t fakect_pretrained_cases <<< "$fakect_pretrained_list"
fakect_pretrained_case=${fakect_pretrained_cases[${SLURM_ARRAY_TASK_ID:?Requires case array index}]:?Invalid case index}
python3 scripts/research/benchmark_pretrained_aorta.py --config "$fakect_pretrained_config" --stage prepare --case-id "$fakect_pretrained_case"
