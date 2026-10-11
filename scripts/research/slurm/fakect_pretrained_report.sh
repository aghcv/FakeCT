#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_pretrained_review
#SBATCH --output=logs/aorta_pretrained_review.%j.out
#SBATCH --error=logs/aorta_pretrained_review.%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_pretrained_config=${1:?Pass benchmark INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env pytorch-gpu/2.8.0
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/fakect-pretrained-review-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/benchmark_pretrained_aorta.py --config "$fakect_pretrained_config" --stage report
