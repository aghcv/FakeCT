#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_ft_review
#SBATCH --output=logs/aorta_ft_review.%j.out
#SBATCH --error=logs/aorta_ft_review.%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
set -euo pipefail
fakect_ft_config=${1:?Pass fine-tuning INI}
fakect_ft_revision=${2:-review-r1}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env pytorch-gpu/2.8.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/fakect-ft-review-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/finetune_aorta.py --config "$fakect_ft_config" --stage report --revision "$fakect_ft_revision"
