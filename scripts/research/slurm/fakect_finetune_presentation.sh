#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_ft_style
#SBATCH --output=logs/aorta_ft_style.%j.out
#SBATCH --error=logs/aorta_ft_style.%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
set -euo pipefail
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env pytorch-gpu/2.8.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/style_pretrained_review.py \
  --source-report outputs/experiments/aorta/pretrained-finetune-r1/review-comparison-r1 \
  --examples-report outputs/experiments/aorta/pretrained-finetune-r1/zoom-prespecified-r2 \
  --output-directory outputs/experiments/aorta/pretrained-finetune-r1/review-presentation-r2
