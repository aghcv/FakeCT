#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_branch_view
#SBATCH --output=logs/aorta_branch_view.%j.out
#SBATCH --error=logs/aorta_branch_view.%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
set -euo pipefail
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env pytorch-gpu/2.8.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
fakect_branch_study=outputs/experiments/aorta/pretrained-finetune-branches-r3
crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/evaluation/aorta_two_arm_ct.py \
  --source-review "$fakect_branch_study/review-comparison-r1" \
  --output-directory "$fakect_branch_study/two-arm-ct-r1"
crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/publish_aorta_two_model_review.py \
  --source-report "$fakect_branch_study/review-comparison-r1" \
  --ct-assets "$fakect_branch_study/two-arm-ct-r1" \
  --output-directory "$fakect_branch_study/review-pretrained-finetuned-r1"
