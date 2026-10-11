#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=aorta_finetune
#SBATCH --output=logs/aorta_finetune.%A_%a.out
#SBATCH --error=logs/aorta_finetune.%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
set -euo pipefail
fakect_ft_config=${1:?Pass fine-tuning INI}
fakect_ft_stage=${2:-fit}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env pytorch-gpu/2.8.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nnUNet_compile=false
export MPLCONFIGDIR="/tmp/fakect-ft-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
if [[ "$fakect_ft_stage" == smoke ]]; then
 crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/finetune_aorta.py --config "$fakect_ft_config" --stage smoke
else
 fakect_ft_list=$(python3 scripts/research/finetune_aorta.py --config "$fakect_ft_config" --stage list --fold all --ids-only)
 mapfile -t fakect_ft_folds <<< "$fakect_ft_list"
 fakect_ft_index=${SLURM_ARRAY_TASK_ID:?Requires fold/arm array index}
 if [[ "$fakect_ft_stage" == pretrained ]]; then
  fakect_ft_fold=${fakect_ft_folds[$fakect_ft_index]:?Invalid fold index}
  crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/finetune_aorta.py --config "$fakect_ft_config" --stage evaluate --fold "$fakect_ft_fold" --arm pretrained
 else
  fakect_ft_fold=${fakect_ft_folds[$((fakect_ft_index / 2))]:?Invalid fold index}
  if (( fakect_ft_index % 2 == 0 )); then fakect_ft_arm=baseline_only; else fakect_ft_arm=siblings; fi
  crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/finetune_aorta.py --config "$fakect_ft_config" --stage fit --fold "$fakect_ft_fold" --arm "$fakect_ft_arm"
  crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/research/finetune_aorta.py --config "$fakect_ft_config" --stage evaluate --fold "$fakect_ft_fold" --arm "$fakect_ft_arm"
 fi
fi
