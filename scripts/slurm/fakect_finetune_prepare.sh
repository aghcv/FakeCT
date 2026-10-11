#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_ft_prepare
#SBATCH --output=logs/aorta_ft_prepare.%A_%a.out
#SBATCH --error=logs/aorta_ft_prepare.%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
set -euo pipefail
fakect_ft_config=${1:?Pass fine-tuning INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
fakect_ft_list=$(python3 scripts/finetune_aorta.py --config "$fakect_ft_config" --stage list --ids-only)
mapfile -t fakect_ft_cases <<< "$fakect_ft_list"
fakect_ft_case=${fakect_ft_cases[${SLURM_ARRAY_TASK_ID:?Requires case index}]:?Invalid case index}
module load container_env pytorch-gpu/2.8.0
crun -p "$PWD/outputs/environments/totalseg-r1" python scripts/finetune_aorta.py --config "$fakect_ft_config" --stage prepare --case-id "$fakect_ft_case"
