#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_sibling_repair
#SBATCH --output=logs/aorta_sibling_repair.%A_%a.out
#SBATCH --error=logs/aorta_sibling_repair.%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_repair_config=${1:?Pass repair INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fakect_repair_list=$(python3 scripts/research/repair_sibling_population.py --config "$fakect_repair_config" --stage list --ids-only)
mapfile -t fakect_repair_ids <<< "$fakect_repair_list"
fakect_repair_index=${SLURM_ARRAY_TASK_ID:?Repair requires array index}
fakect_repair_case=${fakect_repair_ids[$fakect_repair_index]:?Invalid repair array index}
python3 scripts/research/repair_sibling_population.py --config "$fakect_repair_config" --stage repair --case-id "$fakect_repair_case"
