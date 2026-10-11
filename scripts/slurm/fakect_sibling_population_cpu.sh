#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_siblings
#SBATCH --output=logs/aorta_siblings.%A_%a.out
#SBATCH --error=logs/aorta_siblings.%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_population_config=${1:?Pass population INI}
fakect_population_action=${2:-prepare}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/fakect-siblings-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
fakect_population_list=$(python3 scripts/prepare_sibling_population.py --config "$fakect_population_config" --stage list --ids-only)
mapfile -t fakect_population_ids <<< "$fakect_population_list"
fakect_population_index=${SLURM_ARRAY_TASK_ID:?Case generation requires array index}
fakect_population_case=${fakect_population_ids[$fakect_population_index]:?Invalid case index}
python3 scripts/prepare_sibling_population.py --config "$fakect_population_config" --stage "$fakect_population_action" --case-id "$fakect_population_case" --trial-count 7
