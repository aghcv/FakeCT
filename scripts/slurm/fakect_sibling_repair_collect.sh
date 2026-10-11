#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_repair_collect
#SBATCH --output=logs/aorta_repair_collect.%j.out
#SBATCH --error=logs/aorta_repair_collect.%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2

set -euo pipefail
fakect_repair_collect_config=${1:?Pass repair INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 scripts/repair_sibling_population.py --config "$fakect_repair_collect_config" --stage collect
