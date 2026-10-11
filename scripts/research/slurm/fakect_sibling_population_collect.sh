#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_siblings_collect
#SBATCH --output=logs/aorta_siblings_collect.%j.out
#SBATCH --error=logs/aorta_siblings_collect.%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
fakect_population_collect_config=${1:?Pass population INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 scripts/research/prepare_sibling_population.py --config "$fakect_population_collect_config" --stage collect
