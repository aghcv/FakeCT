#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_ct_features
#SBATCH --output=logs/aorta_ct_features.%A_%a.out
#SBATCH --error=logs/aorta_ct_features.%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
fakect_cluster_config=${1:?Pass clustering INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
fakect_cluster_list=$(python3 scripts/research/cluster_sibling_ct.py --config "$fakect_cluster_config" --stage list --ids-only)
mapfile -t fakect_cluster_cases <<< "$fakect_cluster_list"
fakect_cluster_case=${fakect_cluster_cases[${SLURM_ARRAY_TASK_ID:?Requires case array index}]:?Invalid case index}
python3 scripts/research/cluster_sibling_ct.py --config "$fakect_cluster_config" --stage features --case-id "$fakect_cluster_case"
