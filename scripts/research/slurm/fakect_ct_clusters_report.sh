#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_ct_clusters
#SBATCH --output=logs/aorta_ct_clusters.%j.out
#SBATCH --error=logs/aorta_ct_clusters.%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_cluster_config=${1:?Pass clustering INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/fakect-ct-clusters-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
python3 scripts/research/cluster_sibling_ct.py --config "$fakect_cluster_config" --stage analyse
