#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=coa_classreport
#SBATCH --output=logs/coa_classreport.%j.out
#SBATCH --error=logs/coa_classreport.%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
fakect_classreport_config=${1:?Pass the frozen classification-only INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/fakect-coa-classreport-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python scripts/train_coa_classification.py --config "$fakect_classreport_config" --stage report
