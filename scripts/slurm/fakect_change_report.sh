#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_change_report
#SBATCH --output=logs/aorta_change_report.%j.out
#SBATCH --error=logs/aorta_change_report.%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2

set -euo pipefail
fakect_change_report_config=${1:?Pass focused-change INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export MPLCONFIGDIR="/tmp/fakect-change-report-${SLURM_JOB_ID:-local}"
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python scripts/train_focused_change.py --config "$fakect_change_report_config" --stage report
