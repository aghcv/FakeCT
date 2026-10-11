#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_sampling_report
#SBATCH --output=logs/aorta_sampling_report.%j.out
#SBATCH --error=logs/aorta_sampling_report.%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2

set -euo pipefail
fakect_sampling_report_config=${1:?Pass TRAIN diagnostic INI}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export MPLCONFIGDIR="/tmp/fakect-sampling-report-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python scripts/research/train_change_sampling.py --config "$fakect_sampling_report_config" --stage report
