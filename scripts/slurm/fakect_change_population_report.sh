#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=aorta_diverse_report
#SBATCH --output=logs/aorta_diverse_report.%j.out
#SBATCH --error=logs/aorta_diverse_report.%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2

set -euo pipefail
fakect_diverse_report_config=${1:?Pass expanded model INI}
fakect_diverse_report_guard=${2:?Pass the submitted code/input checksum file}
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
sha256sum --check --status "$fakect_diverse_report_guard"
module load container_env tensorflow-gpu/2.17
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/fakect-diverse-report-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python scripts/train_change_population.py --config "$fakect_diverse_report_config" --stage report
