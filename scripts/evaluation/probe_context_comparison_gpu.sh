#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=fakect_probe
#SBATCH --output=logs/fakect_comparison_probe.%j.out
#SBATCH --error=logs/fakect_comparison_probe.%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-probe-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python scripts/evaluation/probe_context_comparison.py \
    --experiment outputs/experiments/coa/context-comparison-r1 \
    --diagnostic outputs/experiments/coa/context-grid-diagnostic-r2 \
    --output outputs/qa/context-comparison-analysis-2026-09-30/checkpoint-probe-r1
