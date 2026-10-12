#!/bin/bash
# Submit after the entire paired extraction array succeeds (afterok dependency).
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=coa-two-data
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the syncoCT integration worktree}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m fakect_two_stage_data prepare \
    --project "${1:-configs/case-studies/coa/project.json}" \
    --output "${2:-outputs/datasets/coa/two-stage-radius115-r1}"
