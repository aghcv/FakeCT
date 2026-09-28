#!/bin/bash
# Submit from this checkout with --array=0-11%3 and explicit --output/--error paths.
# Arguments: anatomy|trial|preflight, batch revision (e.g. r1).
# Each task handles one project case; no pair extraction or freeze occurs here.
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --job-name=coa115
set -euo pipefail

phase=${1:-anatomy}
revision=${2:-r1}
case "$phase" in anatomy|trial|preflight) ;; *) exit 2 ;; esac
[[ "$revision" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
cd "${SLURM_SUBMIT_DIR:?Submit from the FakeCT integration worktree}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/fakect-matplotlib

case_id=$(python3 - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
from pathlib import Path
project = json.loads(Path('configs/cohorts/coa/project-radius115.json').read_text())
print(project['cases'][int(sys.argv[1])]['case_id'])
PY
)
if [[ "$phase" == trial ]]; then
    python3 scripts/evaluation/trial_cohort_recipe.py \
        --config "configs/cohorts/coa/radius115/$case_id.ini" \
        --output "outputs/cohorts/coa/radius115/$case_id/trial-$revision"
else
    python3 scripts/evaluation/run_cohort_batch.py \
        --project configs/cohorts/coa/project-radius115.json \
        --cases "$case_id" --through "$phase" --workers 1 \
        --output "outputs/cohorts/coa/radius115/batch-$revision/$phase/$case_id"
fi
