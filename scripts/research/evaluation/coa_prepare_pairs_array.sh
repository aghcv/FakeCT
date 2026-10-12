#!/bin/bash
# Run from the integration checkout; one frozen phantom per task.
# sbatch --array=0-11%3 --output=outputs/cohorts/coa/radius115/preparation-logs/%A_%a.out \
#   --error=outputs/cohorts/coa/radius115/preparation-logs/%A_%a.err scripts/research/evaluation/coa_prepare_pairs_array.sh
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --job-name=coa-pairs
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the syncoCT integration worktree}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/fakect-matplotlib
case_id=$(python3 - "${SLURM_ARRAY_TASK_ID:?Array index required}" <<'PY'
import json, sys
from pathlib import Path
project = json.loads(Path('configs/case-studies/coa/project.json').read_text())
index = int(sys.argv[1])
if not 0 <= index < len(project['cases']):
    raise ValueError('Array index outside project cases')
print(project['cases'][index]['case_id'])
PY
)
python3 scripts/prepare_cohort.py --config "configs/case-studies/coa/phantoms/$case_id.ini" --stage prepare
