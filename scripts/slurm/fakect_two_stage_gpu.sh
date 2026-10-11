#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=fakect_two_stage
#SBATCH --output=logs/fakect_two_stage.%j.out
#SBATCH --error=logs/fakect_two_stage.%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# From the integration checkout: mkdir -p logs
# sbatch scripts/slurm/fakect_two_stage_gpu.sh configs/models/coa-two-stage.ini
set -euo pipefail
fakect_input=${1:?Pass a two-stage model INI}
fakect_stage=${2:-all}
fakect_checkout=${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd -- "$fakect_checkout"
test -f scripts/train_two_stage.py
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python - <<'PY'
import json
import platform
import tensorflow as tf
devices = tf.config.list_physical_devices('GPU')
print(json.dumps({'python': platform.python_version(), 'tensorflow': tf.__version__,
                  'gpu_devices': [str(d) for d in devices]}), flush=True)
if not devices:
    raise SystemExit('GPU allocation is not visible in the TensorFlow container')
PY
crun -p "$HOME/envs/fakect" python scripts/train_two_stage.py --config "$fakect_input" --stage "$fakect_stage"
