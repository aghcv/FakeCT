#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=fakect_context
#SBATCH --output=logs/fakect_context.%j.out
#SBATCH --error=logs/fakect_context.%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# mkdir -p logs; sbatch scripts/slurm/fakect_context_gpu.sh configs/models/coa-context-diagnostic.ini
set -euo pipefail
fakect_input=${1:?Pass a context-model INI}
fakect_stage=${2:-all}
fakect_checkout=${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd -- "$fakect_checkout"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-context-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python - <<'PY'
import tensorflow as tf
print('TensorFlow',tf.__version__,'GPU',tf.config.list_physical_devices('GPU'),flush=True)
if not tf.config.list_physical_devices('GPU'):
    raise SystemExit('No GPU is visible inside the TensorFlow container')
PY
crun -p "$HOME/envs/fakect" python scripts/train_context_aorta.py --config "$fakect_input" --stage "$fakect_stage"
