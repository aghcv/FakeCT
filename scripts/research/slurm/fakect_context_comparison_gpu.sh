#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=fakect_compare
#SBATCH --array=0-1%1
#SBATCH --output=logs/fakect_compare.%A_%a.out
#SBATCH --error=logs/fakect_compare.%A_%a.err
#SBATCH --time=28:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Freeze the common plan first with compare_context_aorta.py --stage plan.
# mkdir -p logs
# sbatch scripts/research/slurm/fakect_context_comparison_gpu.sh archive/research/configs/models/coa-context-comparison.ini
set -euo pipefail
fakect_input=${1:?Pass the frozen comparison INI}
fakect_checkout=${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd -- "$fakect_checkout"
case "${SLURM_ARRAY_TASK_ID:?Submit the two-arm array}" in
    0) fakect_arm=baseline_only ;;
    1) fakect_arm=baseline_plus_siblings ;;
    *) exit 2 ;;
esac
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-comparison-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python - <<'PY'
import tensorflow as tf
print('TensorFlow', tf.__version__, 'GPU', tf.config.list_physical_devices('GPU'), flush=True)
if not tf.config.list_physical_devices('GPU'):
    raise SystemExit('No GPU visible in TensorFlow container')
PY
crun -p "$HOME/envs/fakect" python scripts/research/compare_context_aorta.py --config "$fakect_input" --stage all --arm "$fakect_arm"
python3 scripts/research/report_context_comparison.py --config "$fakect_input"
