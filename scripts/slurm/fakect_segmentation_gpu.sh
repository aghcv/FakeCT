#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=fakect_segment
#SBATCH --output=logs/fakect_segment.%j.out
#SBATCH --error=logs/fakect_segment.%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

# Submit only after reviewing the study ROI, parameter ranges, and prepared data.
# From the integrated FakeCT checkout, create logs BEFORE submitting:
#   mkdir -p logs
#   sbatch scripts/slurm/fakect_segmentation_gpu.sh /absolute/path/to/study.ini
# The prepared dataset and a new train.model_directory must be set in that INI.
# This wrapper uses the confirmed Wahab TensorFlow container; it installs nothing.
set -euo pipefail

fakect_study_ini=${1:?Pass the reviewed study INI path as the first argument}
fakect_checkout=${FAKECT_CHECKOUT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd -- "$fakect_checkout"
if [[ ! -f scripts/train_study.py ]]; then
    echo 'Submit from the integrated FakeCT checkout, or set FAKECT_CHECKOUT.' >&2
    exit 2
fi
module load container_env tensorflow-gpu/2.17
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=1
crun -p "$HOME/envs/fakect" python scripts/train_study.py --config "$fakect_study_ini" --stage fit
