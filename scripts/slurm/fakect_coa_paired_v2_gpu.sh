#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=coa_paired_v2
#SBATCH --output=logs/coa_paired_v2.%j.out
#SBATCH --error=logs/coa_paired_v2.%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_paired_v2_config=${1:?Pass the frozen paired-classifier v2 INI}
fakect_paired_v2_stage=${2:-diagnostic}
if [[ "$fakect_paired_v2_stage" != diagnostic && "$fakect_paired_v2_stage" != fit ]]; then
    echo 'Second argument must be diagnostic or fit' >&2
    exit 2
fi
cd -- "${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-coa-paired-v2-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
# Use separate processes so smoke allocations are released before training.
if [[ "$fakect_paired_v2_stage" == diagnostic ]]; then
    crun -p "$HOME/envs/fakect" python scripts/train_coa_paired_v2.py --config "$fakect_paired_v2_config" --stage smoke
fi
crun -p "$HOME/envs/fakect" python scripts/train_coa_paired_v2.py --config "$fakect_paired_v2_config" --stage "$fakect_paired_v2_stage"
