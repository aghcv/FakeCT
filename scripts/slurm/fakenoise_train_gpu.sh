#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --job-name=fntrain_gpu
#SBATCH --output=logs/fakenoise_train.%j.out
#SBATCH --error=logs/fakenoise_train.%j.err
#SBATCH --time=28:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
# From the submission directory, create logs/ BEFORE sbatch opens its log files.
# Pass --csv and --out-dir explicitly; extra fakenoise arguments follow unchanged.
# Example: sbatch scripts/slurm/fakenoise_train_gpu.sh --csv /data/pairs.csv --out-dir /scratch/run1
set -euo pipefail
# Slurm can spool this script elsewhere. Submit with --export=ALL,FAKECT_ROOT=/path/to/checkout
# or submit from the repository root, which SLURM_SUBMIT_DIR then records.
FAKECT_ROOT="${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
FAKECT_ENV_PREFIX="${FAKECT_ENV_PREFIX:-$HOME/envs/fakect}"
if [[ ! -f "$FAKECT_ROOT/src/fakenoise.py" ]]; then
    echo "Set FAKECT_ROOT to the integration checkout." >&2
    exit 2
fi
FAKENOISE_ARGS=("$@")
csv_value=""
out_value=""
while (( $# )); do
    case "$1" in
        --csv|--out-dir)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ "$1" == --csv ]]; then csv_value="$2"; else out_value="$2"; fi
            shift 2
            ;;
        --csv=*) csv_value="${1#*=}"; shift ;;
        --out-dir=*) out_value="${1#*=}"; shift ;;
        *) shift ;;
    esac
done
if [[ -z "$csv_value" || -z "$out_value" ]]; then
    echo "Usage: $0 --csv /path/to/pairs.csv --out-dir /path/to/new-run [training options]" >&2
    exit 2
fi
module load container_env tensorflow-gpu/2.17
crun -p "$FAKECT_ENV_PREFIX" python "$FAKECT_ROOT/src/fakenoise.py" \
    --mode train --context 15 --context-step 4 "${FAKENOISE_ARGS[@]}"
