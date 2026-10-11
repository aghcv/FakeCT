#!/bin/bash
#SBATCH --job-name=fnpair
#SBATCH --output=logs/fakenoise_pair.%j.out
#SBATCH --error=logs/fakenoise_pair.%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
module load container_env python3
mkdir -p logs
crun -p ~/.conda/envs/fakect python ~/repo/FakeCT/src/fakenoise.py --mode pair --dataset-dir ~/repo/FakeCT/data/dataset/
