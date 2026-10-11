#!/bin/bash
#SBATCH --job-name=fnpairhvy
#SBATCH --output=logs/fakenoise_pair.%j.out
#SBATCH --error=logs/fakenoise_pair.%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
module load container_env python3
mkdir -p logs
/usr/bin/python3 ~/repo/FakeCT/src/fakenoise.py --mode train --csv ~/repo/FakeCT/data/dataset/paired_datasets/pairs.csv --context 60 --context-step 1
