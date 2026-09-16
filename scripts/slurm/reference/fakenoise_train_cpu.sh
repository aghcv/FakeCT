#!/bin/bash
#SBATCH --job-name=fntrain
#SBATCH --output=logs/fakenoise_train.%j.out
#SBATCH --error=logs/fakenoise_train.%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail

enable_lmod
module load container_env tensorflow-cpu/1.15.0
mkdir -p logs
#/usr/bin/python3 -m pip install --user -r ~/repo/FakeCT/requirements.txt
/usr/bin/python3 ~/repo/FakeCT/src/fakenoise.py --mode train --csv ~/repo/FakeCT/data/dataset/paired_datasets/pairs.csv
