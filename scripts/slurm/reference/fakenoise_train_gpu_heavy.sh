#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --job-name=fnt_gpuhvy
#SBATCH --output=logs/fakenoise_train.%j.out
#SBATCH --error=logs/fakenoise_train.%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# one-time setup: install from requirements.txt
#mkdir -p ~/envs/tf-gpu
#crun -c -p ~/envs/tf-gpu
#crun -p ~/envs/tf-gpu pip install -r /home/aghorban/repo/FakeCT/requirements.txt

enable_lmod
module load container_env tensorflow-gpu/1.15.0
module load gcc/4
module load cuda/9.2
mkdir -p logs
/usr/bin/python3 ~/repo/FakeCT/src/fakenoise.py --mode train --csv ~/repo/FakeCT/data/dataset/paired_datasets/pairs.csv --context 60 --context-step 1
