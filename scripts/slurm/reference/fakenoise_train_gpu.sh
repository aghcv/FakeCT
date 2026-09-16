#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --job-name=fntrain_gpu
#SBATCH --output=logs/fakenoise_train.%j.out
#SBATCH --error=logs/fakenoise_train.%j.err
#SBATCH --time=28:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

# one-time setup: install from requirements.txt
#mkdir -p ~/envs/tf-gpu
#crun -c -p ~/envs/tf-gpu
#crun -p ~/envs/tf-gpu pip install -r /home/aghorban/repo/FakeCT/requirements.txt

module load container_env tensorflow-gpu/2.17

crun -p ~/envs/fakect python ~/repo/FakeCT/src/fakenoise.py --mode train --csv ~/repo/FakeCT/data/dataset/paired_datasets/pairs.csv --context 15 --context-step 4
