#!/bin/bash
#SBATCH --job-name=prep_aug
#SBATCH --output=logs/%j_prep.out
#SBATCH --error=logs/%j_prep.err
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate asr-eval
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

python finetune/prepare_data.py
# If you want offline augmentation, you can run a script here, but currently augment.py is imported in train script.
# python finetune/augment.py
