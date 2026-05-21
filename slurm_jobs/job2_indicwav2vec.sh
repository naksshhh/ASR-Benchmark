#!/bin/bash
#SBATCH --job-name=indicwav2vec_c
#SBATCH --output=logs/%j_indic_c.out
#SBATCH --error=logs/%j_indic_c.err
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00

source ~/.bashrc
conda activate asr-eval
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

python finetune/indicwav2vec_finetune.py --config C
