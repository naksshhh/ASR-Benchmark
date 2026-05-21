#!/bin/bash
#SBATCH --job-name=whisper_c
#SBATCH --output=logs/%j_whisper_c.out
#SBATCH --error=logs/%j_whisper_c.err
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=08:00:00

source ~/.bashrc
conda activate asr-eval
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

python finetune/whisper_finetune.py --config C
