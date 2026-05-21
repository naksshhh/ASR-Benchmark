#!/bin/bash
#SBATCH --job-name=whisper_ablations
#SBATCH --output=logs/%j_whisper_ablations.out
#SBATCH --error=logs/%j_whisper_ablations.err
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate asr-eval
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

for config in A B C; do
    echo "Running Whisper Fine-tuning for Config $config"
    python finetune/whisper_finetune.py --config $config
done
