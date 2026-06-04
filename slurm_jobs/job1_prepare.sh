#!/bin/bash
#SBATCH --job-name=prep_aug
#SBATCH --output=logs/%j_prep.out
#SBATCH --error=logs/%j_prep.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# #SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate asr 2>/dev/null || conda activate asr-eval

# Determine repository root and navigate to it
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    if [[ "$SLURM_SUBMIT_DIR" == */slurm_jobs ]]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR/.."
    else
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    fi
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$SCRIPT_DIR/.."
fi
cd "$REPO_ROOT"

# export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

python finetune/prepare_data.py
# If you want offline augmentation, you can run a script here, but currently augment.py is imported in train script.
# python finetune/augment.py
