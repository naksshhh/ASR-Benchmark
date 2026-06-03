#!/bin/bash
#SBATCH --job-name=configD_finetune
#SBATCH --output=logs/%j_configD_finetune.out
#SBATCH --error=logs/%j_configD_finetune.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

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

echo "====== Starting Config D Fine-Tuning Sweep ======"

echo "1. Running IndicWav2Vec Config D Fine-tuning..."
python finetune/indicwav2vec_finetune.py --config D

echo "2. Running Whisper-medium Config D Fine-tuning..."
python finetune/whisper_finetune.py --config D

echo "====== Config D Fine-Tuning Sweep Completed ======"
