#!/bin/bash
#SBATCH --job-name=whisper_ablations
#SBATCH --output=logs/%j_whisper_ablations.out
#SBATCH --error=logs/%j_whisper_ablations.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

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

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

for config in A B C; do
    echo "Running Whisper Fine-tuning for Config $config"
    python finetune/whisper_finetune.py --config $config
done
