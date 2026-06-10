#!/bin/bash
#SBATCH --job-name=whisper_turbo_e
#SBATCH --output=logs/%j_whisper_turbo_e.out
#SBATCH --error=logs/%j_whisper_turbo_e.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

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

# Set scratch paths
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

# Disable NCCL P2P/IB and tokenizer parallelism to prevent deadlocks on kernel < 5.5.0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

echo "====== Step 1: Preparing Config E Manifest ======"
python finetune/prepare_configE.py

echo "====== Step 2: Starting Whisper Large v3 Turbo Config E Fine-Tuning ======"
python finetune/whisper_turbo_finetune.py --config E --epochs 2.0
echo "====== Whisper Large v3 Turbo Config E Fine-Tuning Completed ======"
