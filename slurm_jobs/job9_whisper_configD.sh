#!/bin/bash
#SBATCH --job-name=whisper_d
#SBATCH --output=logs/%j_whisper_d.out
#SBATCH --error=logs/%j_whisper_d.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# #SBATCH --mem=32G



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

echo "====== Starting Whisper Config D Fine-Tuning ======"
python finetune/whisper_finetune.py --config D --epochs 1.0
echo "====== Whisper Config D Fine-Tuning Completed ======"
