#!/bin/bash
#SBATCH --job-name=indicwav2vec_d
#SBATCH --output=logs/%j_indic_d.out
#SBATCH --error=logs/%j_indic_d.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

echo "====== Starting IndicWav2Vec Config D Fine-Tuning ======"
python finetune/indicwav2vec_finetune.py --config D --epochs 1
echo "====== IndicWav2Vec Config D Fine-Tuning Completed ======"
