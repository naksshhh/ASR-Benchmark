#!/bin/bash
#SBATCH --job-name=indicwav2vec_ablation
#SBATCH --output=logs/%j_indic_ablation.out
#SBATCH --error=logs/%j_indic_ablation.err
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=08:00:00

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

for config in A B; do
    echo "Running Config $config"
    python finetune/indicwav2vec_finetune.py --config $config
done
