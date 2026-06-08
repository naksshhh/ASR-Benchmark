#!/usr/bin/env bash
#SBATCH --job-name=eval_lahaja_all
#SBATCH --output=logs/%j_eval_lahaja_all.out
#SBATCH --error=logs/%j_eval_lahaja_all.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

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
export TOKENIZERS_PARALLELISM=false


echo "====== Step 2: Running Evaluation on All Models on Lahaja ======"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/lahaja.json \
  --models whisper-medium-banking-configD \
  --output results/ \
  --stratify-by accent_group \
  --workers 1

echo "====== Evaluation Completed ======"
