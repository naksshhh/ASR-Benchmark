#!/bin/bash
#SBATCH --job-name=eval_synthetic
#SBATCH --output=logs/%j_eval_synthetic.out
#SBATCH --error=logs/%j_eval_synthetic.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

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

echo "====== Starting Synthetic 100 Evaluation ======"
python -m banking_asr_eval.evaluate \
  --manifest data/synthetic/manifest.json \
  --models indicwav2vec-banking-configC,whisper-medium-banking-configC \
  --output results/ \
  --workers 1
echo "====== Synthetic 100 Evaluation Completed ======"
