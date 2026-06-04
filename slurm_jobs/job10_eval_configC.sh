#!/bin/bash
#SBATCH --job-name=eval_configC
#SBATCH --output=logs/%j_eval_configC.out
#SBATCH --error=logs/%j_eval_configC.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

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

echo "====== Starting Config C Evaluation ======"

echo "1. Evaluating on Kathbath Hindi benchmark..."
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/kathbath_hindi.json \
  --models indicwav2vec-hindi,whisper-medium-hi,indicwav2vec-banking-configC,whisper-medium-banking-configC \
  --output results/ \
  --workers 1

echo "2. Evaluating on Synthetic 100 banking manifest..."
python -m banking_asr_eval.evaluate \
  --manifest data/synthetic/manifest.json \
  --models indicwav2vec-hindi,whisper-medium-hi,indicwav2vec-banking-configC,whisper-medium-banking-configC \
  --output results/ \
  --workers 1

echo "====== Config C Evaluation Completed ======"
