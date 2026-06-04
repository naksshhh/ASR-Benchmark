#!/bin/bash
#SBATCH --job-name=eval_synthetic_d
#SBATCH --output=logs/%j_eval_synthetic_d.out
#SBATCH --error=logs/%j_eval_synthetic_d.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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

echo "====== Starting Synthetic 100 Evaluation Config D (IndicWav2Vec) ======"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/synthetic_100.json \
  --models indicwav2vec-banking-configD \
  --output results/ \
  --workers 1

echo "====== Starting Synthetic 100 Evaluation Config D (Whisper) ======"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/synthetic_100.json \
  --models whisper-medium-banking-configD \
  --output results/ \
  --workers 1
