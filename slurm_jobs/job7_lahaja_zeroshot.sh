#!/bin/bash
#SBATCH --job-name=lahaja_zeroshot
#SBATCH --output=logs/%j_lahaja_zeroshot.out
#SBATCH --error=logs/%j_lahaja_zeroshot.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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

# export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

echo "====== Starting Lahaja Zero-Shot Evaluation (IndicWav2Vec) ======"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/lahaja.json \
  --models indicwav2vec-hindi \
  --output results/ \
  --stratify-by accent_group \
  --workers 1

echo "====== Starting Lahaja Zero-Shot Evaluation (Whisper) ======"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/lahaja.json \
  --models whisper-medium-hi \
  --output results/ \
  --stratify-by accent_group \
  --workers 1

