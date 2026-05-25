#!/bin/bash
#SBATCH --job-name=eval_finetuned
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# #SBATCH --mem=32G
#SBATCH --time=03:00:00

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

echo "Evaluating on Banking 100 Test Set..."
python -m banking_asr_eval.evaluate \
  --config config.yaml \
  --manifest data/manifests/banking_100_test.json \
  --models indicwav2vec-banking,whisper-medium-banking,voxtral-mini-3b

echo "Evaluating on Kathbath Hindi..."
python -m banking_asr_eval.evaluate \
  --config config.yaml \
  --manifest data/manifests/kathbath_hindi.json \
  --models indicwav2vec-banking,indicwav2vec-hindi

echo "Visualizing Results..."
python -m banking_asr_eval.visualize --results results/
