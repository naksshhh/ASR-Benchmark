#!/bin/bash
#SBATCH --job-name=eval_nemotron
#SBATCH --output=logs/%j_eval_nemotron.out
#SBATCH --error=logs/%j_eval_nemotron.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
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
# export HF_HUB_OFFLINE=1 # Commented out since compute nodes have internet connectivity

# Disable NCCL P2P/IB and tokenizer parallelism to prevent deadlocks on kernel < 5.5.0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

# First, ensure the model is pre-downloaded/cached since the compute nodes are offline.
# We will disable HF_HUB_OFFLINE temporarily if needed. Run "python test_nemotron.py"
# on the login node first to pre-cache the model weights and verify loading works.

# echo "====== Starting Kathbath Evaluation (Nemotron-3.5 ASR, target_lang=hi-IN) ======"
# # Make sure target_lang is set to hi-IN
# sed -i 's/target_lang: "auto"/target_lang: "hi-IN"/g' config.yaml
# python -m banking_asr_eval.evaluate \
#   --manifest data/manifests/kathbath_hindi.json \
#   --models nemotron-3.5-asr \
#   --output results/ \
#   --workers 1

echo "====== Starting Synthetic 100 Evaluation (Nemotron-3.5 ASR, target_lang=auto) ======"
# Change target_lang to auto for code-switching/language detection
sed -i 's/target_lang: "hi-IN"/target_lang: "auto"/g' config.yaml
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/synthetic_100.json \
  --models nemotron-3.5-asr \
  --output results/ \
  --workers 1

# echo "====== Starting Lahaja Evaluation (Nemotron-3.5 ASR, target_lang=auto) ======"
# python -m banking_asr_eval.evaluate \
#   --manifest data/manifests/lahaja.json \
#   --models nemotron-3.5-asr \
#   --output results/ \
#   --stratify-by accent_group \
#   --workers 1

# Restore default target_lang to hi-IN
sed -i 's/target_lang: "auto"/target_lang: "hi-IN"/g' config.yaml
echo "====== Evaluation Completed ======"
