#!/bin/bash
#SBATCH --job-name=eval_configE
#SBATCH --output=logs/%j_eval_configE.out
#SBATCH --error=logs/%j_eval_configE.err
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

echo "====== Starting Config E Models Evaluation Sweep ======"

# Models to evaluate
MODELS="whisper-turbo-banking-configE,nemotron-banking-configE"

# 1. Evaluate on Synthetic 100 (Hinglish Banking)
echo "====== Evaluating on Synthetic 100 ======"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/synthetic_100.json \
  --models $MODELS \
  --output results/ \
  --workers 1

# 2. Evaluate on Kathbath Hindi (General Hindi)
echo "====== Evaluating on Kathbath Hindi ======"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/kathbath_hindi.json \
  --models $MODELS \
  --output results/ \
  --workers 1

# 3. Evaluate on Lahaja (Dialect-rich, Stratified by Accent Group)
echo "====== Evaluating on Lahaja (Accent-stratified) ======"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/lahaja.json \
  --models $MODELS \
  --output results/ \
  --stratify-by accent_group \
  --workers 1

echo "====== Config E Evaluation Sweep Completed ======"
