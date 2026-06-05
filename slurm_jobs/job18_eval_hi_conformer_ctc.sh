#!/usr/bin/env bash
#SBATCH --job-name=eval_hi_conformer_ctc
#SBATCH --output=logs/%j_eval_hi_conformer_ctc.out
#SBATCH --error=logs/%j_eval_hi_conformer_ctc.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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

# 1. Run Latency benchmarks using benchmark.py
echo "====== Running Latency + Quality Benchmark (100 samples, Warmup+Timed) ======"

echo "--> Benchmark: Synthetic 100 (Hinglish Banking)"
python -m banking_asr_eval.benchmark \
  --config config.yaml \
  --manifest data/manifests/synthetic_100.json \
  --models stt-hi-conformer-ctc-large \
  --output results/

echo "--> Benchmark: Kathbath Hindi (Subset of 100 samples)"
python -m banking_asr_eval.benchmark \
  --config config.yaml \
  --manifest data/manifests/kathbath_hindi.json \
  --models stt-hi-conformer-ctc-large \
  --output results/ \
  --max-samples 100


# 2. Run Full Quality evaluations using evaluate.py
echo "====== Running Full Quality Evaluations ======"

echo "--> Quality Eval: Synthetic 100 (Hinglish Banking)"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/synthetic_100.json \
  --models stt-hi-conformer-ctc-large \
  --output results/ \
  --workers 1

echo "--> Quality Eval: Kathbath Hindi (3,151 samples)"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/kathbath_hindi.json \
  --models stt-hi-conformer-ctc-large \
  --output results/ \
  --workers 1

echo "--> Quality Eval: Lahaja Multi-Accent Dataset (6,152 samples)"
python -m banking_asr_eval.evaluate \
  --manifest data/manifests/lahaja.json \
  --models stt-hi-conformer-ctc-large \
  --output results/ \
  --stratify-by accent_group \
  --workers 1

echo "====== Conformer-CTC Evaluation Script Completed ======"
