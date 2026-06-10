#!/bin/bash
# ==============================================================================
# Script to launch Nemotron-3.5 ASR Fine-Tuning on Param Rudra
# Fine-tunes nvidia/nemotron-3.5-asr-streaming-0.6b on Config E manifest.
# ==============================================================================

# SBATCH directives for SLURM script allocation (if resubmitting as job)
#SBATCH --job-name=nemotron_ft
#SBATCH --output=logs/%j_nemotron_ft.out
#SBATCH --error=logs/%j_nemotron_ft.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate asr 2>/dev/null || conda activate asr-eval

# Determine repository root and navigate to it
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    if [[ "$SLURM_SUBMIT_DIR" == */finetune ]]; then
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
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

# Setup directory for checkpoints
OUT_DIR="/scratch/$USER/checkpoints/nemotron-3.5-banking-configE"
os.makedirs -p "$OUT_DIR"

echo "====== Step 1: Pre-downloading Nemotron Checkpoint (Login Node equivalent) ======"
# Pull the .nemo model weights locally before running offline on compute node
python -c "
import nemo.collections.asr as nemo_asr
try:
    nemo_asr.models.ASRModel.from_pretrained('nvidia/nemotron-3.5-asr-streaming-0.6b')
    print('Successfully downloaded and cached Nemotron-3.5!')
except Exception as e:
    print('Error pre-downloading model:', e)
"

echo "====== Step 2: Preparing Config E Manifest ======"
python finetune/prepare_configE.py

# Launch NeMo training wrapper script
# NeMo models are fine-tuned using NeMo's speech_to_text_hybrid_rnnt_ctc_bpe script
# We point to our local Config E manifest and load the pre-trained weights.
echo "====== Step 3: Launching NeMo Fine-Tuning ======"

# Check if NeMo example script is accessible
NEMO_DIR="$REPO_ROOT/NeMo"
if [ ! -d "$NEMO_DIR" ]; then
    echo "NeMo repository not found at $NEMO_DIR. Cloning AI4Bharat's fork..."
    git clone https://github.com/AI4Bharat/NeMo.git -b nemo-v2
    cd NeMo && bash reinstall.sh && cd ..
fi

python "$NEMO_DIR/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py" \
    --config-path="$NEMO_DIR/examples/asr/conf/conformer/" \
    --config-name="conformer_hybrid_transducer_ctc_bpe.yaml" \
    model.train_ds.manifest_filepath="data/manifests/finetune_configE.json" \
    model.validation_ds.manifest_filepath="data/manifests/finetune_configE.json" \
    model.train_ds.batch_size=8 \
    model.validation_ds.batch_size=8 \
    model.train_ds.num_workers=4 \
    model.validation_ds.num_workers=4 \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=3 \
    trainer.val_check_interval=1.0 \
    model.optim.lr=1e-5 \
    model.optim.sched.warmup_steps=500 \
    exp_manager.checkpoint_callback_params.save_top_k=2 \
    exp_manager.checkpoint_callback_params.monitor="val_wer" \
    exp_manager.checkpoint_callback_params.mode="min" \
    exp_manager.exp_dir="$OUT_DIR" \
    +model.encoder.from_pretrained="nvidia/nemotron-3.5-asr-streaming-0.6b" \
    +model.decoder.from_pretrained="nvidia/nemotron-3.5-asr-streaming-0.6b" \
    +model.joint.from_pretrained="nvidia/nemotron-3.5-asr-streaming-0.6b"

echo "====== Nemotron Fine-Tuning Script Completed ======"
