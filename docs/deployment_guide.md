# Krim.ai Voice ASR & TTS AWS Deployment Guide

This document outlines the architectural recommendations, software versions, critical dependency overrides, and compiler/infrastructure configurations required to deploy Krim.ai's production voice processing pipeline to AWS.

---

## 1. Production Model Recommendations

For a production banking voice bot scaling to **500+ concurrent calls**, the pipeline requires specific Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) choices:

### Text-to-Speech (TTS) Recommendation
*   **Primary Choice (Cloud API):** **Microsoft Azure Neural TTS (via `edge-tts`)**
    *   **Voices:** `hi-IN-SwaraNeural` (Female) and `hi-IN-MadhurNeural` (Male).
    *   **Rationale:** Standard monolingual TTS engines fail on Hinglish code-switched dialogues (e.g. *"Mera loan amount check kijiye"*). Azure's neural voices blend Hindi and English terms natively in a single query with near-perfect accentuation.
    *   **Self-Hosted / Open-Source Alternative:** **Coqui TTS** or **IndicTTS**. While stitching two monolingual engines (like English **Kokoro** + Hindi **IndicTTS**) is possible, it introduces audible segment-transition latency and alignment issues.
*   **Deployment Vector:** Deploy as a microservice wrapping `edge-tts` or access via official Azure Cognitive Services SDK on AWS.

### Automatic Speech Recognition (ASR) Recommendation
Depending on the latency, throughput, and system resource requirements, choose one of the following ASR topologies:

| Deployment Objective | Recommended Model | Deployment Backend | Offline RTF (A100) | Perceived Latency | Hardware Cost |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Bilingual Live Streaming** (Conversational Bot) | **`Nemotron-3.5-ASR-Streaming-0.6b`** | NeMo Streaming / Triton | 0.174 | **Ultra-Low (<100ms)** | Medium |
| **High-Throughput / Concurrency** (500+ Calls) | **`stt_hi_conformer_ctc_large`** | Triton Server + TensorRT-ASR | **0.013** | Low (49ms chunk) | **Very Low** |
| **Offline Analytics / High Precision** (Batch) | **`Whisper-Medium-Banking` (Config C)** | Faster-Whisper / vLLM | 0.730 | High (Segment end) | High |

#### Architectural Decisions:
1.  **For Interactive Bot Loops:** Deploy **Nemotron-3.5-ASR-Streaming-0.6B**. Since it is an autoregressive streaming model (RNN-T), it transcribes speech chunk-by-chunk as the user is speaking, making the user-perceived delay negligible. Set `target_lang="auto"` to handle dynamic Hindi/English vocabulary shifting.
2.  **For Concurrency Scaling (Cost-Saving):** Deploy the Bilingual **Conformer-CTC** model on **NVIDIA Triton Inference Server** with TensorRT-ASR. Non-autoregressive CTC models do not have repetition loop risks, process chunk batches in parallel on GPU tensor cores, and minimize compute footprints.
3.  **Avoid Multimodal LLMs in ASR loops:** Do not use models like Voxtral-Mini-3B in the live voice loop. Although zero-shot accuracy is high, they are computationally intensive, do not support native chunk-based streaming, and are cost-prohibitive for 500+ parallel calls.

---

## 2. Environment & Dependency Stack

The deployment team must set up the runtime environment according to these specific versions and sources:

### Core Environment Variables
```bash
# Force PyTorch/HuggingFace to run completely offline on compute instances
export HF_HUB_OFFLINE=1
# Route heavy Hugging Face downloads to fast local NVMe partitions (scratch space)
export HF_HOME=/scratch/$USER/hf_cache
```

### Python Package Constraints (`requirements.txt`)
Ensure the following package bounds are respected:
```text
torch>=2.2.0
transformers>=4.41.0
accelerate>=0.28.0
datasets>=2.18.0
jiwer>=3.0.3
soundfile>=0.12.1
librosa>=0.10.1
soundfile>=0.12.1
omegaconf>=2.3.0
edge-tts>=6.1.10
```

### Critical NeMo Dependency Conflict (IndicConformer)
*   **The Issue:** Standard PyPI packages (`pip install nemo_toolkit[asr]`) **cannot** load the AI4Bharat Hindi Conformer checkpoint (`ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large`). It contains custom config keys (multisoftmax, multi-tokenizer, etc.) that throw configuration parser errors in vanilla NeMo.
*   **The Fix:** The deployment team must install **AI4Bharat's custom fork of NeMo (`nemo-v2` branch)** from source:
    ```bash
    git clone https://github.com/AI4Bharat/NeMo.git
    cd NeMo
    git checkout nemo-v2
    bash reinstall.sh
    ```

---

## 3. High-Performance Caching & Parallelization Fixes

To achieve real-time scale, three bottlenecks identified during Param Rudra cluster runs must be patched in the container templates:

### A. Parallel Audio Scanning (Avoiding NFS Latency)
*   **Problem:** Network filesystems (like GPFS or AWS EFS) suffer from massive file-stat latency when sequentially reading folders with thousands of audio segments.
*   **Fix:** Parallelize metadata scans and duration queries using a multi-threaded pool (e.g. 32 threads using `ThreadPoolExecutor`). This reduces dataset loading times from 15 minutes to under 20 seconds.

### B. Early Text Filtering for Whisper Feature Extraction
*   **Problem:** Standard HuggingFace preprocessing maps all audio samples to log-mel spectrogram tensors *before* verifying if the token/sequence length exceeds the model limit, writing gigabytes of redundant arrays to scratch disks.
*   **Fix:** Pre-filter dataset manifests on raw text strings *prior* to calling the feature extractor. This reduces model training initialization times from 1.5 hours to less than 15 minutes.

---

## 4. Nemotron-3.5-ASR Production Monkeypatches

To run the bilingual `nvidia/nemotron-3.5-asr-streaming-0.6b` model in production, the following source code workarounds must be implemented in the inference wrappers (e.g. `nemo_local.py`):

### A. Non-Strict State Dict Restorations
```python
# Force torch/pytorch-lightning/nemo state-dict loaders to load with strict=False
# This prevents crashes due to unexpected prompt-tuning keys (e.g. prompt_kernel)
def make_patched_load_state_dict(original_fn):
    def patched(self, state_dict, strict=True):
        class_name = self.__class__.__name__
        if any(p in class_name for p in ["EncDec", "RNNT", "CTC", "Joint", "Model", "Prompt"]):
            strict = False
        return original_fn(self, state_dict, strict=strict)
    return patched

import torch
torch.nn.Module.load_state_dict = make_patched_load_state_dict(torch.nn.Module.load_state_dict)
```

### B. Model Registry Aliasing
```python
# Nemotron references a deprecated/internal 'rnnt_bpe_models_prompt' module. 
# Explicitly map it to the hybrid module to prevent import crashes:
import sys
import types
import nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt as hybrid_prompt
sys.modules['nemo.collections.asr.models.rnnt_bpe_models_prompt'] = hybrid_prompt
```

### C. Lhotse DataLoader Language Tag Mapping
*   **Problem:** NeMo’s Lhotse loader fetches `cut.supervisions[0].language`. Under real-time streaming or raw-path inference, this field resolves to `None` or language tags (e.g., `"hindi"`, `"mixed"`) not found in Nemotron’s strict prompt dictionary. This throws `ValueError: Unknown prompt key`.
*   **Fix:** Monkeypatch `_get_prompt_index` to route empty or custom labels to valid locales (`hi-IN` or `en-US`):
```python
import nemo.collections.asr.data.audio_to_text_lhotse_prompt as lhotse_prompt_mod

def patch_prompt_index(orig_fn):
    def patched(self, lang):
        if lang is None or str(lang) == 'None' or str(lang).strip() == '':
            lang = 'hi-IN' # Default to Hindi-Belt locale
        
        lang_str = str(lang).lower().strip()
        if lang_str in ['hi', 'hindi', 'hi-in', 'mixed', 'auto']:
            lang = 'hi-IN'
        elif lang_str in ['en', 'english', 'en-us']:
            lang = 'en-US'
        else:
            lang = 'hi-IN' # Fallback
        return orig_fn(self, lang)
    return patched

lhotse_prompt_mod.PromptedAudioToTextLhotseDataset._get_prompt_index = patch_prompt_index(
    lhotse_prompt_mod.PromptedAudioToTextLhotseDataset._get_prompt_index
)
```

### D. Prompt Shape Mismatch (Off-by-One Fix)
*   **Problem:** A prompt sequence length mismatch between the encoder output and target prompts causes `torch.cat` to fail on dimension alignment.
*   **Fix:** Wrap the model's forward pass to intercept `torch.cat` and pad or slice mismatching prompt tensors dynamically.

---

## 5. Whisper Hallucination & Repetition Control

During accent-stratified evaluations (e.g., LAHAJA Dialect ASR), Whisper models often enter infinite loops or truncate speech prefixes on noisy audio.
*   **Fix:** Set the following generation configurations explicitly in `whisper_local.py` or the Hugging Face generation configuration:
```python
# Suppress infinite repetition loops and context carrying failures
generation_config = {
    "condition_on_prev_tokens": False,       # Disables sequence carry-over history
    "repetition_penalty": 1.1,               # Penalizes loops
    "no_repeat_ngram_size": 4,               # Strict n-gram limits
    "compression_ratio_threshold": 1.35      # Triggers fallback on compression loop detection
}
```

---

## 6. Infrastructure & Compiler Constraints

### A. CUDA Driver Mismatch (Nemotron Constraint)
*   **Symptom:** `No conditional node support for Cuda. Cuda graphs with while loops are disabled, decoding speed will be slower.`
*   **Requirement:** Standard AWS GPU instances (e.g., `g5.xlarge`, `p4de.24xlarge`) must run CUDA driver versions supporting at least **CUDA 12.6** (Param Rudra was restricted to CUDA 12.4, resulting in slower autoregressive decoding speeds).

### B. NCCL Inter-GPU & Tokenizer Deadlocks
*   **Symptom:** Processes freeze indefinitely during multi-GPU runs or tokenization execution on older Linux kernels (e.g., RHEL `4.18.0`).
*   **Fix:** Add the following environment overrides in the Docker container launch configurations:
```bash
# Prevent NCCL peer-to-peer/InfiniBand initialization deadlocks on older host kernels
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Suppress tokenizer deadlocks under PyTorch multiprocessing
export TOKENIZERS_PARALLELISM=false
```

### C. Host Memory Exhaustion (SIGKILL 9)
*   **Symptom:** Large-model evaluation jobs are terminated instantly with exit code `0:9` (SIGKILL).
*   **Fix:** Multi-GPU processes duplicate model memory footprints on host CPU memory (RAM) before copying to VRAM. Ensure instances are configured with a minimum of **32GB Host RAM** per A100/A10 GPU. If memory is constrained, set CPU process workers to 1 (`--workers 1`).
