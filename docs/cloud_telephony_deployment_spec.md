# Production Cloud Specification: Telephony Voice Bot (LiveKit + Vobiz SIP)
### Target Platform: AWS L4 GPU (24GB VRAM) on AWS `g6.xlarge` / `g6.2xlarge`

This document specifies the target architecture, VRAM allocation, container configurations, and operational parameters for deploying the Krim.ai voice bot. This sheet should be handed directly to the AWS Cloud Infrastructure and DevOps teams.

---

## 1. Capacity & VRAM Allocation Spec (Per Node)

Each physical EC2 node represents a **single NVIDIA L4 GPU (24GB VRAM)**. 
*   **Target Capacity:** Max **8 concurrent calls** per node.
*   **VRAM Allocation Map (CUDA MPS Enabled):**

| Service | Container Runtime | VRAM Budget | Key Configuration |
| :--- | :--- | :---: | :--- |
| **vLLM (LLM Brain)** | `vllm/vllm-openai:latest` | **~13.2 GB** (55%) | Qwen3-8B-AWQ (int4), max len 8192, guided decoding |
| **ASR Server** | Custom CT2 / PyTorch | **~1.5 GB** | `ASR_ENGINE=whisper`, CT2 int8_float16, utterance-final VAD |
| **TTS (Kokoro)** | `ghcr.io/remsky/kokoro-fastapi-gpu` | **~2.0 GB** (2×1GB) | `TTS_ENGINE=kokoro`, 2 load-balanced replicas |
| **CUDA Overhead** | — | **~0.3 GB** | VRAM context allocations |
| **System Headroom** | Spikes / MPS scheduling | **~7.0 GB** | Safe margins for concurrent LLM KV-cache allocation |
| **TOTAL VRAM** | | **24.0 GB** | |

---

## 2. Infrastructure Operations: CUDA MPS Setup

Because three separate containerized processes (vLLM, ASR, TTS) share a single L4 GPU, the host system **must run CUDA Multi-Process Service (MPS)**.
*   **Why:** Default CUDA scheduling serializes kernel execution across processes, causing context-switching latency spikes. CUDA MPS allows parallel kernel execution and partition-level resource limits, preventing VRAM spikes from crashing neighboring containers.
*   **Host Script (run on host system before starting Docker containers):**
    ```bash
    # Enable persistence mode
    nvidia-smi -pm 1
    
    # Start the MPS control daemon
    export CUDA_VISIBLE_DEVICES=0
    nvidia-cuda-mps-control -d
    
    # Optional: Pin resource shares to prevent vLLM from starving ASR/TTS
    # Allocates 60% of compute to vLLM, 20% to ASR, 20% to TTS
    echo "set_active_device_percentage 0 60" | nvidia-cuda-mps-control
    ```

---

## 3. Container Configuration Specs

### Service A: LLM serving (vLLM / Qwen3-8B AWQ-int4)
*   **Image:** `vllm/vllm-openai:latest`
*   **Environment Variables:**
    ```bash
    HF_HUB_OFFLINE=1
    HF_HOME=/opt/ml/metadata/cache
    ```
*   **Startup Command:**
    ```bash
    python3 -m vllm.entrypoints.openai.api_server \
        --model /opt/ml/models/qwen3-8b-awq \
        --quantization awq \
        --gpu-memory-utilization 0.55 \
        --max-model-len 8192 \
        --disable-log-requests \
        --port 8000
    ```
*   **Runtime Protocol:** Guided decoding (structured JSON schemas) **must** be enforced on the API client requests when Qwen triggers tools/function calls, while conversational speech outputs should stream in raw plain text to avoid parsing delays.

### Service B: ASR Server (CTranslate2 Whisper Config D)
*   **Target Image:** Custom CUDA-accelerated Python 3.10 container.
*   **Hugging Face Repository ID:** `<username>/whisper-medium-banking-configD-ct2` (private repository containing converted CTranslate2 `int8_float16` files).
*   **Environment Variables:**
    ```bash
    ASR_ENGINE=whisper
    MODEL_PATH=/opt/ml/models/ct2_whisper_medium_banking
    COMPUTE_TYPE=int8_float16
    HF_TOKEN=<your_hugging_face_read_token>
    ```

#### 1. Pulling the Model (Container Initialization)
The deployment pipeline or container entrypoint script must pull the converted model weights from Hugging Face before launching the service:
```bash
# Install Hugging Face Hub CLI (if not in base image)
pip install huggingface_hub

# Authenticate and download the CTranslate2 model files directly to target directory
huggingface-cli download \
  <username>/whisper-medium-banking-configD-ct2 \
  --local-dir /opt/ml/models/ct2_whisper_medium_banking \
  --token $HF_TOKEN
```

#### 2. Production Dockerfile Spec
Hand this Dockerfile to the DevOps team to build the GPU-accelerated ASR image:
```dockerfile
# Use official PyTorch base image with CUDA 12.1 and cuDNN support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python packages (pinned for stability)
RUN pip install --no-cache-dir \
    faster-whisper==1.0.3 \
    fastapi==0.110.0 \
    uvicorn==0.28.0 \
    numpy<2.0.0 \
    huggingface_hub==0.23.0

# Copy application server code
COPY main.py /app/main.py

# Expose port
EXPOSE 8001

# Command to run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

#### 3. FastAPI Service Code (`main.py`)
This is the core execution logic that runs inside the container. It loads the CTranslate2 model on GPU with `int8_float16` quantization and transcribes incoming audio files:
```python
import os
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI(title="Krim ASR Service - CTranslate2")

# Read environment variables
model_path = os.environ.get("MODEL_PATH", "/opt/ml/models/ct2_whisper_medium_banking")
compute_type = os.environ.get("COMPUTE_TYPE", "int8_float16")

print(f"Loading CTranslate2 model from {model_path} on CUDA...")
# Lazy-load model onto CUDA using MPS partition
try:
    model = WhisperModel(
        model_path,
        device="cuda",
        compute_type=compute_type,
        cpu_threads=4
    )
    print("ASR Model loaded successfully!")
except Exception as e:
    print(f"Failed to load Whisper Model: {e}")
    model = None

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    if model is None:
        raise HTTPException(status_code=500, detail="ASR Model is not loaded on server.")
    
    # Save uploaded file bytes to a temporary location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Transcribe with target params for telephony (suppressing repetitions)
        segments, info = model.transcribe(
            tmp_path,
            language="hi",
            task="transcribe",
            condition_on_prev_tokens=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            beam_size=5
        )
        
        # Merge segments to build final transcript
        transcript = " ".join([segment.text for segment in segments]).strip()
        
        return JSONResponse({
            "text": transcript,
            "language": info.language,
            "language_probability": info.language_probability
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR transcription failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
```

#### 4. Architecture (Utterance-Final Mode):
*   **No sliding-window streaming:** The server **must not** run a 300ms chunk loop (this prevents word fragmentation and CPU bottlenecks).
*   **VAD-Bounded Execution:** Use **Silero VAD** on the incoming LiveKit audio stream. The stream buffers raw audio in-memory during speech, and transcribes the entire utterance in a single pass **only** when Silero VAD flags an *End-of-Utterance (EOU)* state.
*   *Suppression Arguments:* Set `condition_on_prev_tokens=False`, `repetition_penalty=1.1`, and `no_repeat_ngram_size=4` on transcription parameters.

### Service C: TTS Server (Kokoro FastAPI)
*   **Image:** `ghcr.io/remsky/kokoro-fastapi-gpu:latest`
*   **Port:** 8888
*   **Replicas:** Deploy **2 identical containers** behind a local `least-conn` load balancer (e.g. Nginx or HAProxy) to handle simultaneous speech synthesis calls without blocking.
*   **TTS Sanitizer Layer (Orchestration Pre-processor):**
    Before sending text from Qwen to Kokoro, the orchestrator must pass the string through a deterministic sanitizer:
    1.  **Regex Transliteration:** Map known Latin characters/brand names (e.g., *"credit card"*, *"OTP"*) to Devanagari phonetics.
    2.  **Verbalization:** Convert numbers (e.g., `5000`, `2026`) and currencies (e.g. `₹500`) to verbal Hindi words (e.g., `पाँच हज़ार`, `पाँच सौ रुपये`).
    3.  **Language Code routing:** Pass code `h` (Hindi) and voice `hf_alpha` for Devanagari text, and code `a`/`b` (English) and voice `af_bella`/`am_adam` for English text.

### Service D: LiveKit Agent Workers (CPU Node)
*   **Runtime:** CPU-only instances (gated separate from L4 GPUs).
*   **Execution:** Run **4 to 8 parallel Python processes** utilizing LiveKit Agent framework per EC2 CPU node.
*   **VAD Configuration:** Integrate `silero-vad` locally on the agent worker. The agent handles audio resampling (8kHz Vobiz -> 16kHz) and sends VAD-bounded chunks via WebSockets to the GPU ASR service.

---

## 4. Operational & Deployment Requirements

1.  **Static Engine Registry (S3 versioned):**
    *   All model checkpoints (CT2 directories, NeMo `.nemo` weights, AWQ weights) must be pulled from a versioned AWS S3 bucket during container initialization.
    *   Model swaps (e.g. replacing Whisper Config D with fine-tuned Nemotron 3.5 on Day 8) **must be triggered via environment variables only**, requiring a simple container restart, not code changes.
2.  **Telephony Quality Assurance & Call Auditing:**
    *   Log per-stage latency metrics: `ASR Finalize`, `LLM Time-to-First-Token (TTFT)`, `TTS Time-to-First-Byte (TTFB)`, and `End-to-End User Turn`.
    *   Persist **raw audio recordings (.wav)** and matching **transcripts** directly to an S3 logging bucket. This logs real production data to serve as the QA test corpus and the base dataset for future offline fine-tuning.
3.  **Acceptance Gate (Load Testing):**
    *   Rerun the latency/WER benchmark harness on the target AWS L4 GPU, with vLLM generating responses concurrently under simulated call concurrency levels ($c=1, 4, 8$).
    *   **Pass Condition:** End-to-end user-perceived turn latency $P95 < 2.0\text{ seconds}$ at $c=8$.
