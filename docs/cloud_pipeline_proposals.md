# AWS Cloud Deployment & Pipeline Optimization Proposals
### Telephony Voice Bot (LiveKit + Vobiz SIP Trunk + Qwen-8B + Kokoro TTS)

This document outlines the architecture, model selections, and serving optimization proposals to review with the Cloud Infrastructure team for deploying the Krim.ai voice bot on AWS L4 GPU instances.

---

## 1. ASR (Speech-to-Text) Choices & Optimization

We evaluated both **Bilingual Conformer-CTC Large** (`stt_hi_conformer_ctc_large`) and **Whisper-Medium** on our Hinglish banking dataset.

### Option A: Bilingual Conformer-CTC Large (NVIDIA NeMo)
*   **Behavior:** Transcribes all speech (including English terms) phonetically in **pure Devanagari script** (e.g. *"credit card"* $\rightarrow$ `"क्रेडिट कार्ड"`, *"account statement"* $\rightarrow$ `"अकाउंट स्टेटमेंट"`).
*   **Pros:** Ultra-low latency (49ms chunk processing, RTF 0.013). Non-autoregressive decoding prevents infinite repetition loops on silence or background line noise. Very low VRAM footprint (50+ streams per L4 GPU).
*   **LLM Integration:** Qwen-8B natively understands Devanagari phonetics. The LLM can interpret `"क्रेडिट कार्ड ब्लॉक करना है"` as the semantic action `block_credit_card` without any pre-parsing or regex translations.

### Option B: Fine-Tuned Whisper-Medium (Converted to CTranslate2 / Faster-Whisper)
*   **Behavior:** Transcribes in true **mixed scripts** (Latin for English/Hinglish, Devanagari for Hindi).
*   **Pros (Maximum Quality):** The absolute best performance across all targets. Whisper is the industry standard for pure English (including foreign accents). By using our fine-tuned **Whisper-Medium (Config C/D)** weights, we achieve the lowest Hinglish banking WER (**37.74%** vs. Conformer-CTC's 73.61%) and maintain excellent regional Hindi dialect performance (**20.57% WER** on Kathbath).
*   **Solving the Latency Issue (CTranslate2):** Standard HuggingFace Whisper-Medium is too slow for real-time streaming. However, converting our fine-tuned PyTorch model to **CTranslate2 (Faster-Whisper)** format and running it in `float16` on the L4 GPU slashes chunk transcription latency to **~150ms–200ms** (RTF ~0.15), making it real-time telephony compatible.
*   **Implementation:**
    1. Convert the fine-tuned PyTorch/Transformers checkpoint to CTranslate2 format:
       ```bash
       pip install ctranslate2
       ct2-transformers-converter --model /path/to/whisper-medium-banking-configC/final --output_dir /path/to/ct2_whisper_medium --quantization float16
       ```
    2. Serve using the `faster-whisper` package, passing our repetition-suppression parameters:
       ```python
       from faster_whisper import WhisperModel
       model = WhisperModel("/path/to/ct2_whisper_medium", device="cuda", compute_type="float16")
       
       # Enforce generation controls to eliminate duplication loops in live audio
       segments, info = model.transcribe(
           audio_path,
           condition_on_prev_tokens=False,
           repetition_penalty=1.1,
           no_repeat_ngram_size=4,
           compression_ratio_threshold=1.35
       )
       ```

---

## 2. LLM Serving Optimization (Qwen-8B on AWS L4)

Qwen3-8B is already deployed on an L4 GPU. To guarantee sub-second Response Generation Latency (TTFT < 100ms) for live telephony calls, the cloud team should implement the following serving parameters:

1.  **Deploy with vLLM:**
    Use **vLLM** as the execution backend to leverage PagedAttention, continuous batching, and kernel-level optimizations.
2.  **Guided Decoding (Structured JSON):**
    For slots and tool-calling triggers, force Qwen to output strictly structured JSON using vLLM's guided decoding (`guided_json` or outlines framework). This slashes token generation time by pruning the vocabulary search space and preventing conversational filler text.
3.  **Enable FlashAttention-2:**
    Compile the serving container with FlashAttention-2 enabled for the Ada Lovelace (L4) architecture to maximize attention computation speeds.
4.  **Speculative Decoding:**
    Set up speculative decoding inside vLLM using **Qwen-1.5B** as the draft model. The draft model proposes tokens which the main 8B model verifies in parallel, speeding up throughput by up to 1.5x.

---

## 3. TTS (Text-to-Speech) Integration: Kokoro TTS

Kokoro-82M is an ultra-lightweight, high-quality, open-source TTS model. It runs in milliseconds on L4 GPUs.

### Handling Hinglish Code-Switching with Kokoro
*   **The Problem:** Mixing Latin characters (English) and Devanagari characters (Hindi) in a single text string (e.g., `"आपका account balance पाँच सौ रुपये है"`) confuses the Kokoro phonemizer, resulting in distorted pronunciation or skipped words.
*   **The Solution (All-Devanagari Output):** 
    Prompt Qwen-8B to output its spoken responses **entirely in Devanagari script**, spelling English banking words phonetically.
    *   *Input to Kokoro:* `"आपका अकाउंट बैलेंस पाँच सौ रुपये है"`
    *   *Language Code:* Set to **`h`** (Hindi).
    *   *Voice:* Use **`hf_alpha`** (Female) or **`hm_omega`** (Male).
*   **Why this works:** When Qwen writes English words in Devanagari, Kokoro's Hindi phonemizer reads them natively and pronounces both the Hindi and English words with a natural Indian accent. This completely bypasses the need for multi-script text splitting or stitching separate English and Hindi voice models, which would ruin real-time latency.

### Serving Stack on L4
Since Kokoro is extremely small (82M parameters, ~330MB footprint), it can be served co-located with the ASR or LLM on the same L4 GPU instance without any resource contention. Serve via a lightweight FastAPI wrapper utilizing the `kokoro-onnx` or `kokoro` Python packages.

---

## 4. Handling Pure English Speakers

If a customer speaks entirely in English, the pipeline dynamically routes both script generation and TTS voices.

### A. If using Whisper ASR (`faster-whisper-small`):
*   **Detection:** Whisper returns a metadata language token (`language="en"`).
*   **Routing:** The orchestration agent sets `mode = "en"` and configures the LLM prompt to respond in standard Latin script English.

### B. If using Conformer-CTC ASR (`stt_hi_conformer_ctc_large`):
*   **ASR Transcription:** The English speech is transcribed phonetically in Devanagari (e.g. *"I want to block my credit card"* $\rightarrow$ `"आई वॉन्ट टू ब्लॉक माई क्रेडिट कार्ड"`).
*   **Detection & Routing (LLM-in-the-loop):** 
    We task Qwen-8B to detect the language and output a JSON payload in one single pass:
    ```json
    {
      "detected_language": "en",
      "response": "Sure, I have blocked your credit card. Is there anything else?"
    }
    ```
    Even when reading phonetic Devanagari, Qwen's attention layers easily recognize it as spoken English and output a grammatically correct English response in Latin script.

### C. TTS Execution (Voice Switching)
When the pipeline receives the response from Qwen:
1.  **If `detected_language == "en"`:**
    *   **Text:** Send Qwen's Latin script response directly to Kokoro.
    *   **Language Code:** Set to **`a`** (American English) or **`b`** (British English).
    *   **Voice:** Switch to Kokoro's native English voices (e.g. `af_bella` or `am_adam`).
2.  **If `detected_language == "hi"`:**
    *   **Text:** Send the Devanagari script response.
    *   **Language Code:** Set to **`h`** (Hindi).
    *   **Voice:** Use Hindi voices (`hf_alpha` or `hm_omega`).

This dual-routing approach ensures that pure English speakers receive standard English audio with natural accents, while Hinglish/Hindi speakers hear standard Indian phonetic pronunciations.
