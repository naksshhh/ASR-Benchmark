# Banking ASR Evaluation Pipeline

A comprehensive evaluation framework for benchmarking ASR models on Indian banking domain audio. Built for the ASR fine-tuning research project targeting Hindi-English code-switched banking call center audio.

## Why this exists

Standard WER isn't enough for banking ASR. This pipeline adds:
- **Number Error Rate (NER)** — account numbers, amounts, dates must be perfect
- **Named Entity Accuracy** — HDFC, CIBIL, Aadhaar can't be mangled
- **Code-switching WER** — Hindi-English mixing is the norm in Indian banking
- **Banking-aware text normalization** — `₹5000` = `five thousand rupees` = `paanch hazaar rupaye`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test the pipeline (no GPU needed)
python test_pipeline.py

# Generate synthetic test data
python -c "from banking_asr_eval.data.synthetic import generate_manifest_only; generate_manifest_only('./data/synthetic', 100)"

# Run evaluation (whisper-tiny, Mac Mini)
python -m banking_asr_eval.evaluate \
  --config config.yaml \
  --manifest data/synthetic/manifest.json \
  --max-samples 10

# Generate visualizations
python -m banking_asr_eval.visualize --results results/eval_results_*.csv
```

## Zero-Shot Benchmark Results — Kathbath Hindi (3,151 samples)

**Cluster:** Param Rudra A100 GPU  
**Dataset:** ai4bharat/Kathbath Hindi validation split

| Rank | Model | Backend | WER ↓ | CER ↓ | NER ↓ | Hindi Native? |
|------|-------|---------|-------|-------|-------|---------------|
| 🥇 | **IndicWav2Vec-Hindi** | HuggingFace (CTC) | **11.64%** | **3.30%** | **2.53%** | ✅ Yes |
| 🥈 | **Voxtral-Mini-3B** | Voxtral | **17.66%** | **6.72%** | **9.79%** | Multilingual |
| 🥉 | Whisper-large-v3 | Whisper | 28.82% | 9.44% | 14.52% | Multilingual |
| 4 | Whisper-large-v3-turbo | Whisper | 32.01% | 10.37% | 13.36% | Multilingual |
| 5 | Whisper-medium | Whisper | 41.64% | 15.85% | 21.54% | Multilingual |
| 6 | Whisper-tiny | Whisper | 254.00% | 244.76% | 915.55% | Multilingual |

### Excluded Models (language mismatch — no Hindi support)

| Model | Backend | Reason |
|-------|---------|--------|
| Parakeet-TDT-0.6B | NeMo | English-only — romanized Hindi audio |
| Canary-1B-Flash | NeMo | EN/DE/FR/ES only — no Hindi support |
| Streaming Zipformer | Sherpa-ONNX | English-only (LibriSpeech) — output nonsense on Hindi |
| IndicConformer-Hindi | NeMo | Requires AI4Bharat's NeMo fork (`nemo-v2`), incompatible with standard NeMo |

## Project Structure

```
banking_asr_eval/
├── data/
│   ├── synthetic/              # Banking dialogue scripts + TTS generation
│   │   ├── banking_scripts.py      # 60 banking dialogue templates
│   │   └── generate_banking_data.py
│   └── loaders.py              # Manifest + HuggingFace dataset loading
├── models/
│   ├── model_registry.py       # Lazy-loading model registry (6 backends)
│   └── inference/
│       ├── whisper_local.py        # Whisper (tiny→large-v3)
│       ├── nemo_local.py           # NeMo (Parakeet, Canary, IndicConformer)
│       ├── huggingface_generic.py  # AI4Bharat CTC models (IndicWav2Vec)
│       ├── voxtral_local.py        # Mistral Voxtral Mini-3B
│       └── sherpa_onnx_local.py    # Sherpa-ONNX streaming models (Zipformer)
├── metrics/
│   ├── normalize.py            # Hindi/English/banking text normalization
│   ├── core.py                 # WER, CER, MER, WIL via jiwer
│   ├── number_er.py            # Number Error Rate
│   ├── entity_accuracy.py      # Banking entity accuracy
│   └── codeswitching.py        # Per-language-segment WER
├── evaluate.py                 # Main evaluation loop (with checkpointing)
├── benchmark.py                # Latency + quality (RTF measurement)
└── visualize.py                # Pareto plots, heatmaps, breakdowns
```

## Metric Layers

| Layer | Metric | What it catches |
|-------|--------|----------------|
| 1 | WER | Baseline transcription accuracy |
| 1 | CER | Character-level accuracy (better for Hindi) |
| 1 | MER | Balanced error rate |
| 1 | WIL | Semantic information lost |
| 2 | NER | Number transcription accuracy (banking-critical) |
| 2 | Entity Acc | Banking term recognition (HDFC, EMI, etc.) |
| 2 | CS-WER | Per-language accuracy in code-switched text |
| 3 | RTF | Real-time factor (latency/duration) |

## Supported Model Backends

| Backend | Models | Notes |
|---------|--------|-------|
| `whisper` | OpenAI Whisper (tiny → large-v3) | `transformers` pipeline, auto GPU |
| `nemo` | NVIDIA Parakeet, Canary, IndicConformer | Requires `nemo_toolkit[asr]` |
| `huggingface` | AI4Bharat IndicWav2Vec | Manual CTC loading for GPU compatibility |
| `voxtral` | Mistral Voxtral-Mini-3B | Requires `mistral-common[audio]` |
| `sherpa-onnx` | k2-fsa Zipformer | CPU/ONNX streaming inference |

## Development Workflow

```
Mac Mini (local)          Param Rudra (A100)
─────────────────         ──────────────────
Write scripts             Run full evaluation sweeps
Test on 10 samples        Whisper fine-tuning (12hr jobs)
Build synthetic data      RTF benchmarking
Analyze result CSVs       Model comparison
Make plots                ← git push/pull results back
Write blog drafts
```

### Offline Cluster Usage & Parallelization

For compute nodes without internet access:
```bash
# 1. On login node (has internet): cache all models
python pre_download.py

# 2. On compute node (no internet): run offline
HF_HUB_OFFLINE=1 python -m banking_asr_eval.evaluate \
  --config config.yaml \
  --manifest ./data/manifests/kathbath_hindi.json
```

To speed up evaluation across multiple GPUs or CPU cores, you can run in parallel. The script splits the dataset into chunks, isolates each process to a specific GPU, and aggregates the results:
```bash
# Run with 4 parallel processes distributed across GPU 0 and GPU 1
HF_HUB_OFFLINE=1 python -m banking_asr_eval.evaluate \
  --config config.yaml \
  --manifest ./data/manifests/kathbath_hindi.json \
  --workers 4 \
  --gpus 0,1
```

> **Note:** Voxtral requires internet due to `mistral_common` tokenizer limitations. Run it on an internet-connected node.

### Benchmarking Latency Guidelines

`benchmark.py` performs **3 warmup runs + 5 timed runs** per sample to compute statistically robust latency percentiles (P50, P95, mean). 
*   **Subset Recommendation:** Running this 8x multiplier on the full Kathbath dataset (3,151 samples × 8 = 25,208 evaluations) will take a very long time. For latency/RTF benchmarking, we recommend running on a subset of **50–100 samples** using the `--max-samples` flag, or using the 100-sample synthetic dataset:
    ```bash
    HF_HUB_OFFLINE=1 python -m banking_asr_eval.benchmark \
      --config config.yaml \
      --manifest ./data/manifests/kathbath_hindi.json \
      --max-samples 100
    ```
*   **RAM Disk Optimization:** `benchmark.py` automatically detects and uses `/dev/shm` (RAM disk) to cache audio files during benchmarking. This bypasses the cluster's network filesystem (NFS/GPFS) bottlenecks so that file read times do not skew the latency results.

### Evaluating with a Local IndicSUPERB / Kathbath Copy

If you already have a local copy of the IndicSUPERB (Kathbath) dataset on the cluster, you can prepare the evaluation manifest in two ways:

#### Option A: If it is already in your HuggingFace Cache
If the dataset has been loaded before on the cluster, HuggingFace will resolve it from `~/.cache/huggingface/datasets`. You can build the manifest offline using:
```bash
HF_HUB_OFFLINE=1 python prepare_indic_superb.py \
  --dataset ai4bharat/Kathbath \
  --language hindi \
  --split valid \
  --output ./data/manifests/kathbath_hindi.json
```

#### Option B: If you have a raw directory of audio and transcription files
If the dataset is saved in a raw directory (containing audio files like `.wav`/`.m4a` and transcription files like `transcription.txt` or `text`), our updated script will recursively find, match, compute durations, and generate the manifest:
```bash
python prepare_indic_superb.py \
  --dataset /path/to/local/indicsuperb/kathbath/hindi \
  --language hindi \
  --output ./data/manifests/kathbath_hindi.json
```
The script supports common transcription structures (e.g. Kaldi `text` format, tab-separated metadata transcripts) and uses `librosa` to compute durations automatically.

## Config

Edit `config.yaml` to:
- Enable/disable models
- Set device (cpu/cuda)
- Configure normalization rules
- Set checkpoint intervals

---

## Pillar 5 — ASR Quality for Indian Banking: Detailed Roadmap

### Progress Tracker

| Phase | Task | Status |
|-------|------|--------|
| **Week 1** | Set up eval pipeline | ✅ Complete |
| **Week 1** | Implement metric stack (WER/CER/NER/Entity/CS-WER) | ✅ Complete |
| **Week 1** | Run all models on Kathbath Hindi (3,151 samples) | ✅ Complete (10 models benchmarked) |
| **Week 1** | Integrate new model backends (Voxtral, Sherpa-ONNX) | ✅ Complete |
| **Week 2** | Build synthetic banking test set (100+ samples) | ✅ Complete (100 samples, 6 banking domains, Hindi/English/mixed) |
| **Week 2** | Run evaluation on synthetic banking dataset | ✅ Complete (6 models × 100 samples on A100) |
| **Week 2** | Run RTF latency benchmarks on A100 | ✅ Complete (RTF measured from synthetic eval on A100) |
| **Week 2** | Generate Pareto plots (WER vs RTF) | ✅ Complete (Kathbath + Synthetic plots generated) |
| **Week 3** | Fine-tune Whisper-medium on Kathbath + synthetic data | 🔲 Ready (fine-tuning script created) |
| **Week 4** | Fine-tune IndicWav2Vec on banking data | 🔲 Not started |
| **Week 5** | Error analysis, final visualizations | 🔲 Not started |
| **Week 6** | Write blog draft | 🔲 Not started |

### Key Findings (Week 1 & 2)

1. **IndicWav2Vec dominates** on Kathbath Hindi (11.6% WER, 3.3% CER) — native Hindi CTC model with clean Devanagari output and the **fastest RTF (0.09)**.
2. **Voxtral-Mini-3B is surprisingly strong** at 17.7% WER on Kathbath — a 3B multimodal LLM competitive with much larger Whisper models, with RTF 0.54.
3. **Whisper-large-v3 is the best Whisper variant** at 28.8% WER on Kathbath, but has the highest RTF (1.49) — not real-time on A100.
4. **English-only models fail completely** on Hindi — Parakeet (100.1% WER), Canary (105.4%), Streaming-Zipformer (106.5%), and IndicConformer (100.0%) produce English romanization or gibberish.
5. **Banking domain performance differs from general Hindi** — On the synthetic banking dataset, IndicWav2Vec degrades to 75.5% WER (vs 11.6% on Kathbath), while Voxtral (46.0%) and Whisper-large-v3 (46.3%) hold up much better on code-switched banking utterances.
6. **Entity accuracy reveals production gaps** — On banking scenarios, Voxtral achieves the best entity recognition (70.5% accuracy), while IndicWav2Vec drops to 29.0% — suggesting it struggles with English banking terms in mixed speech.
7. **RTF-quality Pareto frontier** — The optimal models are: Voxtral-Mini-3B (best WER-per-RTF), Whisper-large-v3-turbo (good balance), and IndicWav2Vec (lowest latency if Hindi-only).

---

### First, understand what WER actually measures (and where it breaks)

WER is defined as:

```
WER = (S + D + I) / N

S = substitutions ("five" heard as "fine")
D = deletions (word missed entirely)
I = insertions (hallucinated word)
N = total words in reference transcript
```

It's computed via dynamic programming (same algorithm as edit distance). Two things to understand before you trust any WER number:

**Normalization matters enormously.** "₹5000" vs "five thousand" vs "5,000" are the same thing. If your reference says "five thousand" and your ASR outputs "5000", naive WER counts it as wrong. Every serious ASR evaluation pipeline normalizes text before scoring — lowercasing, expanding numbers, stripping punctuation, standardizing currency. If two papers report WER on the same dataset with different normalization, the numbers are not comparable.

**WER is symmetric in a bad way.** A single long word deletion ("Aadhaar") costs the same as deleting "a". For banking, missing "Aadhaar" is catastrophic; missing "a" is irrelevant. WER doesn't know this.

---

### The metric stack you actually need

Build these in layers — each one catches failures the previous one misses.

#### Layer 1 — Core transcription metrics

* **WER** — your baseline, required for comparison against published numbers.
* **CER (Character Error Rate)** — same formula as WER but at character level. More informative for Hindi written in Devanagari, long compound words, and morphologically rich environments.
* **MER (Match Error Rate)** — penalizes both reference and hypothesis for errors, slightly more balanced than WER.
* **WIL (Word Information Lost)** — captures how much semantic content was lost.

Use the `jiwer` library for all of these:

```python
import jiwer

transforms = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

measures = jiwer.compute_measures(
    reference, hypothesis,
    truth_transform=transforms,
    hypothesis_transform=transforms
)

print(measures['wer'], measures['cer'], measures['mer'], measures['wil'])
```

#### Layer 2 — Domain-critical metrics (banking-specific)

This is where you differentiate your work from generic ASR benchmarks.

* **Number Error Rate (NER)** — extract all numeric entities from reference and hypothesis, compute WER only over those tokens.
  
  ```python
  import re
  import jiwer

  def extract_numbers(text):
      pattern = r'\b(\d[\d,\.]*|\b(?:ek|do|teen|char|paanch|chhe|saat|aath|nau|das|' \
                r'gyarah|barah|tera|chaudah|pandrah|solah|satrah|atharah|unnees|bees|' \
                r'one|two|three|four|five|six|seven|eight|nine|ten|' \
                r'hundred|thousand|lakh|crore|million)\b)'
      return re.findall(pattern, text.lower())

  def number_error_rate(reference, hypothesis):
      ref_nums = extract_numbers(reference)
      hyp_nums = extract_numbers(hypothesis)
      if not ref_nums and not hyp_nums:
          return 0.0
      return jiwer.wer(' '.join(ref_nums), ' '.join(hyp_nums))
  ```

* **Named Entity Accuracy** — extract named entities (bank names, product names, customer names, city names like HDFC, SBI, Aadhaar, EMI, CIBIL, NACH) and compute accuracy separately.
* **OOV Rate (Out-of-Vocabulary)** — what fraction of words in your test set don't appear in the ASR model's vocabulary/training data.
* **Code-switching accuracy** — measure WER separately on Hindi-only segments, English-only segments, and mixed segments.

#### Layer 3 — Latency-quality tradeoff metrics

* **RTF (Real-Time Factor)** = processing time / audio duration.
* **Quality-Latency Pareto frontier** — plot WER on x-axis, RTF on y-axis for every ASR model to find Pareto-optimal models.

---

### Datasets to use

#### Published datasets (use for reproducible benchmarks)

| Dataset | Languages | Size | Notes |
|---|---|---|---|
| **IndicSUPERB** | 12 Indian languages | ~100h total | Standard benchmark, IIT Madras; use this as your primary |
| **Kathbath** | Hindi (multiple accents) | 1750h | Largest Hindi ASR dataset; has accent metadata |
| **Shrutilipi** | 12 Indic languages | 6400h | Massive, use subset |
| **MUCS 2021** | Hindi, Marathi | ~100h | Has code-switching subset |
| **OGI-MLTS** | Various | — | Older but has telephone-quality audio |
| **IISc-MILE** | Kannada, Tamil | ~150h | South Indian languages |
| **IndicTTS** | 13 languages | — | Primarily TTS but has read speech |

#### Banking-specific test set (build this yourself)

Build a small (50–100 hours) test set:
* **Synthetic**: Use TTS to generate banking dialogue scripts in Hindi/English/mixed.
* **Real (from krim.ai)**: Anonymized call center recordings (even 5 hours of real data is extremely valuable).

**Annotation schema:**
```json
{
  "audio_id": "call_001_turn_003",
  "duration_seconds": 8.4,
  "language_segments": [
    {"start": 0.0, "end": 3.2, "language": "hindi"},
    {"start": 3.2, "end": 5.1, "language": "mixed"},
    {"start": 5.1, "end": 8.4, "language": "english"}
  ],
  "reference_transcript": "mera loan amount hai fifty thousand rupees",
  "entities": [
    {"text": "fifty thousand", "type": "AMOUNT", "normalized": "50000"}
  ],
  "accent_region": "maharashtra",
  "noise_level": "moderate"
}
```

---

### Models to benchmark

* **Category 1 — English-first models**: `whisper large-v3-turbo`, `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/canary-1b-flash`
* **Category 2 — Indic-native models**: `ai4bharat/indicwav2vec-hindi`, `ai4bharat/indicconformer-hi`
* **Category 3 — Multilingual**: `Voxtral-Mini-3B-2507`, `openai/whisper-large-v3`
* **Category 4 — Fine-tuned by you**: IndicWav2Vec or Whisper-medium fine-tuned on banking Hindi.

---

### Fine-tuning roadmap

* **Stage 1 — Establish baselines (week 1–2)**: Run zero-shot models on Kathbath Hindi. ✅ Complete
* **Stage 2 — Data preparation (week 2–3)**: Build banking-specific synthetic test set. ✅ Complete
* **Stage 3 — Fine-tune Whisper (week 3–4)**: Fine-tune `whisper-medium` using HF `Seq2SeqTrainer` with 1e-5 learning rate. 🔲 Ready
* **Stage 4 — Fine-tune IndicWav2Vec (week 4–5)**: Fine-tune on banking domain data.
* **Stage 5 — Error analysis (week 5–6)**: Breakdown error categories (substitutions of numbers, entities, code-switched text, etc.).

---

### Kathbath Hindi Benchmark (3,151 samples — General Hindi ASR)

| Model | Category | Hindi WER ↓ | CER ↓ | Number ER ↓ | Fine-tuned? |
|---|---|---|---|---|---|
| IndicWav2Vec-Hindi | Indic-native | **11.64%** | **3.30%** | **2.53%** | No |
| Voxtral-Mini-3B | Multilingual | 17.66% | 6.72% | 9.79% | No |
| Whisper large-v3 | Multilingual | 28.82% | 9.44% | 14.52% | No |
| Whisper large-v3-turbo | English-first | 32.01% | 10.37% | 13.36% | No |
| Whisper-medium-hi | Hindi-tuned | 41.64% | 15.85% | 21.54% | No |
| IndicConformer-Hindi | Indic-native | 100.00% | 100.00% | 100.00% | No |
| Parakeet-TDT-0.6B | English-first | 100.14% | 92.78% | 100.00% | No |
| Canary-1B-Flash | English-first | 105.35% | 99.87% | 100.46% | No |
| Streaming-Zipformer | English-first | 106.49% | 96.38% | 100.69% | No |
| Whisper-tiny | English-first | 254.00% | 244.76% | 915.55% | No |
| **Whisper-medium (yours)** | **Fine-tuned** | **20.57%** | **7.30%** | **4.49%** | **Yes (Config D)** |
| **IndicWav2Vec (yours)** | **Fine-tuned** | **14.52%** | **4.36%** | **2.76%** | **Yes (Config D)** |

### Synthetic Banking Dataset (100 samples — Domain-Specific Evaluation on A100)

| Model | Banking WER ↓ | Banking CER ↓ | Entity Accuracy ↑ | RTF (A100) ↓ | Real-time? |
|---|---|---|---|---|---|
| **Whisper-medium (yours)** | **48.73%** | **35.83%** | — | **0.88** | ✅ Yes (Config D) |
| **IndicWav2Vec (yours)** | **73.35%** | **67.39%** | — | **0.09** | ✅ Yes (Config D) |
| Voxtral-Mini-3B | **46.01%** | **34.69%** | **70.5%** | 0.54 | ✅ Yes |
| Whisper large-v3 | 46.32% | 33.53% | 67.0% | 1.49 | ❌ No |
| Whisper large-v3-turbo | 55.52% | 39.72% | 62.0% | 0.50 | ✅ Yes |
| **Nemotron-3.5-ASR** | **67.81%** | — | — | **0.17** | ✅ Yes (Streaming) |
| IndicWav2Vec-Hindi | 75.49% | 71.41% | 29.0% | **0.09** | ✅ Yes |
| Whisper-medium-hi | 179.17% | 167.22% | 74.0% | 0.96 | ✅ Yes |
| Whisper-tiny | 299.93% | 239.27% | 66.0% | 0.59 | ✅ Yes |

> **Key Insight:** IndicWav2Vec is the best model for pure Hindi transcription (Kathbath), but struggles on code-switched banking speech. Voxtral-Mini-3B offers the best quality-latency tradeoff for real-world banking scenarios with mixed Hindi-English input.
> 
> **Streaming ASR Latency Nuance:** Nemotron-3.5-ASR is a native **streaming** model. Comparing streaming models to offline models purely on offline RTF (0.17 for Nemotron vs 0.50 for Whisper Turbo) can be misleading. In production, a streaming ASR processes audio chunk-by-chunk as the audio is spoken, leading to extremely low user-perceived final delay (chunk latency), whereas offline models must wait for the entire audio segment to end before starting transcription.
>
> **Other Streaming ASRs in the Registry:**
> - **Streaming Zipformer (Sherpa-ONNX)**: Lightweight CPU/ONNX streaming pruned transducer model. Low latency, but currently English-only.
> - **Parakeet-TDT (NeMo)**: FastConformer with Time-Delay Transducer decoding. Skips frames dynamically for high-speed streaming capability.
> - **Canary-1B-Flash (NeMo)**: FastConformer RNN-T model designed with streaming-compatible architectures.

---

### Sequencing for the blog

1. **Why WER isn't enough for Indian banking** — Motivate NER and CS-WER metrics.
2. **The dataset problem** — Constructing banking Hindi datasets.
3. **Zero-shot benchmark** — Models, metrics, and Pareto plots.
4. **Fine-tuning** — Process, improvements, and error analysis.
5. **What it means for production** — Given latency/quality tradeoffs, find the optimal deployment model.
