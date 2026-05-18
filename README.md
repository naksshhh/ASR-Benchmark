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

## Project Structure

```
banking_asr_eval/
├── data/
│   ├── synthetic/           # Banking dialogue scripts + TTS generation
│   │   ├── banking_scripts.py   # 60 banking dialogue templates
│   │   └── generate_banking_data.py
│   └── loaders.py           # Manifest + HuggingFace dataset loading
├── models/
│   ├── model_registry.py    # Lazy-loading model registry
│   └── inference/           # Per-backend inference wrappers
│       ├── whisper_local.py     # Whisper (tiny→large-v3)
│       ├── nemo_local.py        # NeMo (Parakeet, Canary)
│       └── huggingface_generic.py  # AI4Bharat models
├── metrics/
│   ├── normalize.py         # Hindi/English/banking text normalization
│   ├── core.py              # WER, CER, MER, WIL via jiwer
│   ├── number_er.py         # Number Error Rate
│   ├── entity_accuracy.py   # Banking entity accuracy
│   └── codeswitching.py     # Per-language-segment WER
├── evaluate.py              # Main evaluation loop (with checkpointing)
├── benchmark.py             # Latency + quality (RTF measurement)
└── visualize.py             # Pareto plots, heatmaps, breakdowns
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

## Models

| Model | Backend | Mac Mini | Param Rudra |
|-------|---------|----------|-------------|
| whisper-tiny | whisper | ✓ (standin) | ✓ |
| whisper-large-v3-turbo | whisper | ✗ | ✓ |
| parakeet-tdt-0.6b-v3 | nemo | ✗ | ✓ |
| canary-1b-flash | nemo | ✗ | ✓ |
| indicwav2vec-hindi | huggingface | ✗ | ✓ |
| indicconformer-hindi | huggingface | ✗ | ✓ |

## Development Workflow

```
Mac Mini (local)          Param Rudra (A100)
─────────────────         ──────────────────
Write scripts             Run full evaluation sweeps
Test on 10 samples        Whisper fine-tuning (12hr jobs)
Build synthetic data      RTF benchmarking
Analyze result CSVs       Model comparison
Make plots                ← rsync results back
Write blog drafts
```

## Config

Edit `config.yaml` to:
- Enable/disable models
- Set device (cpu/cuda)
- Configure normalization rules
- Set checkpoint intervals

---

## Pillar 5 — ASR Quality for Indian Banking: Detailed Roadmap

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

* **Category 1 — English-first models**: `faster-whisper large-v3-turbo`, `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/canary-1b-flash`
* **Category 2 — Indic-native models**: `ai4bharat/indicwav2vec-v2-hindi`, `ai4bharat/indicconformer-hi`, `ai4bharat/whisper-medium-hi`
* **Category 3 — Multilingual vLLM-compatible**: `Voxtral-Mini-3B-2507`, `openai/whisper-large-v3`
* **Category 4 — Fine-tuned by you**: Parakeet or Whisper-medium fine-tuned on banking Hindi.

---

### Evaluation pipeline architecture

```
banking_asr_eval/
├── data/
│   ├── datasets/          # downloaded datasets
│   ├── synthetic/         # your generated test set
│   └── manifests/         # JSON metadata files
├── models/
│   ├── model_registry.py  # maps model names to inference functions
│   └── inference/         # per-model inference wrappers
├── metrics/
│   ├── wer.py             # WER/CER/MER/WIL via jiwer
│   ├── ner_accuracy.py    # named entity accuracy
│   ├── number_er.py       # number error rate
│   ├── codeswitching.py   # language segment detection + per-segment WER
│   └── normalize.py       # text normalization (critical)
├── evaluate.py            # main evaluation loop
├── benchmark.py           # latency + quality combined (RTF + WER)
└── visualize.py           # Pareto plots, heatmaps, error analysis
```

**The evaluation loop logic:**
```python
for model_name, model_fn in model_registry.items():
    for dataset_name, dataset in datasets.items():
        results = []
        for sample in dataset:
            t0 = time.perf_counter()
            hypothesis = model_fn(sample['audio_path'])
            latency = time.perf_counter() - t0
            
            ref_norm = normalize(sample['reference'])
            hyp_norm = normalize(hypothesis)
            
            results.append({
                'model': model_name,
                'dataset': dataset_name,
                'accent': sample['accent_region'],
                'language_mix': sample['dominant_language'],
                'wer': jiwer.wer(ref_norm, hyp_norm),
                'cer': jiwer.cer(ref_norm, hyp_norm),
                'ner': number_error_rate(ref_norm, hyp_norm),
                'rtf': latency / sample['duration'],
            })
        
        df = pd.DataFrame(results)
```

---

### Fine-tuning roadmap

* **Stage 1 — Establish baselines (week 1–2)**: Run zero-shot models on IndicSUPERB + synthetic set.
* **Stage 2 — Data preparation (week 2–3)**: Collect 10–50 hours of training data in HF `datasets` format.
* **Stage 3 — Fine-tune Whisper (week 3–4)**: Fine-tune `whisper-medium` using HF `Seq2SeqTrainer` with 1e-5 learning rate.
* **Stage 4 — Fine-tune IndicConformer or IndicWav2Vec (week 4–5)**: Fine-tune specialized models.
* **Stage 5 — Error analysis (week 5–6)**: Breakdown error categories (substitutions of numbers, entities, code-switched text, etc.).

---

### The comparison table your blog should land on

| Model | Hindi WER | Code-switch WER | Number ER | RTF (A100) | Fine-tuned? |
|---|---|---|---|---|---|
| Whisper large-v3-turbo | ? | ? | ? | 0.45 | No |
| Parakeet-TDT-0.6B | ? | ? | ? | 0.097 | No |
| Canary-1B-Flash | ? | ? | ? | 0.132 | No |
| IndicConformer-Hi | ? | ? | ? | ? | No |
| IndicWav2Vec-v2 | ? | ? | ? | ? | No |
| Whisper-medium-hi (AI4B) | ? | ? | ? | ? | No |
| **Whisper-medium (yours)** | ? | ? | ? | ? | **Yes** |
| **Parakeet (yours)** | ? | ? | ? | ? | **Yes** |

---

### Sequencing for the blog

1. **Why WER isn't enough for Indian banking** — Motivate NER and CS-WER metrics.
2. **The dataset problem** — Constructing banking Hindi datasets.
3. **Zero-shot benchmark** — Models, metrics, and Pareto plots.
4. **Fine-tuning** — Process, improvements, and error analysis.
5. **What it means for production** — Given latency/quality tradeoffs, find the optimal deployment model.

---

### Realistic timeline

| Week | Work |
|---|---|
| 1 | Set up eval pipeline, run all models on IndicSUPERB, get baseline numbers |
| 2 | Build synthetic banking test set (100 samples minimum), run all models on it |
| 3 | Fine-tune Whisper-medium on Kathbath subset + synthetic data |
| 4 | Fine-tune IndicWav2Vec or IndicConformer on same data |
| 5 | Error analysis, Pareto plots, connect back to latency harness |
| 6 | Write Blog 6 draft |
