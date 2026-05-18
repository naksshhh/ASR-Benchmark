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
