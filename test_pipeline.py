#!/usr/bin/env python3
"""
Quick test: validate the entire pipeline on 10 synthetic samples.

This is the "never submit a script you haven't run on 10 samples locally first" script.
Run this on Mac Mini before anything goes to Param Rudra.

Usage:
    python test_pipeline.py
"""

import os
import sys
import json

# ── Step 1: Test text normalization ──────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Text Normalization")
print("="*60)

from banking_asr_eval.metrics.normalize import normalize

test_cases = [
    ("₹5000 ka EMI bharna hai", "rupees 5000 ka emi bharna hai"),
    ("HDFC bank mein mera account hai", "hdfc bank mein mera account hai"),
    ("My PAN number is ABCDE1234F", "my pan number is abcde1234f"),
    ("मेरा आधार नंबर बताइए", "मेरा आधार नंबर बताइए"),
    ("fifty thousand rupees transfer", "50 1000 rupees transfer"),  # number words normalized
    ("Rs. 50,000 ka loan chahiye", "rupees 50000 ka loan chahiye"),
]

passed = 0
for i, (input_text, expected) in enumerate(test_cases):
    result = normalize(input_text)
    status = "✓" if result == expected else "✗"
    if status == "✓":
        passed += 1
    else:
        print(f"  {status} Case {i+1}: '{input_text}'")
        print(f"    Expected: '{expected}'")
        print(f"    Got:      '{result}'")

print(f"  Normalization: {passed}/{len(test_cases)} passed")


# ── Step 2: Test core metrics ────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Core Metrics (WER/CER/MER/WIL)")
print("="*60)

from banking_asr_eval.metrics.core import compute_core_metrics

ref = "mera account number hai nau do paanch teen ek chaar saat aath chhe"
hyp = "mera account number nau do paanch teen ek chaar saat aath"

metrics = compute_core_metrics(ref, hyp)
print(f"  Reference: {ref}")
print(f"  Hypothesis: {hyp}")
print(f"  WER: {metrics['wer']:.4f}")
print(f"  CER: {metrics['cer']:.4f}")
print(f"  MER: {metrics['mer']:.4f}")
print(f"  WIL: {metrics['wil']:.4f}")
print(f"  S/D/I: {metrics['substitutions']}/{metrics['deletions']}/{metrics['insertions']}")


# ── Step 3: Test Number Error Rate ───────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Number Error Rate (NER)")
print("="*60)

from banking_asr_eval.metrics.number_er import number_error_rate, analyze_number_errors

ref_ner = "account number nau do paanch teen ek chaar with amount fifty thousand"
hyp_ner = "account number nau do paanch teen ek chaar with amount fifty hundred"

ner_result = number_error_rate(ref_ner, hyp_ner)
print(f"  Reference: {ref_ner}")
print(f"  Hypothesis: {hyp_ner}")
print(f"  NER: {ner_result['ner']:.4f}")
print(f"  Ref numbers: {ner_result['ref_numbers']}")
print(f"  Hyp numbers: {ner_result['hyp_numbers']}")

errors = analyze_number_errors(ref_ner, hyp_ner)
if errors:
    print(f"  Number errors:")
    for e in errors:
        print(f"    {e['error_type']}: {e['reference']} → {e['hypothesis']}")


# ── Step 4: Test Entity Accuracy ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Entity Accuracy")
print("="*60)

from banking_asr_eval.metrics.entity_accuracy import entity_accuracy

ref_ent = "I want to check my HDFC credit card EMI using UPI"
hyp_ent = "I want to check my HDFC credit card using UPI"

ent_result = entity_accuracy(ref_ent, hyp_ent)
print(f"  Reference: {ref_ent}")
print(f"  Hypothesis: {hyp_ent}")
print(f"  Entity Accuracy: {ent_result['entity_accuracy']:.4f}")
print(f"  Matched: {ent_result['matched']}")
print(f"  Missed: {ent_result['missed']}")


# ── Step 5: Test Code-switching WER ──────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Code-switching WER")
print("="*60)

from banking_asr_eval.metrics.codeswitching import codeswitching_wer, classify_segment_language

ref_cs = "मेरा savings account balance कितना है"
hyp_cs = "मेरा saving account balance कितना"

cs_result = codeswitching_wer(ref_cs, hyp_cs)
print(f"  Reference: {ref_cs}")
print(f"  Hypothesis: {hyp_cs}")
print(f"  Dominant language: {cs_result['dominant_language']}")
print(f"  Overall WER: {cs_result['overall_wer']:.4f}")
print(f"  Hindi WER: {cs_result.get('hindi_wer', 'N/A')}")
print(f"  English WER: {cs_result.get('english_wer', 'N/A')}")
print(f"  Hindi words: {cs_result['hindi_word_count']}, English words: {cs_result['english_word_count']}")


# ── Step 6: Test synthetic data generation ───────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Synthetic Data Generation (manifest only)")
print("="*60)

from banking_asr_eval.data.synthetic import generate_manifest_only

manifest_path = generate_manifest_only(
    output_dir="./data/synthetic",
    n_samples=10,
)

with open(manifest_path) as f:
    manifest = json.load(f)

print(f"  Generated {len(manifest)} samples")
print(f"  Sample entry:")
sample = manifest[0]
for k, v in sample.items():
    print(f"    {k}: {v}")


# ── Step 7: Test model registry ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: Model Registry")
print("="*60)

from banking_asr_eval.models import ModelRegistry

registry = ModelRegistry.from_config("config.yaml")
models = registry.list_models()
print(f"  Registered models: {len(models)}")
for name, enabled in models.items():
    status = "✓ enabled" if enabled else "  disabled"
    print(f"    {status} : {name}")


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE TEST COMPLETE")
print("="*60)
print("""
Next steps:
  1. Install whisper-tiny: pip install transformers torch
  2. Generate audio: python -c "from banking_asr_eval.data.synthetic import generate_with_gtts; generate_with_gtts('./data/synthetic', 10)"
  3. Run evaluation: python -m banking_asr_eval.evaluate --config config.yaml --manifest data/synthetic/manifest.json --max-samples 10
  4. Generate plots: python -m banking_asr_eval.visualize --results results/eval_results_*.csv
""")
