"""
Main evaluation loop.

Runs all enabled models against all specified datasets, computes all metrics,
and saves results with checkpointing for Param Rudra job resilience.

Usage:
    python -m banking_asr_eval.evaluate --config config.yaml
    python -m banking_asr_eval.evaluate --config config.yaml --max-samples 10  # Mac Mini test
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from tqdm import tqdm

from .data.loaders import load_manifest, iterate_manifest
from .metrics import (
    compute_core_metrics,
    number_error_rate,
    entity_accuracy,
    codeswitching_wer,
    normalize,
)
from .models import ModelRegistry


def evaluate_sample(
    model_fn,
    sample: Dict,
    compute_all_metrics: bool = True,
) -> Dict:
    """
    Evaluate a single sample: run inference + compute all metrics.

    Args:
        model_fn: Callable that takes audio_path → transcript
        sample: Manifest entry with audio_path and reference_transcript
        compute_all_metrics: If True, compute NER, entity, code-switching metrics

    Returns:
        Dict with all metric values + metadata
    """
    reference = sample.get("reference_transcript", "")
    audio_path = sample.get("audio_path", "")

    # ── Inference ──
    t0 = time.perf_counter()
    try:
        hypothesis = model_fn(audio_path)
    except Exception as e:
        return {
            "audio_id": sample.get("audio_id", "unknown"),
            "error": str(e),
            "wer": None,
        }
    latency = time.perf_counter() - t0

    # ── Normalize ──
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    # ── Layer 1: Core metrics ──
    core = compute_core_metrics(reference, hypothesis)

    result = {
        "audio_id": sample.get("audio_id", "unknown"),
        "reference": reference,
        "hypothesis": hypothesis,
        "ref_normalized": ref_norm,
        "hyp_normalized": hyp_norm,
        "latency_seconds": latency,
        "duration_seconds": sample.get("duration_seconds", 0),
        "language": sample.get("language", "unknown"),
        "scenario": sample.get("scenario", "unknown"),
        "accent_region": sample.get("accent_region", "unknown"),
        "noise_level": sample.get("noise_level", "unknown"),
        # Core metrics
        "wer": core["wer"],
        "cer": core["cer"],
        "mer": core["mer"],
        "wil": core["wil"],
        "substitutions": core["substitutions"],
        "deletions": core["deletions"],
        "insertions": core["insertions"],
    }

    # RTF
    duration = sample.get("duration_seconds", 0)
    if duration and duration > 0:
        result["rtf"] = latency / duration
    else:
        result["rtf"] = None

    # ── Layer 2: Domain metrics ──
    if compute_all_metrics:
        # Number Error Rate
        ner_result = number_error_rate(reference, hypothesis)
        result["ner"] = ner_result["ner"]
        result["ner_has_numbers"] = ner_result["has_numbers"]
        result["ner_ref_count"] = ner_result["ref_count"]

        # Entity Accuracy
        ent_result = entity_accuracy(reference, hypothesis)
        result["entity_accuracy"] = ent_result["entity_accuracy"]
        result["entity_has_entities"] = ent_result["has_entities"]
        result["entities_missed"] = json.dumps(ent_result.get("missed", []))

        # Code-switching
        cs_result = codeswitching_wer(reference, hypothesis)
        result["dominant_language"] = cs_result["dominant_language"]
        result["hindi_wer"] = cs_result.get("hindi_wer")
        result["english_wer"] = cs_result.get("english_wer")

    return result


def run_evaluation(
    config_path: str,
    manifest_path: str,
    output_dir: str = "./results",
    max_samples: Optional[int] = None,
    checkpoint_interval: int = 50,
) -> str:
    """
    Run full evaluation: all enabled models × all samples.

    Args:
        config_path: Path to config.yaml
        manifest_path: Path to evaluation manifest JSON
        output_dir: Directory to save results
        max_samples: Limit samples (for testing)
        checkpoint_interval: Save checkpoint every N samples

    Returns:
        Path to results CSV
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load data
    manifest = load_manifest(manifest_path)
    if max_samples:
        manifest = manifest[:max_samples]
    print(f"\n{'='*60}")
    print(f"Evaluation: {len(manifest)} samples")
    print(f"{'='*60}\n")

    # Load models
    registry = ModelRegistry.from_config(config_path)
    enabled = list(registry.enabled_models())
    print(f"Enabled models: {[name for name, _ in enabled]}\n")

    all_results = []

    for model_name, model_fn in enabled:
        print(f"\n── Model: {model_name} ──")
        checkpoint_path = os.path.join(output_dir, f"checkpoint_{model_name}_{timestamp}.json")

        # Check for existing checkpoint
        start_idx = 0
        model_results = []
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                model_results = json.load(f)
            start_idx = len(model_results)
            print(f"  Resuming from checkpoint: {start_idx} samples done")

        for i, sample in enumerate(tqdm(manifest[start_idx:], initial=start_idx, total=len(manifest))):
            result = evaluate_sample(model_fn, sample)
            result["model"] = model_name
            result["dataset"] = os.path.basename(manifest_path)
            model_results.append(result)

            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(model_results, f, ensure_ascii=False, indent=2)
                print(f"\n  [Checkpoint] Saved {len(model_results)} results")

        # Final save for this model
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)

        all_results.extend(model_results)

    # ── Save combined results ──
    results_csv = os.path.join(output_dir, f"eval_results_{timestamp}.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv, index=False)

    results_json = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        valid = model_df[model_df["wer"].notna()]
        errored = model_df[model_df["wer"].isna()]
        print(f"\n{model_name}:")
        print(f"  Samples: {len(valid)} valid, {len(errored)} errored")

        if len(valid) == 0:
            if len(errored) > 0:
                print(f"  First error: {errored.iloc[0].get('error', 'unknown')}")
            continue

        print(f"  WER:  {valid['wer'].mean():.4f} (mean)")
        if "cer" in valid.columns:
            print(f"  CER:  {valid['cer'].mean():.4f} (mean)")
        if "ner" in valid.columns:
            ner_valid = valid[valid["ner_has_numbers"] == True]
            if len(ner_valid) > 0:
                print(f"  NER:  {ner_valid['ner'].mean():.4f} (mean, {len(ner_valid)} samples with numbers)")
        if "rtf" in valid.columns:
            rtf_valid = valid[valid["rtf"].notna()]
            if len(rtf_valid) > 0:
                print(f"  RTF:  {rtf_valid['rtf'].mean():.4f} (mean)")

    print(f"\nResults saved to: {results_csv}")
    return results_csv


def main():
    parser = argparse.ArgumentParser(description="Banking ASR Evaluation Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--manifest", required=True, help="Path to evaluation manifest JSON")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint every N samples")

    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        manifest_path=args.manifest,
        output_dir=args.output,
        max_samples=args.max_samples,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
