"""
Latency + Quality combined benchmark.

Measures RTF (Real-Time Factor) with proper warmup and statistical rigor,
then cross-correlates with WER for Pareto frontier analysis.

Usage:
    python -m banking_asr_eval.benchmark --config config.yaml --manifest data/synthetic/manifest.json
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .data.loaders import load_manifest
from .metrics import compute_wer, normalize
from .models import ModelRegistry


def benchmark_model(
    model_fn,
    audio_path: str,
    reference: str,
    warmup_runs: int = 3,
    timed_runs: int = 5,
    duration_seconds: float = 0,
) -> Dict:
    """
    Benchmark a single audio file with warmup and multiple timed runs.

    Returns latency stats (mean, std, min, max, p50, p95) + WER.
    """
    # Warmup
    for _ in range(warmup_runs):
        try:
            _ = model_fn(audio_path)
        except Exception:
            break

    # Timed runs
    latencies = []
    hypothesis = ""
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        try:
            hypothesis = model_fn(audio_path)
        except Exception as e:
            return {"error": str(e)}
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies)

    # WER
    wer = compute_wer(reference, hypothesis)

    result = {
        "latency_mean": float(latencies.mean()),
        "latency_std": float(latencies.std()),
        "latency_min": float(latencies.min()),
        "latency_max": float(latencies.max()),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "wer": wer,
        "hypothesis": hypothesis,
    }

    if duration_seconds and duration_seconds > 0:
        result["rtf_mean"] = float(latencies.mean() / duration_seconds)
        result["rtf_p95"] = float(np.percentile(latencies, 95) / duration_seconds)

    return result


def run_benchmark(
    config_path: str,
    manifest_path: str,
    output_dir: str = "./results",
    max_samples: Optional[int] = None,
) -> str:
    """
    Run latency+quality benchmark for all enabled models.

    Returns path to results CSV.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    bench_config = config.get("benchmark", {})
    warmup = bench_config.get("warmup_runs", 3)
    timed = bench_config.get("timed_runs", 5)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    manifest = load_manifest(manifest_path)
    if max_samples:
        manifest = manifest[:max_samples]

    # Filter to samples with existing audio
    manifest = [s for s in manifest if os.path.exists(s.get("audio_path", ""))]
    print(f"\nBenchmark: {len(manifest)} samples, {warmup} warmup, {timed} timed runs")

    registry = ModelRegistry.from_config(config_path)
    all_results = []

    for model_name, model_fn in registry.enabled_models():
        print(f"\n── Benchmarking: {model_name} ──")

        for sample in tqdm(manifest):
            result = benchmark_model(
                model_fn=model_fn,
                audio_path=sample["audio_path"],
                reference=sample.get("reference_transcript", ""),
                warmup_runs=warmup,
                timed_runs=timed,
                duration_seconds=sample.get("duration_seconds", 0),
            )
            result["model"] = model_name
            result["audio_id"] = sample.get("audio_id", "")
            result["duration_seconds"] = sample.get("duration_seconds", 0)
            result["language"] = sample.get("language", "unknown")
            result["scenario"] = sample.get("scenario", "unknown")
            all_results.append(result)

    # Save
    results_csv = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv, index=False)

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for model_name in df["model"].unique():
        m = df[df["model"] == model_name]
        print(f"\n{model_name}:")
        print(f"  Latency (mean): {m['latency_mean'].mean():.3f}s")
        print(f"  Latency (p95):  {m['latency_p95'].mean():.3f}s")
        if "rtf_mean" in m.columns:
            rtf = m["rtf_mean"].dropna()
            if len(rtf) > 0:
                print(f"  RTF (mean):     {rtf.mean():.3f}")
        print(f"  WER (mean):     {m['wer'].mean():.4f}")

    print(f"\nResults saved to: {results_csv}")
    return results_csv


def main():
    parser = argparse.ArgumentParser(description="Banking ASR Benchmark (Latency + Quality)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--manifest", required=True, help="Path to evaluation manifest JSON")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")

    args = parser.parse_args()
    run_benchmark(
        config_path=args.config,
        manifest_path=args.manifest,
        output_dir=args.output,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
