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

    # Copy any additional metadata fields from the sample (e.g. accent_group, native_language, gender, age_group, etc.)
    for k, v in sample.items():
        if k not in result and k not in ["audio_path", "reference_transcript"]:
            result[k] = v

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


def evaluate_chunk_worker(args) -> List[Dict]:
    """
    Worker process to evaluate a chunk of the manifest for a specific model.
    Loads the model and imports libraries locally in this process to avoid pickling/sharing issues.
    """
    model_name, config_path, chunk, device_id, worker_idx, dataset_name = args
    import os
    import json

    # Force this subprocess to see only the assigned GPU
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        # Import torch inside worker to initialize CUDA cleanly
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass

    # Lazy imports to ensure clean multiprocessing launch
    from banking_asr_eval.models import ModelRegistry
    from banking_asr_eval.evaluate import evaluate_sample

    registry = ModelRegistry.from_config(config_path)
    try:
        model_fn = registry.get_model(model_name)
    except Exception as e:
        print(f"[Worker {worker_idx}] Failed to load model {model_name}: {e}")
        return [{
            "audio_id": s.get("audio_id", "unknown"),
            "error": f"Failed to load model: {e}",
            "wer": None,
            "model": model_name,
            "dataset": dataset_name
        } for s in chunk]

    results = []
    checkpoint_path = f"./results/checkpoint_{model_name}_worker_{worker_idx}.json"
    
    for i, sample in enumerate(chunk):
        try:
            res = evaluate_sample(model_fn, sample)
            res["model"] = model_name
            res["dataset"] = dataset_name
            results.append(res)
        except Exception as e:
            results.append({
                "audio_id": sample.get("audio_id", "unknown"),
                "error": str(e),
                "wer": None,
                "model": model_name,
                "dataset": dataset_name
            })
            
        # Optional local checkpointing inside worker
        if (i + 1) % 50 == 0:
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
                
    # Cleanup local checkpoint if successfully completed
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass
            
    return results


def run_evaluation(
    config_path: str,
    manifest_path: str,
    output_dir: str = "./results",
    max_samples: Optional[int] = None,
    checkpoint_interval: int = 50,
    num_workers: Optional[int] = None,
    gpus: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    stratify_by: Optional[str] = None,
) -> str:
    """
    Run full evaluation: all enabled models × all samples.
    Can be run sequentially or in parallel using multiprocessing spawn.

    Args:
        config_path: Path to config.yaml
        manifest_path: Path to evaluation manifest JSON
        output_dir: Directory to save results
        max_samples: Limit samples (for testing)
        checkpoint_interval: Save checkpoint every N samples (sequential mode only)
        num_workers: Number of parallel worker processes (overrides config)
        gpus: List of GPU IDs to distribute workers across (overrides config)

    Returns:
        Path to results CSV
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device_config = config.get("device", {})
    compute = device_config.get("compute", "cpu")

    # Resolve workers and GPUs
    if num_workers is None:
        num_workers = device_config.get("parallel_workers", 1)
    if gpus is None:
        gpus = device_config.get("gpus", None)
        if isinstance(gpus, str):
            gpus = [int(x.strip()) for x in gpus.split(",") if x.strip()]
        elif isinstance(gpus, int):
            gpus = [gpus]

    # Clean up GPU assignment if CPU execution is requested
    if compute != "cuda":
        gpus = None

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

    # Load models registry to discover enabled models
    registry = ModelRegistry.from_config(config_path)
    if models:
        for m in models:
            if m not in registry._configs:
                raise KeyError(f"Model '{m}' is not registered in config.yaml")
        enabled_models = models
    else:
        enabled_models = [name for name, cfg in registry._configs.items() if cfg.get("enabled", False)]
    print(f"Models to evaluate: {enabled_models}\n")

    all_results = []

    # Configure multiprocessing context
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    for model_name in enabled_models:
        print(f"\n── Model: {model_name} ──")

        # Decide whether to execute sequentially or in parallel
        actual_workers = min(num_workers, len(manifest))
        
        if actual_workers <= 1:
            print("  Running in sequential single-process mode...")
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{model_name}_{timestamp}.json")

            # Check for existing checkpoint
            start_idx = 0
            model_results = []
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path) as f:
                    model_results = json.load(f)
                start_idx = len(model_results)
                print(f"    Resuming from checkpoint: {start_idx} samples done")

            # Load model function in main thread
            model_fn = registry.get_model(model_name)

            for i, sample in enumerate(tqdm(manifest[start_idx:], initial=start_idx, total=len(manifest))):
                result = evaluate_sample(model_fn, sample)
                result["model"] = model_name
                result["dataset"] = os.path.basename(manifest_path)
                model_results.append(result)

                # Checkpoint
                if (i + 1) % checkpoint_interval == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(model_results, f, ensure_ascii=False, indent=2)
                    print(f"\n    [Checkpoint] Saved {len(model_results)} results")

            # Final save for this model
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(model_results, f, ensure_ascii=False, indent=2)

        else:
            print(f"  Running in parallel mode with {actual_workers} workers...")
            # Chunk the manifest
            chunk_size = (len(manifest) + actual_workers - 1) // actual_workers
            chunks = [manifest[i:i + chunk_size] for i in range(0, len(manifest), chunk_size)]
            
            # Prepare task arguments
            tasks = []
            for idx, chunk in enumerate(chunks):
                # Distribute workers across available GPUs in round-robin fashion
                device_id = gpus[idx % len(gpus)] if gpus else None
                tasks.append((model_name, config_path, chunk, device_id, idx, os.path.basename(manifest_path)))

            # Execute in process pool
            model_results = []
            with multiprocessing.get_context("spawn").Pool(processes=actual_workers) as pool:
                chunk_results = pool.map(evaluate_chunk_worker, tasks)
                
            for r in chunk_results:
                model_results.extend(r)

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

        if stratify_by and stratify_by in valid.columns:
            print(f"  Stratified by {stratify_by}:")
            for val, val_df in valid.groupby(stratify_by):
                print(f"    {val}: WER = {val_df['wer'].mean():.4f} (count: {len(val_df)})")

    print(f"\nResults saved to: {results_csv}")
    return results_csv


def main():
    parser = argparse.ArgumentParser(description="Banking ASR Evaluation Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--manifest", required=True, help="Path to evaluation manifest JSON")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint every N samples")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--gpus", default=None, help="Comma-separated list of GPU IDs to use (e.g. 0,1)")
    parser.add_argument("--models", default=None, help="Comma-separated list of models to evaluate")
    parser.add_argument("--stratify-by", default=None, help="Field to stratify evaluation by (e.g. accent_group)")

    args = parser.parse_args()

    gpus_list = None
    if args.gpus:
        gpus_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    models_list = None
    if args.models:
        models_list = [m.strip() for m in args.models.split(",") if m.strip()]

    run_evaluation(
        config_path=args.config,
        manifest_path=args.manifest,
        output_dir=args.output,
        max_samples=args.max_samples,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.workers,
        gpus=gpus_list,
        models=models_list,
        stratify_by=args.stratify_by,
    )


if __name__ == "__main__":
    main()
