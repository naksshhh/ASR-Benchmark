"""
Dataset loaders for ASR evaluation.

Loads data from:
1. Our custom banking manifest format (JSON)
2. HuggingFace datasets (IndicSUPERB, Kathbath, etc.)
3. Local audio directories with transcripts
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Iterator

import yaml


def load_manifest(manifest_path: str) -> List[Dict]:
    """
    Load our custom banking evaluation manifest.

    Expected format: JSON array of objects with at minimum:
    - audio_path: path to audio file
    - reference_transcript: ground truth text

    Optional fields: audio_id, duration_seconds, language,
    scenario, accent_region, noise_level
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Resolve relative audio paths
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    for sample in manifest:
        audio_path = sample.get("audio_path", "")
        if audio_path and not os.path.isabs(audio_path):
            sample["audio_path"] = os.path.join(manifest_dir, audio_path)

    return manifest


def load_hf_dataset(
    dataset_name: str,
    split: str = "test",
    language: str = "hi",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Load a HuggingFace ASR dataset and convert to our manifest format.

    Supports:
    - ai4bharat/IndicSUPERB
    - google/fleurs (Hindi subset)
    - mozilla-foundation/common_voice
    - Custom datasets with 'audio' and 'sentence'/'text' columns

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split (train/validation/test)
        language: Language filter
        max_samples: Limit number of samples (for testing)
        cache_dir: Local cache directory for downloads

    Returns:
        List of manifest-format dicts
    """
    from datasets import load_dataset, Audio

    print(f"[Loader] Loading {dataset_name} ({split}, lang={language})...")

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # Different datasets have different loading patterns
    if "fleurs" in dataset_name.lower():
        ds = load_dataset(dataset_name, f"{language}_in", split=split, **kwargs)
        text_col = "transcription"
    elif "common_voice" in dataset_name.lower():
        ds = load_dataset(dataset_name, language, split=split, **kwargs)
        text_col = "sentence"
    else:
        # Generic: try to auto-detect
        try:
            ds = load_dataset(dataset_name, language, split=split, **kwargs)
        except Exception:
            ds = load_dataset(dataset_name, split=split, **kwargs)
        text_col = "sentence" if "sentence" in ds.column_names else "text"

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Ensure audio column is properly typed
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    manifest = []
    for i, sample in enumerate(ds):
        entry = {
            "audio_id": f"{dataset_name.split('/')[-1]}_{i:05d}",
            "reference_transcript": sample.get(text_col, ""),
            "language": language,
            "dataset_source": dataset_name,
        }

        # Handle audio — either path or array
        if "audio" in sample:
            audio = sample["audio"]
            if isinstance(audio, dict):
                entry["audio_path"] = audio.get("path", "")
                entry["audio_array"] = audio.get("array")
                entry["sampling_rate"] = audio.get("sampling_rate", 16000)
            else:
                entry["audio_path"] = str(audio)

        # Optional metadata
        for field in ["accent", "gender", "age", "duration"]:
            if field in sample:
                entry[field] = sample[field]

        manifest.append(entry)

    print(f"[Loader] Loaded {len(manifest)} samples from {dataset_name}")
    return manifest


def iterate_manifest(
    manifest: List[Dict],
    require_audio: bool = True,
) -> Iterator[Dict]:
    """
    Iterate over manifest, optionally filtering to samples with existing audio.

    Yields manifest entries one at a time (memory-efficient for large datasets).
    """
    for sample in manifest:
        if require_audio:
            audio_path = sample.get("audio_path", "")
            if audio_path and os.path.exists(audio_path):
                yield sample
            elif "audio_array" in sample:
                yield sample
        else:
            yield sample
