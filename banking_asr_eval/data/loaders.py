import os
import json
from typing import Dict, List, Iterator, Optional

from datasets import load_dataset, Audio


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
    split: str = "valid",
    language: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """
    Load a HuggingFace ASR dataset and convert it into a manifest format.

    This version avoids TorchCodec / FFmpeg decoding issues by disabling
    automatic audio decoding and only storing audio file paths.
    """

    print(f"[Loader] Loading {dataset_name} ({split}, lang={language})...")

    # Load dataset
    if language:
        ds = load_dataset(
            dataset_name,
            language,
            split=split,
            **kwargs,
        )
    else:
        ds = load_dataset(
            dataset_name,
            split=split,
            **kwargs,
        )

    # Disable automatic audio decoding
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))

    # Detect transcript column
    if "sentence" in ds.column_names:
        text_col = "sentence"
    elif "text" in ds.column_names:
        text_col = "text"
    elif "transcript" in ds.column_names:
        text_col = "transcript"
    else:
        raise ValueError(
            f"No transcript column found in dataset columns: {ds.column_names}"
        )

    # Limit samples if requested
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    manifest = []

    for i, sample in enumerate(ds):
        entry = {
            "audio_id": f"{dataset_name.split('/')[-1]}_{i:05d}",
            "reference_transcript": sample.get(text_col, ""),
            "language": language,
            "dataset_source": dataset_name,
        }

        # Store only audio path (NO decoding)
        if "audio" in sample:
            audio = sample["audio"]

            if isinstance(audio, dict):
                entry["audio_path"] = audio.get("path", "")
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
    Iterate over manifest entries.

    If require_audio=True:
    - only yields entries with valid audio_path
    """

    for sample in manifest:
        if require_audio:
            audio_path = sample.get("audio_path", "")

            if audio_path and os.path.exists(audio_path):
                yield sample
        else:
            yield sample