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
        try:
            content = f.read()
            manifest = json.loads(content)
        except json.JSONDecodeError:
            f.seek(0)
            manifest = [json.loads(line) for line in f if line.strip()]

    # Resolve relative audio paths or mismatched absolute paths from other machines
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    for sample in manifest:
        # Standardize keys for downstream evaluation compatibility
        if "audio_path" not in sample and "audio_filepath" in sample:
            sample["audio_path"] = sample["audio_filepath"]
        if "reference_transcript" not in sample and "text" in sample:
            sample["reference_transcript"] = sample["text"]

        audio_path = sample.get("audio_path", "")
        if audio_path:
            resolved_path = None
            
            # 1. If it's absolute, check if it exists
            if os.path.isabs(audio_path) and os.path.exists(audio_path):
                resolved_path = audio_path
            else:
                # 2. Try resolving relative to current working directory (project root)
                candidate_cwd = os.path.abspath(audio_path)
                # 3. Try resolving relative to manifest directory
                candidate_manifest = os.path.abspath(os.path.join(manifest_dir, audio_path))
                
                if os.path.exists(candidate_cwd):
                    resolved_path = candidate_cwd
                elif os.path.exists(candidate_manifest):
                    resolved_path = candidate_manifest
                else:
                    # 4. Try standard candidate directories using the filename
                    filename = os.path.basename(audio_path)
                    candidate_same_dir = os.path.abspath(os.path.join(manifest_dir, filename))
                    candidate_audio_dir = os.path.abspath(os.path.join(manifest_dir, "audio", filename))
                    
                    project_root = os.path.abspath(os.path.join(manifest_dir, "..", ".."))
                    candidate_root_data = os.path.abspath(os.path.join(project_root, "data", "audio", filename))
                    
                    if os.path.exists(candidate_audio_dir):
                        resolved_path = candidate_audio_dir
                    elif os.path.exists(candidate_same_dir):
                        resolved_path = candidate_same_dir
                    elif os.path.exists(candidate_root_data):
                        resolved_path = candidate_root_data

            # Fallback if nothing was resolved
            if resolved_path is None:
                if os.path.isabs(audio_path):
                    resolved_path = audio_path
                else:
                    resolved_path = os.path.abspath(os.path.join(manifest_dir, audio_path))
                    
            sample["audio_path"] = resolved_path

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

    # Enable automatic audio decoding to extract files and get valid paths
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=True))

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

        # Detect audio column (different datasets use different names)
        audio_col = None
        for col in ["audio", "audio_filepath"]:
            if col in sample:
                audio_col = col
                break

        if audio_col is not None:
            audio = sample[audio_col]
            audio_id = entry["audio_id"]

            # Check if audio is a decoded dict with array data
            if isinstance(audio, dict) and "array" in audio:
                import soundfile as sf
                save_dir = os.path.join("data", "audio", dataset_name.split("/")[-1].lower())
                os.makedirs(save_dir, exist_ok=True)
                local_path = os.path.join(save_dir, f"{audio_id}.wav")

                if not os.path.exists(local_path):
                    sf.write(local_path, audio["array"], audio["sampling_rate"])

                entry["audio_path"] = os.path.abspath(local_path)
            elif isinstance(audio, str) and os.path.exists(audio):
                entry["audio_path"] = os.path.abspath(audio)
            elif isinstance(audio, dict) and "path" in audio:
                p = audio["path"]
                if p and os.path.exists(p):
                    entry["audio_path"] = os.path.abspath(p)

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