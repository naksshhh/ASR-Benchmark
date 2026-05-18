"""
Synthetic banking audio data generator.

Uses TTS to generate audio from banking dialogue scripts,
then saves as WAV with manifest metadata.

For Mac Mini testing: uses gTTS (Google TTS) — free, decent quality.
For production quality: use Coqui TTS or IndicTTS models.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .banking_scripts import get_random_scripts, ALL_SCENARIOS


def generate_manifest_only(
    output_dir: str,
    n_samples: int = 100,
    scenarios: List[str] = None,
    languages: List[str] = None,
) -> str:
    """
    Generate just the manifest (no audio) for pipeline testing.

    This is the fast path for Mac Mini — creates the metadata JSON
    without needing TTS. Useful for testing the evaluation pipeline
    with pre-existing audio or for dry runs.

    Args:
        output_dir: Directory to write manifest
        n_samples: Number of samples
        scenarios: Filter scenarios
        languages: Filter languages

    Returns:
        Path to manifest JSON file
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")

    scripts = get_random_scripts(n_samples, scenarios, languages)

    manifest = []
    for i, script in enumerate(scripts):
        sample = {
            "audio_id": f"banking_{i:04d}",
            "audio_path": os.path.join(output_dir, "audio", f"banking_{i:04d}.wav"),
            "duration_seconds": 5.0 + (i % 10),  # placeholder durations
            "reference_transcript": script["text"],
            "language": script["language"],
            "scenario": script["scenario"],
            "accent_region": "neutral",  # placeholder
            "noise_level": "clean",  # placeholder
        }
        manifest.append(sample)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[SyntheticGen] Wrote manifest with {len(manifest)} samples to {manifest_path}")
    return manifest_path


def generate_with_gtts(
    output_dir: str,
    n_samples: int = 20,
    scenarios: List[str] = None,
    languages: List[str] = None,
) -> str:
    """
    Generate audio using Google TTS (gTTS).

    Requires: pip install gTTS
    Limitations: only supports one language at a time, so mixed
    Hindi-English won't sound natural. Fine for pipeline testing.

    Args:
        output_dir: Directory to write audio + manifest
        n_samples: Number of samples
        scenarios: Filter scenarios
        languages: Filter languages

    Returns:
        Path to manifest JSON file
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("gTTS not installed. Run: pip install gTTS")

    output_dir = os.path.abspath(output_dir)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")

    scripts = get_random_scripts(n_samples, scenarios, languages)

    manifest = []
    for i, script in enumerate(scripts):
        audio_path = os.path.join(audio_dir, f"banking_{i:04d}.mp3")

        # gTTS language mapping
        lang_code = "hi" if script["language"] == "hindi" else "en"

        try:
            tts = gTTS(text=script["text"], lang=lang_code, slow=False)
            tts.save(audio_path)
            print(f"  [{i+1}/{n_samples}] Generated: {audio_path}")
        except Exception as e:
            print(f"  [{i+1}/{n_samples}] FAILED: {e}")
            continue

        sample = {
            "audio_id": f"banking_{i:04d}",
            "audio_path": audio_path,
            "duration_seconds": None,  # will be computed after
            "reference_transcript": script["text"],
            "language": script["language"],
            "scenario": script["scenario"],
            "accent_region": "neutral",
            "noise_level": "clean",
        }
        manifest.append(sample)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[SyntheticGen] Generated {len(manifest)} audio files + manifest at {manifest_path}")
    return manifest_path


def update_durations(manifest_path: str) -> None:
    """Update manifest with actual audio durations using librosa."""
    import librosa

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    updated = 0
    for sample in manifest:
        audio_path = sample["audio_path"]
        if os.path.exists(audio_path):
            try:
                duration = librosa.get_duration(path=audio_path)
                sample["duration_seconds"] = round(duration, 2)
                updated += 1
            except Exception as e:
                print(f"  Could not get duration for {audio_path}: {e}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[SyntheticGen] Updated durations for {updated}/{len(manifest)} samples")
