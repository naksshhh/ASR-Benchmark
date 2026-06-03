import os
import sys

# Ensure Hugging Face uses scratch for caching on the cluster to avoid home quota limits
user = os.environ.get("USER", "nakshatrak_iitp")
scratch_cache = f"/scratch/{user}/hf_cache"
if os.path.exists(f"/scratch/{user}"):
    os.makedirs(scratch_cache, exist_ok=True)
    os.environ["HF_HOME"] = scratch_cache
    os.environ["HF_DATASETS_CACHE"] = os.path.join(scratch_cache, "datasets")

import json
from datasets import load_dataset

def main():
    # Hindi-belt districts prioritized for banking customer base
    PRIORITY_DISTRICTS = [
        # Bihar
        ("Bihar", "Patna"), ("Bihar", "Gaya"), ("Bihar", "Muzaffarpur"),
        ("Bihar", "Bhagalpur"), ("Bihar", "Darbhanga"),
        # Uttar Pradesh
        ("UttarPradesh", "Lucknow"), ("UttarPradesh", "Varanasi"),
        ("UttarPradesh", "Kanpur"), ("UttarPradesh", "Agra"),
        ("UttarPradesh", "Meerut"), ("UttarPradesh", "Allahabad"),
        # Punjab / Haryana
        ("Punjab", "Ludhiana"), ("Punjab", "Amritsar"),
        ("Haryana", "Rohtak"), ("Haryana", "Gurugram"),
        # Rajasthan
        ("Rajasthan", "Jaipur"), ("Rajasthan", "Jodhpur"),
    ]

    manifest = []
    hf_token = os.environ.get("HF_TOKEN")

    print("Starting Vaani download and preparation (this requires internet and HF access)...")

    for state, district in PRIORITY_DISTRICTS:
        subset_name = f"{state}_{district}"
        try:
            print(f"Loading subset: {subset_name}...")
            ds = load_dataset(
                "ARTPARK-IISc/VAANI",
                subset_name,
                token=hf_token,
                split="train"
            )
            # Vaani is ~10% transcribed — filter to transcribed only
            transcribed = ds.filter(lambda x: x["transcription"] and len(x["transcription"].strip()) > 5)
            
            for sample in transcribed:
                audio_data = sample["audio"]
                # Calculate correct duration in seconds
                if "array" in audio_data and "sampling_rate" in audio_data:
                    duration = len(audio_data["array"]) / audio_data["sampling_rate"]
                else:
                    duration = 5.0

                manifest.append({
                    "audio_id": f"vaani_{subset_name}_{os.path.basename(audio_data['path'])}",
                    # Dual compatibility
                    "audio_path": audio_data["path"],
                    "audio_filepath": audio_data["path"],
                    "reference_transcript": sample["transcription"],
                    "text": sample["transcription"],
                    "duration_seconds": round(duration, 2),
                    "duration": round(duration, 2),
                    "state": state,
                    "district": district,
                    "accent_group": "hindi_belt" if state in ["Bihar", "UttarPradesh"] else "punjab_haryana" if state in ["Punjab", "Haryana"] else "west_india",
                })
            print(f"  Loaded {state}/{district}: {len(transcribed)} transcribed samples")
        except Exception as e:
            print(f"  Failed loading {state}/{district}: {e}")

    output_dir = "data/manifests"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vaani_hindi_belt.json")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nTotal Vaani samples prepared: {len(manifest)}")
    print(f"Manifest written to: {output_path}")

if __name__ == "__main__":
    main()
