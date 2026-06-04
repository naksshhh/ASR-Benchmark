import os
import sys

# Ensure Hugging Face uses scratch for caching on the cluster to avoid home quota limits
user = os.environ.get("USER", "nakshatrak_iitp")
scratch_cache = f"/scratch/{user}/hf_cache"
if os.path.exists(f"/scratch/{user}"):
    os.makedirs(scratch_cache, exist_ok=True)
    os.environ["HF_HOME"] = scratch_cache
    os.environ["HF_DATASETS_CACHE"] = os.path.join(scratch_cache, "datasets")

# Increase Hugging Face download timeout to 10 minutes to prevent timeouts on large files
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

import json
import soundfile as sf
from datasets import load_dataset

def main():
    # Core Hindi-belt districts capturing the major regional accents (Bhojpuri, Awadhi, Haryana, West)
    # Reduced from 17 to 6 to make the download highly manageable (~2 hours instead of 10 hours)
    PRIORITY_DISTRICTS = [
        # Bihar (East / Bhojpuri)
        ("Bihar", "Patna"), ("Bihar", "Gaya"),
        # Uttar Pradesh (Central / Awadhi)
        ("UttarPradesh", "Lucknow"), ("UttarPradesh", "Varanasi"),
        # Haryana / Rajasthan (North / West)
        ("Haryana", "Rohtak"), ("Rajasthan", "Jaipur"),
    ]

    manifest = []
    existing_districts = set()
    output_dir = "data/manifests"
    output_path = os.path.join(output_dir, "vaani_hindi_belt.json")
    
    if os.path.exists(f"/scratch/{user}"):
        vaani_wav_dir = f"/scratch/{user}/vaani_wavs"
    else:
        vaani_wav_dir = os.path.abspath("data/datasets/vaani/wavs")
    os.makedirs(vaani_wav_dir, exist_ok=True)


    # Load existing manifest to skip already processed districts
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        manifest.append(item)
                        existing_districts.add((item.get("state"), item.get("district")))
            print(f"Loaded {len(manifest)} existing samples from {output_path} (covering districts: {list(existing_districts)})")
        except Exception as e:
            print(f"Could not read existing manifest: {e}. Starting fresh.")
            manifest = []
            existing_districts = set()

    hf_token = os.environ.get("HF_TOKEN")

    print("Starting Vaani download and preparation (this requires internet and HF access)...")

    for state, district in PRIORITY_DISTRICTS:
        if (state, district) in existing_districts:
            print(f"Skipping {state}/{district} (already processed).")
            continue
        subset_name = f"{state}_{district}"
        try:
            print(f"Loading subset: {subset_name}...")
            ds = load_dataset(
                "ARTPARK-IISc/VAANI",
                subset_name,
                token=hf_token,
                split="train"
            )
            
            # Verify if the transcript column exists (some untranscribed subsets lack this column entirely)
            if "transcript" not in ds.column_names:
                print(f"  Skipped {state}/{district}: 'transcript' column not found in dataset features.")
                continue

            # Vaani is ~10% transcribed — filter to transcribed only
            transcribed = ds.filter(lambda x: x.get("isTranscriptionAvailable") == "Yes" and x.get("transcript") is not None and len(str(x["transcript"]).strip()) > 5)
            
            for sample in transcribed:
                audio_data = sample["audio"]
                # Calculate correct duration in seconds
                if "array" in audio_data and "sampling_rate" in audio_data:
                    duration = len(audio_data["array"]) / audio_data["sampling_rate"]
                else:
                    duration = 5.0

                # Save the audio array to a physical wav file on scratch
                raw_filename = os.path.basename(audio_data['path'])
                clean_filename = raw_filename.replace("$", "").replace("?", "").replace("&", "")
                if not clean_filename.endswith(".wav"):
                    clean_filename = f"{clean_filename}.wav"
                
                wav_path = os.path.join(vaani_wav_dir, f"vaani_{subset_name}_{clean_filename}")
                
                # Write to disk if it doesn't exist
                if not os.path.exists(wav_path):
                    try:
                        sf.write(wav_path, audio_data["array"], audio_data["sampling_rate"])
                    except Exception as e:
                        print(f"Error writing wav file {wav_path}: {e}")
                        continue

                manifest.append({
                    "audio_id": f"vaani_{subset_name}_{clean_filename}",
                    # Dual compatibility
                    "audio_path": os.path.abspath(wav_path),
                    "audio_filepath": os.path.abspath(wav_path),
                    "reference_transcript": sample["transcript"],
                    "text": sample["transcript"],
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
