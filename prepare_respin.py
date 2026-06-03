import os
import json
import argparse
import soundfile as sf

def main():
    parser = argparse.ArgumentParser(description="Prepare RESPIN manifest from raw directory")
    parser.add_argument("--respin-dir", default="data/datasets/respin/hindi_finance", help="Path to RESPIN root directory containing extracted folders")
    parser.add_argument("--output", default="data/manifests/respin_finance_train.json", help="Output manifest path")
    parser.add_argument("--domain", default="BANK", help="Filter by domain in filename (e.g. BANK)")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train", help="Include train or test splits")
    args = parser.parse_args()

    respin_dir = args.respin_dir
    if not os.path.exists(respin_dir):
        print(f"Error: RESPIN directory not found at {respin_dir}")
        print("Please ensure the RESPIN dataset is downloaded and extracted.")
        return

    manifest = []
    missing_audio = 0
    total_parsed = 0

    print(f"Scanning for transcript files in {respin_dir}...")
    for root, _, files in os.walk(respin_dir):
        # Exclude 'test' subdirectory for train split and vice versa
        if args.split == "train" and "test" in root.lower():
            continue
        if args.split == "test" and "test" not in root.lower():
            continue

        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)
                # Parse transcript file
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) != 2:
                            continue
                        
                        audio_id, transcript = parts
                        
                        # Filter by domain (e.g., BANK for finance)
                        if args.domain and args.domain not in audio_id:
                            continue
                            
                        # Audio file should be in the same directory
                        audio_path = os.path.join(root, f"{audio_id}.wav")
                        if not os.path.exists(audio_path):
                            missing_audio += 1
                            continue
                            
                        total_parsed += 1
                        
                        # Get duration
                        try:
                            duration = sf.info(audio_path).duration
                        except Exception:
                            duration = 5.0 # fallback default
                            
                        manifest.append({
                            "audio_id": audio_id,
                            # Dual compatibility
                            "audio_path": os.path.abspath(audio_path),
                            "audio_filepath": os.path.abspath(audio_path),
                            "reference_transcript": transcript,
                            "text": transcript,
                            "duration_seconds": round(duration, 2),
                            "duration": round(duration, 2),
                            "source": "respin",
                            "language": "hi",
                        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"RESPIN manifest prepared successfully!")
    print(f"  Total matched samples: {len(manifest)}")
    print(f"  Missing audio files: {missing_audio}")
    print(f"  Manifest path: {args.output}")

if __name__ == "__main__":
    main()
