import os
import json
import argparse
import soundfile as sf

def main():
    parser = argparse.ArgumentParser(description="Prepare RESPIN manifest from Kaldi directory")
    parser.add_argument("--respin-dir", default="data/datasets/respin/hindi_finance", help="Path to RESPIN subset directory containing Kaldi files")
    parser.add_argument("--output", default="data/manifests/respin_finance_train.json", help="Output manifest path")
    args = parser.parse_args()

    respin_dir = args.respin_dir
    if not os.path.exists(respin_dir):
        print(f"Error: RESPIN directory not found at {respin_dir}")
        print("Please ensure the RESPIN dataset is downloaded and extracted.")
        return

    # Check for standard Kaldi files
    wav_scp_path = os.path.join(respin_dir, "wav.scp")
    text_path = os.path.join(respin_dir, "text")
    utt2dur_path = os.path.join(respin_dir, "utt2dur")

    if not os.path.exists(wav_scp_path) or not os.path.exists(text_path):
        print(f"Error: wav.scp or text file not found under {respin_dir}")
        return

    print("Parsing text/transcripts...")
    transcripts = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]

    print("Parsing durations if utt2dur exists...")
    durations = {}
    if os.path.exists(utt2dur_path):
        with open(utt2dur_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    durations[parts[0]] = float(parts[1])

    print("Parsing wav.scp and generating manifest entries...")
    manifest = []
    missing_audio = 0
    
    with open(wav_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
                
            utt_id, audio_path = parts
            
            # Resolve path if it is relative to respin_dir
            if not os.path.isabs(audio_path):
                # Try relative to current working dir, then relative to respin_dir
                candidate1 = os.path.join(respin_dir, audio_path)
                candidate2 = os.path.join(respin_dir, os.path.basename(audio_path))
                if os.path.exists(candidate1):
                    audio_path = os.path.abspath(candidate1)
                elif os.path.exists(candidate2):
                    audio_path = os.path.abspath(candidate2)
                else:
                    # Look recursively for the audio file in respin_dir if not found directly
                    found = False
                    for root, _, files in os.walk(respin_dir):
                        if os.path.basename(audio_path) in files:
                            audio_path = os.path.abspath(os.path.join(root, os.path.basename(audio_path)))
                            found = True
                            break
                    if not found:
                        missing_audio += 1
                        continue

            if not os.path.exists(audio_path):
                missing_audio += 1
                continue

            transcript = transcripts.get(utt_id)
            if not transcript:
                continue

            # Get duration
            duration = durations.get(utt_id)
            if duration is None:
                try:
                    duration = sf.info(audio_path).duration
                except Exception:
                    duration = 5.0 # fallback default

            manifest.append({
                "audio_id": utt_id,
                # Dual compatibility
                "audio_path": audio_path,
                "audio_filepath": audio_path,
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
