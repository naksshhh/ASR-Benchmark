import os
import json
import soundfile as sf
from pathlib import Path

def build_manifest():
    data_dir = Path("data/datasets/mucs/train")
    transcripts_dir = data_dir / "transcripts"
    text_file = transcripts_dir / "text"
    
    if not text_file.exists():
        print(f"Error: {text_file} not found!")
        return

    manifest = []
    missing_audio = 0
    
    print("Reading MUCS transcripts...")
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
                
            utt_id = parts[0]
            transcript = parts[1]
            
            # Kaldi format sometimes has utterance IDs that map to a specific wav file
            # Let's check if the wav file exists
            wav_path = data_dir / f"{utt_id}.wav"
            
            if not wav_path.exists():
                # Some datasets prefix or suffix the ID, let's just skip if not found directly
                missing_audio += 1
                continue
                
            try:
                # Use soundfile to quickly get duration without loading entire audio
                info = sf.info(str(wav_path))
                duration = info.duration
                
                manifest.append({
                    "audio_filepath": str(wav_path.absolute()),
                    "text": transcript,
                    "duration": round(duration, 2),
                    "language": "mixed",
                    "source": "mucs"
                })
            except Exception as e:
                print(f"Error reading {wav_path}: {e}")
                
    os.makedirs("data/manifests", exist_ok=True)
    manifest_path = "data/manifests/mucs_finance_train.json"
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        # Write as JSON Lines
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\nSuccess! Built manifest with {len(manifest)} samples.")
    if missing_audio > 0:
        print(f"Note: Skipped {missing_audio} transcripts because the .wav file was missing.")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    build_manifest()
