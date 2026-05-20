import json
import os
import argparse
from banking_asr_eval.data.loaders import load_hf_dataset

def prepare_from_local_dir(dir_path: str, output_path: str, language: str = "hindi") -> None:
    """
    Search recursively in a local directory for audio files and transcription files,
    pair them up, calculate durations, and write the manifest.
    """
    import librosa
    
    print(f"Scanning local directory: {dir_path}")
    
    # 1. Recursively find audio files
    audio_extensions = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}
    audio_files = {}
    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in audio_extensions:
                basename = os.path.splitext(f)[0]
                audio_path = os.path.join(root, f)
                audio_files[basename] = os.path.abspath(audio_path)
                # Also store key with extension
                audio_files[f] = os.path.abspath(audio_path)

    print(f"Found {len(set(audio_files.values()))} audio files.")

    # 2. Recursively find transcription files
    trans_files = []
    trans_names = {"transcription.txt", "transcription_n2w.txt", "transcripts.txt", "text", "transcript.txt"}
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower() in trans_names or f.lower().endswith("_transcription.txt") or f.lower().endswith(".trans.txt"):
                trans_files.append(os.path.join(root, f))

    print(f"Found {len(trans_files)} potential transcription files: {trans_files}")

    # 3. Parse transcripts
    transcripts = {}
    for tf in trans_files:
        print(f"Parsing transcription file: {tf}")
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split by tab first, then by space if tab not found
                if "\t" in line:
                    parts = line.split("\t", 1)
                else:
                    parts = line.split(" ", 1)
                
                if len(parts) == 2:
                    audio_key, text = parts
                    audio_key = audio_key.strip()
                    text = text.strip()
                    key_no_ext = os.path.splitext(audio_key)[0]
                    transcripts[audio_key] = text
                    transcripts[key_no_ext] = text

    print(f"Loaded {len(transcripts)} transcription entries.")

    # 4. Pair them up
    manifest = []
    matched_paths = set()
    for key, audio_path in audio_files.items():
        if audio_path in matched_paths:
            continue
        
        ref_text = None
        if key in transcripts:
            ref_text = transcripts[key]
        elif os.path.splitext(key)[0] in transcripts:
            ref_text = transcripts[os.path.splitext(key)[0]]
        
        if ref_text:
            matched_paths.add(audio_path)
            try:
                duration = librosa.get_duration(path=audio_path)
            except Exception:
                duration = 5.0
            
            manifest.append({
                "audio_id": os.path.splitext(os.path.basename(audio_path))[0],
                "audio_path": audio_path,
                "duration_seconds": round(duration, 2),
                "reference_transcript": ref_text,
                "language": language,
                "dataset_source": "local_directory",
            })

    if not manifest:
        print("Warning: No audio-transcription matches found!")
    else:
        print(f"Successfully matched {len(manifest)} samples.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done! Manifest saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ASR datasets (local or HF)")
    parser.add_argument("--dataset", default="ai4bharat/Kathbath", help="HuggingFace dataset name OR path to local dataset directory")
    parser.add_argument("--split", default="valid", help="Dataset split (HuggingFace only)")
    parser.add_argument("--language", default="hindi", help="Language filter")
    parser.add_argument("--output", default="./data/manifests/kathbath_hindi.json", help="Output manifest path")
    args = parser.parse_args()

    if os.path.isdir(args.dataset):
        print(f"Detected local directory: {args.dataset}")
        prepare_from_local_dir(args.dataset, args.output, args.language)
    else:
        print(f"Downloading {args.dataset} ({args.language}) from HuggingFace...")
        # This automatically downloads the dataset and caches the audio files
        manifest = load_hf_dataset(args.dataset, split=args.split, language=args.language)

        # HuggingFace loads audio as numpy arrays which aren't JSON serializable.
        # Since HF also caches the raw .wav file paths, we just keep the path.
        for sample in manifest:
            if "audio_array" in sample:
                del sample["audio_array"]  

        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Done! Manifest saved to {args.output}")

    print("You can now evaluate with:")
    print(f"python -m banking_asr_eval.evaluate --config config.yaml --manifest {args.output}")


if __name__ == "__main__":
    main()
