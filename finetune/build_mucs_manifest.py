import os
import json
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_recording(rec_id, wav_filename, rec_segments, data_dir, output_audio_dir):
    wav_path = data_dir / wav_filename
    if not wav_path.exists():
        return [], 0
        
    try:
        # Load the whole recording once
        y, sr = sf.read(str(wav_path))
    except Exception as e:
        print(f"Failed to read {wav_path}: {e}")
        return [], 0
        
    manifest_entries = []
    
    for utt_id, transcript, start, end in rec_segments:
        start_sample = int(float(start) * sr)
        end_sample = int(float(end) * sr)
        
        out_filepath = output_audio_dir / f"{utt_id}.wav"
        
        # Save the slice
        segment_audio = y[start_sample:end_sample]
        sf.write(str(out_filepath), segment_audio, sr)
        
        duration = round(float(end) - float(start), 2)
        
        manifest_entries.append({
            "audio_filepath": str(out_filepath.absolute()),
            "text": transcript,
            "duration": duration,
            "language": "mixed",
            "source": "mucs"
        })
        
    return manifest_entries, len(rec_segments)

def build_manifest():
    data_dir = Path("data/datasets/mucs/train")
    transcripts_dir = data_dir / "transcripts"
    
    text_file = transcripts_dir / "text"
    segments_file = transcripts_dir / "segments"
    wav_scp_file = transcripts_dir / "wav.scp"
    
    output_audio_dir = data_dir / "segments_wav"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    print("Reading text, segments, and wav.scp...")
    
    # 1. Parse text -> utt_id: transcript
    transcripts = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
                
    # 2. Parse wav.scp -> rec_id: filename
    recordings = {}
    with open(wav_scp_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                recordings[parts[0]] = parts[1]
                
    # 3. Parse segments and group by recording
    # We group by recording so we only load each huge WAV file once
    grouped_segments = {}
    with open(segments_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                utt_id, rec_id, start, end = parts
                if utt_id in transcripts:
                    if rec_id not in grouped_segments:
                        grouped_segments[rec_id] = []
                    grouped_segments[rec_id].append(
                        (utt_id, transcripts[utt_id], start, end)
                    )
                    
    print(f"Found {len(grouped_segments)} recordings to slice. Starting parallel processing...")
    
    manifest = []
    total_processed = 0
    
    # Process in parallel to make it lightning fast
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for rec_id, rec_segments in grouped_segments.items():
            if rec_id in recordings:
                wav_filename = recordings[rec_id]
                futures.append(
                    executor.submit(process_recording, rec_id, wav_filename, rec_segments, data_dir, output_audio_dir)
                )
                
        for future in as_completed(futures):
            entries, count = future.result()
            manifest.extend(entries)
            total_processed += count
            if total_processed % 5000 < len(entries):
                print(f"Sliced {total_processed} segments...")

    os.makedirs("data/manifests", exist_ok=True)
    manifest_path = "data/manifests/mucs_finance_train.json"
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\nSuccess! Built manifest with {len(manifest)} samples.")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Sliced audio saved to: {output_audio_dir}")

if __name__ == "__main__":
    build_manifest()
