import os
import json
import glob
import pandas as pd

def main():
    parquet_files = glob.glob("lahaja/data/*.parquet")
    if not parquet_files:
        print("Error: No LAHAJA parquet files found at lahaja/data/*.parquet. Please check the dataset download.")
        return

    # Map native languages to regional accent groups
    accent_map = {
        "Bhojpuri": "hindi_belt", 
        "Awadhi": "hindi_belt",
        "Maithili": "hindi_belt", 
        "Magahi": "hindi_belt",
        "Punjabi": "punjab_haryana", 
        "Haryanvi": "punjab_haryana",
        "Tamil": "south_india", 
        "Telugu": "south_india",
        "Kannada": "south_india", 
        "Malayalam": "south_india",
        "Bengali": "east_india", 
        "Odia": "east_india",
        "Marathi": "west_india", 
        "Gujarati": "west_india",
    }

    manifest = []
    os.makedirs("lahaja/audio", exist_ok=True)
    print("Extracting audio and building manifest from parquet files...")

    for parquet_file in parquet_files:
        print(f"Processing {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        for _, row in df.iterrows():
            filename = row["fname"]
            audio_path = f"lahaja/audio/{filename}.wav"
            
            # Write the raw wav bytes if not already extracted
            if not os.path.exists(audio_path):
                audio_data = row["audio_filepath"]
                if isinstance(audio_data, dict) and "bytes" in audio_data:
                    with open(audio_path, "wb") as f:
                        f.write(audio_data["bytes"])
                else:
                    # Fallback in case format is slightly different
                    continue
            
            manifest.append({
                "audio_id": str(filename),
                # Dual compatibility: evaluate.py (audio_path/reference_transcript) & HF/NeMo (audio_filepath/text)
                "audio_path": os.path.abspath(audio_path),
                "audio_filepath": os.path.abspath(audio_path),
                "reference_transcript": row["text"],
                "text": row["text"],
                "duration_seconds": float(row["duration"]),
                "duration": float(row["duration"]),
                "accent_group": accent_map.get(row["native_language"], "other"),
                "native_language": row["native_language"],
                "occupation_domain": row.get("occupation_domain", "other"),
                "age_group": row["age_group"],
                "gender": row["gender"],
            })

    os.makedirs("data/manifests", exist_ok=True)
    output_path = "data/manifests/lahaja.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    print(f"LAHAJA manifest prepared successfully!")
    print(f"  Total matched samples: {len(manifest)}")
    print(f"  Manifest path: {output_path}")

    # Print distribution
    df_manifest = pd.DataFrame(manifest)
    if not df_manifest.empty:
        print("\nAccent group distribution:")
        print(df_manifest.groupby("accent_group").size())

if __name__ == "__main__":
    main()
