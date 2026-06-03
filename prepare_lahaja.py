import os
import json
import pandas as pd

def main():
    csv_path = "lahaja/meta-data-lahaja.csv"
    if not os.path.exists(csv_path):
        print(f"Error: LAHAJA metadata not found at {csv_path}. Please clone the LAHAJA dataset repository.")
        return

    meta = pd.read_csv(csv_path)
    
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
    missing_count = 0
    
    for _, row in meta.iterrows():
        # Check filename/audio path
        filename = row["filename"]
        audio_path = f"lahaja/audio/{filename}.wav"
        
        if not os.path.exists(audio_path):
            missing_count += 1
            continue
            
        manifest.append({
            "audio_id": filename,
            # Dual compatibility: evaluate.py (audio_path/reference_transcript) & HF/NeMo (audio_filepath/text)
            "audio_path": os.path.abspath(audio_path),
            "audio_filepath": os.path.abspath(audio_path),
            "reference_transcript": row["transcript"],
            "text": row["transcript"],
            "duration_seconds": row["duration"],
            "duration": row["duration"],
            "accent_group": accent_map.get(row["native_language"], "other"),
            "native_language": row["native_language"],
            "occupation_domain": row["occupation_domain"],
            "age_group": row["age_group"],
            "gender": row["gender"],
        })

    os.makedirs("data/manifests", exist_ok=True)
    output_path = "data/manifests/lahaja.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    print(f"LAHAJA manifest prepared successfully!")
    print(f"  Total matched samples: {len(manifest)}")
    print(f"  Missing audio files: {missing_count}")
    print(f"  Manifest path: {output_path}")

    # Print distribution
    df = pd.DataFrame(manifest)
    if not df.empty:
        print("\nAccent group distribution:")
        print(df.groupby("accent_group").size())

if __name__ == "__main__":
    main()
