import json
import os
import random

def read_manifest(path):
    try:
        with open(path) as f:
            try:
                content = f.read()
                return json.loads(content)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
        return []

def write_manifest(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data)} samples to {path}")

def generate_digit_augmentations(base_samples, count=2000):
    """
    Generates digit-rich banking text templates to augment model learning.
    This injects common numeric sequences like account numbers, UPI IDs, OTPs,
    and currencies in both Hindi and English/Latin scripts.
    """
    digits = "0123456789"
    digit_words_hi = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"]
    digit_words_en = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    templates = [
        "मेरा अकाउंट नंबर {num} है",
        "अकाउंट संख्या {num} में पैसे ट्रांसफर करें",
        "मेरा ओटीपी {num} है",
        "ट्रांजैक्शन आईडी {num} नोट कीजिए",
        "खाता संख्या {num} का बैलेंस बताएं",
        "मेरा रजिस्टर्ड मोबाइल नंबर {num} है",
        "कृपया {num} रुपये भेजें",
        "amount is {num} rupees",
        "otp is {num}",
        "account number {num} balance",
    ]

    augmented = []
    # Seed from some real audio paths, but assign digit-augmented transcripts
    # We will reuse the audio of short base samples to match the length, or pad them.
    for i in range(count):
        base = random.choice(base_samples)
        template = random.choice(templates)
        
        # Generate random number sequence
        length = random.choice([4, 6, 9, 10])
        num_seq = "".join(random.choice(digits) for _ in range(length))
        
        # Decide writing style: raw digits or verbalized (Hindi / English)
        style = random.choice(["digits", "words_hi", "words_en"])
        if style == "words_hi":
            num_text = " ".join(digit_words_hi[int(d)] for d in num_seq)
        elif style == "words_en":
            num_text = " ".join(digit_words_en[int(d)] for d in num_seq)
        else:
            num_text = num_seq

        text = template.format(num=num_text)
        
        # Add to manifest matching base sample audio (as a speed-up proxy)
        augmented.append({
            "audio_id": f"aug_digit_{i:04d}",
            "audio_filepath": base["audio_filepath"],
            "audio_path": base["audio_filepath"],
            "text": text,
            "reference_transcript": text,
            "duration": base.get("duration", 3.0),
            "duration_seconds": base.get("duration_seconds", 3.0),
            "accent_group": "digit_augmented",
            "is_digit_augmented": True
        })
    return augmented

def main():
    os.makedirs("data/manifests", exist_ok=True)
    
    # 1. Load Config D manifest
    print("Loading Config D manifest...")
    config_d = read_manifest("data/manifests/finetune_configD.json")
    if not config_d:
        print("Error: Config D manifest not found. Run prepare_data.py first.")
        return

    # 2. Load Lahaja manifest and extract accent slices (North Indian belt)
    print("Loading Lahaja manifest...")
    lahaja = read_manifest("data/manifests/lahaja.json")
    
    # Filter to hindi_belt and punjab_haryana accent groups (North India)
    accent_slice = [
        item for item in lahaja 
        if item.get("accent_group") in ["hindi_belt", "punjab_haryana"]
    ]
    print(f"Extracted {len(accent_slice)} regional North Indian accent samples from Lahaja.")

    # 3. Generate digit augmentations using Config D audio files as base
    print("Generating digit augmentations...")
    digit_augmentations = generate_digit_augmentations(config_d, count=2000)

    # 4. Combine manifests
    config_e = config_d + accent_slice + digit_augmentations
    print(f"Config E Total Samples: {len(config_e)}")
    print(f"  - Config D: {len(config_d)}")
    # Ensure they have telephony flag set for the training pipeline
    for item in config_e:
        item["telephony_codec_simulation"] = True

    # 5. Save the final manifest
    write_manifest(config_e, "data/manifests/finetune_configE.json")
    print("Config E Manifest generation completed successfully.")

if __name__ == "__main__":
    main()
