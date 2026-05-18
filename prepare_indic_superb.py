import json
import os
import argparse
from banking_asr_eval.data.loaders import load_hf_dataset

def main():
    parser = argparse.ArgumentParser(description="Download and prepare HuggingFace ASR datasets")
    parser.add_argument("--dataset", default="ai4bharat/Kathbath", help="HuggingFace dataset name")
    parser.add_argument("--split", default="valid", help="Dataset split")
    parser.add_argument("--language", default="hindi", help="Language filter")
    parser.add_argument("--output", default="./data/manifests/kathbath_hindi.json", help="Output manifest path")
    args = parser.parse_args()

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
