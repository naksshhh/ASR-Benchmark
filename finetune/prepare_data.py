import json
import os

def read_manifest(path):
    try:
        with open(path) as f:
            # First try reading as a single JSON array
            try:
                # Seek back to start in case of multiple attempts (though read() resets it)
                content = f.read()
                return json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try reading as JSON Lines
                f.seek(0)
                return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
        return []

def write_manifest(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data)} samples to {path}")

def main():
    os.makedirs("data/manifests", exist_ok=True)
    
    mucs_train = read_manifest("data/manifests/mucs_finance_train.json")
    synthetic_train = read_manifest("data/synthetic/manifest.json")
    
    # Config A: MUCS Finance subset (~94h)
    write_manifest(mucs_train, "data/manifests/finetune_configA.json")
    
    # Config B: Synthetic banking corpus
    write_manifest(synthetic_train, "data/manifests/finetune_configB.json")
    
    # Config C: MUCS Finance + Synthetic banking
    write_manifest(mucs_train + synthetic_train, "data/manifests/finetune_configC.json")

if __name__ == "__main__":
    main()
