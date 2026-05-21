import json
import os

def read_manifest(path):
    try:
        with open(path) as f:
            return [json.loads(l) for l in f]
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
