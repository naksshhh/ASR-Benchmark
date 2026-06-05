import os
import json
import pandas as pd

results_dir = r"c:\Users\naksh\OneDrive\Desktop\Sem 6\Krim\ASR-Benchmark\results"
print("Scanning results directory...")

for fname in os.listdir(results_dir):
    if fname.startswith("eval_results_") and fname.endswith(".json"):
        path = os.path.join(results_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check if whisper-medium-banking-configD is in this run
            df = pd.DataFrame(data)
            if "model" in df.columns and "whisper-medium-banking-configD" in df["model"].values:
                # Filter by model and dataset
                sub = df[(df["model"] == "whisper-medium-banking-configD") & (df["dataset"] == "lahaja.json")]
                if len(sub) > 0:
                    print(f"\nFile: {fname}")
                    print(f"Total samples for whisper-medium-banking-configD on Lahaja: {len(sub)}")
                    valid = sub[sub["wer"].notna()]
                    errored = sub[sub["wer"].isna()]
                    print(f"  Valid: {len(valid)}, Errored: {len(errored)}")
                    if len(valid) > 0:
                        print(f"  Mean WER: {valid['wer'].mean() * 100:.2f}%")
                        if "accent_group" in valid.columns:
                            print("  By accent group:")
                            for grp, grp_df in valid.groupby("accent_group"):
                                print(f"    {grp}: WER = {grp_df['wer'].mean() * 100:.2f}% (n={len(grp_df)})")
        except Exception as e:
            print(f"Error reading {fname}: {e}")
