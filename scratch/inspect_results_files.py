import pandas as pd
import glob
import os

files = glob.glob("results/eval_results_*.csv")
print(f"Found {len(files)} result files:")

data = []
for f in files:
    try:
        df = pd.read_csv(f)
        models = df['model'].unique() if 'model' in df.columns else []
        dataset = df['dataset'].unique() if 'dataset' in df.columns else []
        samples = len(df)
        has_rtf = 'rtf' in df.columns
        
        data.append({
            "file": os.path.basename(f),
            "samples": samples,
            "models": list(models),
            "dataset": list(dataset),
            "has_rtf": has_rtf
        })
    except Exception as e:
        print(f"Error reading {f}: {e}")

df_info = pd.DataFrame(data)
pd.set_option('display.max_colwidth', None)
print(df_info.to_string())
