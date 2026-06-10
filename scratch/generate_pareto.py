import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style setup for premium look
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("talk")

COLORS = {
    # Baselines
    "whisper-tiny": "#636EFA",
    "whisper-large-v3-turbo": "#EF553B",
    "whisper-large-v3": "#00CC96",
    "whisper-medium-hi": "#B6E880",
    "indicwav2vec-hindi": "#19D3F3",
    "indicconformer-hindi": "#FF6692",
    "voxtral-mini-3b": "#E67E22",
    "streaming-zipformer": "#7F7F7F",
    "parakeet-tdt-0.6b": "#AB63FA",
    "canary-1b-flash": "#FFA15A",
    "nemotron-3.5-asr": "#9B59B6",
    "stt-hi-conformer-ctc-large": "#34495E",
    # Fine-tuned models
    "indicwav2vec-banking-configC": "#FF6B6B",
    "whisper-medium-banking-configC": "#4D96FF",
    "indicwav2vec-banking-configD": "#6BCB77",
    "whisper-medium-banking-configD": "#D1512D",
}

MARKERS = {
    # Fine-tuned models get a star, baseline models get circles
    "indicwav2vec-banking-configC": "*",
    "whisper-medium-banking-configC": "*",
    "indicwav2vec-banking-configD": "*",
    "whisper-medium-banking-configD": "*",
}

def load_kathbath_durations():
    try:
        manifest_path = "data/manifests/kathbath_hindi.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["audio_id"]: item["duration"] for item in data}
    except Exception as e:
        print(f"Error loading Kathbath manifest: {e}")
        return {}

def load_all_results():
    files = glob.glob("results/eval_results_*.csv")
    print(f"Found {len(files)} result files.")
    
    # Load Kathbath durations for mapping
    kathbath_durs = load_kathbath_durations()
    print(f"Loaded {len(kathbath_durs)} durations from Kathbath manifest.")
    
    file_candidates = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            
            if "dataset" not in df.columns or "model" not in df.columns:
                continue
                
            # Prepare space for computed RTF
            df["computed_rtf"] = df.get("rtf", np.nan)
            
            # Determine if this file contains Kathbath evaluations
            is_kathbath_file = any("kathbath" in str(d).lower() for d in df["dataset"].unique())
            
            for idx, row in df.iterrows():
                audio_id_val = row.get("audio_id")
                latency_val = row.get("latency_seconds")
                rtf_val = row.get("rtf")
                
                if is_kathbath_file:
                    duration = kathbath_durs.get(audio_id_val, 0)
                    if duration > 0 and pd.notna(latency_val):
                        df.at[idx, "computed_rtf"] = latency_val / duration
                    elif "latency_mean" in row and duration > 0:
                        df.at[idx, "computed_rtf"] = row["latency_mean"] / duration
                else:
                    if pd.isna(rtf_val) and pd.notna(latency_val):
                        dur_val = row.get("duration_seconds")
                        if pd.notna(dur_val) and dur_val > 0:
                            df.at[idx, "computed_rtf"] = latency_val / dur_val
            
            # Group by model and dataset *within this file*
            for (model_val, dataset_val), sub_df in df.groupby(["model", "dataset"]):
                dataset_val_str = str(dataset_val).lower()
                dataset_group = None
                if "kathbath" in dataset_val_str:
                    dataset_group = "kathbath"
                elif "synthetic" in dataset_val_str:
                    dataset_group = "synthetic"
                    
                if not dataset_group:
                    continue
                    
                wer_mean = sub_df["wer"].mean()
                rtf_mean = sub_df["computed_rtf"].dropna().mean()
                
                if pd.notna(wer_mean):
                    file_candidates.append({
                        "dataset": dataset_group,
                        "model": model_val,
                        "wer": wer_mean,
                        "rtf": rtf_mean,
                        "file": os.path.basename(f)
                    })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    df_candidates = pd.DataFrame(file_candidates)
    if df_candidates.empty:
        return pd.DataFrame()
        
    # For each dataset and model, pick the run with the lowest WER
    best_runs = []
    for (dataset_group, model_val), group in df_candidates.groupby(["dataset", "model"]):
        best_row = group.sort_values("wer").iloc[0]
        best_runs.append(best_row)
        
    return pd.DataFrame(best_runs)

def plot_pareto(df_group, dataset_name, output_path):
    # Convert WER to percentage
    df_group = df_group.copy()
    df_group["wer_pct"] = df_group["wer"] * 100
    
    print(f"\nStats for {dataset_name}:")
    print(df_group[["model", "wer", "rtf", "wer_pct", "file"]].to_string())
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot points
    for _, row in df_group.iterrows():
        model = row["model"]
        wer = row["wer_pct"]
        rtf = row["rtf"]
        
        # Don't plot if RTF is missing
        if pd.isna(rtf):
            continue
            
        color = COLORS.get(model, "#888888")
        marker = MARKERS.get(model, "o")
        size = 350 if marker == "*" else 200
        
        ax.scatter(wer, rtf, s=size, c=color, marker=marker, edgecolors="black", linewidths=1.5, zorder=5, label=model)
        
        # Adjust label placement dynamically
        ax.annotate(
            model,
            (wer, rtf),
            textcoords="offset points",
            xytext=(12, 5),
            fontsize=10,
            fontweight="bold",
            color="#2c3e50"
        )
        
    # Calculate and plot Pareto frontier line
    points = df_group[["wer_pct", "rtf", "model"]].dropna().values.tolist()
    
    if points:
        # Sort by WER ascending
        points.sort(key=lambda x: x[0])
        
        pareto_points = [points[0]]
        for p in points[1:]:
            if p[1] <= pareto_points[-1][1]:
                pareto_points.append(p)
                
        # Draw Pareto frontier line
        if len(pareto_points) > 1:
            px = [p[0] for p in pareto_points]
            py = [p[1] for p in pareto_points]
            ax.plot(px, py, "--", color="#7f8c8d", alpha=0.6, linewidth=2.5, zorder=2, label="Pareto Frontier")
            
    # Reference Line for Real-Time Barrier
    ax.axhline(y=1.0, color="#e74c3c", linestyle=":", alpha=0.7, linewidth=2, zorder=1)
    ax.text(ax.get_xlim()[0] + 0.5, 1.05, "Real-Time Limit (RTF = 1.0)", color="#e74c3c", fontsize=10, fontweight="bold")
    
    # Title and Labels
    dataset_title = "General Hindi (Kathbath)" if dataset_name == "kathbath" else "Banking Hinglish (Synthetic 100)"
    ax.set_title(f"ASR Quality-Latency Pareto Frontier ({dataset_title})", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Word Error Rate (WER %) ↓", fontsize=13, labelpad=10)
    ax.set_ylabel("Real-Time Factor (RTF) ↓", fontsize=13, labelpad=10)
    
    # Log scale for Y (RTF) if range is large
    if not df_group["rtf"].dropna().empty:
        min_rtf = df_group["rtf"].dropna().min()
        max_rtf = df_group["rtf"].dropna().max()
        if min_rtf > 0 and max_rtf / min_rtf > 10:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
            ax.set_ylabel("Real-Time Factor (RTF, Log Scale) ↓", fontsize=13, labelpad=10)
        
    # Clean grid
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto plot to: {output_path}")

def main():
    df = load_all_results()
    if df.empty:
        print("No evaluation data found. Check results/ folder.")
        return
        
    os.makedirs("results/plots", exist_ok=True)
    
    # Generate Kathbath Pareto plot
    df_kathbath = df[df["dataset"] == "kathbath"]
    if not df_kathbath.empty:
        plot_pareto(df_kathbath, "kathbath", "results/plots/kathbath_pareto.png")
    else:
        print("No Kathbath results found.")
        
    # Generate Synthetic Pareto plot
    df_synthetic = df[df["dataset"] == "synthetic"]
    if not df_synthetic.empty:
        plot_pareto(df_synthetic, "synthetic", "results/plots/synthetic_pareto.png")
    else:
        print("No Synthetic results found.")

if __name__ == "__main__":
    main()
