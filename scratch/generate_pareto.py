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
    
    all_rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            
            if "dataset" not in df.columns or "model" not in df.columns:
                continue
                
            # Compute/override RTF values
            for _, row in df.iterrows():
                dataset_val = str(row["dataset"]).lower()
                model_val = str(row["model"])
                wer_val = row.get("wer")
                rtf_val = row.get("rtf")
                audio_id_val = row.get("audio_id")
                latency_val = row.get("latency_seconds")
                
                # Categorize dataset
                dataset_group = None
                if "kathbath" in dataset_val:
                    dataset_group = "kathbath"
                elif "synthetic" in dataset_val:
                    dataset_group = "synthetic"
                    
                if not dataset_group or pd.isna(wer_val):
                    continue
                
                # For Kathbath, check if we need to calculate RTF manually
                if dataset_group == "kathbath":
                    duration = kathbath_durs.get(audio_id_val, 0)
                    if duration > 0 and pd.notna(latency_val):
                        rtf_val = latency_val / duration
                    elif "latency_mean" in row and duration > 0:
                        rtf_val = row["latency_mean"] / duration
                
                # Fallback calculation if rtf is NaN but latency and duration exist in row
                if pd.isna(rtf_val) and pd.notna(latency_val):
                    dur_val = row.get("duration_seconds")
                    if pd.notna(dur_val) and dur_val > 0:
                        rtf_val = latency_val / dur_val
                        
                all_rows.append({
                    "dataset": dataset_group,
                    "model": model_val,
                    "wer": wer_val,
                    "rtf": rtf_val
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.DataFrame(all_rows)

def plot_pareto(df_group, dataset_name, output_path):
    # Group by model
    model_stats = df_group.groupby("model").agg(
        wer_mean=("wer", "mean"),
        rtf_mean=("rtf", "mean")
    ).reset_index()
    
    # Convert WER to percentage
    model_stats["wer_pct"] = model_stats["wer_mean"] * 100
    
    print(f"\nStats for {dataset_name}:")
    print(model_stats.to_string())
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot points
    for _, row in model_stats.iterrows():
        model = row["model"]
        wer = row["wer_pct"]
        rtf = row["rtf_mean"]
        
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
    # A point is Pareto optimal if no other point has BOTH lower WER and lower RTF
    points = model_stats[["wer_pct", "rtf_mean", "model"]].dropna().values.tolist()
    
    if points:
        # Sort by WER ascending
        points.sort(key=lambda x: x[0])
        
        pareto_points = [points[0]]
        for p in points[1:]:
            # Since sorted by WER, if this point has lower or equal RTF than the last pareto point, it's on the frontier
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
    if not model_stats["rtf_mean"].dropna().empty:
        min_rtf = model_stats["rtf_mean"].dropna().min()
        max_rtf = model_stats["rtf_mean"].dropna().max()
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
