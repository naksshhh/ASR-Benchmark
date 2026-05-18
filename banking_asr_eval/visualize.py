"""
Visualization module — Pareto plots, heatmaps, error analysis charts.

These are the figures that go in the blog and paper.

Usage:
    python -m banking_asr_eval.visualize --results results/eval_results_*.csv
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

# Use non-interactive backend for server environments
matplotlib.use("Agg")

# Style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

COLORS = {
    "whisper-tiny": "#636EFA",
    "whisper-large-v3-turbo": "#EF553B",
    "whisper-large-v3": "#00CC96",
    "parakeet-tdt-0.6b": "#AB63FA",
    "canary-1b-flash": "#FFA15A",
    "indicwav2vec-hindi": "#19D3F3",
    "indicconformer-hindi": "#FF6692",
    "whisper-medium-hi": "#B6E880",
}


def plot_wer_comparison(df: pd.DataFrame, output_path: str, title: str = "WER by Model"):
    """Bar chart comparing WER across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_wer = df.groupby("model")["wer"].mean().sort_values()
    colors = [COLORS.get(m, "#888888") for m in model_wer.index]

    bars = ax.barh(model_wer.index, model_wer.values, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, model_wer.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontweight="bold", fontsize=10)

    ax.set_xlabel("Word Error Rate (WER)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(model_wer.values) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pareto_frontier(df: pd.DataFrame, output_path: str):
    """
    WER vs RTF Pareto frontier plot.

    This is THE visualization for Pillar 5 — shows which models are
    on the optimal quality-latency frontier.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name in df["model"].unique():
        m = df[df["model"] == model_name]
        wer = m["wer"].mean()
        rtf = m["rtf"].dropna().mean() if "rtf" in m.columns else m.get("rtf_mean", pd.Series()).dropna().mean()

        if pd.isna(rtf):
            continue

        color = COLORS.get(model_name, "#888888")
        ax.scatter(wer, rtf, s=200, c=color, edgecolors="white", linewidth=2, zorder=5)
        ax.annotate(model_name, (wer, rtf), textcoords="offset points",
                    xytext=(10, 10), fontsize=9, fontweight="bold")

    # Pareto frontier line
    points = []
    for model_name in df["model"].unique():
        m = df[df["model"] == model_name]
        wer = m["wer"].mean()
        rtf_col = "rtf" if "rtf" in m.columns else "rtf_mean"
        rtf = m[rtf_col].dropna().mean() if rtf_col in m.columns else np.nan
        if not pd.isna(rtf):
            points.append((wer, rtf, model_name))

    if points:
        points.sort(key=lambda x: x[0])
        pareto = [points[0]]
        for p in points[1:]:
            if p[1] <= pareto[-1][1]:
                pareto.append(p)

        if len(pareto) > 1:
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.plot(px, py, "--", color="gray", alpha=0.5, linewidth=2, label="Pareto Frontier")

    # Reference lines
    ax.axhline(y=1.0, color="red", linestyle=":", alpha=0.3, label="RTF = 1.0 (real-time)")

    ax.set_xlabel("Word Error Rate (WER) ↓", fontsize=12)
    ax.set_ylabel("Real-Time Factor (RTF) ↓", fontsize=12)
    ax.set_title("Quality-Latency Pareto Frontier", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_metric_heatmap(df: pd.DataFrame, output_path: str):
    """
    Heatmap of all metrics across models.

    Rows = models, Columns = metrics. The comparison table from the blog.
    """
    metrics = ["wer", "cer", "mer", "wil"]
    if "ner" in df.columns:
        metrics.append("ner")
    if "entity_accuracy" in df.columns:
        metrics.append("entity_accuracy")

    # Only include metrics that are actually in the DataFrame
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("  Skipping metric heatmap (no metrics found)")
        return

    pivot = df.groupby("model")[metrics].mean()

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.8)))

    # For entity_accuracy, higher is better; for others, lower is better
    # Use a custom colormap
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Error Rate"},
    )

    ax.set_title("Metric Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_language_breakdown(df: pd.DataFrame, output_path: str):
    """WER breakdown by language (Hindi / English / Mixed)."""
    if "language" not in df.columns:
        print("  Skipping language breakdown (no 'language' column)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    pivot = df.groupby(["model", "language"])["wer"].mean().unstack(fill_value=0)

    pivot.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Word Error Rate", fontsize=12)
    ax.set_title("WER by Language per Model", fontsize=14, fontweight="bold")
    ax.legend(title="Language")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scenario_breakdown(df: pd.DataFrame, output_path: str):
    """WER breakdown by banking scenario."""
    if "scenario" not in df.columns:
        print("  Skipping scenario breakdown (no 'scenario' column)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    pivot = df.groupby(["model", "scenario"])["wer"].mean().unstack(fill_value=0)
    pivot.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Word Error Rate", fontsize=12)
    ax.set_title("WER by Banking Scenario per Model", fontsize=14, fontweight="bold")
    ax.legend(title="Scenario", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_plots(results_path: str, output_dir: str = "./results/plots"):
    """Generate all visualizations from evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating plots from: {results_path}")
    df = pd.read_csv(results_path)

    plot_wer_comparison(df, os.path.join(output_dir, "wer_comparison.png"))
    plot_metric_heatmap(df, os.path.join(output_dir, "metric_heatmap.png"))
    plot_language_breakdown(df, os.path.join(output_dir, "language_breakdown.png"))
    plot_scenario_breakdown(df, os.path.join(output_dir, "scenario_breakdown.png"))

    # Pareto only if RTF data exists
    if "rtf" in df.columns or "rtf_mean" in df.columns:
        plot_pareto_frontier(df, os.path.join(output_dir, "pareto_frontier.png"))

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument("--results", required=True, help="Path to results CSV")
    parser.add_argument("--output", default="./results/plots", help="Output directory for plots")

    args = parser.parse_args()
    generate_all_plots(args.results, args.output)


if __name__ == "__main__":
    main()
