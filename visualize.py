"""
Visualize Benchmark Results
============================
Run this AFTER both teammates have pushed their CSVs.
It merges cuda_cpu_results.csv + mps_cpu_results.csv and plots everything.

Usage:
    python visualize.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"

COLORS = {
    "cpu":  "#6c757d",
    "cuda": "#00b4d8",
    "mps":  "#f77f00",
}

def load_all_results():
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{RESULTS_DIR}/'")
    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    # Drop duplicate rows (cpu appears in both CSVs — keep first)
    df  = df.drop_duplicates(subset=["model", "backend", "precision", "batch_size"])
    print(f"  Loaded {len(df)} rows from: {[os.path.basename(f) for f in files]}")
    return df


def plot_latency_vs_batch(df, ax, model, precision):
    subset = df[(df.model == model) & (df.precision == precision)]
    for backend, grp in subset.groupby("backend"):
        grp = grp.sort_values("batch_size")
        ax.plot(grp.batch_size, grp.latency_ms,
                marker="o", label=backend.upper(),
                color=COLORS.get(backend, "black"), linewidth=2)
    ax.set_title(f"{model} — {precision.upper()} Latency", fontsize=10, fontweight="bold")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(df.batch_size.unique()))


def plot_throughput_vs_batch(df, ax, model, precision):
    subset = df[(df.model == model) & (df.precision == precision)]
    for backend, grp in subset.groupby("backend"):
        grp = grp.sort_values("batch_size")
        ax.plot(grp.batch_size, grp.throughput_img_s,
                marker="s", label=backend.upper(),
                color=COLORS.get(backend, "black"), linewidth=2)
    ax.set_title(f"{model} — {precision.upper()} Throughput", fontsize=10, fontweight="bold")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (img/s)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(df.batch_size.unique()))


def plot_memory_bar(df, ax, model):
    subset = df[(df.model == model) & (df.precision == "fp32") & (df.batch_size == 16)]
    backends = subset.backend.tolist()
    memory   = subset.memory_mb.tolist()
    bars = ax.bar(backends, memory,
                  color=[COLORS.get(b, "gray") for b in backends],
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, memory):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, f"{val:.0f} MB",
                ha="center", fontsize=8)
    ax.set_title(f"{model} — Peak Memory (fp32, bs=16)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Memory (MB)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def plot_fp32_vs_fp16(df, ax, model, backend):
    subset = df[(df.model == model) & (df.backend == backend)]
    if subset.empty:
        ax.set_visible(False)
        return
    for precision, grp in subset.groupby("precision"):
        grp = grp.sort_values("batch_size")
        style = "-" if precision == "fp32" else "--"
        ax.plot(grp.batch_size, grp.throughput_img_s,
                linestyle=style, marker="o",
                label=precision.upper(), linewidth=2)
    ax.set_title(f"{model} — {backend.upper()} FP32 vs FP16", fontsize=10, fontweight="bold")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (img/s)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(df.batch_size.unique()))


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df     = load_all_results()
    models = df.model.unique().tolist()

    # ── Figure 1: Latency & Throughput vs Batch Size ─────────────
    fig, axes = plt.subplots(len(models), 4, figsize=(20, 5 * len(models)))
    fig.suptitle("Cross-Platform Deep Learning Benchmark\nLatency & Throughput vs Batch Size",
                 fontsize=14, fontweight="bold", y=1.01)

    for row_idx, model in enumerate(models):
        plot_latency_vs_batch(df,   axes[row_idx][0], model, "fp32")
        plot_throughput_vs_batch(df, axes[row_idx][1], model, "fp32")
        plot_latency_vs_batch(df,   axes[row_idx][2], model, "fp16")
        plot_throughput_vs_batch(df, axes[row_idx][3], model, "fp16")

    plt.tight_layout()
    path1 = os.path.join(PLOTS_DIR, "latency_throughput.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path1}")
    plt.close()

    # ── Figure 2: Memory Usage ────────────────────────────────────
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("Peak GPU Memory Usage (FP32, Batch=16)",
                 fontsize=14, fontweight="bold")
    for ax, model in zip(axes, models):
        plot_memory_bar(df, ax, model)
    plt.tight_layout()
    path2 = os.path.join(PLOTS_DIR, "memory_usage.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path2}")
    plt.close()

    # ── Figure 3: FP32 vs FP16 Comparison ────────────────────────
    backends   = [b for b in ["cuda", "mps"] if b in df.backend.values]
    fig, axes  = plt.subplots(len(models), len(backends),
                              figsize=(8 * len(backends), 5 * len(models)))
    if len(models) == 1 and len(backends) == 1:
        axes = [[axes]]
    elif len(models) == 1:
        axes = [axes]
    elif len(backends) == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle("FP32 vs FP16 Throughput Comparison",
                 fontsize=14, fontweight="bold")
    for r, model in enumerate(models):
        for c, backend in enumerate(backends):
            plot_fp32_vs_fp16(df, axes[r][c], model, backend)
    plt.tight_layout()
    path3 = os.path.join(PLOTS_DIR, "fp32_vs_fp16.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path3}")
    plt.close()

    print("\n  All plots saved to plots/")


if __name__ == "__main__":
    main()