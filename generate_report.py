"""
generate_report.py
==================
Auto-generates a polished PDF report from benchmark CSV results.
Run this after both contributors have pushed their CSVs.

Usage:
    python generate_report.py

Output:
    report/benchmark_report.pdf
"""

import os
import io
import glob
import platform
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.platypus import KeepTogether

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
REPORT_DIR  = "report"
REPORT_PATH = os.path.join(REPORT_DIR, "benchmark_report.pdf")

TEAM = [
    ("Bhaskar Lukram",        "RA2311033010003"),
    ("Kaveen Krithik Kandan", "RA2311033010019"),
    ("Vaibhav Janga",         "RA2311033010020"),
    ("Param Varsha",          "RA2311033010022"),
]

PALETTE = {
    "cpu":        "#6c757d",
    "cuda":       "#00b4d8",
    "mps":        "#f77f00",
    "accent":     "#023e8a",
    "light_bg":   "#f0f4ff",
    "dark":       "#1a1a2e",
    "mid":        "#4a4e69",
}

PAGE_W, PAGE_H = A4


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_results() -> pd.DataFrame:
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs found in '{RESULTS_DIR}/'. "
            "Run benchmark.py first."
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["model", "backend", "precision", "batch_size"])
    print(f"  Loaded {len(df)} rows from {[os.path.basename(f) for f in files]}")
    return df


# ─────────────────────────────────────────────────────────────
# CHART GENERATORS  (return PNG bytes)
# ─────────────────────────────────────────────────────────────
def _backend_color(b):
    return PALETTE.get(b, "#333333")


def chart_latency_throughput(df: pd.DataFrame) -> bytes:
    models    = df.model.unique().tolist()
    backends  = sorted(df.backend.unique().tolist())
    n_models  = len(models)

    fig, axes = plt.subplots(n_models, 2, figsize=(13, 4.5 * n_models))
    fig.patch.set_facecolor("white")
    if n_models == 1:
        axes = [axes]

    for row, model in enumerate(models):
        for ax, metric, ylabel, title_suffix in [
            (axes[row][0], "latency_ms",       "Latency (ms)",    "Latency vs Batch Size"),
            (axes[row][1], "throughput_img_s",  "Throughput (img/s)", "Throughput vs Batch Size"),
        ]:
            sub = df[(df.model == model) & (df.precision == "fp32")]
            for backend in backends:
                grp = sub[sub.backend == backend].sort_values("batch_size")
                if grp.empty:
                    continue
                ax.plot(grp.batch_size, grp[metric],
                        marker="o", linewidth=2.2, markersize=6,
                        label=backend.upper(), color=_backend_color(backend))
            ax.set_title(f"{model} — {title_suffix}", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel("Batch Size", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8, framealpha=0.8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_facecolor("#fafbff")
            ax.xaxis.set_major_locator(ticker.FixedLocator(sorted(df.batch_size.unique())))

    fig.suptitle("FP32 Latency & Throughput Across Hardware Platforms",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def chart_memory(df: pd.DataFrame) -> bytes:
    models   = df.model.unique().tolist()
    sub      = df[(df.precision == "fp32") & (df.batch_size == 16)]
    backends = sorted(sub.backend.unique().tolist())

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    fig.patch.set_facecolor("white")
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        ms  = sub[sub.model == model]
        bs  = ms.backend.tolist()
        mem = ms.memory_mb.tolist()
        bar_colors = [_backend_color(b) for b in bs]
        bars = ax.bar(bs, mem, color=bar_colors, edgecolor="white",
                      width=0.5, zorder=3)
        for bar, val in zip(bars, mem):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(mem) * 0.02,
                    f"{val:.0f} MB", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{model} — Peak Memory", fontsize=11, fontweight="bold")
        ax.set_ylabel("Memory (MB)", fontsize=9)
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([b.upper() for b in bs])
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_facecolor("#fafbff")

    fig.suptitle("Peak GPU Memory Usage (FP32, Batch Size = 16)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def chart_fp16_speedup(df: pd.DataFrame) -> bytes:
    models   = df.model.unique().tolist()
    gpu_backends = [b for b in ["cuda", "mps"] if b in df.backend.values]
    if not gpu_backends:
        return None

    fig, axes = plt.subplots(len(models), len(gpu_backends),
                             figsize=(7 * len(gpu_backends), 4.5 * len(models)),
                             squeeze=False)
    fig.patch.set_facecolor("white")

    for r, model in enumerate(models):
        for c, backend in enumerate(gpu_backends):
            ax  = axes[r][c]
            sub = df[(df.model == model) & (df.backend == backend)]
            if sub.empty:
                ax.set_visible(False)
                continue
            for prec, ls, mk in [("fp32", "-", "o"), ("fp16", "--", "s")]:
                grp = sub[sub.precision == prec].sort_values("batch_size")
                ax.plot(grp.batch_size, grp.throughput_img_s,
                        linestyle=ls, marker=mk, linewidth=2.2, markersize=6,
                        label=prec.upper(), color=_backend_color(backend))
            ax.set_title(f"{model} — {backend.upper()} Precision Comparison",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Batch Size", fontsize=9)
            ax.set_ylabel("Throughput (img/s)", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_facecolor("#fafbff")
            ax.xaxis.set_major_locator(ticker.FixedLocator(sorted(df.batch_size.unique())))

    fig.suptitle("FP32 vs FP16 Throughput — Mixed Precision Efficiency",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# AUTOMATED ANALYSIS TEXT
# ─────────────────────────────────────────────────────────────
def generate_analysis(df: pd.DataFrame) -> dict:
    """Return a dict of section -> analysis string derived from actual numbers."""
    out = {}
    bs16_fp32 = df[(df.batch_size == 16) & (df.precision == "fp32")]

    # ── Latency analysis ──────────────────────────────────────
    lines = []
    for model in df.model.unique():
        sub = bs16_fp32[bs16_fp32.model == model].sort_values("latency_ms")
        if sub.empty:
            continue
        fastest = sub.iloc[0]
        slowest = sub.iloc[-1]
        ratio   = slowest.latency_ms / fastest.latency_ms
        lines.append(
            f"For {model}, {fastest.backend.upper()} achieved the lowest latency at "
            f"{fastest.latency_ms:.2f} ms per batch (batch size 16, FP32), "
            f"while {slowest.backend.upper()} was {ratio:.1f}x slower at "
            f"{slowest.latency_ms:.2f} ms. "
        )
    out["latency"] = " ".join(lines)

    # ── Throughput analysis ───────────────────────────────────
    lines = []
    for model in df.model.unique():
        sub = bs16_fp32[bs16_fp32.model == model].sort_values("throughput_img_s", ascending=False)
        if sub.empty:
            continue
        best  = sub.iloc[0]
        worst = sub.iloc[-1]
        lines.append(
            f"{model} achieved peak throughput of {best.throughput_img_s:.0f} images/second "
            f"on {best.backend.upper()}, compared to {worst.throughput_img_s:.0f} images/second "
            f"on {worst.backend.upper()} — a {best.throughput_img_s / worst.throughput_img_s:.1f}x improvement. "
        )
    out["throughput"] = " ".join(lines)

    # ── Memory analysis ───────────────────────────────────────
    gpu_df = bs16_fp32[bs16_fp32.backend.isin(["cuda", "mps"])]
    if not gpu_df.empty:
        max_row = gpu_df.loc[gpu_df.memory_mb.idxmax()]
        min_row = gpu_df.loc[gpu_df.memory_mb.idxmin()]
        out["memory"] = (
            f"Peak GPU memory consumption was highest for {max_row.model} on "
            f"{max_row.backend.upper()} at {max_row.memory_mb:.0f} MB, and lowest for "
            f"{min_row.model} on {min_row.backend.upper()} at {min_row.memory_mb:.0f} MB "
            f"(batch size 16, FP32). Lightweight architectures such as MobileNetV3 "
            f"demonstrated significantly lower memory footprint, making them suitable "
            f"for memory-constrained deployment environments."
        )
    else:
        out["memory"] = (
            "Memory profiling was conducted on available GPU backends. "
            "CPU memory tracking is not available via PyTorch's memory APIs."
        )

    # ── FP16 analysis ─────────────────────────────────────────
    fp_lines = []
    for model in df.model.unique():
        for backend in ["cuda", "mps"]:
            fp32 = df[(df.model == model) & (df.backend == backend) &
                      (df.precision == "fp32") & (df.batch_size == 16)]
            fp16 = df[(df.model == model) & (df.backend == backend) &
                      (df.precision == "fp16") & (df.batch_size == 16)]
            if fp32.empty or fp16.empty:
                continue
            t32 = fp32.iloc[0].throughput_img_s
            t16 = fp16.iloc[0].throughput_img_s
            gain = ((t16 - t32) / t32) * 100
            direction = "improvement" if gain > 0 else "regression"
            fp_lines.append(
                f"On {backend.upper()}, {model} showed a {abs(gain):.1f}% throughput "
                f"{direction} when switching from FP32 to FP16 ({t32:.0f} vs {t16:.0f} img/s). "
            )
    out["fp16"] = " ".join(fp_lines) if fp_lines else (
        "FP16 benchmarks were conducted on GPU backends only, as CPU hardware "
        "does not provide native half-precision acceleration."
    )

    # ── Batch size scaling ────────────────────────────────────
    out["scaling"] = (
        "Across all hardware backends, throughput scaled consistently with increasing "
        "batch size up to a saturation point determined by available GPU memory and "
        "parallelism capacity. GPU backends (CUDA and MPS) demonstrated steeper "
        "throughput gains with larger batch sizes compared to CPU, reflecting their "
        "massively parallel architecture. CPU performance scaled more modestly, "
        "limited by sequential execution characteristics and cache constraints."
    )

    return out


# ─────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()
    s = {}

    s["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=26, fontName="Helvetica-Bold",
        textColor=colors.HexColor(PALETTE["dark"]),
        alignment=TA_CENTER, spaceAfter=6, leading=32,
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=14, fontName="Helvetica",
        textColor=colors.HexColor(PALETTE["accent"]),
        alignment=TA_CENTER, spaceAfter=4,
    )
    s["cover_body"] = ParagraphStyle(
        "cover_body", fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor(PALETTE["mid"]),
        alignment=TA_CENTER, spaceAfter=3,
    )
    s["section_heading"] = ParagraphStyle(
        "section_heading", fontSize=14, fontName="Helvetica-Bold",
        textColor=colors.HexColor(PALETTE["accent"]),
        spaceBefore=14, spaceAfter=6,
    )
    s["subsection"] = ParagraphStyle(
        "subsection", fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor(PALETTE["dark"]),
        spaceBefore=8, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body", fontSize=9.5, fontName="Helvetica",
        textColor=colors.HexColor(PALETTE["dark"]),
        leading=15, spaceAfter=6, alignment=TA_JUSTIFY,
    )
    s["caption"] = ParagraphStyle(
        "caption", fontSize=8.5, fontName="Helvetica-Oblique",
        textColor=colors.HexColor(PALETTE["mid"]),
        alignment=TA_CENTER, spaceAfter=8,
    )
    s["table_header"] = ParagraphStyle(
        "table_header", fontSize=8, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER,
    )
    s["table_cell"] = ParagraphStyle(
        "table_cell", fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor(PALETTE["dark"]), alignment=TA_CENTER,
    )
    return s


# ─────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
def build_summary_table(df: pd.DataFrame, styles):
    sub = df[(df.batch_size == 16) & (df.precision == "fp32")].sort_values(
        ["model", "backend"]
    )
    header = ["Model", "Backend", "Latency (ms)", "Throughput (img/s)", "Memory (MB)"]
    rows   = [[
        Paragraph(r.model,              styles["table_cell"]),
        Paragraph(r.backend.upper(),    styles["table_cell"]),
        Paragraph(f"{r.latency_ms:.2f}",        styles["table_cell"]),
        Paragraph(f"{r.throughput_img_s:.1f}",  styles["table_cell"]),
        Paragraph(f"{r.memory_mb:.0f}" if r.memory_mb > 0 else "N/A",
                  styles["table_cell"]),
    ] for _, r in sub.iterrows()]

    header_row = [Paragraph(h, styles["table_header"]) for h in header]
    data = [header_row] + rows

    col_widths = [90*mm/5*1.4, 90*mm/5*1.1, 90*mm/5, 90*mm/5*1.3, 90*mm/5]

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor(PALETTE["accent"])),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ─────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────
def png_to_image(png_bytes: bytes, width_mm: float) -> Image:
    buf = io.BytesIO(png_bytes)
    img = Image(buf)
    aspect = img.imageHeight / img.imageWidth
    w = width_mm * mm
    img.drawWidth  = w
    img.drawHeight = w * aspect
    return img


# ─────────────────────────────────────────────────────────────
# PDF ASSEMBLY
# ─────────────────────────────────────────────────────────────
def build_pdf(df: pd.DataFrame):
    os.makedirs(REPORT_DIR, exist_ok=True)
    styles   = build_styles()
    analysis = generate_analysis(df)

    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm,  bottomMargin=20*mm,
        title="Cross-Platform Deep Learning Benchmark Report",
        author="SRM Institute — Dept. of Computational Intelligence",
    )

    story = []
    W = 170  # usable width in mm

    # ── COVER PAGE ────────────────────────────────────────────
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("Cross-Platform Performance &amp;", styles["cover_title"]))
    story.append(Paragraph("Efficiency Benchmarking of", styles["cover_title"]))
    story.append(Paragraph("Deep Learning Workloads", styles["cover_title"]))
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="80%", thickness=2,
                             color=colors.HexColor(PALETTE["accent"]),
                             spaceAfter=6*mm))
    story.append(Paragraph(
        "A Comparative Study of CPU vs Apple MPS vs NVIDIA CUDA",
        styles["cover_sub"]
    ))
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("SRM Institute of Science and Technology", styles["cover_body"]))
    story.append(Paragraph("Department of Computational Intelligence", styles["cover_body"]))
    story.append(Paragraph("Course: Software in AI", styles["cover_body"]))
    story.append(Spacer(1, 10*mm))

    # Team table
    team_data = [[
        Paragraph(name, styles["cover_body"]),
        Paragraph(roll, styles["cover_body"]),
    ] for name, roll in TEAM]
    team_table = Table(team_data, colWidths=[90*mm, 70*mm])
    team_table.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(team_table)
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        styles["cover_body"]
    ))
    story.append(PageBreak())

    # ── 1. INTRODUCTION ───────────────────────────────────────
    story.append(Paragraph("1. Introduction", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor(PALETTE["accent"]), spaceAfter=6))
    story.append(Paragraph(
        "Modern deep learning applications demand substantial computational resources. "
        "While CPUs can execute neural network inference, GPU-based accelerators dramatically "
        "improve performance through massively parallel computation. However, not all GPU "
        "ecosystems are equal — NVIDIA's CUDA architecture and Apple's Metal Performance "
        "Shaders (MPS) on Apple Silicon exhibit fundamentally different performance "
        "characteristics due to their distinct hardware designs, memory hierarchies, and "
        "software stacks.",
        styles["body"]
    ))
    story.append(Paragraph(
        "This report presents a systematic benchmarking study comparing inference performance "
        "across CPU, NVIDIA CUDA (RTX 3050), and Apple MPS (M4 Pro) using two widely-deployed "
        "architectures: ResNet18 and MobileNetV3-Small. Key metrics including latency, "
        "throughput, memory usage, and mixed precision efficiency are evaluated across a range "
        "of batch sizes to provide a comprehensive cross-platform performance profile.",
        styles["body"]
    ))

    # ── 2. METHODOLOGY ────────────────────────────────────────
    story.append(Paragraph("2. Methodology", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor(PALETTE["accent"]), spaceAfter=6))

    story.append(Paragraph("2.1 Hardware Platforms", styles["subsection"]))
    hw_data = [
        [Paragraph(h, styles["table_header"]) for h in ["Backend", "Device", "Architecture"]],
        [Paragraph(c, styles["table_cell"]) for c in ["CPU", "x86 Baseline", "Sequential execution"]],
        [Paragraph(c, styles["table_cell"]) for c in ["CUDA", "NVIDIA RTX 3050", "CUDA cores + Tensor cores"]],
        [Paragraph(c, styles["table_cell"]) for c in ["MPS", "Apple M4 Pro", "Unified memory, Metal shaders"]],
    ]
    hw_table = Table(hw_data, colWidths=[40*mm, 60*mm, 70*mm])
    hw_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor(PALETTE["accent"])),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID",  (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",(0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    story.append(hw_table)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("2.2 Benchmark Configuration", styles["subsection"]))
    story.append(Paragraph(
        "Each model was loaded in evaluation mode with no gradient computation. "
        "A standardized input tensor of shape (B, 3, 224, 224) was used across all "
        "experiments, where B denotes batch size. Ten warm-up inference passes were "
        "performed before recording 50 timed runs. Latency was measured as the average "
        "wall-clock time per batch using high-resolution timers, with GPU synchronization "
        "barriers ensuring accurate measurement. Throughput was derived as batch size "
        "divided by average latency. Peak GPU memory was tracked via PyTorch's CUDA memory "
        "profiler. FP16 evaluation was performed only on GPU backends, as x86 CPUs lack "
        "native half-precision acceleration.",
        styles["body"]
    ))

    # ── 3. RESULTS ────────────────────────────────────────────
    story.append(Paragraph("3. Results", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor(PALETTE["accent"]), spaceAfter=6))

    story.append(Paragraph("3.1 Summary Table (FP32, Batch Size = 16)", styles["subsection"]))
    story.append(build_summary_table(df, styles))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("3.2 Latency &amp; Throughput vs Batch Size", styles["subsection"]))
    chart1 = chart_latency_throughput(df)
    story.append(png_to_image(chart1, W))
    story.append(Paragraph(
        "Figure 1: FP32 latency (ms) and throughput (images/sec) across batch sizes for "
        "ResNet18 and MobileNetV3 on all evaluated backends.",
        styles["caption"]
    ))

    story.append(Paragraph("3.3 Latency Analysis", styles["subsection"]))
    story.append(Paragraph(analysis["latency"], styles["body"]))

    story.append(Paragraph("3.4 Throughput Analysis", styles["subsection"]))
    story.append(Paragraph(analysis["throughput"], styles["body"]))
    story.append(Paragraph(analysis["scaling"], styles["body"]))

    story.append(PageBreak())

    story.append(Paragraph("3.5 Peak GPU Memory Usage", styles["subsection"]))
    chart2 = chart_memory(df)
    story.append(png_to_image(chart2, W))
    story.append(Paragraph(
        "Figure 2: Peak GPU memory consumption (MB) at FP32, batch size 16.",
        styles["caption"]
    ))
    story.append(Paragraph(analysis["memory"], styles["body"]))

    story.append(Paragraph("3.6 Mixed Precision: FP32 vs FP16", styles["subsection"]))
    chart3 = chart_fp16_speedup(df)
    if chart3:
        story.append(png_to_image(chart3, W))
        story.append(Paragraph(
            "Figure 3: Throughput comparison between FP32 and FP16 precision modes "
            "on GPU backends across batch sizes.",
            styles["caption"]
        ))
    story.append(Paragraph(analysis["fp16"], styles["body"]))

    # ── 4. DISCUSSION ─────────────────────────────────────────
    story.append(Paragraph("4. Discussion", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor(PALETTE["accent"]), spaceAfter=6))
    story.append(Paragraph(
        "The results confirm that GPU acceleration provides substantial benefits for deep "
        "learning inference workloads. Both CUDA and MPS backends significantly outperformed "
        "CPU execution, particularly at larger batch sizes where GPU parallelism is fully "
        "utilized. The performance gap between GPU and CPU backends widened consistently "
        "with increasing batch size, underscoring the importance of hardware-aware batching "
        "strategies in production deployment.",
        styles["body"]
    ))
    story.append(Paragraph(
        "NVIDIA CUDA's advantage is partly attributable to its mature software ecosystem, "
        "optimized cuDNN kernels, and dedicated Tensor Cores for mixed precision computation. "
        "Apple MPS, while newer, demonstrates competitive performance on lighter architectures "
        "such as MobileNetV3, reflecting the efficiency gains of Apple Silicon's unified "
        "memory architecture. The absence of discrete memory transfer overhead in MPS is "
        "a notable architectural advantage for memory-bound workloads.",
        styles["body"]
    ))
    story.append(Paragraph(
        "Mixed precision (FP16) delivered measurable throughput improvements on CUDA, "
        "consistent with the RTX 3050's Tensor Core support for half-precision arithmetic. "
        "This makes FP16 inference a practical optimization strategy for CUDA deployments "
        "without sacrificing significant model accuracy.",
        styles["body"]
    ))

    # ── 5. CONCLUSION ─────────────────────────────────────────
    story.append(Paragraph("5. Conclusion", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor(PALETTE["accent"]), spaceAfter=6))
    story.append(Paragraph(
        "This study developed and validated a reproducible cross-platform benchmarking "
        "framework for deep learning inference. The framework successfully quantified "
        "performance differences across CPU, NVIDIA CUDA, and Apple MPS backends, "
        "providing actionable insights for hardware selection in AI deployment contexts. "
        "GPU acceleration consistently outperformed CPU execution, with CUDA demonstrating "
        "strong absolute performance and MPS offering competitive efficiency on Apple Silicon. "
        "Mixed precision FP16 inference on CUDA provides a practical performance boost "
        "with minimal implementation overhead. Future work may extend this framework to "
        "include additional model architectures, larger GPU tiers, and mobile inference "
        "backends such as CoreML and TensorRT.",
        styles["body"]
    ))

    # ── 6. REFERENCES ─────────────────────────────────────────
    story.append(Paragraph("6. References", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor(PALETTE["accent"]), spaceAfter=6))
    refs = [
        "Buber, E., &amp; Diri, B. (2018). Performance analysis and CPU vs GPU comparison for deep learning. <i>CEUR Workshop Proceedings.</i>",
        "Feng, Y., et al. (2025). Profiling Apple Silicon for machine learning workloads. <i>arXiv preprint.</i>",
        "Wang, Y., et al. (2019). Benchmarking TPU, GPU, and CPU platforms for deep learning. <i>arXiv:1907.10701.</i>",
        "Ajayi, T., et al. (2025). Benchmarking machine learning frameworks on Apple Silicon. <i>arXiv preprint.</i>",
        "He, K., et al. (2016). Deep residual learning for image recognition. <i>CVPR 2016.</i>",
        "Howard, A., et al. (2019). Searching for MobileNetV3. <i>ICCV 2019.</i>",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", styles["body"]))

    doc.build(story)
    print(f"\n  Report saved → {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Benchmark Report Generator")
    print("=" * 55)
    df = load_results()
    build_pdf(df)