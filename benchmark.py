"""
Cross-Platform Deep Learning Benchmarking Framework
====================================================
Benchmarks ResNet18 and MobileNetV3 across available backends.
Each contributor runs this script on their hardware:
  - CUDA machine  → saves results/cuda_cpu_results.csv
  - Apple Silicon → saves results/mps_cpu_results.csv

Requirements:
    pip install torch torchvision tabulate pandas

Usage:
    python benchmark.py
"""

import time
import csv
import gc
import os
import platform

import torch
import torchvision.models as models
import pandas as pd

# ──────────────────────────────────────────────────────────────
# CONFIG — tweak these if needed
# ──────────────────────────────────────────────────────────────
BATCH_SIZES  = [1, 4, 8, 16, 32, 64]
WARMUP_RUNS  = 10
TIMED_RUNS   = 50
INPUT_SIZE   = (3, 224, 224)   # Standard ImageNet spatial size

MODEL_REGISTRY = {
    "ResNet18":    lambda: models.resnet18(weights=None),
    "MobileNetV3": lambda: models.mobilenet_v3_small(weights=None),
}

PRECISIONS = ["fp32", "fp16"]


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def get_backends():
    """Return backends available on this machine."""
    backends = ["cpu"]
    if torch.cuda.is_available():
        backends.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.append("mps")
    return backends


def load_model(name, device, precision):
    model = MODEL_REGISTRY[name]().to(device).eval()
    if precision == "fp16" and device.type in ("cuda", "mps"):
        model = model.half()
    return model


def make_dummy_input(batch_size, device, precision):
    x = torch.randn(batch_size, *INPUT_SIZE, device=device)
    if precision == "fp16" and device.type in ("cuda", "mps"):
        x = x.half()
    return x


def reset_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return 0.0   # CPU/MPS — not tracked via torch


def sync(device):
    """Wait for async GPU ops to finish before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ──────────────────────────────────────────────────────────────
# CORE BENCHMARK
# ──────────────────────────────────────────────────────────────
def run_benchmark(model_name, backend, precision, batch_size):
    """
    Returns a dict with latency_ms, throughput_img_per_sec, memory_mb.
    Returns None if the config is unsupported (e.g. fp16 on CPU).
    """
    # FP16 on CPU is not meaningful — skip
    if precision == "fp16" and backend == "cpu":
        return None

    device = torch.device(backend)

    try:
        model = load_model(model_name, device, precision)
        x     = make_dummy_input(batch_size, device, precision)
    except Exception as e:
        print(f"  [SKIP] {model_name} | {backend} | {precision} | bs={batch_size} — {e}")
        return None

    reset_memory(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(x)
    sync(device)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(TIMED_RUNS):
            sync(device)
            t0 = time.perf_counter()
            _ = model(x)
            sync(device)
            latencies.append(time.perf_counter() - t0)

    mem_mb          = peak_memory_mb(device)
    avg_latency_ms  = (sum(latencies) / len(latencies)) * 1000
    throughput      = batch_size / (avg_latency_ms / 1000)

    # Clean up
    del model, x
    reset_memory(device)

    return {
        "model":            model_name,
        "backend":          backend,
        "precision":        precision,
        "batch_size":       batch_size,
        "latency_ms":       round(avg_latency_ms, 4),
        "throughput_img_s": round(throughput, 2),
        "memory_mb":        round(mem_mb, 2),
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    backends = get_backends()
    os.makedirs("results", exist_ok=True)

    # Name the output file based on what hardware is present
    if "cuda" in backends:
        out_file = "results/cuda_cpu_results.csv"
    elif "mps" in backends:
        out_file = "results/mps_cpu_results.csv"
    else:
        out_file = "results/cpu_only_results.csv"

    print("=" * 65)
    print("  Deep Learning Benchmarking Framework")
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Backends : {backends}")
    print(f"  Output   : {out_file}")
    print("=" * 65)

    fieldnames = ["model", "backend", "precision", "batch_size",
                  "latency_ms", "throughput_img_s", "memory_mb"]

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = len(MODEL_REGISTRY) * len(backends) * len(PRECISIONS) * len(BATCH_SIZES)
        done  = 0

        for model_name in MODEL_REGISTRY:
            for backend in backends:
                for precision in PRECISIONS:
                    for batch_size in BATCH_SIZES:
                        done += 1
                        label = f"[{done}/{total}] {model_name:12s} | {backend:4s} | {precision} | bs={batch_size:2d}"
                        print(f"  Running {label} ...", end="\r")

                        row = run_benchmark(model_name, backend, precision, batch_size)
                        if row:
                            writer.writerow(row)
                            f.flush()
                            print(f"  ✓ {label}  →  {row['latency_ms']:.2f} ms  |  {row['throughput_img_s']:.1f} img/s")
                        else:
                            print(f"  — {label}  [skipped]")

    print()
    print(f"  Results saved → {out_file}")

    # Quick summary table
    df = pd.read_csv(out_file)
    print()
    print("  ── Summary (avg latency ms, fp32, bs=16) ──────────────────")
    summary = df[(df.precision == "fp32") & (df.batch_size == 16)]
    print(summary[["model", "backend", "latency_ms", "throughput_img_s", "memory_mb"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()