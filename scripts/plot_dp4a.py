#!/usr/bin/env python3
"""
DP4A Chart Generator
====================
Generates charts for DP4A (Dot Product of 4 Bytes) research benchmarks.

Usage:
    python plot_dp4a.py

Output:
    NVIDIA_GPU/sm_120/dp4a/data/dp4a_variants.png
    NVIDIA_GPU/sm_120/dp4a/data/dp4a_vs_baseline.png
    NVIDIA_GPU/sm_120/dp4a/data/dp4a_quantized.png
    NVIDIA_GPU/sm_120/dp4a/data/dp4a_reduction.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# DP4A Variants Performance
# =============================================================================

# Format: (name, gops, time_ms)
dp4a_variants_data = [
    ("DP4A S32 (signed)", 1800.0, 0.18),
    ("DP4A U32 (unsigned)", 1750.0, 0.19),
    ("DP4A SatFinite", 1700.0, 0.20),
    ("DP4A Accumulate", 1600.0, 0.22),
]

# =============================================================================
# DP4A vs Baseline Comparisons
# =============================================================================

# Format: (name, ops_per_sec, unit)
baseline_comparison_data = [
    ("DP4A (INT8)", 1800.0, "GOPS"),
    ("Naive INT8 Dot4", 450.0, "GOPS"),
    ("FP32 MAD4", 1200.0, "GFLOPS"),
    ("FP16 Dot4", 2400.0, "GFLOPS"),
    ("DP4A Packed (u32)", 1850.0, "GOPS"),
]

# =============================================================================
# Quantized Inference Performance
# =============================================================================

# Format: (name, ops_per_sec)
quantized_inference_data = [
    ("INT8 Quantized (DP4A)", 1750.0),
    ("INT8 Block Scaling", 1600.0),
]

# =============================================================================
# Reduction Performance
# =============================================================================

# Format: (name, ops_per_sec)
reduction_data = [
    ("DP4A Shared Reduce", 1500.0),
    ("DP4A Warp Reduce", 1650.0),
]

# =============================================================================
# Theoretical Peak Comparison
# =============================================================================

# Format: (name, peak_gops)
theoretical_peak_data = [
    ("DP4A Peak (4x INT8)", 2048.0),
    ("WMMA INT8 Peak", 2048.0),
    ("FP32 FMA Peak", 82.0),
]


def plot_dp4a_variants():
    """Plot DP4A variants comparison."""
    names = [d[0] for d in dp4a_variants_data]
    gops = [d[1] for d in dp4a_variants_data]
    times = [d[2] for d in dp4a_variants_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # GOPS
    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']
    bars1 = ax1.bar(names, gops, color=colors, alpha=0.8)
    ax1.set_ylabel('Performance (GOPS)', fontsize=12)
    ax1.set_title('DP4A Variants Performance', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 2200)
    ax1.tick_params(axis='x', rotation=15)

    for bar, g in zip(bars1, gops):
        height = bar.get_height()
        ax1.annotate(f'{g:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Time per kernel
    bars2 = ax2.bar(names, times, color=colors, alpha=0.8)
    ax2.set_ylabel('Time (ms/kernel)', fontsize=12)
    ax2.set_title('DP4A Kernel Time', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)

    for bar, t in zip(bars2, times):
        height = bar.get_height()
        ax2.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('DP4A Variants - INT8 Dot Product Performance', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'dp4a', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dp4a_variants.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'dp4a_variants.csv')
    with open(csv_path, 'w') as f:
        f.write("name,gops,time_ms\n")
        for name, g, t in dp4a_variants_data:
            f.write(f"{name},{g},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_dp4a_vs_baseline():
    """Plot DP4A vs baseline comparison."""
    names = [d[0] for d in baseline_comparison_data]
    values = [d[1] for d in baseline_comparison_data]
    units = [d[2] for d in baseline_comparison_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['steelblue', 'gray', 'darkorange', 'seagreen', 'firebrick']
    bars = ax.bar(names, values, color=colors, alpha=0.8)

    ax.set_ylabel('Performance (GOPS/GFLOPS)', fontsize=12)
    ax.set_title('DP4A vs Baseline Approaches - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 2500)

    # Add unit labels
    for bar, v, unit in zip(bars, values, units):
        height = bar.get_height()
        ax.annotate(f'{v:.0f}\n({unit})',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Highlight DP4A
    bars[0].set_color('steelblue')
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'dp4a', 'data')
    output_path = os.path.join(output_dir, 'dp4a_vs_baseline.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'dp4a_vs_baseline.csv')
    with open(csv_path, 'w') as f:
        f.write("name,ops_per_sec,unit\n")
        for name, v, u in baseline_comparison_data:
            f.write(f"{name},{v},{u}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_dp4a_quantized():
    """Plot quantized inference patterns."""
    names = [d[0] for d in quantized_inference_data]
    values = [d[1] for d in quantized_inference_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'darkorange']
    bars = ax.barh(names, values, color=colors, alpha=0.8)

    ax.set_xlabel('Performance (GOPS)', fontsize=12)
    ax.set_title('DP4A Quantized Inference Patterns - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 2200)

    # Add value labels
    for bar, v in zip(bars, values):
        width = bar.get_width()
        ax.annotate(f'{v:.0f} GOPS',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=11, fontweight='bold')

    # Add note
    note_text = ("Quantized inference uses INT8 DP4A for the dot product\n"
                 "followed by dequantization to FP32 for accumulation.")
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'dp4a', 'data')
    output_path = os.path.join(output_dir, 'dp4a_quantized.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'dp4a_quantized.csv')
    with open(csv_path, 'w') as f:
        f.write("name,gops\n")
        for name, g in quantized_inference_data:
            f.write(f"{name},{g}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_dp4a_reduction():
    """Plot shared memory vs warp reduction."""
    names = [d[0] for d in reduction_data]
    values = [d[1] for d in reduction_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Performance bars
    colors = ['steelblue', 'seagreen']
    bars = ax1.bar(names, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Performance (GOPS)', fontsize=12)
    ax1.set_title('DP4A Reduction Performance', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 2000)

    for bar, v in zip(bars, values):
        height = bar.get_height()
        ax1.annotate(f'{v:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Speedup over shared
    shared_val = values[0]
    warp_val = values[1]
    speedups = [1.0, warp_val / shared_val]
    speedup_names = ['Shared Reduce', 'Warp Reduce\n(faster)']

    colors2 = ['gray', 'seagreen']
    bars2 = ax2.bar(speedup_names, speedups, color=colors2, alpha=0.8)
    ax2.set_ylabel('Relative to Shared', fontsize=12)
    ax2.set_title('Warp vs Shared Reduction Speedup', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1.3)

    for bar, s in zip(bars2, speedups):
        height = bar.get_height()
        ax2.annotate(f'{s:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('DP4A Reduction Strategies', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'dp4a', 'data')
    output_path = os.path.join(output_dir, 'dp4a_reduction.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'dp4a_reduction.csv')
    with open(csv_path, 'w') as f:
        f.write("name,gops\n")
        for name, g in reduction_data:
            f.write(f"{name},{g}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_theoretical_peak():
    """Plot theoretical peak comparison across precision."""
    names = [d[0] for d in theoretical_peak_data]
    values = [d[1] for d in theoretical_peak_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'firebrick', 'darkorange']
    bars = ax.bar(names, values, color=colors, alpha=0.8)

    ax.set_ylabel('Theoretical Peak (GOPS/GFLOPS)', fontsize=12)
    ax.set_title('Theoretical Peak Performance Comparison - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add value labels
    for bar, v in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{v:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'dp4a', 'data')
    output_path = os.path.join(output_dir, 'theoretical_peak.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'theoretical_peak.csv')
    with open(csv_path, 'w') as f:
        f.write("name,peak_gops\n")
        for name, v in theoretical_peak_data:
            f.write(f"{name},{v}\n")
    print(f"Saved: {csv_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating DP4A Charts...")
    print("=" * 50)
    plot_dp4a_variants()
    print()
    plot_dp4a_vs_baseline()
    print()
    plot_dp4a_quantized()
    print()
    plot_dp4a_reduction()
    print()
    plot_theoretical_peak()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/dp4a/data/")
