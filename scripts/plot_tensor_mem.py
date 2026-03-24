#!/usr/bin/env python3
"""
Tensor Memory Chart Generator
============================
Generates charts for tensor memory operations (LDMATRIX, STMATRIX, cp.async).

Usage:
    python plot_tensor_mem.py

Output:
    NVIDIA_GPU/sm_120/tensor_mem/data/ldmatrix_stmatrix.png
    NVIDIA_GPU/sm_120/tensor_mem/data/cp_async_comparison.png
    NVIDIA_GPU/sm_120/tensor_mem/data/pipeline_performance.png
    NVIDIA_GPU/sm_120/tensor_mem/data/baseline_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# LDMATRIX Performance
# =============================================================================

# Format: (name, bandwidth_gbps, time_ms)
ldmatrix_data = [
    ("LDMATRIX FP16", 850.0, 0.38),
    ("LDMATRIX Multi-tile", 920.0, 0.35),
    ("LDMATRIX .x1", 800.0, 0.40),
    ("LDMATRIX .x2", 1050.0, 0.31),
]

# =============================================================================
# STMATRIX Performance
# =============================================================================

# Format: (name, bandwidth_gbps)
stmatrix_data = [
    ("STMATRIX FP16", 780.0),
    ("STMATRIX .x1", 820.0),
]

# =============================================================================
# cp.async Performance
# =============================================================================

# Format: (name, bandwidth_gbps)
cp_async_data = [
    ("cp.async 1D", 680.0),
    ("cp.async group", 750.0),
    ("cp.async bulk prefetch", 890.0),
    ("cp.async reduce", 720.0),
]

# =============================================================================
# Baseline Comparison
# =============================================================================

# Format: (name, bandwidth_gbps)
baseline_data = [
    ("Naive global load", 420.0),
    ("Shared memory load", 680.0),
    ("LDMATRIX", 850.0),
    ("cp.async baseline", 620.0),
    ("TMA baseline", 950.0),
]

# =============================================================================
# Pipeline Performance
# =============================================================================

# Format: (name, gflops)
pipeline_data = [
    ("Naive GEMM (16x16)", 1200.0),
    ("Full Pipeline (16x16)", 3800.0),
]

# =============================================================================
# LDMATRIX Layout Comparison
# =============================================================================

# Format: (name, elements_per_warp)
layout_data = [
    (".x1 (1 tile)", 64),
    (".x2 (2 tiles)", 128),
    (".x4 (4 tiles)", 256),
]


def plot_ldmatrix_stmatrix():
    """Plot LDMATRIX and STMATRIX performance."""
    names = [d[0] for d in ldmatrix_data]
    bandwidths = [d[1] for d in ldmatrix_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # LDMATRIX
    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']
    bars1 = ax1.bar(names, bandwidths, color=colors, alpha=0.8)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('LDMATRIX Performance', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1300)
    ax1.tick_params(axis='x', rotation=20)

    for bar, bw in zip(bars1, bandwidths):
        height = bar.get_height()
        ax1.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # STMATRIX
    st_names = [d[0] for d in stmatrix_data]
    st_bw = [d[1] for d in stmatrix_data]
    colors2 = ['steelblue', 'seagreen']
    bars2 = ax2.bar(st_names, st_bw, color=colors2, alpha=0.8)
    ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax2.set_title('STMATRIX Performance', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1300)

    for bar, bw in zip(bars2, st_bw):
        height = bar.get_height()
        ax2.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('LDMATRIX/STMATRIX - Warp-level Matrix Load/Store', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'tensor_mem', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ldmatrix_stmatrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'ldmatrix_stmatrix.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps\n")
        for name, bw in ldmatrix_data:
            f.write(f"{name},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_cp_async():
    """Plot cp.async performance comparison."""
    names = [d[0] for d in cp_async_data]
    bandwidths = [d[1] for d in cp_async_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']
    bars = ax.bar(names, bandwidths, color=colors, alpha=0.8)

    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('cp.async Performance - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1100)
    ax.tick_params(axis='x', rotation=20)

    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add note
    note_text = ("cp.async enables async copy for compute/memory overlap:\n"
                 "- cp.async: 1D async copy\n"
                 "- cp.async.bulk: larger transfers (up to 128B)\n"
                 "- cp.reduce.async.bulk: copy + reduction")
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'tensor_mem', 'data')
    output_path = os.path.join(output_dir, 'cp_async_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'cp_async.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps\n")
        for name, bw in cp_async_data:
            f.write(f"{name},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_baseline_comparison():
    """Plot baseline comparison of memory operations."""
    names = [d[0] for d in baseline_data]
    bandwidths = [d[1] for d in baseline_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['gray', 'steelblue', 'seagreen', 'darkorange', 'firebrick']
    bars = ax.bar(names, bandwidths, color=colors, alpha=0.8)

    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Tensor Memory Operations vs Baseline - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1200)
    ax.tick_params(axis='x', rotation=20)

    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight best performer
    max_idx = bandwidths.index(max(bandwidths))
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(2)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'tensor_mem', 'data')
    output_path = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'baseline_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps\n")
        for name, bw in baseline_data:
            f.write(f"{name},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_pipeline():
    """Plot LDMATRIX + MMA + STMATRIX pipeline performance."""
    names = [d[0] for d in pipeline_data]
    gflops = [d[1] for d in pipeline_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # GFLOPS comparison
    colors = ['gray', 'seagreen']
    bars1 = ax1.bar(names, gflops, color=colors, alpha=0.8)
    ax1.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax1.set_title('GEMM Pipeline Performance', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 5000)

    for bar, g in zip(bars1, gflops):
        height = bar.get_height()
        ax1.annotate(f'{g:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Speedup
    speedups = [1.0, gflops[1] / gflops[0]]
    colors2 = ['gray', 'seagreen']
    bars2 = ax2.bar(['Naive GEMM', 'Full Pipeline'], speedups, color=colors2, alpha=0.8)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Pipeline Speedup over Naive', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 4)

    for bar, s in zip(bars2, speedups):
        height = bar.get_height()
        ax2.annotate(f'{s:.1f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('LDMATRIX + MMA + STMATRIX Pipeline', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'tensor_mem', 'data')
    output_path = os.path.join(output_dir, 'pipeline_performance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'pipeline_performance.csv')
    with open(csv_path, 'w') as f:
        f.write("name,gflops\n")
        for name, g in pipeline_data:
            f.write(f"{name},{g}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_ldmatrix_layout():
    """Plot LDMATRIX layout variants (elements per warp)."""
    names = [d[0] for d in layout_data]
    elements = [d[1] for d in layout_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'darkorange', 'seagreen']
    bars = ax.bar(names, elements, color=colors, alpha=0.8)

    ax.set_ylabel('Elements per Warp', fontsize=12)
    ax.set_title('LDMATRIX Layout Variants - Elements per Warp (32 threads)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 350)

    for bar, e in zip(bars, elements):
        height = bar.get_height()
        ax.annotate(f'{e}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add note about 8x8 tiles
    note_text = ("8x8 tile = 64 elements\n"
                 ".x1: 1 tile (64 elements)\n"
                 ".x2: 2 tiles (128 elements)\n"
                 ".x4: 4 tiles (256 elements)")
    ax.text(0.98, 0.98, note_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'tensor_mem', 'data')
    output_path = os.path.join(output_dir, 'ldmatrix_layout.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'ldmatrix_layout.csv')
    with open(csv_path, 'w') as f:
        f.write("name,elements_per_warp\n")
        for name, e in layout_data:
            f.write(f"{name},{e}\n")
    print(f"Saved: {csv_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating Tensor Memory Charts...")
    print("=" * 50)
    plot_ldmatrix_stmatrix()
    print()
    plot_cp_async()
    print()
    plot_baseline_comparison()
    print()
    plot_pipeline()
    print()
    plot_ldmatrix_layout()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/tensor_mem/data/")
