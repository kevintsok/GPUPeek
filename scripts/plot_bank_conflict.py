#!/usr/bin/env python3
"""
Bank Conflict Analysis Chart Generator
=====================================
Generates charts for bank conflict research benchmark results.

Usage:
    python plot_bank_conflict.py

Output:
    NVIDIA_GPU/sm_120/bank_conflict/data/*.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Stride vs Bandwidth Data (RTX 5080)
# Format: (stride, bandwidth_gbps, relative_perf_pct)
stride_data = [
    (1, 729.41, 100.0),
    (2, 736.80, 101.0),
    (4, 489.65, 67.1),
    (8, 285.56, 39.1),
    (16, 285.94, 39.2),
    (32, 266.00, 36.5),
    (64, 464.22, 63.6),
    (128, 669.90, 91.8),
]

# Padding Effect Data
# Format: (padding_words, storage_factor, bandwidth_gbps, relative_perf_pct)
padding_data = [
    (0, 1.0, 332.14, 100.0),
    (1, 2.0, 351.45, 105.8),
    (2, 3.0, 339.98, 102.4),
]

# Broadcast vs Strided Access
# Format: (pattern, bandwidth_gbps)
broadcast_data = [
    ("Broadcast\n(same addr)", 1101.75),
    ("Strided\n(stride=32)", 287.80),
]

# Matrix Transpose with/without Padding
# Format: (method, bandwidth_gbps)
transpose_data = [
    ("With Padding\n(33 cols)", 251.3),
    ("No Padding\n(32 cols)", 186.1),
]


def plot_stride_bandwidth():
    """Plot stride vs bandwidth and relative performance."""
    strides = [d[0] for d in stride_data]
    bw = [d[1] for d in stride_data]
    rel = [d[2] for d in stride_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth chart
    bars1 = ax1.bar(range(len(strides)), bw, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(strides)))
    ax1.set_xticklabels(strides)
    ax1.set_xlabel('Stride (elements)', fontsize=12)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Bank Conflict: Stride vs Bandwidth', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars1, bw):
        ax1.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Highlight stride=32 (worst case)
    idx_32 = strides.index(32)
    bars1[idx_32].set_color('red')
    bars1[idx_32].set_alpha(0.8)

    # Relative performance chart
    bars2 = ax2.bar(range(len(strides)), rel, color='coral', alpha=0.8)
    ax2.set_xticks(range(len(strides)))
    ax2.set_xticklabels(strides)
    ax2.set_xlabel('Stride (elements)', fontsize=12)
    ax2.set_ylabel('Relative Performance (%)', fontsize=12)
    ax2.set_title('Bank Conflict: Relative Performance', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Baseline (stride=1)')

    # Add values on bars
    for bar, val in zip(bars2, rel):
        ax2.annotate(f'{val:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Highlight stride=32
    bars2[idx_32].set_color('darkred')
    bars2[idx_32].set_alpha(0.8)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'bank_conflict', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'stride_bandwidth.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_padding_effect():
    """Plot padding effectiveness for bank conflict mitigation."""
    paddings = [d[0] for d in padding_data]
    storage = [d[1] for d in padding_data]
    bw = [d[2] for d in padding_data]
    rel = [d[3] for d in padding_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Storage vs Bandwidth
    ax1.bar(range(len(paddings)), bw, color='teal', alpha=0.8)
    ax1.set_xticks(range(len(paddings)))
    ax1.set_xticklabels([f'+{p}' for p in paddings])
    ax1.set_xlabel('Padding (words)', fontsize=12)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Padding Effect on Bandwidth', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (b, s) in enumerate(zip(bw, storage)):
        ax1.annotate(f'{b:.0f}\n({s:.1f}x)',
                    xy=(i, b), xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Relative performance
    bars = ax2.bar(range(len(paddings)), rel, color='forestgreen', alpha=0.8)
    ax2.set_xticks(range(len(paddings)))
    ax2.set_xticklabels([f'+{p}' for p in paddings])
    ax2.set_xlabel('Padding (words)', fontsize=12)
    ax2.set_ylabel('Relative Performance (%)', fontsize=12)
    ax2.set_title('Padding: Relative Performance', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, rel):
        ax2.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'bank_conflict', 'data')
    output_path = os.path.join(output_dir, 'padding_effect.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_broadcast_comparison():
    """Plot broadcast vs strided access comparison."""
    patterns = [d[0] for d in broadcast_data]
    bw = [d[1] for d in broadcast_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['green', 'red']
    bars = ax.bar(patterns, bw, color=colors, alpha=0.8)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Broadcast vs Strided Access (Stride=32)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, bw):
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add speedup annotation
    speedup = bw[0] / bw[1]
    ax.annotate(f'{speedup:.1f}x faster',
                xy=(0.5, max(bw) * 0.7),
                fontsize=14, ha='center', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'bank_conflict', 'data')
    output_path = os.path.join(output_dir, 'broadcast_vs_strided.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_transpose_padding():
    """Plot matrix transpose with/without padding."""
    methods = [d[0] for d in transpose_data]
    bw = [d[1] for d in transpose_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['forestgreen', 'gray']
    bars = ax.bar(methods, bw, color=colors, alpha=0.8)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Matrix Transpose: Padding Benefit', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, bw):
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement annotation
    improvement = (bw[0] / bw[1] - 1) * 100
    ax.annotate(f'+{improvement:.1f}% improvement',
                xy=(0.5, max(bw) * 0.85),
                fontsize=14, ha='center', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'bank_conflict', 'data')
    output_path = os.path.join(output_dir, 'transpose_padding.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("Generating Bank Conflict Charts...")
    print("=" * 50)
    plot_stride_bandwidth()
    print()
    plot_padding_effect()
    print()
    plot_broadcast_comparison()
    print()
    plot_transpose_padding()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/bank_conflict/data/")
