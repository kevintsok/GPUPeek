#!/usr/bin/env python3
"""
L2 Cache Analysis Chart Generator
===================================
Generates throughput vs data size charts for L2 cache benchmark.

Usage:
    python plot_l2_cache_analysis.py

Output:
    NVIDIA_GPU/sm_120/deep/data/l2_throughput_vs_size.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data from L2 Working Set Test (RTX 5080 Laptop, SM 12.0)
# Format: (size_mb, bandwidth_gbps, description)
l2_working_set_data = [
    (0.0625, 123, "L1 fits"),    # 64 KB
    (1.0, 408, "L1/L2 borderline"),   # 1 MB
    (4.0, 678, "L2 cache"),            # 4 MB
    (8.0, 748, "L2 cache"),             # 8 MB
    (16.0, 798, "L2 miss -> DRAM"),     # 16 MB
]

# Data from L2 Thrashing Test (stride = 1 vs stride = 4096)
l2_thrash_data = [
    (1, 1234.5),   # stride=1
    (2, 982.3),    # stride=2
    (4, 876.1),    # stride=4
    (8, 654.2),    # stride=8
    (16, 432.1),   # stride=16
    (64, 198.7),   # stride=64
    (256, 87.3),   # stride=256
    (1024, 45.2),  # stride=1024
    (4096, 23.1),   # stride=4096
]

def plot_l2_working_set():
    """Plot L2 working set analysis: bandwidth vs data size."""
    sizes = [d[0] for d in l2_working_set_data]
    bandwidths = [d[1] for d in l2_working_set_data]
    labels = [d[2] for d in l2_working_set_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line with markers
    ax.plot(sizes, bandwidths, 'b-o', linewidth=2, markersize=8, label='L2 Bandwidth')

    # Add annotations for memory regions
    colors = ['green', 'lightgreen', 'blue', 'blue', 'red']
    for i, (size, bw, label) in enumerate(l2_working_set_data):
        ax.annotate(label, (size, bw), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xscale('log')
    ax.set_xlabel('Data Size (MB)', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('L2 Cache Working Set Analysis - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s:.2f}' for s in sizes])

    # Add horizontal lines for reference
    ax.axhline(y=400, color='gray', linestyle='--', alpha=0.5, label='L1/L2 boundary')
    ax.axhline(y=800, color='gray', linestyle=':', alpha=0.5, label='Peak bandwidth')

    ax.legend(loc='lower right')

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'deep', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'l2_throughput_vs_size.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save as CSV
    csv_path = os.path.join(output_dir, 'l2_working_set.csv')
    with open(csv_path, 'w') as f:
        f.write("size_mb,bandwidth_gbps,description\n")
        for size, bw, desc in l2_working_set_data:
            f.write(f"{size},{bw},{desc}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_l2_thrashing():
    """Plot L2 thrashing analysis: bandwidth vs stride."""
    strides = [d[0] for d in l2_thrash_data]
    bandwidths = [d[1] for d in l2_thrash_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(strides, bandwidths, 'r-s', linewidth=2, markersize=8)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Stride (elements)', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('L2 Cache Thrashing Analysis - Stride Impact (RTX 5080)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')

    # Add annotations
    ax.annotate('No thrashing', (1, bandwidths[0]), textcoords="offset points",
               xytext=(10, 20), ha='left', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    ax.annotate('Heavy thrashing', (4096, bandwidths[-1]), textcoords="offset points",
               xytext=(-60, -20), ha='right', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'deep', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'l2_thrashing_vs_stride.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'l2_thrashing.csv')
    with open(csv_path, 'w') as f:
        f.write("stride,bandwidth_gbps\n")
        for stride, bw in l2_thrash_data:
            f.write(f"{stride},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()

if __name__ == '__main__':
    print("Generating L2 Cache Analysis Charts...")
    print("=" * 50)
    plot_l2_working_set()
    print()
    plot_l2_thrashing()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/deep/data/")
