#!/usr/bin/env python3
"""
Memory Bandwidth Analysis Chart Generator
==========================================
Generates throughput vs data size charts for memory subsystem benchmark.

Usage:
    python plot_memory_bandwidth.py

Output:
    NVIDIA_GPU/sm_120/memory/data/memory_bandwidth_vs_size.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Memory Bandwidth vs Data Size (RTX 5080, SM 12.0)
# Format: (size_mb, read_bw_gbps, write_bw_gbps)
memory_bandwidth_data = [
    (0.001, 7.25, 7.25),    # 1 KB
    (0.0625, 7.25, 7.25),  # 64 KB - L1 fits
    (0.25, 32.39, 32.39),   # 256 KB - L1 cache
    (1.0, 73.97, 73.97),    # 1 MB - L1/L2 borderline
    (4.0, 296.36, 296.36),  # 4 MB - L2 cache
    (16.0, 643.02, 643.02), # 16 MB - Peak (1st)
    (64.0, 376.08, 376.08), # 64 MB - L2 miss
    (128.0, 502.44, 502.44),# 128 MB - Recovering
    (256.0, 614.93, 614.93),# 256 MB - Peak (2nd)
]

# Stride Access Efficiency (RTX 5080)
# Format: (stride, efficiency_pct)
stride_efficiency_data = [
    (1, 100.0),    # Sequential
    (2, 86.0),
    (4, 85.9),
    (8, 80.4),
    (16, 62.2),
    (32, 35.3),
    (64, 22.8),
    (128, 11.4),
    (256, 5.9),
]

# Data Type Bandwidth Comparison
# Format: (dtype, bandwidth_gbps)
dtype_bandwidth_data = [
    ("FP32", 878.19),
    ("INT32", 882.25),
    ("FP64", 468.73),
    ("FP16", 410.20),
    ("INT8", 380.50),
]

def plot_memory_bandwidth_vs_size():
    """Plot memory bandwidth vs data size."""
    sizes = [d[0] for d in memory_bandwidth_data]
    read_bw = [d[1] for d in memory_bandwidth_data]
    write_bw = [d[2] for d in memory_bandwidth_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, read_bw, 'b-o', linewidth=2, markersize=8, label='Read')
    ax.plot(sizes, write_bw, 'r-s', linewidth=2, markersize=8, label='Write')

    ax.set_xscale('log')
    ax.set_xlabel('Data Size (MB)', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Memory Bandwidth vs Data Size - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add memory region annotations
    regions = [
        (0.01, 50, "L1"),
        (0.5, 150, "L1/L2"),
        (4, 450, "L2"),
        (16, 650, "Peak"),
        (64, 300, "L2 miss"),
    ]
    for x, y, label in regions:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
        ax.annotate(label, (x, y), fontsize=9, color='gray')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'memory', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'memory_bandwidth_vs_size.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'memory_bandwidth.csv')
    with open(csv_path, 'w') as f:
        f.write("size_mb,read_bw_gbps,write_bw_gbps\n")
        for d in memory_bandwidth_data:
            f.write(f"{d[0]},{d[1]},{d[2]}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_stride_efficiency():
    """Plot stride access efficiency."""
    strides = [d[0] for d in stride_efficiency_data]
    efficiency = [d[1] for d in stride_efficiency_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(strides)), efficiency, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(strides)))
    ax.set_xticklabels(strides)
    ax.set_xlabel('Stride (elements)', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Stride Access Efficiency - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.annotate(f'{eff:.0f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'memory', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'stride_efficiency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'stride_efficiency.csv')
    with open(csv_path, 'w') as f:
        f.write("stride,efficiency_pct\n")
        for stride, eff in stride_efficiency_data:
            f.write(f"{stride},{eff}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_dtype_comparison():
    """Plot data type bandwidth comparison."""
    dtypes = [d[0] for d in dtype_bandwidth_data]
    bandwidths = [d[1] for d in dtype_bandwidth_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['blue', 'green', 'orange', 'red', 'purple']
    bars = ax.bar(dtypes, bandwidths, color=colors, alpha=0.8)
    ax.set_xlabel('Data Type', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Memory Bandwidth by Data Type - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'memory', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dtype_bandwidth_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'dtype_bandwidth.csv')
    with open(csv_path, 'w') as f:
        f.write("dtype,bandwidth_gbps\n")
        for dtype, bw in dtype_bandwidth_data:
            f.write(f"{dtype},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()

if __name__ == '__main__':
    print("Generating Memory Bandwidth Charts...")
    print("=" * 50)
    plot_memory_bandwidth_vs_size()
    print()
    plot_stride_efficiency()
    print()
    plot_dtype_comparison()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/memory/data/")
