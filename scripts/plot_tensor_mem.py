#!/usr/bin/env python3
"""
Tensor Memory Chart Generator
============================
Generates charts for tensor memory operations from benchmark data.
Uses actual benchmark data from RTX 5080.

Usage:
    python3 plot_tensor_mem.py

Output:
    ../NVIDIA_GPU/sm_120/tensor_mem/data/*.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# Actual Benchmark Data (RTX 5080 Laptop GPU)
# =============================================================================

size_data = [
    # (size_bytes, naive_GB/s, shared_GB/s, regular_GB/s, cp_async_GB/s)
    (1024, 0.36, 0.50, 0.57, 0.46),
    (4096, 1.84, 2.51, 2.16, 1.76),
    (8192, 4.10, 3.90, 4.26, 4.86),
    (16384, 7.96, 47.32, 50.22, 50.49),
    (32768, 14.89, 100.98, 101.84, 103.86),
    (65536, 29.01, 183.32, 197.70, 197.40),
    (131072, 61.05, 416.10, 426.60, 424.87),
    (262144, 125.46, 760.94, 816.65, 818.56),
    (524288, 252.36, 1605.78, 1643.54, 1643.54),
    (1048576, 386.68, 3172.70, 3318.28, 3127.74),
    (2097152, 253.51, 6043.67, 6428.05, 6408.41),
    (4194304, 1066.57, 11839.96, 12671.61, 12642.97),
    (8388608, 1095.73, 21495.47, 22399.49, 23797.47),
    (16777216, 1206.34, 47730.34, 50533.78, 41995.54),
    (33554432, 1365.28, 90565.27, 103883.69, 104530.94),
    (67108864, 1513.40, 163182.64, 196368.30, 196512.05),
    (134217728, 1690.53, 329166.72, 399160.53, 426088.03),
    (1073741824, 1817.16, 582605.44, 834298.25, 840831.50),
]

sizes = [d[0] for d in size_data]
naive_bw = [d[1] for d in size_data]
shared_bw = [d[2] for d in size_data]
regular_bw = [d[3] for d in size_data]
cp_async_bw = [d[4] for d in size_data]

# Size labels for x-axis
size_labels = ['1KB', '4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB',
               '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', '256MB']


def format_size(size_bytes):
    """Format size in bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024*1024:
        return f"{size_bytes//1024}KB"
    elif size_bytes < 1024*1024*1024:
        return f"{size_bytes//(1024*1024)}MB"
    else:
        return f"{size_bytes//(1024*1024*1024)}GB"


def plot_bandwidth_vs_size():
    """Plot bandwidth vs data size for different copy methods (log scale)."""
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(sizes))

    ax.plot(x, naive_bw, 'o-', label='Naive Global Load', linewidth=2, markersize=6)
    ax.plot(x, shared_bw, 's-', label='Shared Memory Load', linewidth=2, markersize=6)
    ax.plot(x, regular_bw, '^-', label='Regular Copy (Shared)', linewidth=2, markersize=6)
    ax.plot(x, cp_async_bw, 'd-', label='cp.async 16B', linewidth=2, markersize=6)

    ax.set_xlabel('Data Size', fontsize=14)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=14)
    ax.set_title('Memory Copy Bandwidth vs Data Size\nRTX 5080 Laptop GPU (Blackwell SM 12.0)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Add annotations for key transitions
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)  # 64KB - L1 cache
    ax.axvline(x=9, color='gray', linestyle='--', alpha=0.5)  # 1MB - L2 cache

    plt.tight_layout()

    data_dir = '../NVIDIA_GPU/sm_120/tensor_mem/data'
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'bandwidth_vs_size.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def plot_bandwidth_linear():
    """Plot bandwidth vs data size with linear scale (zoomed in for small sizes)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Small sizes (up to 1MB)
    small_count = 10
    x_small = np.arange(small_count)

    axes[0].plot(x_small, naive_bw[:small_count], 'o-', label='Naive Global Load', linewidth=2, markersize=6)
    axes[0].plot(x_small, shared_bw[:small_count], 's-', label='Shared Memory Load', linewidth=2, markersize=6)
    axes[0].plot(x_small, regular_bw[:small_count], '^-', label='Regular Copy', linewidth=2, markersize=6)
    axes[0].plot(x_small, cp_async_bw[:small_count], 'd-', label='cp.async 16B', linewidth=2, markersize=6)
    axes[0].set_xlabel('Data Size', fontsize=12)
    axes[0].set_ylabel('Bandwidth (GB/s)', fontsize=12)
    axes[0].set_title('Small Data (up to 1MB)\nKernel Launch Overhead + L1 Cache', fontsize=13)
    axes[0].set_xticks(x_small)
    axes[0].set_xticklabels(size_labels[:small_count], rotation=45, ha='right', fontsize=9)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Large sizes (1MB to 256MB)
    large_sizes = sizes[9:]
    x_large = np.arange(len(large_sizes))

    axes[1].plot(x_large, naive_bw[9:], 'o-', label='Naive Global Load', linewidth=2, markersize=6)
    axes[1].plot(x_large, shared_bw[9:], 's-', label='Shared Memory Load', linewidth=2, markersize=6)
    axes[1].plot(x_large, regular_bw[9:], '^-', label='Regular Copy', linewidth=2, markersize=6)
    axes[1].plot(x_large, cp_async_bw[9:], 'd-', label='cp.async 16B', linewidth=2, markersize=6)
    axes[1].set_xlabel('Data Size', fontsize=12)
    axes[1].set_ylabel('Bandwidth (GB/s)', fontsize=12)
    axes[1].set_title('Large Data (1MB to 256MB)\nL2 Cache + DRAM Bandwidth', fontsize=13)
    axes[1].set_xticks(x_large)
    axes[1].set_xticklabels(size_labels[9:], rotation=45, ha='right', fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Memory Copy Bandwidth Analysis\nRTX 5080 Laptop GPU (Blackwell SM 12.0)', fontsize=16, y=1.02)
    plt.tight_layout()

    data_dir = '../NVIDIA_GPU/sm_120/tensor_mem/data'
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'bandwidth_vs_size_linear.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def plot_memory_hierarchy():
    """Plot categorized by memory hierarchy with annotations."""
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(sizes))

    ax.fill_between(x, 0, naive_bw, alpha=0.3, label='_nolegend_')
    ax.plot(x, naive_bw, 'o-', label='Naive Global Load', linewidth=2, markersize=5)
    ax.plot(x, shared_bw, 's-', label='Shared Memory Load', linewidth=2, markersize=5)
    ax.plot(x, regular_bw, '^-', label='Regular Copy', linewidth=2, markersize=5)
    ax.plot(x, cp_async_bw, 'd-', label='cp.async 16B', linewidth=2, markersize=5)

    # Add memory hierarchy regions
    ax.axvspan(-0.5, 5.5, alpha=0.1, color='blue', label='L1 Cache Region')
    ax.axvspan(5.5, 9.5, alpha=0.1, color='green', label='L2 Cache Region')
    ax.axvspan(9.5, 17.5, alpha=0.1, color='red', label='DRAM Region')

    ax.set_xlabel('Data Size', fontsize=14)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=14)
    ax.set_title('Memory Copy Bandwidth by Memory Hierarchy\nRTX 5080 Laptop GPU (Blackwell SM 12.0)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Add region labels
    ax.text(2.5, ax.get_ylim()[1]*0.5, 'L1 Cache\n(<64KB)', ha='center', fontsize=10, color='blue')
    ax.text(7, ax.get_ylim()[1]*0.5, 'L2 Cache\n(64KB-1MB)', ha='center', fontsize=10, color='green')
    ax.text(13, ax.get_ylim()[1]*0.5, 'DRAM\n(>1MB)', ha='center', fontsize=10, color='red')

    plt.tight_layout()

    data_dir = '../NVIDIA_GPU/sm_120/tensor_mem/data'
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'bandwidth_by_hierarchy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def plot_cp_async_comparison():
    """Plot cp.async vs regular copy comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(sizes))

    # Calculate speedup
    speedup = [cp_async_bw[i] / regular_bw[i] if regular_bw[i] > 0 else 0 for i in range(len(sizes))]

    ax.bar(x, speedup, color='steelblue', alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No Speedup (1.0x)')

    ax.set_xlabel('Data Size', fontsize=14)
    ax.set_ylabel('Speedup (cp.async / Regular)', fontsize=14)
    ax.set_title('cp.async vs Regular Copy Speedup\nRTX 5080 Laptop GPU (Blackwell SM 12.0)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(speedup):
        if v > 0:
            ax.text(i, v + 0.02, f'{v:.2f}x', ha='center', fontsize=8, rotation=0)

    plt.tight_layout()

    data_dir = '../NVIDIA_GPU/sm_120/tensor_mem/data'
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'cp_async_speedup.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


if __name__ == "__main__":
    print("Generating Tensor Memory Charts...")
    print()

    plot_bandwidth_vs_size()
    plot_bandwidth_linear()
    plot_memory_hierarchy()
    plot_cp_async_comparison()

    print()
    print("Chart generation complete!")