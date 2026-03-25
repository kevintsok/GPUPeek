#!/usr/bin/env python3
"""
Tensor Memory Chart Generator
============================
Generates charts for tensor memory operations (LDMATRIX, STMATRIX, cp.async).
Uses actual benchmark data from RTX 5080.

Usage:
    python plot_tensor_mem.py

Output:
    NVIDIA_GPU/sm_120/tensor_mem/data/bandwidth_vs_size.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# Actual Benchmark Data (RTX 5080 Laptop GPU)
# =============================================================================

# Size sweep data from benchmark
size_data = [
    # (size_bytes, naive_GB/s, shared_GB/s, regular_GB/s, cp_async_GB/s)
    (4096, 0.32, 0.53, 0.46, 0.55),
    (16384, 1.78, 2.10, 2.27, 2.03),
    (65536, 7.22, 47.73, 49.99, 51.73),
    (262144, 28.52, 187.11, 194.61, 197.40),
    (1048576, 105.20, 756.00, 775.00, 846.31),
    (4194304, 391.92, 3204.20, 3305.20, 3310.42),
    (16777216, 1099.93, 13179.27, 13584.79, 13210.41),
    (67108864, 1224.82, 48735.56, 53601.33, 54648.91),
    (1073741824, 1784.64, 646054.06, 781471.50, 785473.19),
]

sizes = [d[0] for d in size_data]
naive_bw = [d[1] for d in size_data]
shared_bw = [d[2] for d in size_data]
regular_bw = [d[3] for d in size_data]
cp_async_bw = [d[4] for d in size_data]

# Size labels for x-axis
size_labels = ['4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB', '64MB', '256MB']


def plot_bandwidth_vs_size():
    """Plot bandwidth vs data size for different copy methods."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(sizes))

    ax.plot(x, naive_bw, 'o-', label='Naive Global Load', linewidth=2, markersize=8)
    ax.plot(x, shared_bw, 's-', label='Shared Memory Load', linewidth=2, markersize=8)
    ax.plot(x, regular_bw, '^-', label='Regular Copy', linewidth=2, markersize=8)
    ax.plot(x, cp_async_bw, 'd-', label='cp.async 16B', linewidth=2, markersize=8)

    ax.set_xlabel('Data Size', fontsize=14)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=14)
    ax.set_title('Memory Copy Bandwidth vs Data Size\nRTX 5080 Laptop GPU (Blackwell SM 12.0)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    # Ensure data directory exists
    data_dir = 'D:/Projects/dissecting-sm110/NVIDIA_GPU/sm_120/tensor_mem/data'
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'bandwidth_vs_size.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def plot_bandwidth_linear():
    """Plot bandwidth vs data size with linear scale (zoomed in)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(sizes))

    ax.plot(x, naive_bw, 'o-', label='Naive Global Load', linewidth=2, markersize=8)
    ax.plot(x, shared_bw, 's-', label='Shared Memory Load', linewidth=2, markersize=8)
    ax.plot(x, regular_bw, '^-', label='Regular Copy', linewidth=2, markersize=8)
    ax.plot(x, cp_async_bw, 'd-', label='cp.async 16B', linewidth=2, markersize=8)

    ax.set_xlabel('Data Size', fontsize=14)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=14)
    ax.set_title('Memory Copy Bandwidth vs Data Size (Linear Scale)\nRTX 5080 Laptop GPU (Blackwell SM 12.0)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    data_dir = 'D:/Projects/dissecting-sm110/NVIDIA_GPU/sm_120/tensor_mem/data'
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'bandwidth_vs_size_linear.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def plot_size_categorized():
    """Plot categorized by memory hierarchy."""
    # Split into cache-bound (small) and memory-bound (large) regions
    cache_sizes = sizes[:5]  # 4KB to 1MB
    memory_sizes = sizes[5:]  # 4MB to 256MB

    cache_naive = naive_bw[:5]
    cache_shared = shared_bw[:5]
    cache_regular = regular_bw[:5]
    cache_cp_async = cp_async_bw[:5]

    memory_naive = naive_bw[5:]
    memory_shared = shared_bw[5:]
    memory_regular = regular_bw[5:]
    memory_cp_async = cp_async_bw[5:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Cache-bound region
    x1 = np.arange(len(cache_sizes))
    ax1.plot(x1, cache_naive, 'o-', label='Naive Global Load', linewidth=2, markersize=8)
    ax1.plot(x1, cache_shared, 's-', label='Shared Memory Load', linewidth=2, markersize=8)
    ax1.plot(x1, cache_regular, '^-', label='Regular Copy', linewidth=2, markersize=8)
    ax1.plot(x1, cache_cp_async, 'd-', label='cp.async 16B', linewidth=2, markersize=8)
    ax1.set_xlabel('Data Size', fontsize=12)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Cache-Bound Region\n(L1/L2 Cache)', fontsize=14)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(['4KB', '16KB', '64KB', '256KB', '1MB'], fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Memory-bound region
    x2 = np.arange(len(memory_sizes))
    ax2.plot(x2, memory_naive, 'o-', label='Naive Global Load', linewidth=2, markersize=8)
    ax2.plot(x2, memory_shared, 's-', label='Shared Memory Load', linewidth=2, markersize=8)
    ax2.plot(x2, memory_regular, '^-', label='Regular Copy', linewidth=2, markersize=8)
    ax2.plot(x2, memory_cp_async, 'd-', label='cp.async 16B', linewidth=2, markersize=8)
    ax2.set_xlabel('Data Size', fontsize=12)
    ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax2.set_title('Memory-Bound Region\n(DRAM)', fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['4MB', '16MB', '64MB', '256MB'], fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Memory Copy Bandwidth by Memory Hierarchy\nRTX 5080 Laptop GPU (Blackwell SM 12.0)', fontsize=16, y=1.02)
    plt.tight_layout()

    data_dir = 'D:/Projects/dissecting-sm110/NVIDIA_GPU/sm_120/tensor_mem/data'
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'bandwidth_by_hierarchy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def save_csv():
    """Save benchmark data as CSV."""
    data_dir = 'D:/Projects/dissecting-sm110/NVIDIA_GPU/sm_120/tensor_mem/data'
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'benchmark_results.csv')

    with open(csv_path, 'w') as f:
        f.write("size_bytes,size_label,naive_gb_s,shared_gb_s,regular_gb_s,cp_async_gb_s\n")
        labels = ['4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB', '64MB', '256MB']
        for i, d in enumerate(size_data):
            f.write(f"{d[0]},{labels[i]},{d[1]},{d[2]},{d[3]},{d[4]}\n")

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    print("Generating Tensor Memory Charts...")
    print()

    plot_bandwidth_vs_size()
    plot_bandwidth_linear()
    plot_size_categorized()
    save_csv()

    print()
    print("Chart generation complete!")