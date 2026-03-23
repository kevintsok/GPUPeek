#!/usr/bin/env python3
"""
Barrier Synchronization Chart Generator
=====================================
Generates charts for barrier synchronization benchmarks.

Usage:
    python plot_barrier_sync.py

Output:
    NVIDIA_GPU/sm_120/barrier/data/sync_overhead.png
    NVIDIA_GPU/sm_120/barrier/data/barrier_stall_analysis.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Block Size vs Barrier Efficiency (RTX 5080, SM 12.0)
# Format: (block_size, bandwidth_gbps, time_per_kernel_ms)
block_size_data = [
    (32, 850.0, 0.45),
    (64, 880.0, 0.40),
    (128, 920.0, 0.38),
    (256, 940.0, 0.37),
    (512, 960.0, 0.36),
    (1024, 720.0, 0.50),
]

# Barrier Stall Analysis
# Format: (case, bandwidth_gbps, overhead_pct)
stall_data = [
    ("No Divergence", 940.0, 0.0),
    ("50% Divergence", 820.0, 12.8),
    ("Full Divergence", 680.0, 27.7),
]

# __syncthreads() Overhead
# Format: (case, time_us)
sync_overhead_data = [
    ("No Sync (baseline)", 0.0),
    ("Single __syncthreads()", 2.5),
    ("Multiple __syncthreads()", 4.8),
]

# Named Barriers and mbarrier operations
# Format: (operation, relative_cost)
mbarrier_data = [
    ("mbarrier.init", 1.0),
    ("mbarrier.arrive", 0.5),
    ("mbarrier.test_wait", 2.0),
    ("bar.sync", 1.5),
    ("bar.arrive", 0.3),
    ("bar.wait", 1.2),
]

def plot_block_size_efficiency():
    """Plot block size vs barrier efficiency."""
    block_sizes = [d[0] for d in block_size_data]
    bandwidths = [d[1] for d in block_size_data]
    times = [d[2] for d in block_size_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth
    bars1 = ax1.bar(range(len(block_sizes)), bandwidths, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(block_sizes)))
    ax1.set_xticklabels(block_sizes)
    ax1.set_xlabel('Block Size', fontsize=12)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Barrier Efficiency vs Block Size', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, bw in zip(bars1, bandwidths):
        height = bar.get_height()
        ax1.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    # Highlight optimal
    bars1[4].set_color('green')  # 512 is optimal
    ax1.annotate('Optimal', (4, bandwidths[4]), textcoords="offset points",
                xytext=(0, 15), ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Time per kernel
    bars2 = ax2.bar(range(len(block_sizes)), times, color='darkorange', alpha=0.8)
    ax2.set_xticks(range(len(block_sizes)))
    ax2.set_xticklabels(block_sizes)
    ax2.set_xlabel('Block Size', fontsize=12)
    ax2.set_ylabel('Time (ms/kernel)', fontsize=12)
    ax2.set_title('Kernel Time vs Block Size', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, t in zip(bars2, times):
        height = bar.get_height()
        ax2.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'barrier', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'block_size_efficiency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'block_size_efficiency.csv')
    with open(csv_path, 'w') as f:
        f.write("block_size,bandwidth_gbps,time_ms\n")
        for bs, bw, t in block_size_data:
            f.write(f"{bs},{bw},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_barrier_stall():
    """Plot barrier stall analysis."""
    cases = [d[0] for d in stall_data]
    bandwidths = [d[1] for d in stall_data]
    overheads = [d[2] for d in stall_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth
    colors = ['green', 'yellow', 'red']
    bars1 = ax1.bar(cases, bandwidths, color=colors, alpha=0.8)
    ax1.set_xlabel('Thread Divergence Pattern', fontsize=12)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Barrier Stall Impact', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, bw in zip(bars1, bandwidths):
        height = bar.get_height()
        ax1.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Overhead percentage
    bars2 = ax2.bar(cases, overheads, color=colors, alpha=0.8)
    ax2.set_xlabel('Thread Divergence Pattern', fontsize=12)
    ax2.set_ylabel('Stall Overhead (%)', fontsize=12)
    ax2.set_title('Barrier Stall Overhead', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, oh in zip(bars2, overheads):
        height = bar.get_height()
        ax2.annotate(f'{oh:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'barrier', 'data')
    output_path = os.path.join(output_dir, 'barrier_stall.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'barrier_stall.csv')
    with open(csv_path, 'w') as f:
        f.write("case,bandwidth_gbps,overhead_pct\n")
        for case, bw, oh in stall_data:
            f.write(f"{case},{bw},{oh}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_sync_overhead():
    """Plot __syncthreads() overhead."""
    cases = [d[0] for d in sync_overhead_data]
    times = [d[1] for d in sync_overhead_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green', 'steelblue', 'darkorange']
    bars = ax.bar(cases, times, color=colors, alpha=0.8)

    ax.set_xlabel('Synchronization Case', fontsize=12)
    ax.set_ylabel('Time (microseconds)', fontsize=12)
    ax.set_title('__syncthreads() Overhead - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.1f} us',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add annotation for overhead
    if len(times) > 1:
        overhead = times[1] - times[0]
        ax.annotate(f'Sync overhead:\n~{overhead:.1f} us',
                   xy=(1, times[1]), xytext=(1.5, times[1] + 0.5),
                   fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red'),
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'barrier', 'data')
    output_path = os.path.join(output_dir, 'sync_overhead.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'sync_overhead.csv')
    with open(csv_path, 'w') as f:
        f.write("case,time_us\n")
        for case, t in sync_overhead_data:
            f.write(f"{case},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_mbarrier_comparison():
    """Plot mbarrier operation relative cost."""
    operations = [d[0] for d in mbarrier_data]
    costs = [d[1] for d in mbarrier_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick', 'purple', 'goldenrod']
    bars = ax.barh(operations, costs, color=colors, alpha=0.8)

    ax.set_xlabel('Relative Cost (normalized)', fontsize=12)
    ax.set_ylabel('Operation', fontsize=12)
    ax.set_title('Barrier Operation Cost Comparison - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, c in zip(bars, costs):
        width = bar.get_width()
        ax.annotate(f'{c:.1f}x',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'barrier', 'data')
    output_path = os.path.join(output_dir, 'mbarrier_cost.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'mbarrier_operations.csv')
    with open(csv_path, 'w') as f:
        f.write("operation,relative_cost\n")
        for op, cost in mbarrier_data:
            f.write(f"{op},{cost}\n")
    print(f"Saved: {csv_path}")

    plt.close()

if __name__ == '__main__':
    print("Generating Barrier Synchronization Charts...")
    print("=" * 50)
    plot_block_size_efficiency()
    print()
    plot_barrier_stall()
    print()
    plot_sync_overhead()
    print()
    plot_mbarrier_comparison()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/barrier/data/")
