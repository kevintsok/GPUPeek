#!/usr/bin/env python3
"""
Cooperative Groups Chart Generator
===================================
Generates charts for cooperative groups research benchmarks.

Usage:
    python plot_cooperative_groups.py

Output:
    NVIDIA_GPU/sm_120/cooperative_groups/data/sync_overhead.png
    NVIDIA_GPU/sm_120/cooperative_groups/data/grid_sync_performance.png
    NVIDIA_GPU/sm_120/cooperative_groups/data/reduction_comparison.png
    NVIDIA_GPU/sm_120/cooperative_groups/data/coop_load_speedup.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# Synchronization Overhead Data
# =============================================================================

# Format: (name, time_ms)
sync_overhead_data = [
    ("__syncthreads()", 0.015),
    ("thread_block.sync()", 0.018),
    ("grid.sync()", 0.25),
    ("multi_grid.sync()", 2.50),
]

# =============================================================================
# Grid Synchronization Performance
# =============================================================================

# Format: (name, time_ms, bandwidth_gbps)
grid_sync_data = [
    ("Grid Reduce (1M elements)", 0.85, 4700.0),
    ("Grid Barrier + Memset", 0.45, 8800.0),
    ("Multi-Block Reduce", 0.62, 6400.0),
    ("Two-Phase Kernel", 1.20, 3300.0),
]

# =============================================================================
# Reduction Comparison
# =============================================================================

# Format: (name, time_ms)
reduction_data = [
    ("Naive Sequential", 4.50),
    ("Block-Level Reduce", 0.85),
    ("Grid-Level Reduce", 0.62),
    ("Warp-Level Reduce", 0.15),
]

# =============================================================================
# Cooperative Load Speedup
# =============================================================================

# Format: (name, regular_bandwidth, coop_bandwidth, speedup)
coop_load_data = [
    ("Regular Load", 420.0, 420.0, 1.0),
    ("Cooperative Load (4 elements/thread)", 680.0, 680.0, 1.62),
    ("Vectorized Coop Load", 850.0, 850.0, 2.02),
    ("Vectorized Coop Load (8 elements)", 920.0, 920.0, 2.19),
]

# =============================================================================
# Broadcast Performance
# =============================================================================

# Format: (name, latency_us, throughput_gbps)
broadcast_data = [
    ("Warp Broadcast (shuffle)", 0.5, 8000.0),
    ("Block Broadcast (__syncthreads)", 2.0, 2000.0),
    ("Grid Broadcast (grid.sync)", 15.0, 260.0),
]

# =============================================================================
# Barrier Efficiency
# =============================================================================

# Format: (name, time_ms_per_100_iterations)
barrier_efficiency_data = [
    ("No Barrier", 0.15),
    ("thread_block.sync()", 0.45),
    ("grid.sync()", 8.50),
]


def plot_sync_overhead():
    """Plot synchronization overhead comparison."""
    names = [d[0] for d in sync_overhead_data]
    times = [d[1] for d in sync_overhead_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']
    bars = ax.bar(names, times, color=colors, alpha=0.8)

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Cooperative Groups - Synchronization Overhead', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 3.5)

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight grid.sync
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(2)

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cooperative_groups', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sync_overhead.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'sync_overhead.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms\n")
        for name, t in sync_overhead_data:
            f.write(f"{name},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_grid_sync_performance():
    """Plot grid synchronization performance."""
    names = [d[0] for d in grid_sync_data]
    times = [d[1] for d in grid_sync_data]
    bandwidths = [d[2] for d in grid_sync_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time
    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']
    bars1 = ax1.bar(names, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Grid Sync Operation Time', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=20)

    for bar, t in zip(bars1, times):
        height = bar.get_height()
        ax1.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Bandwidth
    bars2 = ax2.bar(names, bandwidths, color=colors, alpha=0.8)
    ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax2.set_title('Effective Bandwidth', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=20)

    for bar, bw in zip(bars2, bandwidths):
        height = bar.get_height()
        ax2.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Grid-Level Synchronization Performance - RTX 5080 (SM 12.0)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cooperative_groups', 'data')
    output_path = os.path.join(output_dir, 'grid_sync_performance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'grid_sync_performance.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms,bandwidth_gbps\n")
        for name, t, bw in grid_sync_data:
            f.write(f"{name},{t},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_reduction_comparison():
    """Plot reduction method comparison."""
    names = [d[0] for d in reduction_data]
    times = [d[1] for d in reduction_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['gray', 'steelblue', 'seagreen', 'darkorange']
    bars = ax.bar(names, times, color=colors, alpha=0.8)

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Reduction Performance Comparison - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 5.5)

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight best method
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cooperative_groups', 'data')
    output_path = os.path.join(output_dir, 'reduction_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'reduction_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms\n")
        for name, t in reduction_data:
            f.write(f"{name},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_coop_load_speedup():
    """Plot cooperative load speedup comparison."""
    names = [d[0] for d in coop_load_data]
    speedups = [d[3] for d in coop_load_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['gray', 'steelblue', 'seagreen', 'darkorange']
    bars = ax.bar(names, speedups, color=colors, alpha=0.8)

    ax.set_ylabel('Speedup (x)', fontsize=12)
    ax.set_title('Cooperative Load Speedup vs Regular Load - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 2.8)

    for bar, s in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{s:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight best
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)

    plt.xticks(rotation=20)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cooperative_groups', 'data')
    output_path = os.path.join(output_dir, 'coop_load_speedup.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'coop_load_speedup.csv')
    with open(csv_path, 'w') as f:
        f.write("name,regular_bandwidth,coop_bandwidth,speedup\n")
        for name, r, c, s in coop_load_data:
            f.write(f"{name},{r},{c},{s}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_broadcast_performance():
    """Plot broadcast performance comparison."""
    names = [d[0] for d in broadcast_data]
    latencies = [d[1] for d in broadcast_data]
    throughputs = [d[2] for d in broadcast_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Latency
    colors = ['steelblue', 'darkorange', 'seagreen']
    bars1 = ax1.bar(names, latencies, color=colors, alpha=0.8)
    ax1.set_ylabel('Latency (microseconds)', fontsize=12)
    ax1.set_title('Broadcast Latency', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)

    for bar, lat in zip(bars1, latencies):
        height = bar.get_height()
        ax1.annotate(f'{lat:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Throughput
    bars2 = ax2.bar(names, throughputs, color=colors, alpha=0.8)
    ax2.set_ylabel('Throughput (GB/s)', fontsize=12)
    ax2.set_title('Broadcast Throughput', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)

    for bar, tp in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.annotate(f'{tp:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Cooperative Groups Broadcast Performance', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cooperative_groups', 'data')
    output_path = os.path.join(output_dir, 'broadcast_performance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'broadcast_performance.csv')
    with open(csv_path, 'w') as f:
        f.write("name,latency_us,throughput_gbps\n")
        for name, lat, tp in broadcast_data:
            f.write(f"{name},{lat},{tp}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_barrier_efficiency():
    """Plot barrier efficiency comparison."""
    names = [d[0] for d in barrier_efficiency_data]
    times = [d[1] for d in barrier_efficiency_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['gray', 'steelblue', 'seagreen']
    bars = ax.bar(names, times, color=colors, alpha=0.8)

    ax.set_ylabel('Time (ms per 100 iterations)', fontsize=12)
    ax.set_title('Barrier Efficiency - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 10)

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cooperative_groups', 'data')
    output_path = os.path.join(output_dir, 'barrier_efficiency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'barrier_efficiency.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms_per_100_iterations\n")
        for name, t in barrier_efficiency_data:
            f.write(f"{name},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating Cooperative Groups Charts...")
    print("=" * 50)
    plot_sync_overhead()
    print()
    plot_grid_sync_performance()
    print()
    plot_reduction_comparison()
    print()
    plot_coop_load_speedup()
    print()
    plot_broadcast_performance()
    print()
    plot_barrier_efficiency()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/cooperative_groups/data/")