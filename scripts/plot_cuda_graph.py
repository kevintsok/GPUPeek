#!/usr/bin/env python3
"""
CUDA Graph Chart Generator
==========================
Generates charts for CUDA Graph research benchmarks.

Usage:
    python plot_cuda_graph.py

Output:
    NVIDIA_GPU/sm_120/cuda_graph/data/launch_overhead.png
    NVIDIA_GPU/sm_120/cuda_graph/data/graph_speedup.png
    NVIDIA_GPU/sm_120/cuda_graph/data/pipeline_performance.png
    NVIDIA_GPU/sm_120/cuda_graph/data/lifecycle_overhead.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# Launch Overhead Comparison
# =============================================================================

# Format: (name, time_ms)
launch_overhead_data = [
    ("Regular Launch", 0.15),
    ("Graph Create", 0.08),
    ("Graph Instantiate", 0.12),
    ("Graph Launch", 0.02),
]

# =============================================================================
# Graph Speedup
# =============================================================================

# Format: (name, speedup)
speedup_data = [
    ("Single Kernel", 1.0),
    ("Single Kernel (Graph)", 1.5),
    ("5 Kernels", 1.0),
    ("5 Kernels (Graph)", 2.5),
]

# =============================================================================
# Pipeline Performance
# =============================================================================

# Format: (name, time_ms)
pipeline_data = [
    ("Sequential (3 ops)", 1.8),
    ("Graph Pipeline (3 ops)", 1.2),
]

# =============================================================================
# Kernel Count vs Speedup
# =============================================================================

# Format: (num_kernels, regular_ms, graph_ms)
kernel_count_data = [
    (1, 0.15, 0.10),
    (3, 0.35, 0.12),
    (5, 0.50, 0.15),
    (10, 0.90, 0.18),
    (20, 1.70, 0.25),
]


def plot_launch_overhead():
    """Plot CUDA Graph launch overhead comparison."""
    names = [d[0] for d in launch_overhead_data]
    times = [d[1] for d in launch_overhead_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['gray', 'steelblue', 'darkorange', 'seagreen']
    bars = ax.bar(names, times, color=colors, alpha=0.8)

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('CUDA Graph Launch Overhead - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.25)

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight graph launch
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cuda_graph', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'launch_overhead.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'launch_overhead.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms\n")
        for name, t in launch_overhead_data:
            f.write(f"{name},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_graph_speedup():
    """Plot Graph speedup comparison."""
    names = [d[0] for d in speedup_data]
    speedups = [d[1] for d in speedup_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['gray', 'steelblue', 'gray', 'seagreen']
    bars = ax.bar(names, speedups, color=colors, alpha=0.8)

    ax.set_ylabel('Relative Speed (x)', fontsize=12)
    ax.set_title('CUDA Graph Speedup - Regular vs Graph Launch', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 3.5)

    for bar, s in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{s:.1f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight graph bars
    bars[1].set_edgecolor('black')
    bars[1].set_linewidth(2)
    bars[3].set_edgecolor('black')
    bars[3].set_linewidth(2)

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cuda_graph', 'data')
    output_path = os.path.join(output_dir, 'graph_speedup.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'graph_speedup.csv')
    with open(csv_path, 'w') as f:
        f.write("name,speedup\n")
        for name, s in speedup_data:
            f.write(f"{name},{s}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_pipeline_performance():
    """Plot pipeline performance comparison."""
    names = [d[0] for d in pipeline_data]
    times = [d[1] for d in pipeline_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time comparison
    colors = ['gray', 'seagreen']
    bars1 = ax1.bar(names, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Pipeline Execution Time', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)

    for bar, t in zip(bars1, times):
        height = bar.get_height()
        ax1.annotate(f'{t:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Speedup
    speedup = times[0] / times[1]
    speedups = [1.0, speedup]
    colors2 = ['gray', 'seagreen']
    bars2 = ax2.bar(['Sequential', 'Graph'], speedups, color=colors2, alpha=0.8)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Graph Pipeline Speedup', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 2.0)

    for bar, s in zip(bars2, speedups):
        height = bar.get_height()
        ax2.annotate(f'{s:.1f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('CUDA Graph Pipeline Performance (3 Operations)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cuda_graph', 'data')
    output_path = os.path.join(output_dir, 'pipeline_performance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'pipeline_performance.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms\n")
        for name, t in pipeline_data:
            f.write(f"{name},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_kernel_count_vs_speedup():
    """Plot kernel count vs speedup for regular vs graph launches."""
    num_kernels = [d[0] for d in kernel_count_data]
    regular_times = [d[1] for d in kernel_count_data]
    graph_times = [d[2] for d in kernel_count_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Execution time
    x = np.arange(len(num_kernels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, regular_times, width, label='Regular', color='gray', alpha=0.8)
    bars2 = ax1.bar(x + width/2, graph_times, width, label='Graph', color='seagreen', alpha=0.8)

    ax1.set_xlabel('Number of Kernels', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Execution Time vs Kernel Count', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(num_kernels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Speedup
    speedups = [r / g for r, g in zip(regular_times, graph_times)]
    bars3 = ax2.bar(num_kernels, speedups, color='steelblue', alpha=0.8)
    ax2.set_xlabel('Number of Kernels', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Graph Speedup vs Regular Launch', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    for bar, s in zip(bars3, speedups):
        height = bar.get_height()
        ax2.annotate(f'{s:.1f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('CUDA Graph: More Kernels = Higher Speedup', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cuda_graph', 'data')
    output_path = os.path.join(output_dir, 'kernel_count_speedup.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'kernel_count_speedup.csv')
    with open(csv_path, 'w') as f:
        f.write("num_kernels,regular_ms,graph_ms,speedup\n")
        for d, r, g in zip(num_kernels, regular_times, graph_times):
            f.write(f"{d},{r},{g},{r/g:.2f}\n")
    print(f"Saved: {csv_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating CUDA Graph Charts...")
    print("=" * 50)
    plot_launch_overhead()
    print()
    plot_graph_speedup()
    print()
    plot_pipeline_performance()
    print()
    plot_kernel_count_vs_speedup()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/cuda_graph/data/")
