#!/usr/bin/env python3
"""
Redux.sync Benchmark Chart Generator
====================================
Generates throughput comparison charts for redux.sync and warp reduction benchmarks.

Usage:
    python plot_redux_sync.py

Output:
    NVIDIA_GPU/sm_120/redux_sync/data/reduction_throughput.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Redux.sync Benchmark Results (RTX 5080, SM 12.0)
# From redux_sync_research_benchmarks.cu run
# Format: (operation, time_ms_100iter, implementation)
redux_benchmark_data = [
    ("Redux ADD (sequential)", 2.147, "Sequential for-loop"),
    ("Redux MIN (sequential)", 1.858, "Sequential for-loop"),
    ("Redux MAX (sequential)", 1.731, "Sequential for-loop"),
    ("Redux AND (sequential)", 1.829, "Sequential for-loop"),
    ("Redux OR (sequential)", 1.824, "Sequential for-loop"),
    ("Redux XOR (sequential)", 1.729, "Sequential for-loop"),
    ("TRUE redux.sync.add", 0.945, "__reduce_add_sync()"),
    ("TRUE redux.sync.min", 0.991, "__reduce_min_sync()"),
    ("TRUE redux.sync.max", 0.899, "__reduce_max_sync()"),
]

# Reduction Method Comparison
# Format: (method, time_ms_100iter)
reduction_method_data = [
    ("Shuffle\nReduction", 1.074),
    ("Butterfly\nReduction", 1.024),
    ("TRUE redux.sync", 0.899),
]

# Warp Vote Operations
# Format: (operation, time_ms_100iter)
warp_vote_data = [
    ("__any_sync", 0.85),
    ("__all_sync", 0.82),
    ("__ballot_sync", 0.88),
]

def plot_redux_operations():
    """Plot redux.sync operation comparison."""
    operations = [d[0] for d in redux_benchmark_data]
    times = [d[1] for d in redux_benchmark_data]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color sequential operations differently from TRUE redux.sync
    colors = []
    for op in operations:
        if "TRUE redux" in op:
            colors.append('darkgreen')
        else:
            colors.append('steelblue')

    bars = ax.bar(operations, times, color=colors, alpha=0.8)

    ax.set_xlabel('Redux.sync Operation', fontsize=12)
    ax.set_ylabel('Time (ms / 100 iterations)', fontsize=12)
    ax.set_title('Redux.sync Operations Performance - RTX 5080 (SM 12.0)\n(Green = TRUE hardware redux.sync)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    # Add speedup annotation
    seq_time = times[0]  # Redux ADD sequential = 2.147
    true_time = times[6]  # TRUE redux.sync.add = 0.945
    speedup = seq_time / true_time
    ax.annotate(f'2.3x faster\n(TRUE redux.sync)',
               xy=(6, true_time), xytext=(7, true_time + 0.5),
               fontsize=11, color='darkgreen', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'redux_sync', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'redux_operations.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV with implementation info
    csv_path = os.path.join(output_dir, 'redux_benchmark.csv')
    with open(csv_path, 'w') as f:
        f.write("operation,time_ms_100iter,implementation\n")
        for op, t, impl in redux_benchmark_data:
            f.write(f"{op},{t},{impl}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_reduction_methods():
    """Plot reduction method comparison."""
    methods = [d[0] for d in reduction_method_data]
    times = [d[1] for d in reduction_method_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    # TRUE redux.sync is the fastest - highlight it
    colors = ['steelblue', 'darkorange', 'seagreen']
    bars = ax.bar(methods, times, color=colors, alpha=0.8)

    ax.set_xlabel('Reduction Method', fontsize=12)
    ax.set_ylabel('Time (ms / 100 iterations)', fontsize=12)
    ax.set_title('Warp Reduction Methods Comparison - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.3f} ms',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add speedup annotation
    speedup = times[0] / times[2]
    ax.annotate(f'TRUE redux.sync\n{speedup:.1f}x faster\nthan Shuffle',
               xy=(2, times[2]), xytext=(1.5, times[0]),
               fontsize=10, color='darkgreen', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'redux_sync', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reduction_methods.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'reduction_methods.csv')
    with open(csv_path, 'w') as f:
        f.write("method,time_ms_100iter\n")
        for method, t in reduction_method_data:
            f.write(f"{method.replace(chr(10), ' ')},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_warp_vote():
    """Plot warp vote operations comparison."""
    operations = [d[0] for d in warp_vote_data]
    times = [d[1] for d in warp_vote_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['steelblue', 'darkorange', 'seagreen']
    bars = ax.bar(operations, times, color=colors, alpha=0.8)

    ax.set_xlabel('Warp Vote Operation', fontsize=12)
    ax.set_ylabel('Time (ms / 100 iterations)', fontsize=12)
    ax.set_title('Warp Vote Operations Performance - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'redux_sync', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'warp_vote_operations.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'warp_vote.csv')
    with open(csv_path, 'w') as f:
        f.write("operation,time_ms_100iter\n")
        for op, t in warp_vote_data:
            f.write(f"{op},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_redux_speedup():
    """Plot speedup comparison of reduction methods."""
    methods = ['Shuffle\nReduction', 'Butterfly\nReduction', 'TRUE redux.sync']
    times = [1.074, 1.024, 0.899]  # Updated with TRUE redux.sync
    baseline = times[0]
    speedups = [baseline / t for t in times]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['steelblue', 'darkorange', 'seagreen']
    bars = ax.bar(methods, speedups, color=colors, alpha=0.8)

    ax.set_xlabel('Reduction Method', fontsize=12)
    ax.set_ylabel('Speedup (vs Shuffle Baseline)', fontsize=12)
    ax.set_title('Warp Reduction Speedup - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline (Shuffle)')

    # Add value labels
    for bar, su in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{su:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.legend()

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'redux_sync', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reduction_speedup.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()

if __name__ == '__main__':
    print("Generating Redux.sync Benchmark Charts...")
    print("=" * 50)
    plot_redux_operations()
    print()
    plot_reduction_methods()
    print()
    plot_warp_vote()
    print()
    plot_redux_speedup()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/redux_sync/data/")
