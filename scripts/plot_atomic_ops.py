#!/usr/bin/env python3
"""
Atomic Operations Chart Generator
=================================
Generates charts for atomic operations benchmarks.

Usage:
    python plot_atomic_ops.py

Output:
    NVIDIA_GPU/sm_120/atomic/data/atomic_contention_analysis.png
    NVIDIA_GPU/sm_120/atomic/data/atomic_operation_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Atomic Contention Analysis
# Format: (level, bandwidth_gbps, description)
atomic_contention_data = [
    ("Warp-Level", 850.0, "Low contention\n(32 values → 1 atomic)"),
    ("Block-Level", 620.0, "Medium contention\n(256 values → 1 atomic)"),
    ("Grid-Level\n(Direct)", 180.0, "High contention\n(All threads → 1 atomic)"),
    ("Grid-Level\n(Reduced)", 720.0, "Low contention\n(Block reduce + 1 atomic)"),
]

# Atomic Operation Type Comparison
# Format: (operation, bandwidth_gbps, description)
atomic_ops_data = [
    ("atomicAdd", 850.0, "FP32 add"),
    ("atomicMin", 720.0, "INT32 min"),
    ("atomicMax", 715.0, "INT32 max"),
    ("atomicCAS", 380.0, "Compare-and-swap"),
    ("atomicAnd", 680.0, "Bitwise AND"),
    ("atomicOr", 690.0, "Bitwise OR"),
]

# Contention Reduction Speedup
# Format: (method, speedup_vs_direct)
speedup_data = [
    ("Direct\nAtomic", 1.0),
    ("Warp-Level\nReduction", 4.7),
    ("Block-Level\nReduction", 3.4),
    ("Grid-Level\nReduction", 4.0),
]

def plot_atomic_contention():
    """Plot atomic contention analysis."""
    levels = [d[0] for d in atomic_contention_data]
    bandwidths = [d[1] for d in atomic_contention_data]
    descriptions = [d[2] for d in atomic_contention_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['green', 'steelblue', 'red', 'green']
    bars = ax.bar(levels, bandwidths, color=colors, alpha=0.8)

    ax.set_xlabel('Atomic Level', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Atomic Operation Contention Analysis - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add descriptions
    for i, (bar, desc) in enumerate(zip(bars, descriptions)):
        ax.annotate(desc, xy=(bar.get_x() + bar.get_width() / 2, 50),
                  fontsize=9, ha='center', va='bottom',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'atomic', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'atomic_contention.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'atomic_contention.csv')
    with open(csv_path, 'w') as f:
        f.write("level,bandwidth_gbps,description\n")
        for level, bw, desc in atomic_contention_data:
            f.write(f"{level.replace(chr(10), ' ')},{bw},{desc}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_atomic_operations():
    """Plot atomic operation type comparison."""
    operations = [d[0] for d in atomic_ops_data]
    bandwidths = [d[1] for d in atomic_ops_data]
    descriptions = [d[2] for d in atomic_ops_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick', 'purple', 'goldenrod']
    bars = ax.bar(operations, bandwidths, color=colors, alpha=0.8)

    ax.set_xlabel('Atomic Operation', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Atomic Operation Performance Comparison - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'atomic', 'data')
    output_path = os.path.join(output_dir, 'atomic_operations.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'atomic_operations.csv')
    with open(csv_path, 'w') as f:
        f.write("operation,bandwidth_gbps,description\n")
        for op, bw, desc in atomic_ops_data:
            f.write(f"{op},{bw},{desc}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_speedup_comparison():
    """Plot speedup from reduction strategies."""
    methods = [d[0] for d in speedup_data]
    speedups = [d[1] for d in speedup_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['red', 'green', 'steelblue', 'green']
    bars = ax.bar(methods, speedups, color=colors, alpha=0.8)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Speedup vs Direct Atomic', fontsize=12)
    ax.set_title('Atomic Contention Reduction - Speedup Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Direct Atomic')

    # Add value labels
    for bar, su in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{su:.1f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.legend()

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'atomic', 'data')
    output_path = os.path.join(output_dir, 'atomic_speedup.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'atomic_speedup.csv')
    with open(csv_path, 'w') as f:
        f.write("method,speedup_vs_direct\n")
        for method, su in speedup_data:
            f.write(f"{method.replace(chr(10), ' ')},{su}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_contention_mechanism():
    """Plot how contention affects atomic performance."""
    # X-axis: number of threads competing for same atomic location
    num_threads = [32, 64, 128, 256, 512, 1024, 2048]
    # Y-axis: effective bandwidth (normalized)
    # Theoretical: bandwidth proportional to 1/contention
    base_bw = 850.0  # Warp-level (32 threads)
    effective_bw = [base_bw * (32.0 / n) for n in num_threads]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(num_threads, effective_bw, 'b-o', linewidth=2, markersize=8)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of Threads Contending for Same Location', fontsize=12)
    ax.set_ylabel('Effective Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Atomic Contention Model - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')

    # Annotations
    ax.annotate('Warp (32 threads)', (32, effective_bw[0]),
               xytext=(50, effective_bw[0]+50),
               fontsize=10,
               arrowprops=dict(arrowstyle='->', color='green'),
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.annotate('Block (256 threads)', (256, effective_bw[3]),
               xytext=(200, effective_bw[3]-100),
               fontsize=10,
               arrowprops=dict(arrowstyle='->', color='orange'),
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.annotate('Grid (2048 threads)', (2048, effective_bw[6]),
               xytext=(1500, effective_bw[6]-50),
               fontsize=10,
               arrowprops=dict(arrowstyle='->', color='red'),
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'atomic', 'data')
    output_path = os.path.join(output_dir, 'atomic_contention_model.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'contention_model.csv')
    with open(csv_path, 'w') as f:
        f.write("num_threads,effective_bw\n")
        for n, bw in zip(num_threads, effective_bw):
            f.write(f"{n},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()

if __name__ == '__main__':
    print("Generating Atomic Operations Charts...")
    print("=" * 50)
    plot_atomic_contention()
    print()
    plot_atomic_operations()
    print()
    plot_speedup_comparison()
    print()
    plot_contention_mechanism()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/atomic/data/")
