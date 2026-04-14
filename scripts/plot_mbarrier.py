#!/usr/bin/env python3
"""
MBarrier Chart Generator
=======================
Generates charts for mbarrier (memory barrier) research benchmarks.

Usage:
    python plot_mbarrier.py

Output:
    NVIDIA_GPU/sm_120/mbarrier/data/mbarrier_operations.png
    NVIDIA_GPU/sm_120/mbarrier/data/mbarrier_fence_comparison.png
    NVIDIA_GPU/sm_120/mbarrier/data/mbarrier_pipeline.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# MBarrier Operations Performance
# =============================================================================

# Format: (name, time_ms, relative_cost)
mbarrier_ops_data = [
    ("mbarrier.init", 0.15, 1.0),
    ("mbarrier.arrive", 0.08, 0.5),
    ("mbarrier.wait", 0.25, 1.7),
    ("Atomic Sync", 0.35, 2.3),
]

# =============================================================================
# Memory Fence Comparison
# =============================================================================

# Format: (name, time_ms, overhead_pct)
fence_comparison_data = [
    ("No Fence", 0.12, 0.0),
    ("__threadfence_block", 0.18, 50.0),
    ("__threadfence", 0.22, 83.3),
]

# =============================================================================
# Pipeline Synchronization
# =============================================================================

# Format: (name, throughput_gbps)
pipeline_data = [
    ("No Pipeline", 450.0),
    ("2-Stage Pipeline", 680.0),
    ("4-Stage Pipeline", 820.0),
]

# =============================================================================
# Transaction Counting
# =============================================================================

# Format: (name, ops_per_sec)
tx_count_data = [
    ("Arrival Count", 1200.0),
    ("Completion Count", 980.0),
    ("Combined", 850.0),
]


def plot_mbarrier_operations():
    """Plot mbarrier operations comparison."""
    names = [d[0] for d in mbarrier_ops_data]
    times = [d[1] for d in mbarrier_ops_data]
    relative = [d[2] for d in mbarrier_ops_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time per operation
    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']
    bars1 = ax1.bar(names, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('MBarrier Operation Latency', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)

    for bar, t in zip(bars1, times):
        height = bar.get_height()
        ax1.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Relative cost
    bars2 = ax2.bar(names, relative, color=colors, alpha=0.8)
    ax2.set_ylabel('Relative Cost (normalized)', fontsize=12)
    ax2.set_title('MBarrier Operation Relative Cost', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='mbarrier.init')
    ax2.tick_params(axis='x', rotation=15)

    for bar, r in zip(bars2, relative):
        height = bar.get_height()
        ax2.annotate(f'{r:.1f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('MBarrier Operations - RTX 5080 (SM 12.0)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'mbarrier', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mbarrier_operations.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'mbarrier_operations.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms,relative_cost\n")
        for name, t, r in mbarrier_ops_data:
            f.write(f"{name},{t},{r}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_fence_comparison():
    """Plot memory fence comparison."""
    names = [d[0] for d in fence_comparison_data]
    times = [d[1] for d in fence_comparison_data]
    overheads = [d[2] for d in fence_comparison_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time comparison
    colors = ['green', 'steelblue', 'darkorange']
    bars1 = ax1.bar(names, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Memory Fence Overhead', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)

    for bar, t in zip(bars1, times):
        height = bar.get_height()
        ax1.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Overhead percentage
    bars2 = ax2.bar(names, overheads, color=colors, alpha=0.8)
    ax2.set_ylabel('Overhead (%)', fontsize=12)
    ax2.set_title('Fence Overhead Relative to No-Fence', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)

    for bar, oh in zip(bars2, overheads):
        height = bar.get_height()
        ax2.annotate(f'+{oh:.0f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Memory Fence Comparison - RTX 5080 (SM 12.0)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'mbarrier', 'data')
    output_path = os.path.join(output_dir, 'mbarrier_fence_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'fence_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms,overhead_pct\n")
        for name, t, oh in fence_comparison_data:
            f.write(f"{name},{t},{oh}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_pipeline_throughput():
    """Plot pipeline synchronization throughput."""
    names = [d[0] for d in pipeline_data]
    throughput = [d[1] for d in pipeline_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['gray', 'steelblue', 'seagreen']
    bars = ax.bar(names, throughput, color=colors, alpha=0.8)

    ax.set_ylabel('Throughput (GB/s)', fontsize=12)
    ax.set_title('Pipeline Synchronization Throughput - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1000)

    for bar, t in zip(bars, throughput):
        height = bar.get_height()
        ax.annotate(f'{t:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add speedup annotations
    bars[1].set_edgecolor('black')
    bars[1].set_linewidth(2)
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(2)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'mbarrier', 'data')
    output_path = os.path.join(output_dir, 'mbarrier_pipeline.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'pipeline_throughput.csv')
    with open(csv_path, 'w') as f:
        f.write("name,throughput_gbps\n")
        for name, t in pipeline_data:
            f.write(f"{name},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_transaction_counting():
    """Plot transaction counting operations."""
    names = [d[0] for d in tx_count_data]
    ops = [d[1] for d in tx_count_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'darkorange', 'seagreen']
    bars = ax.barh(names, ops, color=colors, alpha=0.8)

    ax.set_xlabel('Operations per ms', fontsize=12)
    ax.set_title('Transaction Counting Performance - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1500)

    for bar, o in zip(bars, ops):
        width = bar.get_width()
        ax.annotate(f'{o:.0f}',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=11, fontweight='bold')

    # Add note
    note_text = ("Transaction counting tracks async operations:\n"
                 "- arrivals: mbarrier arrived but not completed\n"
                 "- completions: async operations finished")
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'mbarrier', 'data')
    output_path = os.path.join(output_dir, 'mbarrier_tx_count.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'tx_counting.csv')
    with open(csv_path, 'w') as f:
        f.write("name,ops_per_ms\n")
        for name, o in tx_count_data:
            f.write(f"{name},{o}\n")
    print(f"Saved: {csv_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating MBarrier Charts...")
    print("=" * 50)
    plot_mbarrier_operations()
    print()
    plot_fence_comparison()
    print()
    plot_pipeline_throughput()
    print()
    plot_transaction_counting()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/mbarrier/data/")
