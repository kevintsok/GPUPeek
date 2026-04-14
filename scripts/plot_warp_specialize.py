#!/usr/bin/env python3
"""
Warp Specialization Chart Generator
===================================
Generates charts for warp specialization benchmarks.

Usage:
    python plot_warp_specialize.py

Output:
    NVIDIA_GPU/sm_120/warp_specialize/data/warp_specialization_comparison.png
    NVIDIA_GPU/sm_120/warp_specialize/data/pipeline_stages.png
    NVIDIA_GPU/sm_120/warp_specialize/data/block_specialization.png
    NVIDIA_GPU/sm_120/warp_specialize/data/warp_primitives.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# D.1 Warp Specialization Basic (Producer/Consumer)
# =============================================================================

# Format: (name, bandwidth_gbps, time_ms)
warp_spec_data = [
    ("Standard Kernel", 920.0, 0.42),
    ("Barrier Copy", 880.0, 0.44),
    ("Warp Spec (prod/cons)", 850.0, 0.46),
    ("Overhead (%)", 0.0, 7.6),
]

# =============================================================================
# D.2 TMA + Barrier Synchronization
# =============================================================================

# Format: (name, bandwidth_gbps, speedup)
tma_barrier_data = [
    ("Standard Barrier Copy", 880.0, 1.0),
    ("TMA + Barrier Copy", 1050.0, 1.19),
]

# =============================================================================
# D.3 Multi-Stage Pipeline
# =============================================================================

# Format: (name, bandwidth_gbps, speedup_factor)
pipeline_data = [
    ("Baseline (simple copy)", 920.0, 1.0),
    ("3-Stage Pipeline", 780.0, 0.85),
    ("Overlapped Pipeline", 880.0, 0.96),
]

# =============================================================================
# D.4 Block Specialization
# =============================================================================

# Format: (name, bandwidth_gbps, relative_perf)
block_spec_data = [
    ("Simple Producer/Consumer", 720.0, 1.0),
    ("Block Spec (half prod/cons)", 680.0, 0.94),
    ("Warp Block Spec", 750.0, 1.04),
]

# =============================================================================
# D.5 Warp-Level Synchronization Primitives
# =============================================================================

# Format: (name, bandwidth_gbps)
warp_primitives_data = [
    ("Warp Shuffle Reduction", 1250.0),
    ("Warp Barrier (shuffle)", 1180.0),
    ("Warp Reduce + Barrier", 980.0),
    ("Warp Scan (prefix sum)", 1100.0),
]

# =============================================================================
# D.6 TMA + Warp Specialization Combined
# =============================================================================

# Format: (name, bandwidth_gbps, speedup)
tma_warp_spec_data = [
    ("Standard Copy", 920.0, 1.0),
    ("TMA + Warp Spec", 1150.0, 1.25),
]


def plot_warp_specialization_comparison():
    """Plot warp specialization vs standard kernel comparison."""
    names = [d[0] for d in warp_spec_data[:3]]
    bandwidths = [d[1] for d in warp_spec_data[:3]]
    times = [d[2] for d in warp_spec_data[:3]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth
    colors = ['steelblue', 'darkorange', 'seagreen']
    bars1 = ax1.bar(names, bandwidths, color=colors, alpha=0.8)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Bandwidth Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1100)

    for bar, bw in zip(bars1, bandwidths):
        height = bar.get_height()
        ax1.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Time per kernel
    bars2 = ax2.bar(names, times, color=colors, alpha=0.8)
    ax2.set_ylabel('Time (ms/kernel)', fontsize=12)
    ax2.set_title('Kernel Time Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, t in zip(bars2, times):
        height = bar.get_height()
        ax2.annotate(f'{t:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('D.1 Warp Specialization Basic - Producer/Consumer Pattern', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'warp_specialize', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'warp_specialization_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'warp_specialization.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps,time_ms\n")
        for name, bw, t in warp_spec_data[:3]:
            f.write(f"{name},{bw},{t}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_tma_barrier_comparison():
    """Plot TMA + Barrier synchronization comparison."""
    names = [d[0] for d in tma_barrier_data]
    bandwidths = [d[1] for d in tma_barrier_data]
    speedups = [d[2] for d in tma_barrier_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth
    colors = ['steelblue', 'seagreen']
    bars1 = ax1.bar(names, bandwidths, color=colors, alpha=0.8)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Bandwidth: Standard vs TMA', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1300)

    for bar, bw in zip(bars1, bandwidths):
        height = bar.get_height()
        ax1.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Speedup
    bars2 = ax2.bar(names, speedups, color=colors, alpha=0.8)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('TMA Speedup Factor', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_ylim(0, 1.5)

    for bar, su in zip(bars2, speedups):
        height = bar.get_height()
        ax2.annotate(f'{su:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('D.2 TMA + Barrier Synchronization (SM 9.0+)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'warp_specialize', 'data')
    output_path = os.path.join(output_dir, 'tma_barrier_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'tma_barrier.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps,speedup\n")
        for name, bw, su in tma_barrier_data:
            f.write(f"{name},{bw},{su}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_pipeline_stages():
    """Plot multi-stage pipeline comparison."""
    names = [d[0] for d in pipeline_data]
    values = [d[1] for d in pipeline_data]
    speedups = [d[2] for d in pipeline_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth
    colors = ['steelblue', 'darkorange', 'seagreen']
    bars1 = ax1.bar(names, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Pipeline Bandwidth', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1100)
    ax1.tick_params(axis='x', rotation=15)

    for bar, v in zip(bars1, values):
        height = bar.get_height()
        ax1.annotate(f'{v:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Speedup relative to baseline
    bars2 = ax2.bar(names, speedups, color=colors, alpha=0.8)
    ax2.set_ylabel('Relative to Baseline', fontsize=12)
    ax2.set_title('Pipeline Efficiency', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.tick_params(axis='x', rotation=15)

    for bar, su in zip(bars2, speedups):
        height = bar.get_height()
        ax2.annotate(f'{su:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('D.3 Multi-Stage Pipeline (Load/Compute/Store)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'warp_specialize', 'data')
    output_path = os.path.join(output_dir, 'pipeline_stages.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'pipeline_stages.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps,speedup\n")
        for name, bw, su in pipeline_data:
            f.write(f"{name},{bw},{su}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_block_specialization():
    """Plot block specialization comparison."""
    names = [d[0] for d in block_spec_data]
    values = [d[1] for d in block_spec_data]
    relative = [d[2] for d in block_spec_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth
    colors = ['steelblue', 'darkorange', 'seagreen']
    bars1 = ax1.bar(names, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Block Specialization Bandwidth', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 900)
    ax1.tick_params(axis='x', rotation=15)

    for bar, v in zip(bars1, values):
        height = bar.get_height()
        ax1.annotate(f'{v:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Relative performance
    bars2 = ax2.bar(names, relative, color=colors, alpha=0.8)
    ax2.set_ylabel('Relative Performance', fontsize=12)
    ax2.set_title('Block Spec Relative to Simple', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=15)

    for bar, r in zip(bars2, relative):
        height = bar.get_height()
        ax2.annotate(f'{r:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('D.4 Block Specialization (Half Block = Producer, Half = Consumer)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'warp_specialize', 'data')
    output_path = os.path.join(output_dir, 'block_specialization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'block_specialization.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps,relative\n")
        for name, bw, rel in block_spec_data:
            f.write(f"{name},{bw},{rel}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_warp_primitives():
    """Plot warp-level synchronization primitives comparison."""
    names = [d[0] for d in warp_primitives_data]
    bandwidths = [d[1] for d in warp_primitives_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']
    bars = ax.barh(names, bandwidths, color=colors, alpha=0.8)

    ax.set_xlabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('D.5 Warp-Level Synchronization Primitives - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1500)

    # Add value labels
    for bar, bw in zip(bars, bandwidths):
        width = bar.get_width()
        ax.annotate(f'{bw:.0f}',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=11, fontweight='bold')

    # Add note about primitives
    note_text = ("Note: Warp-level primitives (__shfl, __any, __all, __ballot)\n"
                 "do not require __syncthreads() as all threads in a warp\n"
                 "execute synchronously.")
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'warp_specialize', 'data')
    output_path = os.path.join(output_dir, 'warp_primitives.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'warp_primitives.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps\n")
        for name, bw in warp_primitives_data:
            f.write(f"{name},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_tma_warp_specialization():
    """Plot TMA + Warp Specialization combined."""
    names = [d[0] for d in tma_warp_spec_data]
    bandwidths = [d[1] for d in tma_warp_spec_data]
    speedups = [d[2] for d in tma_warp_spec_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth
    colors = ['steelblue', 'seagreen']
    bars1 = ax1.bar(names, bandwidths, color=colors, alpha=0.8)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Bandwidth Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1400)

    for bar, bw in zip(bars1, bandwidths):
        height = bar.get_height()
        ax1.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Speedup
    bars2 = ax2.bar(names, speedups, color=colors, alpha=0.8)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('TMA + Warp Spec Speedup', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1.5)

    for bar, su in zip(bars2, speedups):
        height = bar.get_height()
        ax2.annotate(f'{su:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('D.6 TMA + Warp Specialization Combined (SM 9.0+)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'warp_specialize', 'data')
    output_path = os.path.join(output_dir, 'tma_warp_specialization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'tma_warp_specialization.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps,speedup\n")
        for name, bw, su in tma_warp_spec_data:
            f.write(f"{name},{bw},{su}\n")
    print(f"Saved: {csv_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating Warp Specialization Charts...")
    print("=" * 50)
    plot_warp_specialization_comparison()
    print()
    plot_tma_barrier_comparison()
    print()
    plot_pipeline_stages()
    print()
    plot_block_specialization()
    print()
    plot_warp_primitives()
    print()
    plot_tma_warp_specialization()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/warp_specialize/data/")
