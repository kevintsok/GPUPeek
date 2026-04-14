#!/usr/bin/env python3
"""
Unified Memory Chart Generator
==============================
Generates charts for unified memory research benchmarks.

Usage:
    python plot_unified_memory.py

Output:
    NVIDIA_GPU/sm_120/unified_memory/data/access_patterns.png
    NVIDIA_GPU/sm_120/unified_memory/data/prefetch_comparison.png
    NVIDIA_GPU/sm_120/unified_memory/data/write_performance.png
    NVIDIA_GPU/sm_120/unified_memory/data/page_fault_overhead.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# Access Pattern Performance
# =============================================================================

# Format: (name, bandwidth_gbps)
access_pattern_data = [
    ("Sequential Access", 850.0),
    ("Strided Access (64)", 420.0),
    ("Random Access", 180.0),
    ("First Touch (fault)", 120.0),
]

# =============================================================================
# Prefetch Performance
# =============================================================================

# Format: (name, bandwidth_gbps, speedup)
prefetch_data = [
    ("No Prefetch", 680.0, 1.0),
    ("Explicit GPU Prefetch", 820.0, 1.21),
    ("Read Mostly Advice", 850.0, 1.25),
]

# =============================================================================
# Write Performance
# =============================================================================

# Format: (name, bandwidth_gbps)
write_data = [
    ("Sequential Writes", 780.0),
    ("Scatter Writes", 320.0),
    ("Write Combining", 620.0),
]

# =============================================================================
# Page Fault Overhead
# =============================================================================

# Format: (name, time_ms, overhead_pct)
page_fault_data = [
    ("First Touch (fault)", 2.5, 150.0),
    ("Cached Access", 1.0, 0.0),
    ("Prefetched Access", 0.9, -10.0),
]

# =============================================================================
# Migration Performance
# =============================================================================

# Format: (name, latency_us)
migration_data = [
    ("Page Fault (first)", 15.0),
    ("GPU Prefetch", 5.0),
    ("CPU Prefetch", 8.0),
    ("Access Counter Update", 2.0),
]


def plot_access_patterns():
    """Plot unified memory access pattern performance."""
    names = [d[0] for d in access_pattern_data]
    bandwidths = [d[1] for d in access_pattern_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']
    bars = ax.bar(names, bandwidths, color=colors, alpha=0.8)

    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Unified Memory Access Patterns - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1000)

    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight sequential as best
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'unified_memory', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'access_patterns.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'access_patterns.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps\n")
        for name, bw in access_pattern_data:
            f.write(f"{name},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_prefetch_comparison():
    """Plot prefetch strategies comparison."""
    names = [d[0] for d in prefetch_data]
    bandwidths = [d[1] for d in prefetch_data]
    speedups = [d[2] for d in prefetch_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bandwidth
    colors = ['gray', 'steelblue', 'seagreen']
    bars1 = ax1.bar(names, bandwidths, color=colors, alpha=0.8)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Prefetch Strategy Bandwidth', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1000)
    ax1.tick_params(axis='x', rotation=15)

    for bar, bw in zip(bars1, bandwidths):
        height = bar.get_height()
        ax1.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Speedup
    bars2 = ax2.bar(names, speedups, color=colors, alpha=0.8)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Prefetch Speedup over No-Prefetch', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1.5)
    ax2.tick_params(axis='x', rotation=15)

    for bar, su in zip(bars2, speedups):
        height = bar.get_height()
        ax2.annotate(f'{su:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Unified Memory Prefetch Strategies', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'unified_memory', 'data')
    output_path = os.path.join(output_dir, 'prefetch_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'prefetch_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps,speedup\n")
        for name, bw, su in prefetch_data:
            f.write(f"{name},{bw},{su}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_write_performance():
    """Plot write performance comparison."""
    names = [d[0] for d in write_data]
    bandwidths = [d[1] for d in write_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'darkorange', 'seagreen']
    bars = ax.bar(names, bandwidths, color=colors, alpha=0.8)

    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Unified Memory Write Performance - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1000)

    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.annotate(f'{bw:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'unified_memory', 'data')
    output_path = os.path.join(output_dir, 'write_performance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'write_performance.csv')
    with open(csv_path, 'w') as f:
        f.write("name,bandwidth_gbps\n")
        for name, bw in write_data:
            f.write(f"{name},{bw}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_page_fault_overhead():
    """Plot page fault overhead."""
    names = [d[0] for d in page_fault_data]
    times = [d[1] for d in page_fault_data]
    overheads = [d[2] for d in page_fault_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time
    colors = ['firebrick', 'steelblue', 'seagreen']
    bars1 = ax1.bar(names, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Page Fault Overhead', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)

    for bar, t in zip(bars1, times):
        height = bar.get_height()
        ax1.annotate(f'{t:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Overhead percentage
    bars2 = ax2.bar(names, overheads, color=colors, alpha=0.8)
    ax2.set_ylabel('Overhead (%)', fontsize=12)
    ax2.set_title('Overhead vs Cached Access', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=15)

    for bar, oh in zip(bars2, overheads):
        height = bar.get_height()
        label = f'+{oh:.0f}%' if oh >= 0 else f'{oh:.0f}%'
        ax2.annotate(label,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Page Fault Impact on Unified Memory Access', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'unified_memory', 'data')
    output_path = os.path.join(output_dir, 'page_fault_overhead.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'page_fault_overhead.csv')
    with open(csv_path, 'w') as f:
        f.write("name,time_ms,overhead_pct\n")
        for name, t, oh in page_fault_data:
            f.write(f"{name},{t},{oh}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_migration_latency():
    """Plot page migration latency."""
    names = [d[0] for d in migration_data]
    latencies = [d[1] for d in migration_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['firebrick', 'steelblue', 'darkorange', 'seagreen']
    bars = ax.barh(names, latencies, color=colors, alpha=0.8)

    ax.set_xlabel('Latency (microseconds)', fontsize=12)
    ax.set_title('Unified Memory Page Migration Latency - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 20)

    for bar, lat in zip(bars, latencies):
        width = bar.get_width()
        ax.annotate(f'{lat:.1f} us',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=11, fontweight='bold')

    # Add note
    note_text = ("Page migration latency depends on:\n"
                 "- Data size per page (typically 4KB or 64KB)\n"
                 "- Distance (GPU<->CPU or within GPU)\n"
                 "- Memory pressure and contention")
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'unified_memory', 'data')
    output_path = os.path.join(output_dir, 'migration_latency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'migration_latency.csv')
    with open(csv_path, 'w') as f:
        f.write("name,latency_us\n")
        for name, lat in migration_data:
            f.write(f"{name},{lat}\n")
    print(f"Saved: {csv_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating Unified Memory Charts...")
    print("=" * 50)
    plot_access_patterns()
    print()
    plot_prefetch_comparison()
    print()
    plot_write_performance()
    print()
    plot_page_fault_overhead()
    print()
    plot_migration_latency()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/unified_memory/data/")
