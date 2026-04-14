#!/usr/bin/env python3
"""
Tensor Core / WMMA Chart Generator
==================================
Generates charts for Tensor Core and WMMA benchmarks.

Usage:
    python plot_tensor_core.py

Output:
    NVIDIA_GPU/sm_120/wmma/data/tensor_dtype_comparison.png
    NVIDIA_GPU/sm_120/wmma/data/wmma_throughput.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Tensor Core Data Type Performance (RTX 5080, SM 12.0)
# Format: (dtype, tflops_measured, tflops_theoretical)
tensor_dtype_data = [
    ("FP16", 89.0, 2048.0),
    ("BF16", 82.0, 2048.0),
    ("FP32\n(TF32)", 39.0, 512.0),
    ("FP64", 4.5, 32.0),
    ("INT8", 178.0, 2048.0),
    ("INT4", 356.0, 4096.0),
]

# WMMA Operation Latency
# Format: (operation, latency_cycles)
wmma_latency_data = [
    ("load_matrix\n(A)", 4),
    ("load_matrix\n(B)", 4),
    ("mma_sync", 8),
    ("store_matrix", 4),
]

# WMMA vs CUDA Core Throughput
# Format: (mode, throughput_tflops)
throughput_comparison = [
    ("FP32 CUDA\n(1 Thread)", 0.088),
    ("FP16 WMMA\n(1 Warp)", 89.0),
    ("FP16 WMMA\n(Theoretical Peak)", 2048.0),
]

# Memory Footprint per MMA
# Format: (operation, bytes)
memory_footprint = [
    ("Load A (FP16)", 512),
    ("Load B (FP16)", 512),
    ("MMA Compute", 0),
    ("Store D (FP32)", 1024),
]

def plot_tensor_dtype_comparison():
    """Plot Tensor Core performance by data type."""
    dtypes = [d[0] for d in tensor_dtype_data]
    measured = [d[1] for d in tensor_dtype_data]
    theoretical = [d[2] for d in tensor_dtype_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(dtypes))
    width = 0.35

    bars1 = ax.bar(x - width/2, measured, width, label='Measured', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, theoretical, width, label='Theoretical Peak', color='lightcoral', alpha=0.8)

    ax.set_xlabel('Data Type', fontsize=12)
    ax.set_ylabel('TFLOPS', fontsize=12)
    ax.set_title('Tensor Core Performance by Data Type - RTX 5080 (SM 12.0)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dtypes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, measured):
        height = bar.get_height()
        if val > 1:
            ax.annotate(f'{val:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, theoretical):
        height = bar.get_height()
        if val > 1:
            ax.annotate(f'{val:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, color='red')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'wmma', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tensor_dtype_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'tensor_dtype.csv')
    with open(csv_path, 'w') as f:
        f.write("dtype,measured_tflops,theoretical_tflops\n")
        for dtype, meas, theo in tensor_dtype_data:
            f.write(f"{dtype.replace(chr(10), ' ')},{meas},{theo}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_wmma_latency():
    """Plot WMMA operation latency breakdown."""
    operations = [d[0] for d in wmma_latency_data]
    cycles = [d[1] for d in wmma_latency_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'steelblue', 'darkorange', 'steelblue']
    bars = ax.barh(operations, cycles, color=colors, alpha=0.8)

    ax.set_xlabel('Latency (cycles)', fontsize=12)
    ax.set_ylabel('WMMA Operation', fontsize=12)
    ax.set_title('WMMA Operation Latency Breakdown - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, cyc in zip(bars, cycles):
        width = bar.get_width()
        ax.annotate(f'{cyc} cyc',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=11, fontweight='bold')

    # Total latency annotation
    total = sum(cycles)
    ax.annotate(f'Total: {total} cycles',
               xy=(max(cycles), len(cycles) - 0.5),
               xytext=(max(cycles) + 2, len(cycles) - 0.5),
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'wmma', 'data')
    output_path = os.path.join(output_dir, 'wmma_latency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'wmma_latency.csv')
    with open(csv_path, 'w') as f:
        f.write("operation,latency_cycles\n")
        for op, cyc in wmma_latency_data:
            f.write(f"{op.replace(chr(10), ' ')},{cyc}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_throughput_comparison():
    """Plot WMMA vs CUDA Core throughput."""
    modes = [d[0] for d in throughput_comparison]
    throughputs = [d[1] for d in throughput_comparison]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'darkorange', 'red']
    bars = ax.bar(modes, throughputs, color=colors, alpha=0.8)

    ax.set_xlabel('Execution Mode', fontsize=12)
    ax.set_ylabel('Throughput (TFLOPS)', fontsize=12)
    ax.set_title('WMMA vs CUDA Core Throughput - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, tput in zip(bars, throughputs):
        height = bar.get_height()
        ax.annotate(f'{tput:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Speedup annotation
    speedup = throughputs[1] / throughputs[0]
    ax.annotate(f'WMMA is\n~{speedup:.0f}x faster',
               xy=(1, throughputs[1]), xytext=(0.5, throughputs[1] * 0.7),
               fontsize=11,
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'wmma', 'data')
    output_path = os.path.join(output_dir, 'throughput_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'throughput_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("mode,throughput_tflops\n")
        for mode, tput in throughput_comparison:
            f.write(f"{mode.replace(chr(10), ' ')},{tput}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_memory_footprint():
    """Plot memory footprint per MMA operation."""
    operations = [d[0] for d in memory_footprint]
    bytes_vals = [d[1] for d in memory_footprint]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'steelblue', 'lightgray', 'darkorange']
    bars = ax.bar(operations, bytes_vals, color=colors, alpha=0.8)

    ax.set_xlabel('Operation', fontsize=12)
    ax.set_ylabel('Memory Bytes', fontsize=12)
    ax.set_title('WMMA Memory Footprint per Iteration - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, bytes_vals):
        height = bar.get_height()
        if val > 0:
            ax.annotate(f'{val}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Total annotation
    total = sum(bytes_vals)
    ax.annotate(f'Total: {total} bytes\nper warp MMA',
               xy=(2, 800), fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'wmma', 'data')
    output_path = os.path.join(output_dir, 'memory_footprint.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'memory_footprint.csv')
    with open(csv_path, 'w') as f:
        f.write("operation,bytes\n")
        for op, val in memory_footprint:
            f.write(f"{op},{val}\n")
    print(f"Saved: {csv_path}")

    plt.close()

if __name__ == '__main__':
    print("Generating Tensor Core / WMMA Charts...")
    print("=" * 50)
    plot_tensor_dtype_comparison()
    print()
    plot_wmma_latency()
    print()
    plot_throughput_comparison()
    print()
    plot_memory_footprint()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/wmma/data/")
