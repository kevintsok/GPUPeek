#!/usr/bin/env python3
"""
CUDA Core Compute Throughput Chart Generator
==========================================
Generates throughput comparison charts for CUDA core compute benchmarks.

Usage:
    python plot_cuda_core_throughput.py

Output:
    NVIDIA_GPU/sm_120/cuda_core/data/throughput_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data Type Throughput Comparison (RTX 5080, SM 12.0)
# Format: (dtype, throughput_gflops, latency_ms)
compute_throughput_data = [
    ("FP32", 88.0, 0.068),
    ("FP64", 12.0, 0.350),
    ("FP16", 204.0, 0.021),
    ("BF16", 180.0, 0.025),
    ("INT32", 121.0, 0.035),
    ("INT8", 240.0, 0.015),
]

# Instruction Latency Comparison
# Format: (instruction, latency_cycles)
instruction_latency_data = [
    ("FP32 FMA", 4),
    ("FP64 FMA", 16),
    ("FP16 FMA", 2),
    ("INT32 ADD", 4),
    ("INT32 MUL", 4),
    ("SHFL", 1),
]

# Vector Instruction Throughput (relative to scalar)
# Format: (vector_type, relative_throughput)
vector_throughput_data = [
    ("Scalar (float)", 1.0),
    ("float2", 2.0),
    ("float4", 4.0),
    ("double2", 1.8),
]

def plot_dtype_throughput():
    """Plot data type throughput comparison."""
    dtypes = [d[0] for d in compute_throughput_data]
    throughputs = [d[1] for d in compute_throughput_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown']
    bars = ax.bar(dtypes, throughputs, color=colors, alpha=0.8)

    ax.set_xlabel('Data Type', fontsize=12)
    ax.set_ylabel('Throughput (GFLOPS)', fontsize=12)
    ax.set_title('CUDA Core Compute Throughput by Data Type - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, tput in zip(bars, throughputs):
        height = bar.get_height()
        ax.annotate(f'{tput:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add theoretical peak line
    ax.axhline(y=1024, color='gray', linestyle='--', alpha=0.5, label='FP32 Peak (est.)')

    ax.legend()

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cuda_core', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dtype_throughput_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'compute_throughput.csv')
    with open(csv_path, 'w') as f:
        f.write("dtype,throughput_gflops,latency_ms\n")
        for dtype, tput, lat in compute_throughput_data:
            f.write(f"{dtype},{tput},{lat}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_instruction_latency():
    """Plot instruction latency comparison."""
    instructions = [d[0] for d in instruction_latency_data]
    latencies = [d[1] for d in instruction_latency_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'darkorange', 'firebrick', 'seagreen', 'purple', 'goldenrod']
    bars = ax.barh(instructions, latencies, color=colors, alpha=0.8)

    ax.set_xlabel('Latency (cycles)', fontsize=12)
    ax.set_ylabel('Instruction', fontsize=12)
    ax.set_title('Instruction Latency Comparison - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, lat in zip(bars, latencies):
        width = bar.get_width()
        ax.annotate(f'{lat} cyc',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=10)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cuda_core', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'instruction_latency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'instruction_latency.csv')
    with open(csv_path, 'w') as f:
        f.write("instruction,latency_cycles\n")
        for instr, lat in instruction_latency_data:
            f.write(f"{instr},{lat}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_vector_throughput():
    """Plot vector instruction throughput."""
    vector_types = [d[0] for d in vector_throughput_data]
    throughputs = [d[1] for d in vector_throughput_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['blue', 'green', 'orange', 'red']
    bars = ax.bar(vector_types, throughputs, color=colors, alpha=0.8)

    ax.set_xlabel('Vector Type', fontsize=12)
    ax.set_ylabel('Relative Throughput', fontsize=12)
    ax.set_title('Vector Instruction Throughput - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, tput in zip(bars, throughputs):
        height = bar.get_height()
        ax.annotate(f'{tput:.1f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cuda_core', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'vector_throughput.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'vector_throughput.csv')
    with open(csv_path, 'w') as f:
        f.write("vector_type,relative_throughput\n")
        for vtype, tput in vector_throughput_data:
            f.write(f"{vtype},{tput}\n")
    print(f"Saved: {csv_path}")

    plt.close()

def plot_throughput_efficiency():
    """Plot throughput efficiency (TFLOPS vs theoretical peak)."""
    dtypes = [d[0] for d in compute_throughput_data]
    throughputs = [d[1] for d in compute_throughput_data]

    # Theoretical peaks (approximate for RTX 5080)
    peaks = {
        'FP32': 1024,      # ~1 TFLOPS FP32
        'FP64': 32,        # ~32 GFLOPS FP64
        'FP16': 2048,      # ~2 TFLOPS FP16 (tensor)
        'BF16': 2048,      # ~2 TFLOPS BF16
        'INT32': 512,      # ~512 GIOPS
        'INT8': 2048,      # ~2 TOPS INT8
    }

    efficiencies = [throughputs[i] / peaks.get(dtypes[i], 1) * 100
                   for i in range(len(dtypes))]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown']
    bars = ax.bar(dtypes, efficiencies, color=colors, alpha=0.8)

    ax.set_xlabel('Data Type', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Compute Efficiency by Data Type - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # Add value labels
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.annotate(f'{eff:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'cuda_core', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'compute_efficiency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()

if __name__ == '__main__':
    print("Generating CUDA Core Throughput Charts...")
    print("=" * 50)
    plot_dtype_throughput()
    print()
    plot_instruction_latency()
    print()
    plot_vector_throughput()
    print()
    plot_throughput_efficiency()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/cuda_core/data/")
