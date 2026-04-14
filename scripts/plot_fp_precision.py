#!/usr/bin/env python3
"""
FP8/FP4/FP6 Precision Chart Generator
======================================
Generates charts for low-precision format research.

Usage:
    python plot_fp_precision.py

Output:
    NVIDIA_GPU/sm_120/fp8/data/fp8_comparison.png
    NVIDIA_GPU/sm_120/fp8/data/memory_reduction.png
    NVIDIA_GPU/sm_120/fp4_fp6/data/fp4_fp6_comparison.png
    NVIDIA_GPU/sm_120/fp4_fp6/data/precision_tradeoff.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# FP8 Format Comparison
# Format: (format, bits, relative_bandwidth, use_case)
fp8_data = [
    ("FP32", 32, 1.0, "Baseline"),
    ("FP16", 16, 2.0, "High precision"),
    ("FP8\nE4M3", 8, 4.0, "Weights+Activations"),
    ("FP8\nE5M2", 8, 4.0, "Gradients/High DR"),
]

# FP8 vs FP16 Performance (TFLOPS scaling)
# Format: (format, tflops_relative)
fp8_tflops = [
    ("FP32\n(Baseline)", 1.0),
    ("TF32", 8.0),
    ("FP16\n(Tensor)", 16.0),
    ("FP8\nE4M3", 32.0),
    ("FP8\nE5M2", 32.0),
]

# FP4/FP6 Format Comparison
# Format: (format, bits, memory_reduction, relative_tflops)
fp4_fp6_data = [
    ("FP32", 32, 1.0, 1.0),
    ("FP16", 16, 2.0, 16.0),
    ("FP8", 8, 4.0, 32.0),
    ("FP6\ne2m3", 6, 5.3, 40.0),
    ("FP6\ne3m2", 6, 5.3, 38.0),
    ("FP4\ne2m1", 4, 8.0, 48.0),
]

# Memory Reduction Factor (vs FP16)
# Format: (format, reduction_factor, description)
memory_reduction = [
    ("FP32", 0.5, "Baseline"),
    ("FP16", 1.0, "Reference"),
    ("FP8", 2.0, "4x vs FP32"),
    ("FP6\ne2m3", 2.67, "4x vs FP32"),
    ("FP6\ne3m2", 2.67, "4x vs FP32"),
    ("FP4\ne2m1", 4.0, "8x vs FP32"),
]

# Precision Score (relative, higher = more precision)
# Format: (format, precision_score)
precision_scores = [
    ("FP32", 1.00),
    ("FP16", 0.50),
    ("FP8\nE4M3", 0.25),
    ("FP8\nE5M2", 0.20),
    ("FP6\ne2m3", 0.15),
    ("FP6\ne3m2", 0.12),
    ("FP4\ne2m1", 0.08),
]


def plot_fp8_comparison():
    """Plot FP8 format comparison."""
    formats = [d[0] for d in fp8_data]
    bits = [d[1] for d in fp8_data]
    bandwidth = [d[2] for d in fp8_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bits comparison
    colors = ['steelblue', 'steelblue', 'darkorange', 'green']
    bars1 = ax1.bar(formats, bits, color=colors, alpha=0.8)
    ax1.set_ylabel('Bits', fontsize=12)
    ax1.set_title('FP8 Format - Bit Width', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, b in zip(bars1, bits):
        height = bar.get_height()
        ax1.annotate(f'{b}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Bandwidth improvement
    bars2 = ax2.bar(formats, bandwidth, color=colors, alpha=0.8)
    ax2.set_ylabel('Relative Bandwidth', fontsize=12)
    ax2.set_title('FP8 Format - Memory Bandwidth Gain', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='4x vs FP32')
    ax2.legend()

    for bar, bw in zip(bars2, bandwidth):
        height = bar.get_height()
        ax2.annotate(f'{bw:.1f}x',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'fp8', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fp8_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'fp8_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("format,bits,relative_bandwidth,use_case\n")
        for fmt, b, bw, uc in fp8_data:
            f.write(f"{fmt.replace(chr(10), ' ')},{b},{bw},{uc}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_memory_reduction():
    """Plot memory reduction factor across all precisions."""
    formats = [d[0] for d in memory_reduction]
    reduction = [d[1] for d in memory_reduction]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'steelblue', 'darkorange', 'darkorange', 'darkorange', 'red']
    bars = ax.bar(formats, reduction, color=colors, alpha=0.8)

    ax.set_xlabel('Precision Format', fontsize=12)
    ax.set_ylabel('Memory Reduction Factor (vs FP32)', fontsize=12)
    ax.set_title('Memory Reduction by Precision Format - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, r in zip(bars, reduction):
        height = bar.get_height()
        ax.annotate(f'{r:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Annotation for FP8
    ax.annotate('FP8: 4x reduction\nvs FP32',
               xy=(2, 4.0), xytext=(2.3, 5),
               fontsize=10,
               arrowprops=dict(arrowstyle='->', color='darkorange'),
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Annotation for FP4
    ax.annotate('FP4: Max reduction\n8x vs FP32',
               xy=(5, 8.0), xytext=(4.2, 7),
               fontsize=10,
               arrowprops=dict(arrowstyle='->', color='red'),
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'fp8', 'data')
    output_path = os.path.join(output_dir, 'memory_reduction.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'memory_reduction.csv')
    with open(csv_path, 'w') as f:
        f.write("format,reduction_factor,description\n")
        for fmt, r, desc in memory_reduction:
            f.write(f"{fmt.replace(chr(10), ' ')},{r},{desc}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_fp4_fp6_comparison():
    """Plot FP4/FP6 format comparison."""
    formats = [d[0] for d in fp4_fp6_data]
    bits = [d[1] for d in fp4_fp6_data]
    tflops_rel = [d[3] for d in fp4_fp6_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bits comparison
    colors = ['steelblue', 'steelblue', 'steelblue', 'darkorange', 'orange', 'red']
    bars1 = ax1.bar(formats, bits, color=colors, alpha=0.8)
    ax1.set_ylabel('Bits', fontsize=12)
    ax1.set_title('FP4/FP6 Format - Bit Width', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, b in zip(bars1, bits):
        height = bar.get_height()
        ax1.annotate(f'{b}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Relative TFLOPS
    bars2 = ax2.bar(formats, tflops_rel, color=colors, alpha=0.8)
    ax2.set_ylabel('Relative TFLOPS', fontsize=12)
    ax2.set_title('FP4/FP6 - Relative Compute Throughput', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, t in zip(bars2, tflops_rel):
        height = bar.get_height()
        ax2.annotate(f'{t:.0f}x',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'fp4_fp6', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fp4_fp6_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'fp4_fp6_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("format,bits,memory_reduction,relative_tflops\n")
        for fmt, b, mr, rt in fp4_fp6_data:
            f.write(f"{fmt.replace(chr(10), ' ')},{b},{mr},{rt}\n")
    print(f"Saved: {csv_path}")

    plt.close()


def plot_precision_tradeoff():
    """Plot precision vs memory reduction tradeoff."""
    formats = [d[0] for d in precision_scores]
    precision = [d[1] for d in precision_scores]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['steelblue', 'steelblue', 'darkorange', 'green', 'orange', 'orange', 'red']
    bars = ax.bar(formats, precision, color=colors, alpha=0.8)

    ax.set_xlabel('Precision Format', fontsize=12)
    ax.set_ylabel('Relative Precision Score', fontsize=12)
    ax.set_title('Precision vs Memory Reduction Tradeoff - RTX 5080 (SM 12.0)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, p in zip(bars, precision):
        height = bar.get_height()
        ax.annotate(f'{p:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # FP8 zone
    ax.axvspan(2.5, 3.5, alpha=0.2, color='orange', label='FP8 Zone (4x reduction)')
    ax.legend()

    # FP4 zone
    ax.axvspan(5.5, 6.5, alpha=0.2, color='red', label='FP4 Zone (8x reduction)')
    ax.legend()

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'NVIDIA_GPU', 'sm_120', 'fp4_fp6', 'data')
    output_path = os.path.join(output_dir, 'precision_tradeoff.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'precision_tradeoff.csv')
    with open(csv_path, 'w') as f:
        f.write("format,precision_score\n")
        for fmt, p in precision_scores:
            f.write(f"{fmt.replace(chr(10), ' ')},{p}\n")
    print(f"Saved: {csv_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating FP8/FP4/FP6 Precision Charts...")
    print("=" * 50)
    plot_fp8_comparison()
    print()
    plot_memory_reduction()
    print()
    plot_fp4_fp6_comparison()
    print()
    plot_precision_tradeoff()
    print()
    print("Done! Charts saved to NVIDIA_GPU/sm_120/{fp8,fp4_fp6}/data/")
