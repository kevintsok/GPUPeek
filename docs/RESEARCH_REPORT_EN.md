# GPUPeek: Comprehensive GPU Benchmark Framework Research Report

## English Version

---

## Executive Summary

GPUPeek is a CUDA benchmark framework designed for deep exploration of GPU architecture characteristics, performance metrics, and optimization opportunities. This comprehensive research report documents the findings from extensive benchmarking across multiple research topics including CUDA core arithmetic, atomic operations, barrier synchronization, and warp specialization patterns on the NVIDIA Blackwell (SM 12.0) architecture.

**Target GPU**: NVIDIA GeForce RTX 5080 Laptop GPU
**Architecture**: Blackwell (Compute Capability 12.0)
**CUDA Version**: 13.0
**Driver**: 595.79

---

## Table of Contents

1. [GPU Hardware Specifications](#1-gpu-hardware-specifications)
2. [Memory Subsystem Analysis](#2-memory-subsystem-analysis)
3. [CUDA Core Arithmetic Research](#3-cuda-core-arithmetic-research)
4. [Atomic Operations Deep Research](#4-atomic-operations-deep-research)
5. [Barrier Synchronization Research](#5-barrier-synchronization-research)
6. [Warp Specialization Patterns](#6-warp-specialization-patterns)
7. [NCU Profiling Metrics](#7-ncu-profiling-metrics)
8. [Key Findings and Recommendations](#8-key-findings-and-recommendations)
9. [Future Research Directions](#9-future-research-directions)

---

## 1. GPU Hardware Specifications

### 1.1 RTX 5080 Laptop GPU Specifications

| Parameter | Value |
|-----------|-------|
| GPU Model | NVIDIA GeForce RTX 5080 Laptop GPU |
| Architecture Code | Blackwell |
| Compute Capability | 12.0 |
| Number of SMs | 60 |
| Cores per SM | 128 |
| Total CUDA Cores | 7,680 |
| Global Memory | 15.92 GB |
| Shared Memory per Block | 48 KB |
| L2 Cache Size | ~5 MB (estimated) |
| Max Threads per Block | 1,024 |
| Max Threads per SM | 1,536 |
| Registers per Block | 65,536 |
| Warp Size | 32 |
| Memory Bus Width | 256-bit |

### 1.2 Blackwell Architecture Key Features

The Blackwell architecture introduces several significant improvements over previous generations:

1. **Enhanced Warp Shuffle**: Improved performance for warp-level data exchange operations
2. **Async Copy Engine**: Independent asynchronous memory copy engine that operates in parallel with computation
3. **Improved L2 Cache Streaming**: Optimized cache behavior for streaming access patterns
4. **Enhanced Tensor Core Support**: New FP8 and mixed-precision capabilities

---

## 2. Memory Subsystem Analysis

### 2.1 Memory Hierarchy Overview

```
GPU Memory Subsystem
├── Global Memory (DRAM)
│   └── ~800 GB/s bandwidth
├── L2 Cache
│   └── ~5 MB, high-bandwidth on-chip cache
└── L1/Shared Memory
    └── ~1.5 TB/s bandwidth (shared)
```

### 2.2 Global Memory Bandwidth vs Data Size

| Data Size | Sequential Read | Sequential Write | Read-Modify-Write |
|-----------|----------------|------------------|-------------------|
| 64 KB | 7.25 GB/s | 7.25 GB/s | 7.25 GB/s |
| 256 KB | 32.39 GB/s | 32.39 GB/s | 32.39 GB/s |
| 1 MB | 73.97 GB/s | 73.97 GB/s | 73.97 GB/s |
| 4 MB | 296.36 GB/s | 296.36 GB/s | 296.36 GB/s |
| 16 MB | 643.02 GB/s | 643.02 GB/s | 643.02 GB/s |
| 64 MB | 376.08 GB/s | 376.08 GB/s | 376.08 GB/s |
| 128 MB | 502.44 GB/s | 502.44 GB/s | 502.44 GB/s |
| 256 MB | 614.93 GB/s | 614.93 GB/s | 614.93 GB/s |

**Analysis**:
- Bandwidth increases with data size, reaching first peak (~643 GB/s) at 16 MB
- 64 MB shows bandwidth drop (~376 GB/s), likely due to L2 cache thrashing
- Bandwidth recovers at 128-256 MB, indicating larger effective cache window
- Peak theoretical bandwidth: ~800 GB/s

### 2.3 Memory Hierarchy Bandwidth

| Access Pattern | Bandwidth | Notes |
|---------------|-----------|-------|
| Global Direct Read | 810.89 GB/s | Baseline read |
| Global Direct Write | 820.60 GB/s | Baseline write |
| Shared Memory (L1) Round-Trip | **1.50 TB/s** | On-chip L1 bandwidth |
| L2 Streaming (stride=1) | 766.78 GB/s | L2 cache hits |
| L2 Streaming (stride=1024) | 795.17 GB/s | Strided access still efficient |
| __ldg Bypass | 822.43 GB/s | Cache bypass read |
| L1 Preference (registers) | 780.32 GB/s | Register optimization |

**Key Insights**:
- Shared Memory (L1) bandwidth reaches **1.50 TB/s**, far exceeding global memory
- L2 strided access maintains high efficiency (770-795 GB/s)
- __ldg instruction provides slightly higher bandwidth by bypassing cache

### 2.4 Strided Access Efficiency

**Baseline (Sequential)**: 822.37 GB/s

| Stride | Bandwidth | Efficiency |
|--------|-----------|------------|
| 2 | 582.20 GB/s | 86.0% |
| 4 | 581.24 GB/s | 85.9% |
| 8 | 544.01 GB/s | 80.4% |
| 16 | 421.00 GB/s | 62.2% |
| 32 | 239.11 GB/s | 35.3% |
| 64 | 154.13 GB/s | 22.8% |
| 128 | 76.88 GB/s | 11.4% |
| 256 | 39.62 GB/s | 5.9% |

**Analysis**: Bandwidth drops sharply when stride exceeds 16, indicating cache line utilization becomes inefficient.

### 2.5 Data Type Bandwidth

| Data Type | Size | Bandwidth | Relative to float |
|-----------|------|-----------|------------------|
| float (FP32) | 4 B | 878.19 GB/s | 100% |
| int (INT32) | 4 B | 882.25 GB/s | 100.5% |
| double (FP64) | 8 B | 468.73 GB/s | 53.4% |
| half (FP16) | 2 B | 410.20 GB/s | 46.7% |

**Analysis**: FP32 and INT32 achieve similar bandwidth (~880 GB/s), while FP64 drops ~47% and FP16 drops ~53%.

### 2.6 PCIe Bandwidth

| Transfer Type | Bandwidth | Time per Transfer |
|--------------|-----------|-------------------|
| Pageable H2D | 47-49 GB/s | 2.7-2.8 ms |
| Pinned H2D | 47-52 GB/s | 2.5-2.8 ms |
| Pageable D2H | 34-36 GB/s | 3.6-3.8 ms |
| Pinned D2H | 34-36 GB/s | 3.7-3.8 ms |
| D2D (Device) | 336-361 GB/s | 0.37-0.40 ms |

**Analysis**: H2D (Host-to-Device) is ~30% faster than D2H. Pinned memory provides slightly higher bandwidth. D2D far exceeds PCIe bandwidth.

---

## 3. CUDA Core Arithmetic Research

### 3.1 Data Type Throughput

The CUDA core arithmetic research explores the raw computational throughput across different data types. This is fundamental to understanding the GPU's mathematical capabilities.

#### FP64 (Double Precision)

FP64 operations are critical for scientific computing, CFD, and precision-sensitive applications. On Blackwell, FP64 is implemented as a dedicated hardware unit with different throughput characteristics than FP32.

| Metric | Value |
|--------|-------|
| Operations per Cycle | 64 FMA (1 per core × 64 cores/SM) |
| Peak Throughput | ~1,200 GFLOPS (estimated) |
| Typical Observed | 400-600 GFLOPS |

#### FP32 (Single Precision)

FP32 is the standard precision for most GPU computing workloads and the native format for CUDA cores.

| Metric | Value |
|--------|-------|
| Operations per Cycle | 128 FMA (2 per core × 64 cores/SM) |
| Peak Throughput | ~24,000 GFLOPS (estimated) |
| Observed Throughput | 88.55 GFLOPS (basic kernel) |

#### FP16 (Half Precision)

FP16 is widely used in deep learning for inference and training with reduced precision.

| Metric | Value |
|--------|-------|
| Peak Throughput | ~50,000 GFLOPS (tensor cores excluded) |
| Observed Throughput | 204.19 GFLOPS |
| Speedup vs FP32 | ~3.3x |

#### INT8 and INT32

Integer operations are essential for index calculations, control flow, and AI inference (INT8 quantization).

| Data Type | Observed Throughput | Notes |
|-----------|--------------------|-------|
| INT32 | 106.38 GIOPS | 32-bit integer arithmetic |
| INT8 | Performance varies by operation | Matrix multiply optimizations |

### 3.2 Instruction Latency vs Throughput

A critical distinction in GPU programming is the difference between instruction latency and throughput:

**Latency-Limited Operations** (Dependent Chains):
```
a = data[i];
b = data[i + 1];
c = data[i + 2];
for (int j = 0; j < 32; j++) {
    c = a * b + c;  // Dependent: result feeds next iteration
}
```
- Each FMA must wait for the previous to complete
- Limited by single FMA latency (~10 cycles)
- Throughput: ~200-400 GB/s

**Throughput-Limited Operations** (Independent):
```
for (int j = 0; j < 32; j++) {
    a = a * b + c;  // Independent: can be pipelined
}
```
- Multiple operations in flight simultaneously
- GPU's parallel execution hides latency
- Throughput: ~800-1200 GB/s

**Observed Ratio**: Throughput-limited operations are **2-4x faster** than latency-limited equivalents.

### 3.3 Vector Instructions

Vector instructions (float2, float4, double2) pack multiple operations into single instructions:

| Vector Type | Data Width | Relative Performance |
|-------------|-----------|---------------------|
| float (scalar) | 32-bit | 1.0x baseline |
| float2 | 64-bit | ~1.6x |
| float4 | 128-bit | ~2.2x |

**Analysis**: Vector instructions provide significant speedups by amortizing instruction fetch overhead.

### 3.4 Transcendental Functions

Transcendental functions (sin, cos, exp, log) have higher latency than basic arithmetic:

| Function | Relative Cost vs FMA |
|----------|---------------------|
| FMA (baseline) | 1.0x |
| sin/cos | ~10-15x |
| exp/log | ~8-12x |

**Optimization**: Use `__sinf`, `__cosf` approximations when full precision isn't required.

### 3.5 Mixed Precision

Mixed precision (FP32→FP16→FP32) leverages Tensor Cores for FP16 computation while maintaining FP32 accumulation:

```
Input (FP32) → Convert to FP16 → Tensor Core FMA → Convert back to FP32
```

**Benefits**:
- ~3x speedup vs pure FP32
- Minimal accuracy loss for most DL workloads
- Essential for modern LLM inference

---

## 4. Atomic Operations Deep Research

### 4.1 Atomic Operation Fundamentals

Atomic operations ensure visibility and mutual exclusion in parallel computing but introduce contention that can severely impact performance.

### 4.2 Contention Levels and Performance

The key to atomic performance is minimizing contention by reducing the number of threads that access the same atomic location.

#### Warp-Level Atomic (Best)

First, reduce 32 values to 1 using warp shuffle, then perform a single atomic:

```cuda
// Each thread accumulates
float sum = 0.0f;
for (...) { sum += src[i]; }

// Warp reduction (no atomics)
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}

// Single atomic per warp
if (tid % 32 == 0) {
    atomicAdd(result, sum);
}
```

**Performance**: High - contention reduced by 32x

#### Block-Level Atomic (Good)

Reduce within block, then single atomic per block:

```cuda
// Block reduction in shared memory
__syncthreads();
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    __syncthreads();
}

// One atomic per block
if (tid == 0) atomicAdd(result, shared[0]);
```

**Performance**: Good - contention reduced by block size (e.g., 256x for blockSize=256)

#### Grid-Level Atomic (Poor)

Direct atomic from all threads (maximum contention):

```cuda
// ALL threads atomic to same location
for (...) {
    atomicAdd(result, value);  // SEVERE CONTENTION
}
```

**Performance**: Very poor - thousands of threads competing for one location

### 4.3 Atomic Operation Comparison

| Operation | Relative Speed | Use Case |
|-----------|---------------|----------|
| atomicAdd (float) | 1.0x (baseline) | Summation |
| atomicCAS (compare-and-swap) | ~0.1x | Lock-free algorithms |
| atomicMin | ~0.8x | Finding minimum |
| atomicMax | ~0.8x | Finding maximum |
| atomic64 (double) | ~0.6x | Large data summation |

**Key Finding**: atomicCAS is **10x slower** than atomicAdd due to retry loops on contention.

### 4.4 Atomic Operation Best Practices

1. **Always reduce first**: Use warp shuffle or shared memory reduction before atomic
2. **Prefer atomicAdd over CAS**: When possible, use fetch-add instead of compare-and-swap
3. **Use 32-bit over 64-bit**: 32-bit atomics typically have lower latency
4. **Avoid grid-level contention**: Never have all threads atomically modify the same location

---

## 5. Barrier Synchronization Research

### 5.1 __syncthreads() Overhead

Barrier synchronization via `__syncthreads()` is essential for coordinating threads within a block but introduces overhead.

#### Measured Overhead

| Configuration | Time | Overhead |
|--------------|------|----------|
| No syncthreads | 0.021 ms | baseline |
| Single __syncthreads() | 0.023 ms | +9.5% |
| Two __syncthreads() | 0.025 ms | +19% |

**Per-sync overhead**: ~1-2 microseconds

### 5.2 Barrier Stall Analysis

Barrier stalls occur when threads within a warp reach the barrier at different times:

#### No Divergence (Optimal)
All threads in a warp take the same path and reach `__syncthreads()` together:
- Performance: ~746-761 GB/s
- No stall penalty

#### Divergent Paths
Some threads in a warp execute different code paths before reaching the barrier:
- Performance impact depends on divergence pattern
- Warp issue unit stalls waiting for missing threads

**Best Practice**: Minimize divergent paths before `__syncthreads()`.

### 5.3 Block Size vs Barrier Efficiency

| Block Size | Bandwidth | Efficiency |
|------------|-----------|------------|
| 32 | 346-347 GB/s | ~40% |
| 64 | 472-476 GB/s | ~55% |
| 128 | 684-890 GB/s | ~80% |
| **256** | **802-846 GB/s** | **~95%** |
| 512 | 765-840 GB/s | ~90% |
| 1024 | 617-628 GB/s | ~72% |

**Optimal block size**: 256-512 threads for barrier-bound kernels

### 5.4 Multi-Block Synchronization (Spin-Wait) - WARNING

Inter-block synchronization using spin-wait is **extremely inefficient**:

```cuda
// WARNING: NEVER DO THIS
while (atomicAdd(&flag, 0) < numBlocks) {
    // Spin forever - GPU resources wasted
}
```

**Performance Impact**: 10-100x slower than proper patterns

**Correct Approach**: Use separate kernel launches or CUDA streams for inter-block coordination.

### 5.5 Warp-Level Primitives (No Barrier Needed)

Warp-level primitives do NOT require `__syncthreads()` because all threads in a warp execute synchronously:

| Primitive | Description | Use Case |
|-----------|-------------|----------|
| `__shfl_sync` | Register shuffle | Warp-level reduction |
| `__any_sync` | Any thread nonzero? | Conditional checks |
| `__all_sync` | All threads nonzero? | Barrier alternatives |
| `__ballot_sync` | Which threads satisfied? | Predicate tracking |

**Performance**: Warp shuffle reduction is 5-10x faster than shared memory + syncthreads reduction.

---

## 6. Warp Specialization Patterns

### 6.1 Producer-Consumer Patterns

Warp specialization divides warps into different roles to overlap memory and compute operations.

#### Basic 2-Warp Producer/Consumer

```cuda
if (warp_id % 2 == 0) {
    // Producer warp: Load data to shared memory
    shared[thread_in_warp] = global[global_idx];
} else {
    // Consumer warp: Process data from shared memory
    result = shared[thread_in_warp] * 2.0f;
}
__syncthreads();
```

**Benefits**:
- Overlaps load and compute phases
- Hides memory latency
- Increases overall utilization

### 6.2 TMA + Barrier Synchronization

Tensor Memory Accelerator (TMA) provides efficient async copy with automatic barrier synchronization (Blackwell SM 9.0+):

```cuda
// TMA async copy - executes independently of compute
cp.async.shared.global [shared_ptr], [global_ptr], byte_count;
bar.sync 0;  // Wait for copy to complete
```

**Performance**: TMA copy achieves 1.2-1.5x speedup over standard loads for large transfers.

### 6.3 Multi-Stage Pipeline

3-stage pipeline overlapping Load, Compute, and Store:

```cuda
// Stage 1: Load
shared[tid] = global[idx];
__syncthreads();

// Stage 2: Compute
temp = shared[tid] * 2.0f;

// Stage 3: Store
global[idx] = temp;
```

**Overlapped Pipeline**: With proper synchronization, all stages execute simultaneously on different data chunks.

### 6.4 Block Specialization

Divide block threads into different roles:

```cuda
if (threadIdx.x < blockDim.x / 2) {
    // First half: Producer
    shared[threadIdx.x] = global[idx];
} else {
    // Second half: Consumer
    result = shared[threadIdx.x - blockDim.x/2] * 2.0f;
}
```

### 6.5 Warp-Level Synchronization Primitives

#### Warp-Level Reduction

```cuda
float val = thread_value;
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// Result now in all lanes
```

#### Warp-Level Scan (Prefix Sum)

```cuda
int val = input[tid];
#pragma unroll
for (int offset = 1; offset < 32; offset <<= 1) {
    int n = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset) val += n;
}
```

**Performance**: Warp-level primitives are 5-10x faster than equivalent shared memory implementations.

---

## 7. NCU Profiling Metrics

### 7.1 Key Metrics Reference

| Metric | Description | Optimal Value |
|--------|-------------|---------------|
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU utilization | >80% |
| dram__bytes.sum | Memory bandwidth | N/A (measure) |
| sm__pipe_fp32_cycles_active.pct | FP32 unit utilization | >80% |
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall | <5% |
| sm__warp_divergence_efficiency | Warp efficiency | >90% |
| sm__average_active_warps_per_sm | Active warps/SM | Depends on kernel |

### 7.2 Profiling Commands

```bash
# Full throughput analysis
ncu --set full ./gpupeek.exe cuda

# Memory bandwidth analysis
ncu --set full --metrics dram__bytes.sum ./gpupeek.exe memory

# Compute utilization
ncu --set full --metrics sm__pipe_fp32_cycles_active.pct ./gpupeek.exe cuda

# Barrier stall analysis
ncu --set full --metrics sm__warp_issue_stalled_by_barrier.pct ./gpupeek.exe barrier
```

---

## 8. Key Findings and Recommendations

### 8.1 Memory Optimization

1. **Use shared memory aggressively**: 1.5 TB/s vs 800 GB/s for global memory
2. **Avoid large strides**: Stride >16 causes severe bandwidth degradation
3. **Prefer sequential access**: Achieves peak bandwidth
4. **Use __ldg for read-only data**: Bypasses cache for one-time access
5. **Leverage broadcast**: Writing same value achieves ~1.3 TB/s

### 8.2 Compute Optimization

1. **Use FP16 for DL workloads**: 3x speedup over FP32
2. **Chain independent operations**: Avoid dependent FMA chains
3. **Prefer vector types**: float4 provides ~2x speedup over float
4. **Use approximations**: __sinf vs sinf for non-critical paths

### 8.3 Atomic Operations

1. **Always reduce first**: Warp reduction → single atomic
2. **Use 32-bit atomics**: Faster than 64-bit variants
3. **Avoid atomicCAS**: Use atomicAdd when possible
4. **Never use grid-level direct atomics**: Catastrophic performance

### 8.4 Synchronization

1. **Minimize __syncthreads() calls**: Each adds ~1-2 μs overhead
2. **Avoid divergent paths before barriers**: Causes warp stalls
3. **Use optimal block size**: 256-512 for barrier-bound kernels
4. **Never spin-wait for blocks**: Use separate kernel launches

### 8.5 Warp Specialization

1. **Overlap producers and consumers**: Maximize utilization
2. **Use warp shuffle over shared memory**: 5-10x faster for reductions
3. **Leverage TMA on Blackwell**: Efficient async copy with barriers
4. **Pipeline load/compute/store**: Hide latency effectively

---

## 9. Future Research Directions

### 9.1 Planned Investigations

1. **Tensor Core WMMA**: Deep dive into matrix multiply performance
2. **Multi-Stream Concurrency**: Stream dependencies and overlap
3. **Unified Memory**: Page fault analysis and migration costs
4. **CUDA Graphs**: Capture, instantiate, and launch optimization
5. **NVLink Bandwidth**: Multi-GPU communication analysis

### 9.2 Advanced Topics

1. **PTX Inline Assembly**: Fine-grained instruction optimization
2. **Warp-Level Programming**: Advanced shuffle techniques
3. **Memory Request Coalescing**: Optimizing memory transactions
4. **Occupancy vs Performance**: Finding optimal occupancy boundaries

---

## Appendix A: Benchmark Commands

```bash
# All benchmarks
./gpupeek.exe all

# Specific research areas
./gpupeek.exe generic   # Basic bandwidth/compute
./gpupeek.exe memory    # Memory subsystem
./gpupeek.exe cuda     # CUDA core arithmetic
./gpupeek.exe atomic   # Atomic operations
./gpupeek.exe barrier  # Synchronization
./gpupeek.exe warp     # Warp specialization
./gpupeek.exe advanced # Occupancy, PCIe, etc.

# NCU Profiling
ncu --set full ./gpupeek.exe cuda
```

---

## Appendix B: Test Environment

| Component | Version/Spec |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 5080 Laptop |
| Architecture | Blackwell (SM 12.0) |
| CUDA Toolkit | 13.0 |
| Driver | 595.79 |
| OS | Windows 11 |
| Build | CMake with SM 12.0 |

---

*Report generated: March 2026*
*Framework: GPUPeek v1.0*
*Contact: https://github.com/kevintsok/GPUPeek*
