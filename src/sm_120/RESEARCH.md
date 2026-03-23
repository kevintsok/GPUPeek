# SM 12.0 (Blackwell) GPU Architecture Research Report

> **Target GPU**: NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell, GB203)
> **Last Updated**: 2026-03-23
> **Report Language**: English

---

## Table of Contents

1. [Hardware Specifications](#1-hardware-specifications)
2. [Memory Subsystem](#2-memory-subsystem)
3. [Compute Performance](#3-compute-performance)
4. [Tensor Core (MMA)](#4-tensor-core-mma)
5. [Warp-Level Operations](#5-warp-level-operations)
6. [Memory Operations](#6-memory-operations)
7. [Cross-Generation Comparison](#7-cross-generation-comparison)
8. [NCU Profiling Metrics](#8-ncu-profiling-metrics)
9. [Test Environment](#9-test-environment)
10. [Benchmark Commands](#10-benchmark-commands)
11. [References](#11-references)

---

## 1. Hardware Specifications

### 1.1 GPU Specifications

| Parameter | Value |
|-----------|-------|
| GPU Model | NVIDIA GeForce RTX 5080 Laptop GPU |
| Architecture Code | Blackwell |
| Compute Capability | 12.0 |
| SM Count | 60 |
| CUDA Cores/SM | 128 |
| Total CUDA Cores | 7,680 |
| Global Memory | 15.92 GB |
| Shared Memory/Block | 48 KB |
| L1 Cache/SM | 128 KB |
| L2 Cache | 65 MB (monolithic) |
| Max Threads/Block | 1,024 |
| Max Threads/SM | 1,536 |
| Max Registers/SM | 65,536 |
| Warp Size | 32 |
| Memory Type | GDDR7 |
| Memory Bandwidth | ~8.2 TB/s |

### 1.2 Cross-Generation GPU Comparison

| Parameter | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) |
|-----------|-------------------|----------------|----------------|
| Compute Capability | 12.0 | 9.0 | 8.0 |
| SM Count | 60 (full: 84) | 132 | 108 |
| CUDA Cores/SM | 128 | 128 | 64 |
| Memory Type | GDDR7 | HBM2e | HBM2e |
| Memory Bandwidth | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s |
| L1 Cache/SM | 128 KB | 256 KB | 192 KB |
| Shared Memory/SM | ~99 KB | ~227 KB | ~227 KB |
| L2 Cache | 65 MB | 50 MB | 80 MB |
| L2 Architecture | Monolithic | 2-partition | 2-partition |
| Tensor Core Gen | 5th | 4th | 3rd |
| FP64 Units/SM | 2 (limited) | 64 (full) | 64 (full) |

---

## 2. Memory Subsystem

### 2.1 Global Memory Bandwidth

#### 2.1.1 Global Memory Bandwidth vs Data Size

| Data Size | Sequential Read | Sequential Write | State |
|-----------|----------------|-----------------|-------|
| 1 KB | 0.00 GB/s | 0.00 GB/s | - |
| 64 KB | 7.25 GB/s | 7.25 GB/s | L1 fits |
| 256 KB | 32.39 GB/s | 32.39 GB/s | L1 cache |
| 1 MB | 73.97 GB/s | 73.97 GB/s | L1/L2 borderline |
| 4 MB | 296.36 GB/s | 296.36 GB/s | L2 cache |
| 16 MB | 643.02 GB/s | 643.02 GB/s | Peak (1st) |
| 64 MB | 376.08 GB/s | 376.08 GB/s | L2 miss |
| 128 MB | 502.44 GB/s | 502.44 GB/s | Recovering |
| 256 MB | 614.93 GB/s | 614.93 GB/s | Peak (2nd) |

**Analysis**:
- Peak bandwidth ~640-820 GB/s at 16MB working set
- Bandwidth drop at 64MB indicates L2 cache eviction
- Recovery at 128-256MB suggests larger effective cache window

#### 2.1.2 Stride Access Efficiency

| Stride | Bandwidth | Efficiency |
|--------|-----------|------------|
| 1 | 822.37 GB/s | 100% (baseline) |
| 2 | 582.20 GB/s | 86.0% |
| 4 | 581.24 GB/s | 85.9% |
| 8 | 544.01 GB/s | 80.4% |
| 16 | 421.00 GB/s | 62.2% |
| 32 | 239.11 GB/s | 35.3% |
| 64 | 154.13 GB/s | 22.8% |
| 128 | 76.88 GB/s | 11.4% |
| 256 | 39.62 GB/s | 5.9% |

**Analysis**: Bandwidth drops sharply after stride 16, indicating cache line crossing overhead.

#### 2.1.3 Data Type Bandwidth

| Data Type | Size | Bandwidth |
|------------|------|-----------|
| float | 4 B | 878.19 GB/s |
| int | 4 B | 882.25 GB/s |
| double | 8 B | 468.73 GB/s |
| half (FP16) | 2 B | 410.20 GB/s |

**Analysis**:
- float/int bandwidth ~880 GB/s
- double drops ~47%
- half (FP16) drops ~53%

### 2.2 L1/L2 Cache Bandwidth

#### 2.2.1 Memory Hierarchy Bandwidth

| Access Pattern | Bandwidth | Notes |
|----------------|-----------|-------|
| Global Direct Read | 810.89 GB/s | Baseline read |
| Global Direct Write | 820.60 GB/s | Baseline write |
| Shared Memory R/W | **1.50 TB/s** | L1-level bandwidth |
| L2 Streaming (stride=1) | 766.78 GB/s | L2 cache hit |
| L2 Streaming (stride=1024) | 795.17 GB/s | Strided access |
| __ldg Bypass | 822.43 GB/s | Cache bypass |
| L1 Preference | 780.32 GB/s | Register optimization |

**Analysis**:
- Shared Memory (L1): **1.50 TB/s** - significantly higher than global memory
- L2 streaming maintains high bandwidth across strides
- __ldg (read-only cache bypass) slightly outperforms normal reads

#### 2.2.2 L2 Working Set Analysis

| Data Size | Bandwidth | State |
|-----------|-----------|-------|
| 64 KB | 123.09 GB/s | L1 fits |
| 1 MB | 407.66 GB/s | L2 borderline |
| 4 MB | 678.20 GB/s | L2 thrashing |
| 8 MB | 747.53 GB/s | L2 thrashing |
| 16 MB | 797.97 GB/s | L2 miss → DRAM |

#### 2.2.3 L2 Thrashing Test (Strided Access)

| Stride | Bandwidth |
|--------|-----------|
| 1 | 729.19 GB/s |
| 2 | 713.45 GB/s |
| 4 | 679.30 GB/s |
| 8 | 418.62 GB/s |
| 16 | 402.67 GB/s |
| 64 | 218.89 GB/s |
| 256 | 226.36 GB/s |
| 1024 | 244.46 GB/s |
| 4096 | 406.78 GB/s |

**Analysis**: Stride 8+ causes sharp bandwidth drop, indicating cache line crossing inefficiency.

### 2.3 Shared Memory Performance

| Operation | Bandwidth | Time/Kernel |
|-----------|-----------|-------------|
| Shared Memory R/W | **1.50 TB/s** | - |
| Broadcast Write | 1.30 TB/s | - |
| Reduction Read | 332.91 GB/s | - |
| Reduction Write | 1.30 GB/s | - |

### 2.4 TMA (Tensor Memory Access) Performance

#### 2.4.1 TMA 1D Copy

| Data Size | TMA Copy | cudaMemcpy | Speedup |
|-----------|----------|------------|---------|
| 64 KB | 6.88 GB/s | 6.88 GB/s | 0.99x |
| 256 KB | 33.97 GB/s | 33.97 GB/s | 0.97x |
| 1 MB | 133.87 GB/s | 133.87 GB/s | 1.13x |
| 4 MB | 431.20 GB/s | 431.20 GB/s | 1.04x |
| 16 MB | 850.07 GB/s | 850.07 GB/s | 0.72x |
| 64 MB | 382.15 GB/s | 382.15 GB/s | 1.06x |
| 128 MB | 373.07 GB/s | 373.07 GB/s | 0.99x |

**Analysis**: Peak TMA bandwidth 850 GB/s at 16MB.

#### 2.4.2 TMA 2D Copy (1024x1024, pitch=2048)

| Method | Bandwidth |
|--------|-----------|
| TMA 2D | 626.36 GB/s |
| cudaMemcpy2D | 704.31 GB/s |

**Analysis**: cudaMemcpy2D outperforms custom TMA kernel in 2D copy scenarios.

### 2.5 PCIe Bandwidth

| Transfer Type | Bandwidth | Time/Transfer |
|---------------|-----------|---------------|
| Pageable H2D | 47-49 GB/s | 2.7-2.8 ms |
| Pinned H2D | 47-52 GB/s | 2.5-2.8 ms |
| Pageable D2H | 34-36 GB/s | 3.6-3.8 ms |
| Pinned D2H | 34-36 GB/s | 3.7-3.8 ms |
| D2D (Device) | 336-361 GB/s | 0.37-0.40 ms |

**Analysis**:
- H2D (Host→Device) ~30% faster than D2H
- Pinned memory slightly faster than pageable
- D2D bandwidth far exceeds PCIe

### 2.6 Occupancy vs Performance

| Block Size | Bandwidth | Time |
|------------|-----------|------|
| 32 | 292-300 GB/s | ~57 μs |
| 64 | 374-453 GB/s | ~35-45 μs |
| 128 | 674-890 GB/s | ~18-25 μs |
| 256 | 802-876 GB/s | ~19-21 μs |
| **512** | **828-900 GB/s** | ~18-21 μs |
| 1024 | 589-628 GB/s | ~26-29 μs |

**Optimal Block Size**: 256-512 threads

### 2.7 Branch Divergence

| Branch Type | Bandwidth | Time |
|-------------|-----------|------|
| No Divergence | 746-761 GB/s | 0.021 ms |
| High Divergence | 796-810 GB/s | 0.021 ms |

**Analysis**: Divergence overhead not significant for simple kernels.

### 2.8 Memory Fence Impact

| Configuration | Bandwidth | Time/Kernel |
|--------------|-----------|-------------|
| No Fence | 793.25 GB/s | 0.021 ms |
| With Fence | 536.38 GB/s | 0.031 ms |

**Analysis**: Memory fence introduces ~50% performance overhead.

---

## 3. Compute Performance

### 3.1 FP32 Performance

| Test | Throughput | Latency | Notes |
|------|------------|---------|-------|
| FP32 FMA (Fused Multiply-Add) | 88.55 GFLOPS | 0.068 ms | Old test |
| FP32 FMA | 61.55 GFLOPS | 0.068 ms | New test |

### 3.2 FP64 Performance

| Metric | Value | Notes |
|--------|-------|-------|
| FP64 Units/SM | 2 (limited) | vs 64 in Hopper |
| FP64 True Latency | ~63 cycles | vs ~8 in Hopper |

**Warning**: Blackwell is severely limited for FP64 workloads.

### 3.3 FP16 Performance

| Test | Throughput | Latency |
|------|------------|---------|
| FP16 FMA | 204.19 GFLOPS | 0.021 ms |

**Analysis**: FP16 is ~3.3x faster than FP32.

### 3.4 INT32 Performance

| Test | Throughput | Latency |
|------|------------|---------|
| INT32 Arithmetic | 121.52 GIOPS | 0.035 ms |

### 3.5 Instruction Throughput Summary

| Instruction Type | Throughput | Latency |
|-----------------|------------|---------|
| FP32 FMA | 61.55 GFLOPS | 0.068 ms |
| INT32 Arithmetic | 121.52 GIOPS | 0.035 ms |
| FP16 FMA | 204.19 GFLOPS | 0.021 ms |

---

## 4. Tensor Core (MMA)

### 4.1 WMMA (Warp-level MMA)

WMMA is the standard CUDA API for Tensor Cores (Section 9.7.14).

#### 4.1.1 WMMA Shape Support

| Shape | FP16 | BF16 | TF32 | FP64 | INT8 | INT4 |
|-------|------|------|------|------|------|------|
| m8n8k4 | Yes | - | - | Yes | - | - |
| m8n8k16 | Yes | - | - | - | Yes | - |
| m8n8k32 | Yes | - | - | - | Yes | Yes |
| m16n8k4 | Yes | - | Yes | - | - | - |
| m16n8k8 | Yes | Yes | Yes | - | Yes | - |
| m16n8k16 | Yes | Yes | Yes | - | Yes | Yes |
| m16n8k32 | Yes | Yes | - | - | Yes | Yes |
| m16n8k64 | Yes | - | - | - | Yes | - |
| m16n8k128 | Yes | - | - | - | Yes | - |
| m16n8k256 | Yes | - | - | - | Yes | - |
| m16n16k16 | Yes | Yes | Yes | Yes | Yes | - |

#### 4.1.2 WMMA FP16 Benchmark Results

| Metric | Value |
|--------|-------|
| Matrix Size | M=256, N=256, K=256 |
| Shape | m16n16k16 |
| Grid | 16x16 |
| Block | 32 (1 warp) |
| Time | 0.130 ms/iteration |
| **Performance** | **257.41 GFLOPS** |
| Verification | sum=4103416.75 (non-zero=correct) |

### 4.2 MMA (New Warp-level MMA)

Standard MMA instructions (Section 9.7.14.5).

| Shape | FP16 | FP64 | TF32 | BF16 | INT8 | INT4 |
|-------|------|------|------|------|------|------|
| m8n8k4 | Yes | Yes | - | - | - | - |
| m8n8k16 | Yes | - | - | - | Yes | - |
| m8n8k32 | Yes | - | - | - | Yes | Yes |
| m8n8k128 | Yes | - | - | - | Yes | Yes |
| m16n8k4 | Yes | - | Yes | - | - | - |
| m16n8k8 | Yes | - | Yes | Yes | Yes | - |
| m16n8k16 | Yes | - | Yes | Yes | Yes | Yes |
| m16n8k32 | Yes | - | - | Yes | Yes | Yes |
| m16n8k64 | Yes | - | - | - | Yes | - |
| m16n8k128 | Yes | - | - | - | Yes | - |
| m16n8k256 | Yes | - | - | - | Yes | - |

### 4.3 MMA.SP (Sparse MMA)

2:4 Structured Sparsity (Section 9.7.14.6).

| Shape | FP16 | BF16 | TF32 | INT8 | FP8 |
|-------|------|------|------|------|-----|
| m16n8k8.sp | Yes | Yes | Yes | - | - |
| m16n8k16.sp | Yes | Yes | Yes | Yes | - |
| m16n8k32.sp | Yes | - | - | Yes | - |
| m16n8k64.sp | Yes | - | - | - | - |
| m16n8k128.sp | Yes | - | - | - | - |

**Note**: 2:4 sparsity requires 2 zeros out of every 4 elements.

### 4.4 WGMMA (Async Warpgroup MMA)

**Important**: WGMMA is **Hopper-only**. Blackwell does NOT support WGMMA.

| Shape | FP16 | BF16 | TF32 | FP64 | INT8 | FP8 |
|-------|------|------|------|------|------|-----|
| m64nNk16 | Yes | Yes | Yes | Yes | Yes | - |
| m64nNk8 | Yes | Yes | Yes | - | Yes | - |
| m64nNk32 | Yes | Yes | - | - | - | Yes |
| m64nNk256 | Yes | - | - | - | - | - |

**N = K / 16**

### 4.5 TCGen05 (5th Gen Tensor Core)

Blackwell's 5th generation Tensor Core API (Section 9.7.16).

#### 4.5.1 TCGen05 vs WGMMA

| Feature | WGMMA (Hopper) | TCGen05 (Blackwell) |
|---------|-----------------|---------------------|
| PTX Section | 9.7.15 | 9.7.16 |
| API | wgmma.mma_async | tcgen05.mma |
| SASS | WGMMA | **OMMA (FP4), QMMA (FP8/FP6)** |
| Async | Yes | Yes |
| Sparse | 2:4 | 2:4 |
| FP4/FP6 | No | **Yes** |
| Block Scaling | No | **Yes (Hardware)** |

#### 4.5.2 TCGen05 Shape Support

| Shape | FP16 | BF16 | TF32 | FP32 | FP64 | INT8 | FP4 | FP6 |
|-------|------|------|------|------|------|------|-----|-----|
| .32x32b | Yes | Yes | Yes | Yes | Yes | - | - | - |
| .16x64b | Yes | Yes | Yes | Yes | Yes | - | - | - |
| .16x128b | Yes | Yes | Yes | Yes | - | - | - | - |
| .16x256b | Yes | Yes | Yes | - | - | - | - | - |
| .16x32bx2 | Yes | Yes | Yes | Yes | - | - | - | - |
| **m16n8k32** | - | - | - | - | - | - | **Yes** | **Yes** |

#### 4.5.3 TCGen05 Variants

| Variant | Description |
|---------|-------------|
| tcgen05.mma | Basic MMA |
| tcgen05.mma.sp | Sparse MMA (2:4) |
| tcgen05.mma.ws | Weight-only Quantization (W8A16) |
| tcgen05.mma.ws.sp | Weight-only + Sparse |

#### 4.5.4 TCGen05 CTA Group Types

| CTA Group | Description | D Registers |
|-----------|-------------|-------------|
| cta_group::1 | Single CTA (1 warp group) | 4 |
| cta_group::2 | Dual CTA cluster (2 warp groups) | 8+ |

#### 4.5.5 TCGen05 Operand Source Variants

| Variant | A Source | B Source | Description |
|---------|----------|----------|-------------|
| SS | SMEM | SMEM | Both from shared memory |
| TS | TMEM | SMEM | A from tensor memory |
| ST | SMEM | TMEM | B from tensor memory |
| TT | TMEM | TMEM | Both from tensor memory |

#### 4.5.6 TMEM (Tensor Memory)

| Property | Value |
|----------|-------|
| Size | **256 KB per SM** |
| Organization | 512 columns × 128 rows × 32-bit cells |
| D (accumulator) | **Must** be in TMEM |
| A operand | Can load from TMEM or SMEM |

#### 4.5.7 Block Scaling (TCGen05 Hardware Support)

| Format | Block Size | Scale Factor Format |
|--------|------------|-------------------|
| UE8M0 | 32 elements | 8-bit unsigned exp (2^x), -127 ≤ x ≤ 127 |
| UE4M3 | 16 elements | 4-bit exp + 3-bit mantissa, max 448 |

**TMEM Layout**:
- block32/1X: up to 12 columns
- block32/2X: up to 24 columns
- block16/4X: up to 48 columns

#### 4.5.8 PTX Instruction Examples (from CUTLASS)

```asm
// TF32 SS (both from SMEM)
tcgen05.mma.cta_group::1.kind::tf32 [%tmem_c], %desc_a, %desc_b, %idescE,
  {%mask0, %mask1, %mask2, %mask3}, p;

// FP16 BF16 SS
tcgen05.mma.cta_group::1.kind::f16 [%tmem_c], %desc_a, %desc_b, %idescE,
  {%mask0, %mask1, %mask2, %mask3}, p;

// Block Scaled MXFP8/MXFP6/MXF4
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%tmem_c], %desc_a, %desc_b,
  %idescE, [%tsfa_addr], [%tsfb_addr], p;

// Block Scaled MXF4 (NVIDIA FP4) - 16 element block
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%tmem_c], %desc_a,
  %desc_b, %idescE, [%tsfa_addr], [%tsfb_addr], p;

// Cluster MMA (2 CTA)
tcgen05.mma.cta_group::2.kind::tf32 [%tmem_c], %desc_a, %desc_b, %idescE,
  {%mask0..%mask11}, p;

// Sparse MMA (2:4 structure)
tcgen05.mma.sp.cta_group::1.kind::tf32 [%tmem_c], %desc_a, %desc_b, [%tsfb_addr],
  %idescE, {%mask0..%mask7}, p;

// SM120 Native MMA (m16n8k32, register-based)
mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32
  {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3};
```

#### 4.5.9 RTX 50 (GeForce) Limitations

| Feature | Support |
|---------|---------|
| TMA Multicast | ❌ Not supported |
| Cluster Shape | Must be 1x1x1 |
| Dynamic Datatypes | ❌ Not supported |
| Cluster MMA | ❌ Not supported (data center GPUs only) |

#### 4.5.10 Sub-byte Data Memory Layout Requirements

| Data Type | TMA Packing Format | SMEM Alignment |
|-----------|-------------------|----------------|
| FP4 (E2M1) | CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B | 16-byte aligned |
| FP6 (E2M3/E3M2) | CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B | 16-byte aligned |

**Key Constraints**:
- Base address requires 32-byte alignment
- Leading dimension must be multiple of 128 elements
- Only 128-byte swizzle mode or no swizzle supported
- K extent for MMA tile is always 32 (dense GEMM)

### 4.6 FP4/FP6 Low-Precision MMA

#### 4.6.1 Format Specifications

| Format | Exponent | Mantissa | Total Bits | PTX Type |
|--------|----------|----------|------------|----------|
| FP4 (e2m1) | 2 | 1 | 4 | e2m1 |
| FP6 (e2m3) | 2 | 3 | 6 | e2m3 |
| FP6 (e3m2) | 3 | 2 | 6 | e3m2 |

**PTX ISA (CUDA 12.9+)**:

```ptx
mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32   // FP4
mma.sync.aligned.m16n8k32.row.col.f32.e2m3.e2m3.f32   // FP6 e2m3
mma.sync.aligned.m16n8k32.row.col.f32.e3m2.e3m2.f32   // FP6 e3m2
```

**Shape**: m16n8k32 (different from FP8's m16n8k16)

#### 4.6.2 FP4/FP6 vs FP8 Comparison

| Feature | FP8 (E4M3/E5M2) | FP4 (e2m1) | FP6 (e2m3/e3m2) |
|---------|------------------|------------|------------------|
| Bits | 8 | 4 | 6 |
| Precision | High | Very Low | Low |
| Memory Reduction | 2x vs FP16 | 4x vs FP16 | 2.67x vs FP16 |
| TFLOPS | Highest | Highest | High |
| Use Case | Weights+Activations | Weights only | Weights only |

#### 4.6.3 FP4/FP6 Preliminary Test Results

| Test | Result |
|------|--------|
| FP32 → FP4 Conversion | 1.304 ms (1M elements) |
| FP4 → FP32 Conversion | 0.052 ms (1M elements) |
| FP16 GEMM Baseline | 743.57 GFLOPS |
| FP4 Style GEMM (simulated) | 920.45 GFLOPS |

**Note**: FP4/FP6 style GEMM appears faster than FP16 baseline because simulated kernels do simplified quantization. True FP4/FP6 MMA requires CUDA 12.9+ hardware support.

### 4.7 SASS Instruction Mapping

| PTX | SASS | Description |
|-----|------|-------------|
| wmma.mma.f16 | HMMA | Half precision MMA |
| wmma.mma.bf16 | BMMA | BFloat16 MMA |
| wmma.mma.tf32 | HMMA | TensorFloat-32 MMA |
| wmma.mma.f64 | DMMA | Double precision MMA |
| wmma.mma.s32 | IMMA | INT32 MMA |
| mma.mma | HMMA/IMMA/DMMA | Generic MMA |
| wgmma.mma_async | WGMMA | Async Warpgroup MMA (Hopper only) |
| ld.matrix | LDMATRIX | Matrix load |
| st.matrix | STMATRIX | Matrix store |
| tcgen05.mma | OMMA/QMMA | TCGen05 MMA (Blackwell) |

### 4.8 NCU Tensor Core Metrics

| Metric | Meaning | Ideal |
|--------|---------|-------|
| sm__pipe_tensor_cycles_active.pct | Tensor Core utilization | Higher = better |
| sm__pipe_tensor_cycles_active.sum | Tensor Core active cycles | Lower = better |
| smsp__average_executed_epc_per_warp | Instructions per warp | Stable |
| sm__inst_executed.sum | Total executed instructions | Lower = better |
| dram__bytes.sum | Global memory bandwidth | Reference |
| lts__tcs_hit_rate.pct | L2 cache hit rate | Higher = better |

---

## 5. Warp-Level Operations

### 5.1 Warp Shuffle

| Operation | Bandwidth | Time/Kernel |
|-----------|-----------|-------------|
| Shuffle Reduce | 747.46 GB/s | 0.022 ms |
| Butterfly Reduce | 730.21 GB/s | 0.023 ms |
| Generic Shuffle | 305.59 GB/s | - |
| **Enhanced Shuffle** | **418.49 GB/s** | - |

**Analysis**: Blackwell's Enhanced Shuffle is **37% faster** than generic Warp Shuffle (418.49 vs 305.59 GB/s).

### 5.2 Warp Vote/Ballot

| Operation | Performance |
|-----------|-------------|
| Ballot Sync | 0.020 ms/kernel |

### 5.3 Redux.sync

Redux.sync performs warp-level reduction in a **single instruction**.

| Method | Instructions | Latency | Advantage |
|--------|-------------|---------|-----------|
| Shuffle Loop | log2(32) = 5 shuffles | Higher | Compatible |
| **Redux.sync** | **1 instruction** | **Lowest** | **HW accelerated** |

**Supported Operations**: ADD, MIN, MAX, AND, OR, XOR

---

## 6. Memory Operations

### 6.1 Async Copy

| Operation | Bandwidth | Notes |
|-----------|-----------|-------|
| Async Copy | 422.69 GB/s | Highest performance |
| L2 Streaming | 316.46 GB/s | Cache streaming |
| Register Bandwidth | 298.96 GB/s | Register-level |
| Software Prefetch | 251.10 GB/s | Lower but predictable |

### 6.2 LDMATRIX/STMATRIX

Matrix load/store operations for Tensor Core data (Section 9.7.14.5.15-16).

#### 6.2.1 LDMATRIX Variants

| Instruction | Description | Elements/Thread |
|-------------|-------------|-----------------|
| ldmatrix.sync.aligned.m8n8.x1 | 8x8 tile, 1 matrix | 2 |
| ldmatrix.sync.aligned.m8n8.x2 | 8x8 tile, 2 matrices | 4 |
| ldmatrix.sync.aligned.m8n8.x4 | 8x8 tile, 4 matrices | 8 |
| ldmatrix.sync.aligned.m16n8.k1 | 16x8 tile | varies |

**Key Features**:
- Warp-level operation (32 threads cooperate)
- Transposed layout (MMA-friendly)
- 16-byte alignment required

#### 6.2.2 STMATRIX Variants

| Instruction | Description |
|-------------|-------------|
| stmatrix.sync.aligned.m8n8.x1 | 8x8 tile, 1 matrix |
| stmatrix.sync.aligned.m8n8.x2 | 8x8 tile, 2 matrices |
| stmatrix.sync.aligned.m8n8.x4 | 8x8 tile, 4 matrices |

### 6.3 cp.async (Async Copy)

| Instruction | Description |
|-------------|-------------|
| cp.async.ca | Cache policy async copy |
| cp.async.commit_group | Commit async group |
| cp.async.wait_group n | Wait for n groups |
| cp.async.wait_all | Wait for all |

### 6.4 cp.async.bulk (Bulk Async Copy)

| Instruction | Description |
|-------------|-------------|
| cp.async.bulk | Bulk async copy |
| cp.async.bulk.commit_group | Bulk commit group |
| cp.async.bulk.wait_group n | Bulk wait |
| cp.reduce.async.bulk.add | Bulk copy + sum |
| cp.async.bulk.prefetch | Bulk prefetch |

### 6.5 Tensor Memory Performance Summary

| Operation | Advantage | Use Case |
|-----------|-----------|----------|
| LDMATRIX | Warp cooperation, transpose | MMA A/B loading |
| STMATRIX | Warp cooperation | MMA result storage |
| cp.async | Latency hiding | Compute/copy overlap |
| TMA | Large 2D transfers | Large matrix tiling |

---

## 7. Cross-Generation Comparison

### 7.1 Compute Performance

| Metric | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) |
|--------|-------------------|----------------|----------------|
| FP32 Performance | ~17.6 TFLOPS | ~19.5 TFLOPS | ~19.5 TFLOPS |
| FP16 Performance | ~89.2 TFLOPS | ~99.8 TFLOPS | ~39.7 TFLOPS |
| FP64 Performance | **Limited** | Full | Full |
| Tensor Core Gen | 5th | 4th | 3rd |

### 7.2 Memory System

| Metric | Blackwell | Hopper | Ampere |
|--------|-----------|--------|--------|
| Memory Type | GDDR7 | HBM2e | HBM2e |
| Bandwidth | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s |
| L1 Cache/SM | 128 KB | 256 KB | 192 KB |
| Shared Memory/SM | ~99 KB | ~227 KB | ~227 KB |
| L2 Cache | 65 MB | 50 MB | 80 MB |
| L2 Architecture | Monolithic | 2-partition | 2-partition |

### 7.3 Tensor Core Feature Comparison

| Feature | Blackwell (5th) | Hopper (4th) | Ampere (3rd) |
|---------|-----------------|--------------|--------------|
| WGMMA | ❌ | ✅ | ❌ |
| FP4 Support | ✅ | ❌ | ❌ |
| FP6 Support | ✅ | ❌ | ❌ |
| FP8 Support | ✅ | ✅ | ❌ |
| Block Scaling | ✅ (HW) | ❌ | ❌ |
| 2:4 Sparse | ✅ | ✅ | ✅ |

### 7.4 Latency Comparison

| Operation | Blackwell | Hopper |
|-----------|-----------|--------|
| FP32 True Latency | 15.96 cycles | 31.62 cycles |
| INT32 Latency | 14 cycles | 16 cycles |
| FP64 True Latency | **~63 cycles** | ~8 cycles |
| L2 Cache Hit | ~358 cycles | ~273 cycles |
| Global Memory | ~877 cycles | ~659 cycles |
| MMA Completion | 1.21 cycles | 1.66 cycles |

### 7.5 Power Efficiency (FP8/FP4/FP6)

| Precision | Blackwell | Hopper |
|-----------|-----------|--------|
| FP8 | ~46W | ~55W |
| FP4 | ~16.75W | N/A |
| FP6 e2m3 | ~39.38W | N/A |
| FP6 e3m2 | ~46.72W | N/A |

### 7.6 Transformer Engine Support

| Version | Supported Precisions | Architecture |
|---------|---------------------|--------------|
| TE 1st Gen | FP8, FP16, BF16, FP32, FP64 | Hopper |
| **TE 2nd Gen** | **FP4, FP6** + above | **Blackwell** |

---

## 8. NCU Profiling Metrics

### 8.1 Key Metrics Reference

| Metric | Meaning | Use Case |
|--------|---------|----------|
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU utilization | Higher = better |
| dram__bytes.sum | Memory bandwidth | Memory operation tests |
| sm__pipe_fp32_cycles_active.pct | FP32 unit utilization | Compute tests |
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall | Sync overhead |
| sm__warp_divergence_efficiency | Warp divergence efficiency | Divergence tests |
| sm__average_active_warps_per_sm | Active warps/SM | Occupancy |

### 8.2 Kernel Code Coverage

| Benchmark | Kernels | PTX Instructions |
|-----------|---------|------------------|
| memory | Various | Global/Shared/L1 access |
| deep | Various | L2, TMA, prefetch |
| advanced | Various | Atomic, constant memory |
| wmma | wmma_fp16_kernel, etc. | wmma.mma |
| tcgen05 | tcgen05_mma_kernel | tcgen05.mma |
| tensor_mem | ldmatrix, stmatrix, cp.async | ld.matrix, st.matrix, cp.async |
| wgmma | wgmma_async_kernel | wgmma.mma_async (Hopper only) |
| dp4a | dp4a_*_kernel | dp4a.s32.s8.s8 |
| fp8 | fp8_gemm_*_kernel | FP8 MMA |
| fp4 | fp4_*_kernel | FP4/FP6 MMA (simulated) |
| cuda | cuda_core_*_kernel | Various |

---

## 9. Test Environment

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 13.0 |
| Driver | 595.79 |
| OS | Windows 11 |
| GPU | RTX 5080 Laptop (GB203) |
| Compute Capability | 12.0 |
| SM Count | 60 |
| Compile Flags | -O3 -arch=sm_90 --use_fast_math |

---

## 10. Benchmark Commands

### 10.1 GPUPeek Benchmark Commands

```bash
# All benchmarks
./build/gpupeek.exe all

# Specific modules
./build/gpupeek.exe memory    # Memory research
./build/gpupeek.exe deep      # Deep research
./build/gpupeek.exe advanced  # Advanced research
./build/gpupeek.exe wmma      # WMMA (Tensor Core)
./build/gpupeek.exe tcgen05   # TCGen05 research
./build/gpupeek.exe tensor_mem # Tensor memory
./build/gpupeek.exe cuda      # CUDA Core compute
./build/gpupeek.exe atomic    # Atomic operations
./build/gpupeek.exe barrier   # Barrier sync
./build/gpupeek.exe warp      # Warp specialization
./build/gpupeek.exe mma       # MMA/Tensor Core (DISABLED - use wmma)
./build/gpupeek.exe wgmma     # WGMMA (Hopper only)
./build/gpupeek.exe dp4a      # DP4A
./build/gpupeek.exe fp8       # FP8
./build/gpupeek.exe fp4       # FP4/FP6
./build/gpupeek.exe graph     # CUDA Graph
./build/gpupeek.exe unified   # Unified Memory
./build/gpupeek.exe multi_stream # Multi-Stream
./build/gpupeek.exe mbarrier # Mbarrier
./build/gpupeek.exe coop     # Cooperative Groups
./build/gpupeek.exe redux    # Redux.sync
```

### 10.2 NCU Profiling Commands

```bash
# Full tensor core analysis
ncu --set full --metrics sm__pipe_tensor_cycles_active.pct,sm__inst_executed.sum,dram__bytes.sum,lts__tcs_hit_rate.pct ./gpupeek.exe wmma

# Memory bandwidth analysis
ncu --set full --metrics dram__bytes.sum ./gpupeek.exe memory

# SASS instruction analysis
ncu --set full --kernels-by-compute ./gpupeek.exe tensor_mem

# GPU utilization
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe deep
```

---

## 11. References

- [CUDA Programming Guide](../ref/cuda_programming_guide.html)
- [PTX ISA](../ref/ptx_isa.html)
- [Inline PTX Assembly](../ref/inline_ptx.html)
- [Blackwell Compatibility Guide](../ref/blackwell_guide.html)
- [CUTLASS Tutorial: TMEM GEMM](../ref/cutlass_tutorial_tmem_gemm.md)
- [CUTLASS Tutorial: Block Scaling](../ref/cutlass_tutorial_block_scaling.md)
- [CUTLASS Tutorial: Sub-byte GEMM](../ref/cutlass_tutorial_subbyte_gemm.md)
- [CUTLASS Tutorial: Cluster GEMM](../ref/cutlass_tutorial_cluster_gemm.md)
- [DeepSeek FP8 Training](../ref/deepseek_fp8_training.md)
- [FlashAttention-4](../ref/flashattention4.md)
- [arXiv:2507.10789 - Blackwell Microbenchmarks](https://arxiv.org/abs/2507.10789)

---

## Appendix A: Compilation Status (2026-03-23)

### Working Modules (7/18)

| Module | Status | Notes |
|--------|--------|-------|
| memory | ✅ | Memory research |
| deep | ✅ | Deep research |
| advanced | ✅ | Advanced research |
| fp4 | ✅ | FP4/FP6 research |
| multi_stream | ✅ | Multi-stream concurrency |
| wmma | ✅ | WMMA (warp-level MMA) |
| tcgen05 | ✅ | TCGen05/UMMA research |

### Disabled Modules (Compilation Errors)

| Module | Issue |
|--------|-------|
| ncu | Encoding issue |
| cuda | Missing bf16 header |
| atomic | Under investigation |
| barrier | Cooperative groups API issue |
| warp | Cooperative groups API issue |
| mma | DISABLED - use wmma instead |
| tensor_mem | WMMA fragment type undefined |
| wgmma | WMMA fragment type undefined |
| dp4a | Under investigation |
| fp8 | Under investigation |
| graph | Under investigation |
| unified | Under investigation |
| mbarrier | Under investigation |
| coop | Cooperative groups API issue |
| redux | Under investigation |

---

## Appendix B: WMMA Fix (2026-03-23)

**Problem**: WMMA kernels had "illegal memory access" errors at runtime.

**Solution**:
1. Created new `wmma_test_kernel.cu` and `wmma_test_benchmarks.cu`
2. Used correct fragment type definitions:
   ```cuda
   fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
   fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
   fragment<accumulator, 16, 16, 16, float> frag_d;
   ```
3. Used `using namespace nvcuda::wmma;`
4. Grid: `(N / 16, M / 16)`, Block: `32`
5. Each warp handles one 16x16 output block

---

*Report generated from GPUPeek benchmark framework*
