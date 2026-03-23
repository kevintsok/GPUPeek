# SM 12.0 (Blackwell) GPU Architecture Research Report

> **Target GPU**: NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell, GB203)
> **Last Updated**: 2026-03-23

---

## Table of Contents

1. [Hardware Specifications](#1-hardware-specifications)
2. [Memory Subsystem](#2-memory-subsystem)
   - [2.1 Global Memory Bandwidth](#21-global-memory-bandwidth)
   - [2.2 L1/L2 Cache Bandwidth](#22-l1l2-cache-bandwidth)
   - [2.3 Shared Memory Performance](#23-shared-memory-performance)
   - [2.4 PCIe Bandwidth](#24-pcie-bandwidth)
3. [Compute Performance](#3-compute-performance)
   - [3.1 FP32/FP64 Performance](#31-fp32fp64-performance)
   - [3.2 FP16/BF16 Performance](#32-fp16bf16-performance)
   - [3.3 INT32 Performance](#33-int32-performance)
4. [Tensor Core (MMA)](#4-tensor-core-mma)
   - [4.1 WMMA (Warp-level MMA)](#41-wmma-warp-level-mma)
   - [4.2 TCGen05 (5th Gen Tensor Core)](#42-tcgen05-5th-gen-tensor-core)
   - [4.3 MMA Shapes Reference](#43-mma-shapes-reference)
5. [Warp-Level Operations](#5-warp-level-operations)
   - [5.1 Warp Shuffle](#51-warp-shuffle)
   - [5.2 Warp Vote/Ballot](#52-warp-voteballot)
   - [5.3 Redux.sync](#53-reduxsync)
6. [Memory Operations](#6-memory-operations)
   - [6.1 Async Copy](#61-async-copy)
   - [6.2 LDMATRIX/STMATRIX](#62-ldmatrixstmatrix)
   - [6.3 TMA (Tensor Memory Access)](#63-tma-tensor-memory-access)
7. [Blackwell vs Hopper vs Ampere](#7-blackwell-vs-hopper-vs-ampere)
8. [Microbenchmark Results](#8-microbenchmark-results)
9. [Test Environment](#9-test-environment)
10. [References](#10-references)

---

## 1. Hardware Specifications

### GPU Specifications

| Parameter | Value |
|-----------|-------|
| GPU Model | NVIDIA GeForce RTX 5080 Laptop GPU |
| Architecture | Blackwell |
| Compute Capability | 12.0 |
| SM Count | 60 |
| CUDA Cores/SM | 128 |
| Total CUDA Cores | 7,680 |
| Global Memory | 15.92 GB |
| Shared Memory/SM | 48 KB |
| L1 Cache/SM | 128 KB |
| L2 Cache | 65 MB (monolithic) |
| Max Threads/Block | 1,024 |
| Max Threads/SM | 1,536 |
| Max Registers/SM | 65,536 |
| Warp Size | 32 |
| Memory Type | GDDR7 |
| Memory Bandwidth | ~8.2 TB/s |

### Cross-Generation Comparison

| Parameter | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) |
|-----------|-------------------|----------------|----------------|
| Compute Capability | 12.0 | 9.0 | 8.0 |
| SM Count | 60 (or 84 full) | 132 | 108 |
| CUDA Cores/SM | 128 | 128 | 64 |
| Memory Type | GDDR7 | HBM2e | HBM2e |
| Memory Bandwidth | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s |
| L1 Cache/SM | 128 KB | 256 KB | 192 KB |
| L2 Cache | 65 MB | 50 MB | 80 MB |
| Tensor Core Gen | 5th | 4th | 3rd |
| FP64 Support | Limited (2/SM) | Full (64/SM) | Full |

---

## 2. Memory Subsystem

### 2.1 Global Memory Bandwidth

#### Global Memory Bandwidth vs Data Size

| Data Size | Sequential Read | Sequential Write | Notes |
|-----------|-----------------|------------------|-------|
| 64 KB | 7.25 GB/s | 7.25 GB/s | L1 cache fits |
| 256 KB | 32.39 GB/s | 32.39 GB/s | L1 cache |
| 1 MB | 73.97 GB/s | 73.97 GB/s | L1/L2 borderline |
| 4 MB | 296.36 GB/s | 296.36 GB/s | L2 cache thrashing |
| 16 MB | 643.02 GB/s | 643.02 GB/s | Peak bandwidth (1st) |
| 64 MB | 376.08 GB/s | 376.08 GB/s | L2 miss |
| 128 MB | 502.44 GB/s | 502.44 GB/s | - |
| 256 MB | 614.93 GB/s | 614.93 GB/s | Recovered |

**Analysis**:
- Peak bandwidth: ~640-820 GB/s at 16MB working set
- Bandwidth drop at 64MB indicates L2 cache eviction
- Recovery at 128-256MB suggests larger effective cache window

#### Stride Access Efficiency

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

### 2.2 L1/L2 Cache Bandwidth

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

#### L2 Working Set Analysis

| Data Size | Bandwidth | State |
|-----------|-----------|-------|
| 64 KB | 123.09 GB/s | L1 fits |
| 1 MB | 407.66 GB/s | L2 borderline |
| 4 MB | 678.20 GB/s | L2 thrashing |
| 8 MB | 747.53 GB/s | L2 thrashing |
| 16 MB | 797.97 GB/s | L2 miss → DRAM |

### 2.3 Shared Memory Performance

| Operation | Bandwidth | Notes |
|-----------|-----------|-------|
| Shared Memory R/W | **1.50 TB/s** | L1-level bandwidth |
| Broadcast Write | 1.30 TB/s | All threads write same value |
| Reduction Write | 1.30 GB/s | Read: 332.91 GB/s |

### 2.4 PCIe Bandwidth

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

---

## 3. Compute Performance

### 3.1 FP32/FP64 Performance

| Operation | Throughput | Latency | Notes |
|-----------|------------|---------|-------|
| FP32 FMA | 88.55 GFLOPS | 0.068 ms | Fused Multiply-Add |
| FP64 | Limited | ~63 cycles | Only 2 FP64 units/SM |

**Note**: Blackwell has severely limited FP64 (only 2 units/SM vs 64 in Hopper), making it unsuitable for FP64-heavy workloads.

### 3.2 FP16/BF16 Performance

| Operation | Throughput | Notes |
|-----------|------------|-------|
| FP16 FMA | 204.19 GFLOPS | 3.3x faster than FP32 |
| WMMA FP16 | **257.41 GFLOPS** |实测 (m16n16k16) |
| BF16 | 待实测 | - |

### 3.3 INT32 Performance

| Operation | Throughput | Latency |
|-----------|------------|---------|
| INT32 Arithmetic | 121.52 GIOPS | 0.035 ms |

### Instruction Throughput Summary

| Instruction Type | Throughput | Latency |
|-----------------|------------|---------|
| FP32 FMA | 61.55 GFLOPS | 0.068 ms |
| INT32 Arith | 121.52 GIOPS | 0.035 ms |
| FP16 FMA | 204.19 GFLOPS | 0.021 ms |

---

## 4. Tensor Core (MMA)

### 4.1 WMMA (Warp-level MMA)

WMMA (Warp-level Matrix Multiply Accumulate) is the standard CUDA API for Tensor Cores.

| Shape | FP16 | BF16 | TF32 | FP64 | INT8 | INT4 |
|-------|------|------|------|------|------|------|
| m8n8k4 | Yes | - | - | Yes | - | - |
| m8n8k16 | Yes | - | - | - | Yes | - |
| m8n8k32 | Yes | - | - | - | Yes | Yes |
| m16n8k4 | Yes | - | Yes | - | - | - |
| m16n8k8 | Yes | Yes | Yes | - | Yes | - |
| m16n8k16 | Yes | Yes | Yes | - | Yes | Yes |
| m16n8k32 | Yes | Yes | - | - | Yes | Yes |
| m16n16k16 | Yes | Yes | Yes | Yes | Yes | - |

#### WMMA FP16 Benchmark Results

- **Matrix Size**: M=256, N=256, K=256
- **Shape**: m16n16k16
- **Grid**: 16x16, Block: 32 (1 warp)
- **Time**: 0.130 ms/iteration
- **Performance**: **257.41 GFLOPS**
- **Verification**: sum=4103416.75 (non-zero = correct)

### 4.2 TCGen05 (5th Gen Tensor Core)

> **Important**: TCGen05 is the 5th generation Tensor Core API for Blackwell, replacing WGMMA from Hopper.

#### TCGen05 vs WGMMA

| Feature | WGMMA (Hopper) | TCGen05 (Blackwell) |
|---------|-----------------|---------------------|
| PTX Section | 9.7.15 | 9.7.16 |
| API | wgmma.mma_async | tcgen05.mma |
| SASS | WGMMA | **OMMA (FP4), QMMA (FP8/FP6)** |
| Async | Yes | Yes |
| Sparse | 2:4 | 2:4 |
| FP4/FP6 | No | **Yes** |
| Block Scaling | No | **Yes (HW)** |

#### TCGen05 Precision Support

| Shape | FP16 | BF16 | TF32 | FP32 | FP64 | INT8 | FP4 | FP6 |
|-------|------|------|------|------|------|------|-----|-----|
| .32x32b | Yes | Yes | Yes | Yes | Yes | - | - | - |
| .16x64b | Yes | Yes | Yes | Yes | Yes | - | - | - |
| .16x128b | Yes | Yes | Yes | Yes | - | - | - | - |
| .16x256b | Yes | Yes | Yes | - | - | - | - | - |
| .16x32bx2 | Yes | Yes | Yes | Yes | - | - | - | - |
| **m16n8k32** | - | - | - | - | - | - | **Yes** | **Yes** |

#### TCGen05 Variants

| Variant | Description |
|---------|-------------|
| tcgen05.mma | Basic MMA |
| tcgen05.mma.sp | Sparse MMA (2:4) |
| tcgen05.mma.ws | Weight-only Quantization (W8A16) |
| tcgen05.mma.ws.sp | Weight-only + Sparse |

#### CTA Group Types

| CTA Group | Description | D Registers |
|-----------|-------------|-------------|
| cta_group::1 | Single CTA (1 warp group) | 4 |
| cta_group::2 | Dual CTA cluster (2 warp groups) | 8+ |

#### Operand Source Variants

| Variant | A Source | B Source | Description |
|---------|----------|----------|-------------|
| SS | SMEM | SMEM | Both from shared memory |
| TS | TMEM | SMEM | A from tensor memory |
| ST | SMEM | TMEM | B from tensor memory |
| TT | TMEM | TMEM | Both from tensor memory |

#### TMEM (Tensor Memory)

- **256 KB per SM** on-chip memory for Tensor Cores
- Organization: 512 columns × 128 rows of 32-bit cells
- D (accumulator) **must** be in TMEM
- A operand can optionally load from TMEM

#### Block Scaling (TCGen05 Hardware Support)

| Format | Block Size | Scale Factor |
|--------|------------|--------------|
| UE8M0 | 32 elements | 8-bit unsigned exp (2^x) |
| UE4M3 | 16 elements | 4-bit exp + 3-bit mantissa |

#### RTX 50 (GeForce) Limitations

| Feature | Support |
|---------|---------|
| TMA Multicast | ❌ Not supported |
| Cluster Shape | Must be 1x1x1 |
| Dynamic Datatypes | ❌ Not supported |
| Cluster MMA | ❌ Not supported (数据中心 only) |

### 4.3 MMA Shapes Reference

#### PTX ISA MMA Instructions

| Instruction | Shape | Dtypes |
|------------|-------|--------|
| wmma.load | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |
| wmma.store | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |
| wmma.mma | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |

#### SASS Instruction Mapping

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

#### NCU Tensor Core Metrics

| Metric | Meaning |
|--------|---------|
| sm__pipe_tensor_cycles_active.pct | Tensor Core utilization |
| sm__pipe_tensor_cycles_active.sum | Tensor Core active cycles |
| sm__inst_executed.mma.sum | MMA instruction count |
| dram__bytes.sum | Global memory bandwidth |
| lts__tcs_hit_rate.pct | L2 cache hit rate |

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

Supported operations: ADD, MIN, MAX, AND, OR, XOR

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

Matrix load/store operations for Tensor Core data.

| Instruction | Variant | Description |
|-------------|---------|-------------|
| ldmatrix.sync.aligned | m8n8.x1 | 8x8 tile, 1 matrix |
| ldmatrix.sync.aligned | m8n8.x2 | 8x8 tile, 2 matrices |
| ldmatrix.sync.aligned | m8n8.x4 | 8x8 tile, 4 matrices |
| stmatrix.sync.aligned | m8n8.x1 | Matrix store |

**Key Features**:
- Warp-level operation (32 threads cooperate)
- Transposed layout (MMA-friendly)
- 16-byte alignment required

### 6.3 TMA (Tensor Memory Access)

| Metric | Value |
|--------|-------|
| TMA 1D Copy (16MB peak) | 850.07 GB/s |
| TMA 2D Copy (1024x1024) | 626.36 GB/s |

---

## 7. Blackwell vs Hopper vs Ampere

### Compute Performance

| Metric | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) |
|--------|-------------------|----------------|----------------|
| FP32 Performance | ~17.6 TFLOPS | ~19.5 TFLOPS | ~19.5 TFLOPS |
| FP16 Performance | ~89.2 TFLOPS | ~99.8 TFLOPS | ~39.7 TFLOPS |
| FP64 Performance | **Limited** | Full | Full |
| Tensor Core Gen | 5th | 4th | 3rd |

### Memory System

| Metric | Blackwell | Hopper | Ampere |
|--------|-----------|--------|--------|
| Memory Type | GDDR7 | HBM2e | HBM2e |
| Bandwidth | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s |
| L1 Cache/SM | 128 KB | 256 KB | 192 KB |
| Shared Memory/SM | ~99 KB | ~227 KB | ~227 KB |
| L2 Cache | 65 MB | 50 MB | 80 MB |
| L2 Architecture | Monolithic | 2-partition | 2-partition |

### Tensor Core Comparison

| Feature | Blackwell (5th) | Hopper (4th) | Ampere (3rd) |
|---------|-----------------|--------------|--------------|
| WGMMA | ❌ | ✅ | ❌ |
| FP4 Support | ✅ | ❌ | ❌ |
| FP6 Support | ✅ | ❌ | ❌ |
| FP8 Support | ✅ | ✅ | ❌ |
| Block Scaling | ✅ (HW) | ❌ | ❌ |
| 2:4 Sparse | ✅ | ✅ | ✅ |

### Latency Comparison

| Operation | Blackwell | Hopper |
|-----------|-----------|--------|
| FP32 True Latency | 15.96 cycles | 31.62 cycles |
| INT32 Latency | 14 cycles | 16 cycles |
| FP64 True Latency | **~63 cycles** | ~8 cycles |
| L2 Cache Hit | ~358 cycles | ~273 cycles |
| Global Memory | ~877 cycles | ~659 cycles |
| MMA Completion | 1.21 cycles | 1.66 cycles |

### Power Efficiency (FP8/FP4/FP6)

| Precision | Blackwell | Hopper |
|-----------|-----------|--------|
| FP8 | ~46W | ~55W |
| FP4 | ~16.75W | N/A |
| FP6 e2m3 | ~39.38W | N/A |
| FP6 e3m2 | ~46.72W | N/A |

---

## 8. Microbenchmark Results

### Memory Hierarchy Bandwidth Summary

| Level | Bandwidth | Notes |
|-------|-----------|-------|
| Register | 298.96 GB/s | - |
| Shared Memory (L1) | **1.50 TB/s** | Fastest on-chip |
| L2 Cache | 766-797 GB/s | Streaming |
| Global Memory | 810-820 GB/s | Peak |
| PCIe | 35-50 GB/s | Host-Device |

### Occupancy vs Performance

| Block Size | Bandwidth | Time |
|------------|-----------|------|
| 32 | 292-300 GB/s | ~57 μs |
| 64 | 374-453 GB/s | ~35-45 μs |
| 128 | 674-890 GB/s | ~18-25 μs |
| 256 | 802-876 GB/s | ~19-21 μs |
| **512** | **828-900 GB/s** | ~18-21 μs |
| 1024 | 589-628 GB/s | ~26-29 μs |

**Optimal Block Size**: 256-512 threads

### Branch Divergence

| Branch Type | Bandwidth | Time |
|-------------|-----------|------|
| No Divergence | 746-761 GB/s | 0.021 ms |
| High Divergence | 796-810 GB/s | 0.021 ms |

**Analysis**: Divergence overhead not significant for simple kernels.

---

## 9. Test Environment

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 13.0 |
| Driver | 595.79 |
| OS | Windows 11 |
| GPU | RTX 5080 Laptop (GB203) |
| Compile Flags | -O3 -arch=sm_90 --use_fast_math |

---

## 10. References

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

## 中文版 (Chinese Version)

## 目录

1. [硬件规格](#1-硬件规格)
2. [内存子系统](#2-内存子系统)
   - [2.1 全局内存带宽](#21-全局内存带宽)
   - [2.2 L1/L2 缓存带宽](#22-l1l2-缓存带宽)
   - [2.3 共享内存性能](#23-共享内存性能)
   - [2.4 PCIe 带宽](#24-pcie-带宽)
3. [计算性能](#3-计算性能)
   - [3.1 FP32/FP64 性能](#31-fp32fp64-性能)
   - [3.2 FP16/BF16 性能](#32-fp16bf16-性能)
   - [3.3 INT32 性能](#33-int32-性能)
4. [Tensor Core (MMA)](#4-tensor-core-mma)
   - [4.1 WMMA (Warp级 MMA)](#41-wmma-warp级-mma)
   - [4.2 TCGen05 (第五代 Tensor Core)](#42-tcgen05-第五代-tensor-core)
   - [4.3 MMA Shape 参考](#43-mma-shape-参考)
5. [Warp 级操作](#5-warp-级操作)
   - [5.1 Warp Shuffle](#51-warp-shuffle)
   - [5.2 Warp Vote/Ballot](#52-warp-voteballot)
   - [5.3 Redux.sync](#53-reduxsync)
6. [内存操作](#6-内存操作)
   - [6.1 异步拷贝](#61-异步拷贝)
   - [6.2 LDMATRIX/STMATRIX](#62-ldmatrixstmatrix)
   - [6.3 TMA (张量内存访问)](#63-tma-张量内存访问)
7. [Blackwell vs Hopper vs Ampere 对比](#7-blackwell-vs-hopper-vs-ampere-对比)
8. [微基准测试结果](#8-微基准测试结果)
9. [测试环境](#9-测试环境)
10. [参考文献](#10-参考文献)

---

## 1. 硬件规格

### GPU 规格

| 参数 | 值 |
|------|-----|
| GPU 型号 | NVIDIA GeForce RTX 5080 Laptop GPU |
| 架构代号 | Blackwell |
| Compute Capability | 12.0 |
| SM 数量 | 60 |
| 每 SM CUDA 核心数 | 128 |
| 总 CUDA 核心数 | 7,680 |
| 全局内存 | 15.92 GB |
| 每 SM 共享内存 | 48 KB |
| 每 SM L1 缓存 | 128 KB |
| L2 缓存 | 65 MB (单体设计) |
| 每 Block 最大线程数 | 1,024 |
| 每 SM 最大线程数 | 1,536 |
| 每 SM 最大寄存器数 | 65,536 |
| Warp 大小 | 32 |
| 内存类型 | GDDR7 |
| 内存带宽 | ~8.2 TB/s |

### 跨代对比

| 参数 | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) |
|------|-------------------|----------------|----------------|
| Compute Capability | 12.0 | 9.0 | 8.0 |
| SM 数量 | 60 (完整 84) | 132 | 108 |
| 每 SM CUDA 核心数 | 128 | 128 | 64 |
| 内存类型 | GDDR7 | HBM2e | HBM2e |
| 内存带宽 | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s |
| 每 SM L1 缓存 | 128 KB | 256 KB | 192 KB |
| L2 缓存 | 65 MB | 50 MB | 80 MB |
| Tensor Core 代数 | 5th | 4th | 3rd |
| FP64 支持 | 有限 (仅 2/SM) | 完整 (64/SM) | 完整 |

---

## 2. 内存子系统

### 2.1 全局内存带宽

#### 全局内存带宽 vs 数据大小

| 数据大小 | 顺序读取 | 顺序写入 | 备注 |
|---------|---------|---------|------|
| 64 KB | 7.25 GB/s | 7.25 GB/s | L1 缓存 |
| 256 KB | 32.39 GB/s | 32.39 GB/s | L1 缓存 |
| 1 MB | 73.97 GB/s | 73.97 GB/s | L1/L2 边界 |
| 4 MB | 296.36 GB/s | 296.36 GB/s | L2 缓存 |
| 16 MB | 643.02 GB/s | 643.02 GB/s | 峰值 (第一) |
| 64 MB | 376.08 GB/s | 376.08 GB/s | L2 miss |
| 128 MB | 502.44 GB/s | 502.44 GB/s | - |
| 256 MB | 614.93 GB/s | 614.93 GB/s | 恢复 |

**分析**:
- 峰值带宽: 16MB 工作集时约 640-820 GB/s
- 64MB 时带宽下降表明 L2 缓存失效
- 128-256MB 后恢复，表明内存控制器有更大的有效缓存窗口

#### 跨距访问效率

| Stride | 带宽 | 效率 |
|--------|------|------|
| 1 | 822.37 GB/s | 100% (基线) |
| 2 | 582.20 GB/s | 86.0% |
| 4 | 581.24 GB/s | 85.9% |
| 8 | 544.01 GB/s | 80.4% |
| 16 | 421.00 GB/s | 62.2% |
| 32 | 239.11 GB/s | 35.3% |
| 64 | 154.13 GB/s | 22.8% |
| 128 | 76.88 GB/s | 11.4% |
| 256 | 39.62 GB/s | 5.9% |

**分析**: Stride 超过 16 后带宽急剧下降，表明缓存行跨距访问开销大。

### 2.2 L1/L2 缓存带宽

| 访问模式 | 带宽 | 备注 |
|---------|------|------|
| 全局直接读取 | 810.89 GB/s | 基线读取 |
| 全局直接写入 | 820.60 GB/s | 基线写入 |
| 共享内存 R/W | **1.50 TB/s** | L1 级带宽 |
| L2 Streaming (stride=1) | 766.78 GB/s | L2 缓存命中 |
| L2 Streaming (stride=1024) | 795.17 GB/s | 跨距访问 |
| __ldg Bypass | 822.43 GB/s | 绕过缓存 |
| L1 Preference | 780.32 GB/s | 寄存器优化 |

**分析**:
- 共享内存 (L1): **1.50 TB/s** - 显著高于全局内存
- L2 streaming 在不同跨距下保持高带宽
- __ldg (只读缓存绕过) 略优于普通读取

#### L2 工作集分析

| 数据大小 | 带宽 | 状态 |
|---------|------|------|
| 64 KB | 123.09 GB/s | L1 缓存 |
| 1 MB | 407.66 GB/s | L2 边界 |
| 4 MB | 678.20 GB/s | L2 缓存 |
| 8 MB | 747.53 GB/s | L2 缓存 |
| 16 MB | 797.97 GB/s | L2 miss → DRAM |

### 2.3 共享内存性能

| 操作 | 带宽 | 备注 |
|------|------|------|
| 共享内存 R/W | **1.50 TB/s** | L1 级带宽 |
| 广播写入 | 1.30 TB/s | 所有线程写入相同值 |
| Reduction 写入 | 1.30 GB/s | 读取: 332.91 GB/s |

### 2.4 PCIe 带宽

| 传输类型 | 带宽 | 每次传输时间 |
|---------|------|------------|
| Pageable H2D | 47-49 GB/s | 2.7-2.8 ms |
| Pinned H2D | 47-52 GB/s | 2.5-2.8 ms |
| Pageable D2H | 34-36 GB/s | 3.6-3.8 ms |
| Pinned D2H | 34-36 GB/s | 3.7-3.8 ms |
| D2D (设备内) | 336-361 GB/s | 0.37-0.40 ms |

**分析**:
- H2D (写入 GPU) 比 D2H 快约 30%
- Pinned 内存比 Pageable 略快
- D2D 带宽远高于 PCIe

---

## 3. 计算性能

### 3.1 FP32/FP64 性能

| 操作 | 吞吐量 | 延迟 | 备注 |
|------|--------|------|------|
| FP32 FMA | 88.55 GFLOPS | 0.068 ms | 融合乘加 |
| FP64 | 有限 | ~63 cycles | 仅 2 个 FP64 单元/SM |

**注意**: Blackwell 的 FP64 严重受限 (仅 2 单元/SM，而 Hopper 有 64 单元)，不适合 FP64 密集型工作负载。

### 3.2 FP16/BF16 性能

| 操作 | 吞吐量 | 备注 |
|------|--------|------|
| FP16 FMA | 204.19 GFLOPS | 比 FP32 快 3.3 倍 |
| WMMA FP16 | **257.41 GFLOPS** | 实测 (m16n16k16) |
| BF16 | 待实测 | - |

### 3.3 INT32 性能

| 操作 | 吞吐量 | 延迟 |
|------|--------|------|
| INT32 算术 | 121.52 GIOPS | 0.035 ms |

### 指令吞吐量汇总

| 指令类型 | 吞吐量 | 延迟 |
|---------|--------|------|
| FP32 FMA | 61.55 GFLOPS | 0.068 ms |
| INT32 算术 | 121.52 GIOPS | 0.035 ms |
| FP16 FMA | 204.19 GFLOPS | 0.021 ms |

---

## 4. Tensor Core (MMA)

### 4.1 WMMA (Warp级 MMA)

WMMA 是标准的 CUDA Tensor Core API。

| Shape | FP16 | BF16 | TF32 | FP64 | INT8 | INT4 |
|-------|------|------|------|------|------|------|
| m8n8k4 | Yes | - | - | Yes | - | - |
| m8n8k16 | Yes | - | - | - | Yes | - |
| m8n8k32 | Yes | - | - | - | Yes | Yes |
| m16n8k4 | Yes | - | Yes | - | - | - |
| m16n8k8 | Yes | Yes | Yes | - | Yes | - |
| m16n8k16 | Yes | Yes | Yes | - | Yes | Yes |
| m16n8k32 | Yes | Yes | - | - | Yes | Yes |
| m16n16k16 | Yes | Yes | Yes | Yes | Yes | - |

#### WMMA FP16 基准测试结果

- **矩阵尺寸**: M=256, N=256, K=256
- **Shape**: m16n16k16
- **Grid**: 16x16, Block: 32 (1 个 warp)
- **时间**: 0.130 ms/iteration
- **性能**: **257.41 GFLOPS**
- **验证**: sum=4103416.75 (非零 = 正确)

### 4.2 TCGen05 (第五代 Tensor Core)

> **重要**: TCGen05 是 Blackwell 的第 5 代 Tensor Core API，取代了 Hopper 的 WGMMA。

#### TCGen05 vs WGMMA

| 特性 | WGMMA (Hopper) | TCGen05 (Blackwell) |
|------|----------------|---------------------|
| PTX 章节 | 9.7.15 | 9.7.16 |
| API | wgmma.mma_async | tcgen05.mma |
| SASS | WGMMA | **OMMA (FP4), QMMA (FP8/FP6)** |
| 异步 | Yes | Yes |
| 稀疏 | 2:4 | 2:4 |
| FP4/FP6 | No | **Yes** |
| Block Scaling | No | **Yes (硬件)** |

#### TCGen05 精度支持

| Shape | FP16 | BF16 | TF32 | FP32 | FP64 | INT8 | FP4 | FP6 |
|-------|------|------|------|------|------|------|-----|-----|
| .32x32b | Yes | Yes | Yes | Yes | Yes | - | - | - |
| .16x64b | Yes | Yes | Yes | Yes | Yes | - | - | - |
| .16x128b | Yes | Yes | Yes | Yes | - | - | - | - |
| .16x256b | Yes | Yes | Yes | - | - | - | - | - |
| .16x32bx2 | Yes | Yes | Yes | Yes | - | - | - | - |
| **m16n8k32** | - | - | - | - | - | - | **Yes** | **Yes** |

#### TCGen05 变体

| 变体 | 描述 |
|------|------|
| tcgen05.mma | 基本 MMA |
| tcgen05.mma.sp | 稀疏 MMA (2:4) |
| tcgen05.mma.ws | 仅权重量化 (W8A16) |
| tcgen05.mma.ws.sp | Weight-only + 稀疏 |

#### CTA Group 类型

| CTA Group | 描述 | D 寄存器数 |
|-----------|------|-----------|
| cta_group::1 | 单 CTA (1 个 warp group) | 4 |
| cta_group::2 | 双 CTA 集群 (2 个 warp groups) | 8+ |

#### 操作数源变体

| 变体 | A 来源 | B 来源 | 描述 |
|------|-------|-------|------|
| SS | SMEM | SMEM | 两者都从共享内存 |
| TS | TMEM | SMEM | A 从张量内存 |
| ST | SMEM | TMEM | B 从张量内存 |
| TT | TMEM | TMEM | 两者都从张量内存 |

#### TMEM (张量内存)

- 每 SM **256 KB** 片上内存用于 Tensor Core
- 组织: 512 列 × 128 行 × 32 位单元
- D (累加器) **必须** 放在 TMEM
- A 操作数可选择从 TMEM 或 SMEM 加载

#### Block Scaling (TCGen05 硬件支持)

| 格式 | Block 大小 | 缩放因子 |
|------|------------|----------|
| UE8M0 | 32 元素 | 8-bit unsigned exp (2^x) |
| UE4M3 | 16 元素 | 4-bit exp + 3-bit mantissa |

#### RTX 50 (GeForce) 限制

| 特性 | 支持情况 |
|------|---------|
| TMA Multicast | ❌ 不支持 |
| Cluster Shape | 必须 1x1x1 |
| Dynamic Datatypes | ❌ 不支持 |
| Cluster MMA | ❌ 不支持 (仅数据中心 GPU) |

### 4.3 MMA Shape 参考

#### PTX ISA MMA 指令

| 指令 | Shape | 数据类型 |
|------|-------|---------|
| wmma.load | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |
| wmma.store | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |
| wmma.mma | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |

#### SASS 指令映射

| PTX | SASS | 描述 |
|-----|------|------|
| wmma.mma.f16 | HMMA | 半精度 MMA |
| wmma.mma.bf16 | BMMA | BFloat16 MMA |
| wmma.mma.tf32 | HMMA | TensorFloat-32 MMA |
| wmma.mma.f64 | DMMA | 双精度 MMA |
| wmma.mma.s32 | IMMA | INT32 MMA |
| mma.mma | HMMA/IMMA/DMMA | 通用 MMA |
| wgmma.mma_async | WGMMA | 异步 Warpgroup MMA (仅 Hopper) |
| ld.matrix | LDMATRIX | 矩阵加载 |
| st.matrix | STMATRIX | 矩阵存储 |

#### NCU Tensor Core 指标

| 指标 | 含义 |
|------|------|
| sm__pipe_tensor_cycles_active.pct | Tensor Core 利用率 |
| sm__pipe_tensor_cycles_active.sum | Tensor Core 活跃周期数 |
| sm__inst_executed.mma.sum | MMA 指令数 |
| dram__bytes.sum | 全局内存带宽 |
| lts__tcs_hit_rate.pct | L2 缓存命中率 |

---

## 5. Warp 级操作

### 5.1 Warp Shuffle

| 操作 | 带宽 | 时间/Kernel |
|------|------|------------|
| Shuffle Reduce | 747.46 GB/s | 0.022 ms |
| Butterfly Reduce | 730.21 GB/s | 0.023 ms |
| 通用 Shuffle | 305.59 GB/s | - |
| **增强型 Shuffle** | **418.49 GB/s** | - |

**分析**: Blackwell 的增强型 Shuffle 比通用 Warp Shuffle 快 **37%** (418.49 vs 305.59 GB/s)。

### 5.2 Warp Vote/Ballot

| 操作 | 性能 |
|------|------|
| Ballot Sync | 0.020 ms/kernel |

### 5.3 Redux.sync

Redux.sync 在**单条指令**内完成 warp 级归约。

| 方法 | 指令数 | 延迟 | 优势 |
|------|--------|------|------|
| Shuffle 循环 | log2(32) = 5 次 | 较高 | 兼容性好 |
| **Redux.sync** | **1 条指令** | **最低** | **硬件加速** |

支持的操作: ADD, MIN, MAX, AND, OR, XOR

---

## 6. 内存操作

### 6.1 异步拷贝

| 操作 | 带宽 | 备注 |
|------|------|------|
| Async Copy | 422.69 GB/s | 最高性能 |
| L2 Streaming | 316.46 GB/s | 缓存流式访问 |
| Register Bandwidth | 298.96 GB/s | 寄存器级 |
| Software Prefetch | 251.10 GB/s | 较低但可预测 |

### 6.2 LDMATRIX/STMATRIX

Tensor Core 数据的矩阵加载/存储操作。

| 指令 | 变体 | 描述 |
|------|------|------|
| ldmatrix.sync.aligned | m8n8.x1 | 8x8 tile, 1 个矩阵 |
| ldmatrix.sync.aligned | m8n8.x2 | 8x8 tile, 2 个矩阵 |
| ldmatrix.sync.aligned | m8n8.x4 | 8x8 tile, 4 个矩阵 |
| stmatrix.sync.aligned | m8n8.x1 | 矩阵存储 |

**关键特性**:
- Warp 级操作 (32 线程协作)
- 转置布局 (MMA 友好)
- 需要 16 字节对齐

### 6.3 TMA (张量内存访问)

| 指标 | 值 |
|------|---|
| TMA 1D 拷贝 (16MB 峰值) | 850.07 GB/s |
| TMA 2D 拷贝 (1024x1024) | 626.36 GB/s |

---

## 7. Blackwell vs Hopper vs Ampere 对比

### 计算性能

| 指标 | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) |
|------|-------------------|----------------|----------------|
| FP32 性能 | ~17.6 TFLOPS | ~19.5 TFLOPS | ~19.5 TFLOPS |
| FP16 性能 | ~89.2 TFLOPS | ~99.8 TFLOPS | ~39.7 TFLOPS |
| FP64 性能 | **有限** | 完整 | 完整 |
| Tensor Core 代数 | 5th | 4th | 3rd |

### 内存系统

| 指标 | Blackwell | Hopper | Ampere |
|------|-----------|--------|--------|
| 内存类型 | GDDR7 | HBM2e | HBM2e |
| 带宽 | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s |
| 每 SM L1 缓存 | 128 KB | 256 KB | 192 KB |
| 每 SM 共享内存 | ~99 KB | ~227 KB | ~227 KB |
| L2 缓存 | 65 MB | 50 MB | 80 MB |
| L2 架构 | 单体 | 2 分区 | 2 分区 |

### Tensor Core 对比

| 特性 | Blackwell (5th) | Hopper (4th) | Ampere (3rd) |
|------|-----------------|--------------|--------------|
| WGMMA | ❌ | ✅ | ❌ |
| FP4 支持 | ✅ | ❌ | ❌ |
| FP6 支持 | ✅ | ❌ | ❌ |
| FP8 支持 | ✅ | ✅ | ❌ |
| Block Scaling | ✅ (硬件) | ❌ | ❌ |
| 2:4 稀疏 | ✅ | ✅ | ✅ |

### 延迟对比

| 操作 | Blackwell | Hopper |
|------|-----------|--------|
| FP32 True Latency | 15.96 cycles | 31.62 cycles |
| INT32 Latency | 14 cycles | 16 cycles |
| FP64 True Latency | **~63 cycles** | ~8 cycles |
| L2 缓存命中 | ~358 cycles | ~273 cycles |
| 全局内存 | ~877 cycles | ~659 cycles |
| MMA Completion | 1.21 cycles | 1.66 cycles |

### 能效对比 (FP8/FP4/FP6)

| 精度 | Blackwell | Hopper |
|------|-----------|--------|
| FP8 | ~46W | ~55W |
| FP4 | ~16.75W | N/A |
| FP6 e2m3 | ~39.38W | N/A |
| FP6 e3m2 | ~46.72W | N/A |

---

## 8. 微基准测试结果

### 内存层级带宽汇总

| 层级 | 带宽 | 备注 |
|------|------|------|
| 寄存器 | 298.96 GB/s | - |
| 共享内存 (L1) | **1.50 TB/s** | 最快的片上 |
| L2 缓存 | 766-797 GB/s | Streaming |
| 全局内存 | 810-820 GB/s | 峰值 |
| PCIe | 35-50 GB/s | Host-Device |

### Occupancy vs 性能

| Block Size | 带宽 | 时间 |
|------------|--------|------|
| 32 | 292-300 GB/s | ~57 μs |
| 64 | 374-453 GB/s | ~35-45 μs |
| 128 | 674-890 GB/s | ~18-25 μs |
| 256 | 802-876 GB/s | ~19-21 μs |
| **512** | **828-900 GB/s** | ~18-21 μs |
| 1024 | 589-628 GB/s | ~26-29 μs |

**最佳 Block Size**: 256-512 线程

### 分支分歧

| 分支类型 | 带宽 | 时间 |
|---------|------|------|
| 无分歧 | 746-761 GB/s | 0.021 ms |
| 高分歧 | 796-810 GB/s | 0.021 ms |

**分析**: 对于简单 kernel，分歧开销不明显。

---

## 9. 测试环境

| 组件 | 版本 |
|------|------|
| CUDA Toolkit | 13.0 |
| 驱动版本 | 595.79 |
| 操作系统 | Windows 11 |
| GPU | RTX 5080 Laptop (GB203) |
| 编译选项 | -O3 -arch=sm_90 --use_fast_math |

---

## 10. 参考文献

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
- [arXiv:2507.10789 - Blackwell 微基准测试](https://arxiv.org/abs/2507.10789)
