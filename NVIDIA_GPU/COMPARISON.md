# GPU Architecture Cross-Generation Comparison

> **Last Updated**: 2026-03-23
> **Report Language**: English

---

## Table of Contents

1. [Hardware Specifications Comparison](#1-hardware-specifications-comparison)
2. [Memory System Comparison](#2-memory-system-comparison)
3. [Compute Performance Comparison](#3-compute-performance-comparison)
4. [Tensor Core Feature Comparison](#4-tensor-core-feature-comparison)
5. [Latency Comparison](#5-latency-comparison)
6. [Power Efficiency Comparison](#6-power-efficiency-comparison)
7. [Transformer Engine Support](#7-transformer-engine-support)

---

## 1. Hardware Specifications Comparison

### GPU Architecture Overview

| Parameter | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) | Volta (GV100) |
|-----------|-------------------|----------------|----------------|---------------|
| Compute Capability | 12.0 | 9.0 | 8.0 | 7.0 |
| Architecture Code | Blackwell | Hopper | Ampere | Volta |
| SM Count | 60 (full: 84) | 132 | 108 | 80 |
| CUDA Cores/SM | 128 | 128 | 64 | 64 |
| Total CUDA Cores | 7,680 (10,752 full) | 16,896 | 6,912 | 5,120 |
| Transistors | 92B | 80B | 54.2B | 21.1B |
| Die Size | ~750 mm² | ~814 mm² | ~826 mm² | ~815 mm² |
| Process Node | TSMC 4NP | TSMC 4N | Samsung 8N | TSMC 12FFN |

### Memory Configuration

| Parameter | Blackwell | Hopper | Ampere | Volta |
|-----------|-----------|--------|--------|-------|
| Memory Type | GDDR7 | HBM2e | HBM2 | HBM2 |
| Memory Size | 16 GB | 80 GB | 80 GB | 32 GB |
| Memory Bandwidth | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s | 900 GB/s |
| L1 Cache/SM | 128 KB | 256 KB | 192 KB | 128 KB |
| Shared Memory/SM | ~99 KB | ~227 KB | ~227 KB | 96 KB |
| L2 Cache | 65 MB | 50 MB | 80 MB | 6 MB |
| L2 Architecture | Monolithic | 2-partition | 2-partition | 2-partition |

### Compute Resources

| Parameter | Blackwell | Hopper | Ampere | Volta |
|-----------|-----------|--------|--------|-------|
| FP32 Units/SM | 128 | 128 | 64 | 64 |
| FP64 Units/SM | 2 (limited) | 64 (full) | 64 (full) | 32 (full) |
| INT32 Units/SM | Unified with FP32 | Independent | Independent | Independent |
| Tensor Core Gen | 5th | 4th | 3rd | 1st |
| RT Core Gen | 5th | 4th | 3rd | 2nd |

---

## 2. Memory System Comparison

### Cache Hierarchy

| Level | Blackwell | Hopper | Ampere | Volta |
|-------|-----------|--------|--------|-------|
| L0 (Instruction) | 128 KB/SM | 128 KB/SM | 128 KB/SM | 128 KB/SM |
| L1 (Configurable) | 128 KB/SM | 256 KB/SM | 192 KB/SM | 128 KB/SM |
| Shared Memory | 48 KB/SM | 228 KB/SM | 228 KB/SM | 96 KB/SM |
| L2 Cache | 65 MB | 50 MB | 80 MB | 6 MB |
| L2 Partitions | 1 | 2 | 2 | 2 |

### Key Memory Changes (Blackwell vs Hopper)

| Feature | Change | Impact |
|---------|--------|--------|
| L1 Cache/SM | 256 KB → 128 KB (-50%) | Less cache per SM |
| Shared Memory/SM | 227 KB → ~99 KB (-56%) | Reduced max shared memory |
| L2 Cache | 50 MB → 65 MB (+30%) | Larger but monolithic |
| L2 Architecture | 2-partition → Monolithic | Different access patterns |
| Memory Bandwidth | 15.8 TB/s → 8.2 TB/s (-48%) | GDDR7 vs HBM2e |

### Memory Bandwidth by Operation

| Operation | Blackwell | Hopper | Ampere |
|-----------|-----------|--------|--------|
| Global Memory Peak | ~820 GB/s | ~15.8 TB/s | ~2.0 TB/s |
| Shared Memory (L1) | **1.50 TB/s** | ~20 TB/s (est.) | ~15 TB/s (est.) |
| L2 Cache Hit | ~798 GB/s | ~12 TB/s (est.) | ~3.5 TB/s (est.) |

---

## 3. Compute Performance Comparison

### Theoretical Peak Performance

| Precision | Blackwell | Hopper | Ampere | Volta |
|-----------|-----------|--------|--------|-------|
| FP32 | ~17.6 TFLOPS | ~19.5 TFLOPS | ~19.5 TFLOPS | ~15.7 TFLOPS |
| FP64 | **Limited** | ~19.5 TFLOPS | ~9.7 TFLOPS | ~7.8 TFLOPS |
| FP16 | ~89.2 TFLOPS | ~99.8 TFLOPS | ~39.7 TFLOPS | ~31.4 TFLOPS |
| BF16 | ~89.2 TFLOPS | ~99.8 TFLOPS | ~39.7 TFLOPS | N/A |
| TF32 | ~35.7 TFLOPS | ~39.9 TFLOPS | ~15.9 TFLOPS | N/A |
| FP8 | ~178.4 TFLOPS | ~199.6 TFLOPS | N/A | N/A |
| FP4 | ~356.8 TFLOPS | N/A | N/A | N/A |
| INT8 | ~178.4 TOPS | ~199.6 TOPS | ~79.4 TOPS | ~62.9 TOPS |

###实测 Performance (GPUPeek)

| Operation | Blackwell | Notes |
|-----------|-----------|-------|
| FP32 FMA | 88.55 GFLOPS | Old test |
| FP32 FMA | 61.55 GFLOPS | New test |
| FP16 FMA | 204.19 GFLOPS | - |
| INT32 Arithmetic | 121.52 GIOPS | - |
| WMMA FP16 | **257.41 GFLOPS** | m16n16k16 |

### FP64 Availability Warning

| Architecture | FP64 Units/SM | FP64 True Latency | Suitable for FP64? |
|--------------|---------------|-------------------|-------------------|
| Blackwell | 2 (limited) | ~63 cycles | ❌ No |
| Hopper | 64 (full) | ~8 cycles | ✅ Yes |
| Ampere | 64 (full) | ~10 cycles | ✅ Yes |
| Volta | 32 (full) | ~12 cycles | ✅ Yes |

**Conclusion**: Blackwell (RTX 50 series) is NOT suitable for FP64-heavy workloads. Use Hopper (H100) or Ampere (A100) for FP64 applications.

---

## 4. Tensor Core Feature Comparison

### Tensor Core Generation Timeline

| Generation | Architecture | Year | Key Features |
|-----------|--------------|------|--------------|
| 1st Gen | Volta (V100) | 2017 | FP16, INT8, INT4 |
| 2nd Gen | Turing (T4) | 2018 | FP16, INT8, INT4, INT1 |
| 3rd Gen | Ampere (A100) | 2020 | TF32, FP64, Sparse, WGMMA |
| 4th Gen | Hopper (H100) | 2022 | FP8, WGMMA Async |
| 5th Gen | Blackwell (RTX 50) | 2025 | FP4, FP6, Block Scaling, OMMA/QMMA |

### Feature Matrix

| Feature | Blackwell (5th) | Hopper (4th) | Ampere (3rd) | Volta (1st) |
|---------|-----------------|--------------|--------------|-------------|
| WGMMA | ❌ | ✅ | ❌ | ❌ |
| FP4 Support | ✅ | ❌ | ❌ | ❌ |
| FP6 Support | ✅ | ❌ | ❌ | ❌ |
| FP8 Support | ✅ | ✅ | ❌ | ❌ |
| Block Scaling | ✅ (HW) | ❌ | ❌ | ❌ |
| 2:4 Sparse | ✅ | ✅ | ✅ | ❌ |
| TF32 | ✅ | ✅ | ✅ | ❌ |
| FP64 MMA | ✅ | ✅ | ✅ | ❌ |
| Async MMA | ✅ | ✅ | ❌ | ❌ |
| TMEM | ✅ | ❌ | ❌ | ❌ |

### TCGen05 vs WGMMA

| Feature | WGMMA (Hopper) | TCGen05 (Blackwell) |
|---------|-----------------|---------------------|
| PTX Section | 9.7.15 | 9.7.16 |
| API | wgmma.mma_async | tcgen05.mma |
| SASS | WGMMA | **OMMA (FP4), QMMA (FP8/FP6)** |
| Async | Yes | Yes |
| Sparse | 2:4 | 2:4 |
| FP4/FP6 | No | **Yes** |
| Block Scaling | No | **Yes (Hardware)** |
| CTA Groups | 1, 2 | 1, 2 |
| Operand Sources | SS, TT | SS, TS, ST, TT |

### MMA Shape Support

| Shape | Blackwell | Hopper | Ampere |
|-------|-----------|--------|--------|
| m16n16k16 (WMMA) | ✅ | ✅ | ✅ |
| m16n8k8 (MMA) | ✅ | ✅ | ✅ |
| m16n8k16 | ✅ | ✅ | ✅ |
| m16n8k32 | ✅ | ✅ | ✅ |
| m64nNk16 (WGMMA) | ❌ | ✅ | ❌ |
| m64nNk8 | ❌ | ✅ | ❌ |
| m64nNk32 | ❌ | ✅ | ❌ |
| m16n8k32 (FP4/FP6) | ✅ | ❌ | ❌ |

---

## 5. Latency Comparison

### Instruction Latency

| Operation | Blackwell | Hopper | Ampere |
|-----------|-----------|--------|--------|
| FP32 True Latency | 15.96 cycles | 31.62 cycles | ~20 cycles |
| INT32 Latency | 14 cycles | 16 cycles | ~12 cycles |
| FP64 True Latency | **~63 cycles** | ~8 cycles | ~10 cycles |
| FP64 Completion | ~11 cycles | ~13 cycles | ~10 cycles |

### Memory Latency

| Operation | Blackwell | Hopper | Ampere |
|-----------|-----------|--------|--------|
| L1 Cache Hit | ~25 cycles (est.) | ~25 cycles | ~25 cycles |
| L2 Cache Hit | ~358 cycles | ~273 cycles | ~200 cycles |
| Global Memory | ~877 cycles | ~659 cycles | ~550 cycles |
| Shared Memory | ~25 cycles (est.) | ~25 cycles | ~25 cycles |

### Tensor Core Latency

| Operation | Blackwell | Hopper | Ampere |
|-----------|-----------|--------|--------|
| MMA Completion | **1.21 cycles** | 1.66 cycles | ~2 cycles |
| MMA True Latency | ~6 cycles (est.) | ~8 cycles | ~10 cycles |

**Analysis**: Blackwell has significantly lower FP32/INT32 latency than Hopper, but much higher FP64 latency due to reduced FP64 units.

---

## 6. Power Efficiency Comparison

### Power Consumption by Precision (Blackwell)

| Precision | Power | Notes |
|-----------|-------|-------|
| FP8 | ~46W | Per chip |
| FP6 e2m3 | ~39.38W | Per chip |
| FP6 e3m2 | ~46.72W | Per chip |
| FP4 | ~16.75W | Per chip |

### Power Efficiency (Blackwell vs Hopper)

| Precision | Blackwell | Hopper | Improvement |
|-----------|-----------|--------|-------------|
| FP8 | ~46W | ~55W | **+20%** more efficient |
| FP4 | ~16.75W | N/A | New capability |
| FP6 e2m3 | ~39.38W | N/A | New capability |
| FP6 e3m2 | ~46.72W | N/A | New capability |

**Conclusion**: Blackwell is approximately **20% more power efficient** than Hopper for FP8 operations.

### Performance per Watt

| Metric | Blackwell | Hopper | Notes |
|--------|-----------|--------|-------|
| FP16 TOPS/W | ~1.94 | ~1.77 | +10% efficiency |
| FP8 TOPS/W | ~3.88 | ~3.63 | +7% efficiency |
| FP4 TOPS/W | ~7.75 | N/A | New capability |

---

## 7. Transformer Engine Support

### TE Version Comparison

| Version | Architecture | Supported Precisions |
|---------|--------------|---------------------|
| TE 1st Gen | Hopper (H100) | FP8, FP16, BF16, FP32, FP64 |
| TE 2nd Gen | Blackwell (B100/B200) | FP4, FP6, FP8, FP16, BF16, FP32, FP64 |
| TE 3rd Gen | Blackwell Ultra | FP4, FP6, FP8, FP16, BF16, FP32, FP64, FP64 |

### Blackwell New Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| FP4 Support | 4-bit floating point | 4x memory reduction vs FP16 |
| FP6 Support | 6-bit floating point | 2.67x memory reduction vs FP16 |
| Block Scaling | Hardware dequantization | Efficient low-precision inference |
| TMEM | 256KB/SM on-chip | Faster tensor operations |

---

## Summary

### When to Use Each Architecture

| Use Case | Recommended Architecture |
|----------|------------------------|
| FP64 Heavy Workloads | Hopper (H100), Ampere (A100) |
| FP8/FP4/FP6 Inference | **Blackwell (RTX 50, B100)** |
| General ML Training | Hopper (H100), Blackwell (B200) |
| Cost-Sensitive | Ampere (A100, RTX 40) |
| LLM Inference | **Blackwell (RTX 50)** |
| Scientific Computing | Hopper (H100), Ampere (A100) |

### Key Takeaways

1. **Blackwell excels at low-precision (FP4/FP6/FP8) inference**
2. **Hopper (H100) is best for FP64 and WGMMA workloads**
3. **Blackwell has lower FP32/INT32 latency but higher FP64 latency**
4. **Blackwell is ~20% more power efficient for FP8 than Hopper**
5. **RTX 50 series (GeForce) lacks TMA multicast and Cluster MMA**

---

## References

- [arXiv:2507.10789 - Blackwell Microbenchmarks](https://arxiv.org/abs/2507.10789)
- [NVIDIA Hopper Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA Ampere Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ampere/pdf/NVIDIA-Ampere-Architecture-Whitepaper.pdf)
- [CUDA Programming Guide](../ref/cuda_programming_guide.html)
- [PTX ISA](../ref/ptx_isa.html)
