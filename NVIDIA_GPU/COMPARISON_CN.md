# GPU 架构跨代对比

> **最后更新**: 2026-03-23
> **报告语言**: 中文

---

## 目录

1. [硬件规格对比](#1-硬件规格对比)
2. [内存系统对比](#2-内存系统对比)
3. [计算性能对比](#3-计算性能对比)
4. [Tensor Core 特性对比](#4-tensor-core-特性对比)
5. [延迟对比](#5-延迟对比)
6. [能效对比](#6-能效对比)
7. [Transformer Engine 支持](#7-transformer-engine-支持)

---

## 1. 硬件规格对比

### GPU 架构概览

| 参数 | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) | Volta (GV100) |
|------|-------------------|----------------|----------------|---------------|
| Compute Capability | 12.0 | 9.0 | 8.0 | 7.0 |
| 架构代号 | Blackwell | Hopper | Ampere | Volta |
| SM 数量 | 60 (完整: 84) | 132 | 108 | 80 |
| 每 SM CUDA 核心数 | 128 | 128 | 64 | 64 |
| 总 CUDA 核心数 | 7,680 (10,752 完整) | 16,896 | 6,912 | 5,120 |
| 晶体管数 | 92B | 80B | 54.2B | 21.1B |
| Die 尺寸 | ~750 mm² | ~814 mm² | ~826 mm² | ~815 mm² |
| 制程节点 | TSMC 4NP | TSMC 4N | Samsung 8N | TSMC 12FFN |

### 内存配置

| 参数 | Blackwell | Hopper | Ampere | Volta |
|------|-----------|--------|--------|-------|
| 内存类型 | GDDR7 | HBM2e | HBM2 | HBM2 |
| 内存大小 | 16 GB | 80 GB | 80 GB | 32 GB |
| 内存带宽 | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s | 900 GB/s |
| 每 SM L1 缓存 | 128 KB | 256 KB | 192 KB | 128 KB |
| 每 SM 共享内存 | ~99 KB | ~227 KB | ~227 KB | 96 KB |
| L2 缓存 | 65 MB | 50 MB | 80 MB | 6 MB |
| L2 架构 | 单体 | 2 分区 | 2 分区 | 2 分区 |

### 计算资源

| 参数 | Blackwell | Hopper | Ampere | Volta |
|------|-----------|--------|--------|-------|
| 每 SM FP32 单元数 | 128 | 128 | 64 | 64 |
| 每 SM FP64 单元数 | 2 (有限) | 64 (完整) | 64 (完整) | 32 (完整) |
| 每 SM INT32 单元数 | 与 FP32 统一 | 独立 | 独立 | 独立 |
| Tensor Core 代数 | 5th | 4th | 3rd | 1st |
| RT Core 代数 | 5th | 4th | 3rd | 2nd |

---

## 2. 内存系统对比

### 缓存层级

| 层级 | Blackwell | Hopper | Ampere | Volta |
|------|-----------|--------|--------|-------|
| L0 (指令) | 128 KB/SM | 128 KB/SM | 128 KB/SM | 128 KB/SM |
| L1 (可配置) | 128 KB/SM | 256 KB/SM | 192 KB/SM | 128 KB/SM |
| 共享内存 | 48 KB/SM | 228 KB/SM | 228 KB/SM | 96 KB/SM |
| L2 缓存 | 65 MB | 50 MB | 80 MB | 6 MB |
| L2 分区 | 1 | 2 | 2 | 2 |

### 关键内存变化 (Blackwell vs Hopper)

| 特性 | 变化 | 影响 |
|------|------|------|
| 每 SM L1 缓存 | 256 KB → 128 KB (-50%) | 每 SM 缓存减少 |
| 每 SM 共享内存 | 227 KB → ~99 KB (-56%) | 最大共享内存减少 |
| L2 缓存 | 50 MB → 65 MB (+30%) | 更大但单体架构 |
| L2 架构 | 2 分区 → 单体 | 不同的访问模式 |
| 内存带宽 | 15.8 TB/s → 8.2 TB/s (-48%) | GDDR7 vs HBM2e |

### 各操作内存带宽

| 操作 | Blackwell | Hopper | Ampere |
|------|-----------|--------|--------|
| 全局内存峰值 | ~820 GB/s | ~15.8 TB/s | ~2.0 TB/s |
| 共享内存 (L1) | **1.50 TB/s** | ~20 TB/s (估计) | ~15 TB/s (估计) |
| L2 缓存命中 | ~798 GB/s | ~12 TB/s (估计) | ~3.5 TB/s (估计) |

---

## 3. 计算性能对比

### 理论峰值性能

| 精度 | Blackwell | Hopper | Ampere | Volta |
|------|-----------|--------|--------|-------|
| FP32 | ~17.6 TFLOPS | ~19.5 TFLOPS | ~19.5 TFLOPS | ~15.7 TFLOPS |
| FP64 | **有限** | ~19.5 TFLOPS | ~9.7 TFLOPS | ~7.8 TFLOPS |
| FP16 | ~89.2 TFLOPS | ~99.8 TFLOPS | ~39.7 TFLOPS | ~31.4 TFLOPS |
| BF16 | ~89.2 TFLOPS | ~99.8 TFLOPS | ~39.7 TFLOPS | N/A |
| TF32 | ~35.7 TFLOPS | ~39.9 TFLOPS | ~15.9 TFLOPS | N/A |
| FP8 | ~178.4 TFLOPS | ~199.6 TFLOPS | N/A | N/A |
| FP4 | ~356.8 TFLOPS | N/A | N/A | N/A |
| INT8 | ~178.4 TOPS | ~199.6 TOPS | ~79.4 TOPS | ~62.9 TOPS |

### 实测性能 (GPUPeek)

| 操作 | Blackwell | 备注 |
|------|-----------|------|
| FP32 FMA | 88.55 GFLOPS | 旧测试 |
| FP32 FMA | 61.55 GFLOPS | 新测试 |
| FP16 FMA | 204.19 GFLOPS | - |
| INT32 算术 | 121.52 GIOPS | - |
| WMMA FP16 | **257.41 GFLOPS** | m16n16k16 |

### FP64 可用性警告

| 架构 | 每 SM FP64 单元数 | FP64 True Latency | 适合 FP64? |
|------|-------------------|-------------------|------------|
| Blackwell | 2 (有限) | ~63 cycles | ❌ 否 |
| Hopper | 64 (完整) | ~8 cycles | ✅ 是 |
| Ampere | 64 (完整) | ~10 cycles | ✅ 是 |
| Volta | 32 (完整) | ~12 cycles | ✅ 是 |

**结论**: Blackwell (RTX 50 系列) 不适合 FP64 密集型工作负载。使用 Hopper (H100) 或 Ampere (A100) 处理 FP64 应用。

---

## 4. Tensor Core 特性对比

### Tensor Core 代际时间线

| 代数 | 架构 | 年份 | 关键特性 |
|------|------|------|----------|
| 1st Gen | Volta (V100) | 2017 | FP16, INT8, INT4 |
| 2nd Gen | Turing (T4) | 2018 | FP16, INT8, INT4, INT1 |
| 3rd Gen | Ampere (A100) | 2020 | TF32, FP64, Sparse, WGMMA |
| 4th Gen | Hopper (H100) | 2022 | FP8, WGMMA Async |
| 5th Gen | Blackwell (RTX 50) | 2025 | FP4, FP6, Block Scaling, OMMA/QMMA |

### 特性矩阵

| 特性 | Blackwell (5th) | Hopper (4th) | Ampere (3rd) | Volta (1st) |
|------|-----------------|--------------|--------------|-------------|
| WGMMA | ❌ | ✅ | ❌ | ❌ |
| FP4 支持 | ✅ | ❌ | ❌ | ❌ |
| FP6 支持 | ✅ | ❌ | ❌ | ❌ |
| FP8 支持 | ✅ | ✅ | ❌ | ❌ |
| Block Scaling | ✅ (硬件) | ❌ | ❌ | ❌ |
| 2:4 稀疏 | ✅ | ✅ | ✅ | ❌ |
| TF32 | ✅ | ✅ | ✅ | ❌ |
| FP64 MMA | ✅ | ✅ | ✅ | ❌ |
| 异步 MMA | ✅ | ✅ | ❌ | ❌ |
| TMEM | ✅ | ❌ | ❌ | ❌ |

### TCGen05 vs WGMMA

| 特性 | WGMMA (Hopper) | TCGen05 (Blackwell) |
|------|----------------|---------------------|
| PTX 章节 | 9.7.15 | 9.7.16 |
| API | wgmma.mma_async | tcgen05.mma |
| SASS | WGMMA | **OMMA (FP4), QMMA (FP8/FP6)** |
| 异步 | Yes | Yes |
| 稀疏 | 2:4 | 2:4 |
| FP4/FP6 | No | **Yes** |
| Block Scaling | No | **Yes (硬件)** |
| CTA Groups | 1, 2 | 1, 2 |
| 操作数源 | SS, TT | SS, TS, ST, TT |

### MMA Shape 支持

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

## 5. 延迟对比

### 指令延迟

| 操作 | Blackwell | Hopper | Ampere |
|------|-----------|--------|--------|
| FP32 True Latency | 15.96 cycles | 31.62 cycles | ~20 cycles |
| INT32 Latency | 14 cycles | 16 cycles | ~12 cycles |
| FP64 True Latency | **~63 cycles** | ~8 cycles | ~10 cycles |
| FP64 Completion | ~11 cycles | ~13 cycles | ~10 cycles |

### 内存延迟

| 操作 | Blackwell | Hopper | Ampere |
|------|-----------|--------|--------|
| L1 缓存命中 | ~25 cycles (估计) | ~25 cycles | ~25 cycles |
| L2 缓存命中 | ~358 cycles | ~273 cycles | ~200 cycles |
| 全局内存 | ~877 cycles | ~659 cycles | ~550 cycles |
| 共享内存 | ~25 cycles (估计) | ~25 cycles | ~25 cycles |

### Tensor Core 延迟

| 操作 | Blackwell | Hopper | Ampere |
|------|-----------|--------|--------|
| MMA Completion | **1.21 cycles** | 1.66 cycles | ~2 cycles |
| MMA True Latency | ~6 cycles (估计) | ~8 cycles | ~10 cycles |

**分析**: Blackwell 的 FP32/INT32 延迟显著低于 Hopper，但由于 FP64 单元减少，FP64 延迟高得多。

---

## 6. 能效对比

### 各精度功耗 (Blackwell)

| 精度 | 功耗 | 备注 |
|------|------|------|
| FP8 | ~46W | 每芯片 |
| FP6 e2m3 | ~39.38W | 每芯片 |
| FP6 e3m2 | ~46.72W | 每芯片 |
| FP4 | ~16.75W | 每芯片 |

### 能效对比 (Blackwell vs Hopper)

| 精度 | Blackwell | Hopper | 提升 |
|------|-----------|--------|------|
| FP8 | ~46W | ~55W | **+20%** 能效提升 |
| FP4 | ~16.75W | N/A | 新能力 |
| FP6 e2m3 | ~39.38W | N/A | 新能力 |
| FP6 e3m2 | ~46.72W | N/A | 新能力 |

**结论**: Blackwell 在 FP8 操作上比 Hopper **大约 20%** 能效更高。

### 每瓦性能

| 指标 | Blackwell | Hopper | 备注 |
|------|-----------|--------|------|
| FP16 TOPS/W | ~1.94 | ~1.77 | +10% 能效 |
| FP8 TOPS/W | ~3.88 | ~3.63 | +7% 能效 |
| FP4 TOPS/W | ~7.75 | N/A | 新能力 |

---

## 7. Transformer Engine 支持

### TE 版本对比

| 版本 | 架构 | 支持的精度 |
|------|------|-----------|
| TE 1st Gen | Hopper (H100) | FP8, FP16, BF16, FP32, FP64 |
| TE 2nd Gen | Blackwell (B100/B200) | FP4, FP6, FP8, FP16, BF16, FP32, FP64 |
| TE 3rd Gen | Blackwell Ultra | FP4, FP6, FP8, FP16, BF16, FP32, FP64, FP64 |

### Blackwell 新特性

| 特性 | 描述 | 优势 |
|------|------|------|
| FP4 支持 | 4 位浮点 | 相比 FP16 减少 4x 内存 |
| FP6 支持 | 6 位浮点 | 相比 FP16 减少 2.67x 内存 |
| Block Scaling | 硬件去量化 | 高效低精度推理 |
| TMEM | 256KB/SM 片上 | 更快的张量操作 |

---

## 总结

### 各架构适用场景

| 使用场景 | 推荐架构 |
|----------|----------|
| FP64 密集型工作负载 | Hopper (H100), Ampere (A100) |
| FP8/FP4/FP6 推理 | **Blackwell (RTX 50, B100)** |
| 通用 ML 训练 | Hopper (H100), Blackwell (B200) |
| 成本敏感 | Ampere (A100, RTX 40) |
| LLM 推理 | **Blackwell (RTX 50)** |
| 科学计算 | Hopper (H100), Ampere (A100) |

### 关键要点

1. **Blackwell 在低精度 (FP4/FP6/FP8) 推理方面表现出色**
2. **Hopper (H100) 最适合 FP64 和 WGMMA 工作负载**
3. **Blackwell 有更低的 FP32/INT32 延迟但更高的 FP64 延迟**
4. **Blackwell 在 FP8 上比 Hopper 节能约 20%**
5. **RTX 50 系列 (GeForce) 缺乏 TMA multicast 和 Cluster MMA**

---

## 参考文献

- [arXiv:2507.10789 - Blackwell 微基准测试](https://arxiv.org/abs/2507.10789)
- [NVIDIA Hopper 架构白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA Ampere 架构白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/ampere/pdf/NVIDIA-Ampere-Architecture-Whitepaper.pdf)
- [CUDA 编程指南](../ref/cuda_programming_guide.html)
- [PTX ISA](../ref/ptx_isa.html)
