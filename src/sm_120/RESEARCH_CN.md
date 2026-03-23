# SM 12.0 (Blackwell) GPU 架构研究报告

> **目标 GPU**: NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell, GB203)
> **最后更新**: 2026-03-23
> **报告语言**: 中文

---

## 目录

1. [硬件规格](#1-硬件规格)
2. [内存子系统](#2-内存子系统)
3. [计算性能](#3-计算性能)
4. [Tensor Core (MMA)](#4-tensor-core-mma)
5. [Warp 级操作](#5-warp-级操作)
6. [内存操作](#6-内存操作)
7. [跨代对比](#7-跨代对比)
8. [NCU 性能分析指标](#8-ncu-性能分析指标)
9. [测试环境](#9-测试环境)
10. [基准测试命令](#10-基准测试命令)
11. [参考文献](#11-参考文献)

---

## 1. 硬件规格

### 1.1 GPU 规格

| 参数 | 值 |
|------|-----|
| GPU 型号 | NVIDIA GeForce RTX 5080 Laptop GPU |
| 架构代号 | Blackwell |
| Compute Capability | 12.0 |
| SM 数量 | 60 |
| 每 SM CUDA 核心数 | 128 |
| 总 CUDA 核心数 | 7,680 |
| 全局内存 | 15.92 GB |
| 每 Block 共享内存 | 48 KB |
| 每 SM L1 缓存 | 128 KB |
| L2 缓存 | 65 MB (单体设计) |
| 每 Block 最大线程数 | 1,024 |
| 每 SM 最大线程数 | 1,536 |
| 每 SM 最大寄存器数 | 65,536 |
| Warp 大小 | 32 |
| 内存类型 | GDDR7 |
| 内存带宽 | ~8.2 TB/s |

### 1.2 跨代 GPU 对比

| 参数 | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) |
|------|------------------|----------------|----------------|
| Compute Capability | 12.0 | 9.0 | 8.0 |
| SM 数量 | 60 (完整: 84) | 132 | 108 |
| 每 SM CUDA 核心数 | 128 | 128 | 64 |
| 内存类型 | GDDR7 | HBM2e | HBM2e |
| 内存带宽 | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s |
| 每 SM L1 缓存 | 128 KB | 256 KB | 192 KB |
| 每 SM 共享内存 | ~99 KB | ~227 KB | ~227 KB |
| L2 缓存 | 65 MB | 50 MB | 80 MB |
| L2 架构 | 单体 | 2 分区 | 2 分区 |
| Tensor Core 代数 | 5th | 4th | 3rd |
| 每 SM FP64 单元数 | 2 (有限) | 64 (完整) | 64 (完整) |

---

## 2. 内存子系统

### 2.1 全局内存带宽

#### 2.1.1 全局内存带宽 vs 数据大小

| 数据大小 | 顺序读取 | 顺序写入 | 状态 |
|---------|---------|---------|------|
| 1 KB | 0.00 GB/s | 0.00 GB/s | - |
| 64 KB | 7.25 GB/s | 7.25 GB/s | L1 缓存 |
| 256 KB | 32.39 GB/s | 32.39 GB/s | L1 缓存 |
| 1 MB | 73.97 GB/s | 73.97 GB/s | L1/L2 边界 |
| 4 MB | 296.36 GB/s | 296.36 GB/s | L2 缓存 |
| 16 MB | 643.02 GB/s | 643.02 GB/s | 峰值 (第一) |
| 64 MB | 376.08 GB/s | 376.08 GB/s | L2 miss |
| 128 MB | 502.44 GB/s | 502.44 GB/s | 恢复中 |
| 256 MB | 614.93 GB/s | 614.93 GB/s | 峰值 (第二) |

**分析**:
- 峰值带宽约 640-820 GB/s (16MB 工作集)
- 64MB 时带宽下降，表明 L2 缓存失效
- 128-256MB 后恢复，表明内存控制器有更大的有效缓存窗口

#### 2.1.2 跨距访问效率

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

#### 2.1.3 不同数据类型带宽

| 数据类型 | 大小 | 带宽 |
|----------|------|------|
| float | 4 B | 878.19 GB/s |
| int | 4 B | 882.25 GB/s |
| double | 8 B | 468.73 GB/s |
| half (FP16) | 2 B | 410.20 GB/s |

**分析**:
- float/int 带宽约 880 GB/s
- double 下降约 47%
- half (FP16) 下降约 53%

### 2.2 L1/L2 缓存带宽

#### 2.2.1 内存层级带宽

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

#### 2.2.2 L2 工作集分析

| 数据大小 | 带宽 | 状态 |
|---------|------|------|
| 64 KB | 123.09 GB/s | L1 缓存 |
| 1 MB | 407.66 GB/s | L2 边界 |
| 4 MB | 678.20 GB/s | L2 缓存 |
| 8 MB | 747.53 GB/s | L2 缓存 |
| 16 MB | 797.97 GB/s | L2 miss → DRAM |

#### 2.2.3 L2 Thrashing 测试 (跨距访问)

| Stride | 带宽 |
|--------|------|
| 1 | 729.19 GB/s |
| 2 | 713.45 GB/s |
| 4 | 679.30 GB/s |
| 8 | 418.62 GB/s |
| 16 | 402.67 GB/s |
| 64 | 218.89 GB/s |
| 256 | 226.36 GB/s |
| 1024 | 244.46 GB/s |
| 4096 | 406.78 GB/s |

**分析**: Stride 8+ 导致带宽急剧下降，表明缓存行跨距访问效率低。

### 2.3 共享内存性能

| 操作 | 带宽 | 时间/Kernel |
|------|------|------------|
| 共享内存 R/W | **1.50 TB/s** | - |
| 广播写入 | 1.30 TB/s | - |
| Reduction 读取 | 332.91 GB/s | - |
| Reduction 写入 | 1.30 GB/s | - |

### 2.4 TMA (张量内存访问) 性能

#### 2.4.1 TMA 1D 拷贝

| 数据大小 | TMA 拷贝 | cudaMemcpy | 加速比 |
|---------|----------|------------|--------|
| 64 KB | 6.88 GB/s | 6.88 GB/s | 0.99x |
| 256 KB | 33.97 GB/s | 33.97 GB/s | 0.97x |
| 1 MB | 133.87 GB/s | 133.87 GB/s | 1.13x |
| 4 MB | 431.20 GB/s | 431.20 GB/s | 1.04x |
| 16 MB | 850.07 GB/s | 850.07 GB/s | 0.72x |
| 64 MB | 382.15 GB/s | 382.15 GB/s | 1.06x |
| 128 MB | 373.07 GB/s | 373.07 GB/s | 0.99x |

**分析**: TMA 峰值带宽 850 GB/s (16MB)。

#### 2.4.2 TMA 2D 拷贝 (1024x1024, pitch=2048)

| 方式 | 带宽 |
|------|------|
| TMA 2D | 626.36 GB/s |
| cudaMemcpy2D | 704.31 GB/s |

**分析**: cudaMemcpy2D 在 2D 拷贝场景下优于自定义 TMA kernel。

### 2.5 PCIe 带宽

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

### 2.6 Occupancy vs 性能

| Block Size | 带宽 | 时间 |
|------------|--------|------|
| 32 | 292-300 GB/s | ~57 μs |
| 64 | 374-453 GB/s | ~35-45 μs |
| 128 | 674-890 GB/s | ~18-25 μs |
| 256 | 802-876 GB/s | ~19-21 μs |
| **512** | **828-900 GB/s** | ~18-21 μs |
| 1024 | 589-628 GB/s | ~26-29 μs |

**最佳 Block Size**: 256-512 线程

### 2.7 分支分歧

| 分支类型 | 带宽 | 时间 |
|---------|------|------|
| 无分歧 | 746-761 GB/s | 0.021 ms |
| 高分歧 | 796-810 GB/s | 0.021 ms |

**分析**: 对于简单 kernel，分歧开销不明显。

### 2.8 内存 Fence 影响

| 配置 | 带宽 | 时间/Kernel |
|------|------|------------|
| 无 Fence | 793.25 GB/s | 0.021 ms |
| 有 Fence | 536.38 GB/s | 0.031 ms |

**分析**: Memory fence 引入约 50% 的性能开销。

---

## 3. 计算性能

### 3.1 FP32 性能

| 测试 | 吞吐量 | 延迟 | 备注 |
|------|--------|------|------|
| FP32 FMA (融合乘加) | 88.55 GFLOPS | 0.068 ms | 旧测试 |
| FP32 FMA | 61.55 GFLOPS | 0.068 ms | 新测试 |

### 3.2 FP64 性能

| 指标 | 值 | 备注 |
|------|-----|------|
| 每 SM FP64 单元数 | 2 (有限) | vs Hopper 的 64 |
| FP64 True Latency | ~63 cycles | vs Hopper 的 ~8 |

**警告**: Blackwell 不适合 FP64 密集型工作负载。

### 3.3 FP16 性能

| 测试 | 吞吐量 | 延迟 |
|------|--------|------|
| FP16 FMA | 204.19 GFLOPS | 0.021 ms |

**分析**: FP16 比 FP32 快约 3.3 倍。

### 3.4 INT32 性能

| 测试 | 吞吐量 | 延迟 |
|------|--------|------|
| INT32 算术 | 121.52 GIOPS | 0.035 ms |

### 3.5 指令吞吐量汇总

| 指令类型 | 吞吐量 | 延迟 |
|---------|--------|------|
| FP32 FMA | 61.55 GFLOPS | 0.068 ms |
| INT32 算术 | 121.52 GIOPS | 0.035 ms |
| FP16 FMA | 204.19 GFLOPS | 0.021 ms |

---

## 4. Tensor Core (MMA)

### 4.1 WMMA (Warp级 MMA)

WMMA 是标准的 CUDA Tensor Core API (Section 9.7.14)。

#### 4.1.1 WMMA Shape 支持

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

#### 4.1.2 WMMA FP16 基准测试结果

| 指标 | 值 |
|------|---|
| 矩阵尺寸 | M=256, N=256, K=256 |
| Shape | m16n16k16 |
| Grid | 16x16 |
| Block | 32 (1 个 warp) |
| 时间 | 0.130 ms/iteration |
| **性能** | **257.41 GFLOPS** |
| 验证 | sum=4103416.75 (非零=正确) |

### 4.2 MMA (新型 Warp级 MMA)

标准 MMA 指令 (Section 9.7.14.5)。

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

### 4.3 MMA.SP (稀疏 MMA)

2:4 结构化稀疏 (Section 9.7.14.6)。

| Shape | FP16 | BF16 | TF32 | INT8 | FP8 |
|-------|------|------|------|------|-----|
| m16n8k8.sp | Yes | Yes | Yes | - | - |
| m16n8k16.sp | Yes | Yes | Yes | Yes | - |
| m16n8k32.sp | Yes | - | - | Yes | - |
| m16n8k64.sp | Yes | - | - | - | - |
| m16n8k128.sp | Yes | - | - | - | - |

**注意**: 2:4 稀疏要求每 4 个元素中有 2 个必须是零。

### 4.4 WGMMA (异步 Warpgroup MMA)

**重要**: WGMMA **仅在 Hopper 上支持**。Blackwell **不支持** WGMMA。

| Shape | FP16 | BF16 | TF32 | FP64 | INT8 | FP8 |
|-------|------|------|------|------|------|-----|
| m64nNk16 | Yes | Yes | Yes | Yes | Yes | - |
| m64nNk8 | Yes | Yes | Yes | - | Yes | - |
| m64nNk32 | Yes | Yes | - | - | - | Yes |
| m64nNk256 | Yes | - | - | - | - | - |

**N = K / 16**

### 4.5 TCGen05 (第五代 Tensor Core)

Blackwell 的第 5 代 Tensor Core API (Section 9.7.16)。

#### 4.5.1 TCGen05 vs WGMMA

| 特性 | WGMMA (Hopper) | TCGen05 (Blackwell) |
|------|----------------|---------------------|
| PTX 章节 | 9.7.15 | 9.7.16 |
| API | wgmma.mma_async | tcgen05.mma |
| SASS | WGMMA | **OMMA (FP4), QMMA (FP8/FP6)** |
| 异步 | Yes | Yes |
| 稀疏 | 2:4 | 2:4 |
| FP4/FP6 | No | **Yes** |
| Block Scaling | No | **Yes (硬件)** |

#### 4.5.2 TCGen05 Shape 支持

| Shape | FP16 | BF16 | TF32 | FP32 | FP64 | INT8 | FP4 | FP6 |
|-------|------|------|------|------|------|------|-----|-----|
| .32x32b | Yes | Yes | Yes | Yes | Yes | - | - | - |
| .16x64b | Yes | Yes | Yes | Yes | Yes | - | - | - |
| .16x128b | Yes | Yes | Yes | Yes | - | - | - | - |
| .16x256b | Yes | Yes | Yes | - | - | - | - | - |
| .16x32bx2 | Yes | Yes | Yes | Yes | - | - | - | - |
| **m16n8k32** | - | - | - | - | - | - | **Yes** | **Yes** |

#### 4.5.3 TCGen05 变体

| 变体 | 描述 |
|------|------|
| tcgen05.mma | 基本 MMA |
| tcgen05.mma.sp | 稀疏 MMA (2:4) |
| tcgen05.mma.ws | 仅权重量化 (W8A16) |
| tcgen05.mma.ws.sp | Weight-only + 稀疏 |

#### 4.5.4 TCGen05 CTA Group 类型

| CTA Group | 描述 | D 寄存器数 |
|-----------|------|-----------|
| cta_group::1 | 单 CTA (1 个 warp group) | 4 |
| cta_group::2 | 双 CTA 集群 (2 个 warp groups) | 8+ |

#### 4.5.5 TCGen05 操作数源变体

| 变体 | A 来源 | B 来源 | 描述 |
|------|-------|-------|------|
| SS | SMEM | SMEM | 两者都从共享内存 |
| TS | TMEM | SMEM | A 从张量内存 |
| ST | SMEM | TMEM | B 从张量内存 |
| TT | TMEM | TMEM | 两者都从张量内存 |

#### 4.5.6 TMEM (张量内存)

| 属性 | 值 |
|------|---|
| 大小 | **256 KB per SM** |
| 组织 | 512 列 × 128 行 × 32 位单元 |
| D (累加器) | **必须** 放在 TMEM |
| A 操作数 | 可选择从 TMEM 或 SMEM 加载 |

#### 4.5.7 Block Scaling (TCGen05 硬件支持)

| 格式 | Block 大小 | 缩放因子格式 |
|------|------------|--------------|
| UE8M0 | 32 元素 | 8-bit unsigned exp (2^x), -127 ≤ x ≤ 127 |
| UE4M3 | 16 元素 | 4-bit exp + 3-bit mantissa, 最大 448 |

**TMEM 布局**:
- block32/1X: 最多 12 列
- block32/2X: 最多 24 列
- block16/4X: 最多 48 列

#### 4.5.8 PTX 指令示例 (来自 CUTLASS)

```asm
// TF32 SS (两者都从SMEM)
tcgen05.mma.cta_group::1.kind::tf32 [%tmem_c], %desc_a, %desc_b, %idescE,
  {%mask0, %mask1, %mask2, %mask3}, p;

// FP16 BF16 SS
tcgen05.mma.cta_group::1.kind::f16 [%tmem_c], %desc_a, %desc_b, %idescE,
  {%mask0, %mask1, %mask2, %mask3}, p;

// Block Scaled MXFP8/MXFP6/MXF4
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%tmem_c], %desc_a, %desc_b,
  %idescE, [%tsfa_addr], [%tsfb_addr], p;

// Block Scaled MXF4 (NVIDIA FP4) - 16元素块
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%tmem_c], %desc_a,
  %desc_b, %idescE, [%tsfa_addr], [%tsfb_addr], p;

// Cluster MMA (2 CTA)
tcgen05.mma.cta_group::2.kind::tf32 [%tmem_c], %desc_a, %desc_b, %idescE,
  {%mask0..%mask11}, p;

// Sparse MMA (2:4 结构化稀疏)
tcgen05.mma.sp.cta_group::1.kind::tf32 [%tmem_c], %desc_a, %desc_b, [%tsfb_addr],
  %idescE, {%mask0..%mask7}, p;

// SM120 Native MMA (m16n8k32, 寄存器-based)
mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32
  {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3};
```

#### 4.5.9 RTX 50 (GeForce) 限制

| 特性 | 支持情况 |
|------|---------|
| TMA Multicast | ❌ 不支持 |
| Cluster Shape | 必须 1x1x1 |
| Dynamic Datatypes | ❌ 不支持 |
| Cluster MMA | ❌ 不支持 (仅数据中心 GPU) |

#### 4.5.10 Sub-byte 数据内存布局要求

| 数据类型 | TMA 打包格式 | SMEM 对齐 |
|----------|-------------|----------|
| FP4 (E2M1) | CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B | 16 字节对齐 |
| FP6 (E2M3/E3M2) | CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B | 16 字节对齐 |

**关键约束**:
- 基地址需要 32 字节对齐
- Leading dimension 必须是 128 元素的倍数
- 只支持 128 字节 swizzle 模式或无 swizzle
- MMA tile 的 K extent 永远是 32 (dense GEMM)

### 4.6 FP4/FP6 低精度 MMA

#### 4.6.1 格式规格

| 格式 | 指数位 | 尾数位 | 总位数 | PTX 类型 |
|------|--------|--------|--------|----------|
| FP4 (e2m1) | 2 | 1 | 4 | e2m1 |
| FP6 (e2m3) | 2 | 3 | 6 | e2m3 |
| FP6 (e3m2) | 3 | 2 | 6 | e3m2 |

**PTX ISA (CUDA 12.9+)**:

```ptx
mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32   // FP4
mma.sync.aligned.m16n8k32.row.col.f32.e2m3.e2m3.f32   // FP6 e2m3
mma.sync.aligned.m16n8k32.row.col.f32.e3m2.e3m2.f32   // FP6 e3m2
```

**Shape**: m16n8k32 (与 FP8 的 m16n8k16 不同)

#### 4.6.2 FP4/FP6 vs FP8 对比

| 特性 | FP8 (E4M3/E5M2) | FP4 (e2m1) | FP6 (e2m3/e3m2) |
|------|------------------|------------|------------------|
| 位数 | 8 | 4 | 6 |
| 精度 | 高 | 极低 | 低 |
| 内存减少 | 2x vs FP16 | 4x vs FP16 | 2.67x vs FP16 |
| TFLOPS | 最高 | 最高 | 高 |
| 适用场景 | 权重+激活 | 仅权重 | 仅权重 |

#### 4.6.3 FP4/FP6 初步测试结果

| 测试 | 结果 |
|------|------|
| FP32 → FP4 转换 | 1.304 ms (1M 元素) |
| FP4 → FP32 转换 | 0.052 ms (1M 元素) |
| FP16 GEMM 基线 | 743.57 GFLOPS |
| FP4 风格 GEMM (模拟) | 920.45 GFLOPS |

**注意**: FP4/FP6 风格 GEMM 比 FP16 基线快，这是因为模拟 kernel 做了量化简化。真正的 FP4/FP6 MMA 需要 CUDA 12.9+ 硬件支持。

### 4.7 SASS 指令映射

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
| tcgen05.mma | OMMA/QMMA | TCGen05 MMA (Blackwell) |

### 4.8 NCU Tensor Core 指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| sm__pipe_tensor_cycles_active.pct | Tensor Core 利用率 | 越高越好 |
| sm__pipe_tensor_cycles_active.sum | Tensor Core 活跃周期数 | 越少越好 |
| smsp__average_executed_epc_per_warp | 每 warp 执行指令数 | 稳定 |
| sm__inst_executed.sum | 总执行指令数 | 越少越好 |
| dram__bytes.sum | 全局内存带宽 | 参考 |
| lts__tcs_hit_rate.pct | L2 缓存命中率 | 越高越好 |

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

**支持的操作**: ADD, MIN, MAX, AND, OR, XOR

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

Tensor Core 数据的矩阵加载/存储操作 (Section 9.7.14.5.15-16)。

#### 6.2.1 LDMATRIX 变体

| 指令 | 描述 | 每线程元素数 |
|------|------|-------------|
| ldmatrix.sync.aligned.m8n8.x1 | 8x8 tile, 1 个矩阵 | 2 |
| ldmatrix.sync.aligned.m8n8.x2 | 8x8 tile, 2 个矩阵 | 4 |
| ldmatrix.sync.aligned.m8n8.x4 | 8x8 tile, 4 个矩阵 | 8 |
| ldmatrix.sync.aligned.m16n8.k1 | 16x8 tile | varies |

**关键特性**:
- Warp 级操作 (32 线程协作)
- 转置布局 (MMA 友好)
- 需要 16 字节对齐

#### 6.2.2 STMATRIX 变体

| 指令 | 描述 |
|------|------|
| stmatrix.sync.aligned.m8n8.x1 | 8x8 tile, 1 个矩阵 |
| stmatrix.sync.aligned.m8n8.x2 | 8x8 tile, 2 个矩阵 |
| stmatrix.sync.aligned.m8n8.x4 | 8x8 tile, 4 个矩阵 |

### 6.3 cp.async (异步拷贝)

| 指令 | 描述 |
|------|------|
| cp.async.ca | 缓存策略异步拷贝 |
| cp.async.commit_group | 提交异步组 |
| cp.async.wait_group n | 等待 n 个组 |
| cp.async.wait_all | 等待所有 |

### 6.4 cp.async.bulk (批量异步拷贝)

| 指令 | 描述 |
|------|------|
| cp.async.bulk | 批量异步拷贝 |
| cp.async.bulk.commit_group | 批量提交组 |
| cp.async.bulk.wait_group n | 批量等待 |
| cp.reduce.async.bulk.add | 批量拷贝+求和 |
| cp.async.bulk.prefetch | 批量预取 |

### 6.5 张量内存性能汇总

| 操作 | 优势 | 适用场景 |
|------|------|----------|
| LDMATRIX | Warp 协作、转置 | MMA A/B 加载 |
| STMATRIX | Warp 协作 | MMA 结果存储 |
| cp.async | 延迟隐藏 | 计算/拷贝重叠 |
| TMA | 大块 2D 传输 | 大矩阵分块 |

---

## 7. 跨代对比

### 7.1 计算性能

| 指标 | Blackwell (GB203) | Hopper (GH100) | Ampere (GA100) |
|------|-------------------|----------------|----------------|
| FP32 性能 | ~17.6 TFLOPS | ~19.5 TFLOPS | ~19.5 TFLOPS |
| FP16 性能 | ~89.2 TFLOPS | ~99.8 TFLOPS | ~39.7 TFLOPS |
| FP64 性能 | **有限** | 完整 | 完整 |
| Tensor Core 代数 | 5th | 4th | 3rd |

### 7.2 内存系统

| 指标 | Blackwell | Hopper | Ampere |
|------|-----------|--------|--------|
| 内存类型 | GDDR7 | HBM2e | HBM2e |
| 带宽 | 8.2 TB/s | 15.8 TB/s | 2.0 TB/s |
| 每 SM L1 缓存 | 128 KB | 256 KB | 192 KB |
| 每 SM 共享内存 | ~99 KB | ~227 KB | ~227 KB |
| L2 缓存 | 65 MB | 50 MB | 80 MB |
| L2 架构 | 单体 | 2 分区 | 2 分区 |

### 7.3 Tensor Core 特性对比

| 特性 | Blackwell (5th) | Hopper (4th) | Ampere (3rd) |
|------|-----------------|--------------|--------------|
| WGMMA | ❌ | ✅ | ❌ |
| FP4 支持 | ✅ | ❌ | ❌ |
| FP6 支持 | ✅ | ❌ | ❌ |
| FP8 支持 | ✅ | ✅ | ❌ |
| Block Scaling | ✅ (硬件) | ❌ | ❌ |
| 2:4 稀疏 | ✅ | ✅ | ✅ |

### 7.4 延迟对比

| 操作 | Blackwell | Hopper |
|------|-----------|--------|
| FP32 True Latency | 15.96 cycles | 31.62 cycles |
| INT32 Latency | 14 cycles | 16 cycles |
| FP64 True Latency | **~63 cycles** | ~8 cycles |
| L2 缓存命中 | ~358 cycles | ~273 cycles |
| 全局内存 | ~877 cycles | ~659 cycles |
| MMA Completion | 1.21 cycles | 1.66 cycles |

### 7.5 能效对比 (FP8/FP4/FP6)

| 精度 | Blackwell | Hopper |
|------|-----------|--------|
| FP8 | ~46W | ~55W |
| FP4 | ~16.75W | N/A |
| FP6 e2m3 | ~39.38W | N/A |
| FP6 e3m2 | ~46.72W | N/A |

### 7.6 Transformer Engine 支持

| 版本 | 支持的精度 | 架构 |
|------|-----------|------|
| TE 1st Gen | FP8, FP16, BF16, FP32, FP64 | Hopper |
| **TE 2nd Gen** | **FP4, FP6** + 以上 | **Blackwell** |

---

## 8. NCU 性能分析指标

### 8.1 关键指标参考

| 指标 | 含义 | 用途 |
|------|------|------|
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU 利用率 | 越高越好 |
| dram__bytes.sum | 内存带宽 | 内存操作测试 |
| sm__pipe_fp32_cycles_active.pct | FP32 单元利用率 | 算力测试 |
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall | 同步开销 |
| sm__warp_divergence_efficiency | Warp 分歧效率 | 分歧测试 |
| sm__average_active_warps_per_sm | 每 SM 活跃 warp | Occupancy |

### 8.2 Kernel 代码覆盖

| Benchmark | Kernels | PTX 指令 |
|-----------|---------|----------|
| memory | Various | Global/Shared/L1 access |
| deep | Various | L2, TMA, prefetch |
| advanced | Various | Atomic, constant memory |
| wmma | wmma_fp16_kernel, etc. | wmma.mma |
| tcgen05 | tcgen05_mma_kernel | tcgen05.mma |
| tensor_mem | ldmatrix, stmatrix, cp.async | ld.matrix, st.matrix, cp.async |
| wgmma | wgmma_async_kernel | wgmma.mma_async (仅 Hopper) |
| dp4a | dp4a_*_kernel | dp4a.s32.s8.s8 |
| fp8 | fp8_gemm_*_kernel | FP8 MMA |
| fp4 | fp4_*_kernel | FP4/FP6 MMA (模拟) |
| cuda | cuda_core_*_kernel | Various |

---

## 9. 测试环境

| 组件 | 版本 |
|------|------|
| CUDA Toolkit | 13.0 |
| 驱动版本 | 595.79 |
| 操作系统 | Windows 11 |
| GPU | RTX 5080 Laptop (GB203) |
| Compute Capability | 12.0 |
| SM 数量 | 60 |
| 编译选项 | -O3 -arch=sm_90 --use_fast_math |

---

## 10. 基准测试命令

### 10.1 GPUPeek 基准测试命令

```bash
# 所有基准测试
./build/gpupeek.exe all

# 特定模块
./build/gpupeek.exe memory    # 内存研究
./build/gpupeek.exe deep      # 深度研究
./build/gpupeek.exe advanced  # 高级研究
./build/gpupeek.exe wmma      # WMMA (Tensor Core)
./build/gpupeek.exe tcgen05   # TCGen05 研究
./build/gpupeek.exe tensor_mem # 张量内存
./build/gpupeek.exe cuda      # CUDA Core 算力
./build/gpupeek.exe atomic    # 原子操作
./build/gpupeek.exe barrier   # Barrier 同步
./build/gpupeek.exe warp      # Warp 特化
./build/gpupeek.exe mma       # MMA/Tensor Core (禁用 - 使用 wmma)
./build/gpupeek.exe wgmma     # WGMMA (仅 Hopper)
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

### 10.2 NCU Profiling 命令

```bash
# 完整 tensor core 分析
ncu --set full --metrics sm__pipe_tensor_cycles_active.pct,sm__inst_executed.sum,dram__bytes.sum,lts__tcs_hit_rate.pct ./gpupeek.exe wmma

# 内存带宽分析
ncu --set full --metrics dram__bytes.sum ./gpupeek.exe memory

# SASS 指令分析
ncu --set full --kernels-by-compute ./gpupeek.exe tensor_mem

# GPU 利用率
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe deep
```

---

## 11. 参考文献

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

---

## 附录 A: 编译状态 (2026-03-23)

### 正常工作的模块 (7/18)

| 模块 | 状态 | 备注 |
|------|------|------|
| memory | ✅ | 内存研究 |
| deep | ✅ | 深度研究 |
| advanced | ✅ | 高级研究 |
| fp4 | ✅ | FP4/FP6 研究 |
| multi_stream | ✅ | 多流并发 |
| wmma | ✅ | WMMA (warp级 MMA) |
| tcgen05 | ✅ | TCGen05/UMMA 研究 |

### 禁用的模块 (编译错误)

| 模块 | 问题 |
|------|------|
| ncu | 编码问题 |
| cuda | 缺少 bf16 头文件 |
| atomic | 待查 |
| barrier | cooperative groups API 问题 |
| warp | cooperative groups API 问题 |
| mma | 禁用 - 使用 wmma 代替 |
| tensor_mem | WMMA fragment 类型未定义 |
| wgmma | WMMA fragment 类型未定义 |
| dp4a | 待查 |
| fp8 | 待查 |
| graph | 待查 |
| unified | 待查 |
| mbarrier | 待查 |
| coop | cooperative groups API 问题 |
| redux | 待查 |

---

## 附录 B: WMMA 修复 (2026-03-23)

**问题**: WMMA kernel 在运行时出现 "illegal memory access" 错误。

**解决方案**:
1. 创建了新的 `wmma_test_kernel.cu` 和 `wmma_test_benchmarks.cu`
2. 使用正确的 fragment 类型定义:
   ```cuda
   fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
   fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
   fragment<accumulator, 16, 16, 16, float> frag_d;
   ```
3. 使用 `using namespace nvcuda::wmma;`
4. Grid: `(N / 16, M / 16)`, Block: `32`
5. 每个 warp 处理一个 16x16 输出块

---

*报告由 GPUPeek 基准测试框架生成*
