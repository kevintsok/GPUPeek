# Advanced Research

## 概述

高级研究测试概述模块，提供 Constant Memory、Bank Conflict、Atomic Operations 等主题的基础介绍。

> **注意**: 详细研究请参考各专题模块：
> - Bank Conflict: [`../bank_conflict/`](../bank_conflict/RESEARCH.md)
> - Atomic Operations: [`../atomic/`](../atomic/RESEARCH.md)
> - Memory: [`../memory/`](../memory/RESEARCH.md)
> - Barrier: [`../barrier/`](../barrier/RESEARCH.md)
> - CUDA Core: [`../cuda_core/`](../cuda_core/RESEARCH.md)
> - WMMA/Tensor Core: [`../wmma/`](../wmma/RESEARCH.md)
> - Redux.sync: [`../redux_sync/`](../redux_sync/RESEARCH.md)
> - Cooperative Groups: [`../cooperative_groups/`](../cooperative_groups/RESEARCH.md)
> - CUDA Graph: [`../cuda_graph/`](../cuda_graph/RESEARCH.md)
> - Warp Specialization: [`../warp_specialize/`](../warp_specialize/RESEARCH.md)
> - FP8/FP4/FP6: [`../fp8/`](../fp8/RESEARCH.md), [`../fp4_fp6/`](../fp4_fp6/RESEARCH.md)
> - DP4A: [`../dp4a/`](../dp4a/RESEARCH.md)
> - MBarrier: [`../mbarrier/`](../mbarrier/RESEARCH.md)
> - Multi-Stream: [`../multi_stream/`](../multi_stream/RESEARCH.md)
> - Unified Memory: [`../unified_memory/`](../unified_memory/RESEARCH.md)
> - Deep Research (L2/TMA): [`../deep/`](../deep/RESEARCH.md)

## 1. Occupancy 分析

### Block Size vs Performance

| Block Size | 寄存器压力 | Occupancy | 性能 |
|------------|------------|-----------|------|
| 32 | 低 | 高 | 中等 |
| 64 | 低 | 高 | 高 |
| 128 | 中 | 高 | 高 |
| 256 | 中高 | 中 | 高 |
| 512 | 高 | 中低 | 中 |
| 1024 | 最高 | 低 | 低 |

### 寄存器使用影响

- 高寄存器使用 → 低 Occupancy → 可能降低性能
- 低寄存器使用 → 高 Occupancy → 更好隐藏延迟

## 2. Constant Memory

常量内存是一种专用只读内存，广播读取效率高。

| 特性 | 值 |
|------|---|
| 大小 | 64 KB |
| 读取 | 广播至所有线程 |
| 带宽 | 1.29 TB/s (缓存命中) |

### 访问模式

| 模式 | 带宽 | 描述 |
|------|------|------|
| Broadcast (同一地址) | 1.29 TB/s | 所有线程读相同地址 |
| Strided (多地址) | 573.36 GB/s | 线程读不同地址，无广播 |

## 3. Bank Conflict 分析

> **详细研究**: 完整的 Bank Conflict 研究见 [`../bank_conflict/`](../bank_conflict/RESEARCH.md) 模块。

共享内存分为多个 bank，访问同一 bank 的不同地址会产生冲突。

| 配置 | Bank 大小 |
|------|----------|
| SM 7.x+ (Blackwell) | 4 bytes |
| Bank 数量 | 32 |

### Stride vs 带宽 (RTX 5080 实测)

| Stride | Bank Conflict | 相对带宽 |
|--------|---------------|----------|
| 1 | 无 | 100% |
| 2 | 无 | 6.4% |
| 4 | 中 | 12.6% |
| 8 | 高 | 22.2% |
| 16 | 严重 | 38.5% |
| 32 | **最大** | 47.8% |
| 64 | 中 | 55.2% |
| 128 | 低 | 66.2% |

**实测发现**: Stride=2 时带宽急剧下降至 6.4%，这与预期不符，可能与测试方法相关。

### 避免冲突

- 使用 padding (如 `__shared__ float buf[32][33]`)
- 矩阵转置使用 +1 padding 可获得 35%+ 提升
- 交错访问模式
- 考虑 128-bit 访问

详见 [`../bank_conflict/`](../bank_conflict/RESEARCH.md)。

## 4. Branch Divergence

> **详细数据**: RTX 5080 实测显示简单 kernel 分歧开销约 -24% (反向加速)。

### 分支分歧开销

| 分支模式 | 带宽 | 开销 |
|----------|------|------|
| 无分歧 | 581.92 GB/s | 基准 |
| 高分歧 | 764.59 GB/s | -23.9% (更快) |

**实测发现**: 高分歧情况下反而更快，这与测试 kernel 的具体实现相关。

### 原理
当 warp 内线程走不同分支时，GPU 串行执行各分支，导致利用率下降。

## 5. Atomic Operations

> **详细研究**: 完整的原子操作研究见 [`../atomic/`](../atomic/RESEARCH.md) 模块。

### AtomicAdd 性能

| 粒度 | 带宽 | 描述 |
|------|------|------|
| Global atomic | 3.12 GB/s | 所有线程竞争同一地址 |
| Block reduction + atomic | 284.71 GB/s | Block 内归约后单次 atomic |
| atomicCAS | 9.93 GB/s | Compare-And-Swap |

### 优化策略
- Block-level reduction 减少 atomic 竞争 (3-5x 提升)
- 使用 warp-level shuffle 代替 atomic
- 分散热点数据到不同 atomic 位置

详见 [`../atomic/`](../atomic/RESEARCH.md)。

## 6. Memory vs Compute Bound

### 判定方法

| 比率 (ComputeBw / MemoryBw) | 结论 |
|-----------------------------|------|
| > 2.0 | Memory Bound |
| < 0.5 | Compute Bound |
| 0.5-2.0 | Balanced |

### 优化方向
- Memory Bound: 增加计算密度或使用更好的缓存
- Compute Bound: 减少计算量或使用更低精度

## 7. NCU 指标

| 指标 | 含义 |
|------|------|
| sm__shared_bank_conflict_throughput | 银行冲突吞吐量 |
| sm__average_occupancy | 平均 Occupancy |
| sm__pipe_fp64_cycles_active | FP64 计算利用率 |

## 参考文献

- [CUDA Programming Guide - Shared Memory](../ref/cuda_programming_guide.html)
- [CUDA Best Practices Guide - Bank Conflicts](../ref/cuda_best_practices.html)
