# Advanced Research

## 概述

高级研究测试，包括 Occupancy、Constant Memory、Bank Conflict、Atomic Operations 等。

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
| 带宽 | ~300 GB/s (缓存命中) |

### 访问模式

| 模式 | 带宽 | 描述 |
|------|------|------|
| Broadcast (同一地址) | 高 | 所有线程读相同地址 |
| Strided (多地址) | 中 | 线程读不同地址，无广播 |

## 3. Bank Conflict 分析

共享内存分为多个 bank，访问同一 bank 的不同地址会产生冲突。

| 配置 | Bank 大小 |
|------|----------|
| SM 7.x+ (Blackwell) | 4 bytes |
| Bank 数量 | 32 |

### Stride vs 带宽

| Stride | Bank Conflict | 相对带宽 |
|--------|---------------|----------|
| 1 | 无 | 100% |
| 2 | 中 | ~60% |
| 4 | 高 | ~40% |
| 8 | 严重 | ~25% |

### 避免冲突

- 使用 padding (如 `__shared__ float buf[256][5]`)
- 交错访问模式
- 考虑 128-bit 访问

## 4. Branch Divergence

### 分支分歧开销

| 分支模式 | 开销 |
|----------|------|
| 无分歧 | 基准 |
| 50% 分歧 | ~20-30% 减速 |

### 原理
当 warp 内线程走不同分支时，GPU 串行执行各分支，导致利用率下降。

## 5. Atomic Operations

### AtomicAdd 性能

| 粒度 | 带宽 | 描述 |
|------|------|------|
| Global atomic | 低 | 所有线程竞争同一地址 |
| Block reduction + atomic | 高 | Block 内归约后单次 atomic |

### 优化策略
- Block-level reduction 减少 atomic 竞争
- 使用 warp-level shuffle 代替 atomic

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
