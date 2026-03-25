# Occupancy Analysis Research

## 概述

本专题研究Apple M2 GPU的occupancy（占用率）对性能的影响，包括不同线程组大小和共享内存使用对GPU资源利用率的作用。

## 背景

**Occupancy** = 活跃线程数 / 最大可同时运行线程数

Apple M2 GPU规格:
- 最大线程组大小: 1024线程
- 最大线程组内存: 32KB
- SIMD宽度: 32线程

## 关键发现

### Occupancy 级别对比

| Occupancy级别 | 线程数 | 共享内存 | 性能 |
|--------------|--------|----------|------|
| Low | 32 | 1KB | ~0.06 GOPS |
| Medium | 128 | 256B | ~0.06 GOPS |
| High | 512 | 64B | ~0.06 GOPS |

### Kernel类型对比

| Kernel类型 | Occupancy敏感度 | 原因 |
|-----------|----------------|------|
| Compute Bound | 低 | ALU是瓶颈，高occupancy无帮助 |
| Memory Bound | 高 | 需要高occupancy隐藏内存延迟 |

## 关键洞察

1. **Apple M2上Occupancy影响较小** - 实测各occupancy级别性能相近
2. **32KB共享内存限制** - 超过后必须减少线程数
3. **Memory-bound受益于高occupancy** - 更多线程可隐藏内存延迟
4. **Compute-bound对occupancy不敏感** - 计算单元是瓶颈

## 优化策略

1. **Memory-bound kernels**: 使用高occupancy（512+线程）
2. **Compute-bound kernels**: 优先使用寄存器，occupancy其次
3. **共享内存权衡**: 32KB限制下平衡threads和shared
4. **实测验证**: 不同kernel类型需要不同优化策略

## 相关专题

- [Barriers](../../Synchronization/Barriers/RESEARCH.md) - 同步开销
- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽
- [GEMM](../../Compute/GEMM/RESEARCH.md) - 矩阵乘法中的occupancy优化
