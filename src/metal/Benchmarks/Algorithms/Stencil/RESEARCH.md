# Stencil Computation Research

## 概述

本专题研究GPU上的模板计算(Stencil)算法和共享内存优化。

## 关键发现

### Stencil性能

| 方法 | 性能 | 加速比 |
|------|------|--------|
| Naive | 0.079 GOPS | 基准 |
| Shared Memory | 0.094 GOPS | 1.2x |

## 关键洞察

1. **共享内存减少全局内存访问** - 减少重复加载
2. **1.2x典型加速** - 5点stencil优化效果
3. **Halo单元需要特殊处理** - 边界条件

## 相关专题

- [GEMM](../../Compute/GEMM/RESEARCH.md) - 矩阵乘法中的分块
- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽
