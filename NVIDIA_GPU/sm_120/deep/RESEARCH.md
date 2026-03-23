# Deep Research

## 概述

深度研究测试，包括 L2 缓存、TMA、prefetch 等高级内存操作。

## 1. L2 缓存

### L2 工作集分析

| 数据大小 | 带宽 | 状态 |
|---------|------|------|
| 64 KB | ~123 GB/s | L1 缓存 |
| 1 MB | ~408 GB/s | L2 边界 |
| 4 MB | ~678 GB/s | L2 缓存 |
| 8 MB | ~748 GB/s | L2 缓存 |
| 16 MB | ~798 GB/s | L2 miss → DRAM |

### L2 Thrashing

Stride > 8 导致带宽急剧下降，表明缓存行跨距访问效率低。

## 2. TMA (张量内存访问器)

### TMA 1D 拷贝

| 数据大小 | TMA 带宽 |
|---------|----------|
| 64 KB | ~7 GB/s |
| 1 MB | ~134 GB/s |
| 4 MB | ~431 GB/s |
| 16 MB | **~850 GB/s** |

### TMA 2D 拷贝

TMA 2D 带宽 ~626 GB/s

## 3. Prefetch

软件预取性能 ~251 GB/s

## 4. NCU 指标

| 指标 | 含义 |
|------|------|
| lts__tcs_hit_rate.pct | L2 缓存命中率 |
| dram__bytes.sum | 内存带宽 |
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU 利用率 |

## 参考文献

- [CUDA Programming Guide - L2 Cache](../ref/cuda_programming_guide.html)
