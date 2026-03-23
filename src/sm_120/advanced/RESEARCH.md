# Advanced Research

## 概述

高级研究测试，包括 Constant Memory、Bank Conflict 分析等。

## 1. Constant Memory

常量内存是一种专用只读内存，广播读取效率高。

| 特性 | 值 |
|------|---|
| 大小 | 64 KB |
| 读取 | 广播至所有线程 |

## 2. Bank Conflict 分析

共享内存分为多个 bank，访问同一 bank 的不同地址会产生冲突。

| 配置 | Bank 大小 |
|------|----------|
| SM 7.x+ | 4 bytes |

### 避免冲突

- 使用 padding
- 交错访问模式
- 考虑 128-bit 访问

## 3. Memory Fence 影响

Memory fence 引入约 50% 的性能开销。

## 4. NCU 指标

| 指标 | 含义 |
|------|------|
| sm__shared_bank_conflict_throughput | 银行冲突吞吐量 |

## 参考文献

- [CUDA Programming Guide - Shared Memory](../ref/cuda_programming_guide.html)
- [CUDA Best Practices - Bank Conflicts](../ref/best_practices_guide.html)
