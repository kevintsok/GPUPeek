# MBarrier Research

## 概述

MBarrier (Multi-Block Barrier) 是 Hopper 架构的增强同步机制。

## 1. MBarrier vs Barrier

| 特性 | Barrier | MBarrier |
|------|---------|----------|
| 范围 | 单 Block | 多 Block |
| 同步 | __syncthreads() | mbarrier |
| 适用 | CTA 内 | Grid/多 Block |

## 2. 基本 API

```cuda
__mbarrier_t bar;
// 初始化
mbarrier_init(bar, num_threads);
// 等待
mbarrier_wait(bar, phase);
// 同步
mbarrier_arrive(bar);
```

## 3. Phase Synchronization

```cuda
mbarrier_wait(&bar, my_phase);
mbarrier_arrive(&bar, 32);
```

## 参考文献

- [CUDA Programming Guide - MBarrier](../ref/cuda_programming_guide.html)
