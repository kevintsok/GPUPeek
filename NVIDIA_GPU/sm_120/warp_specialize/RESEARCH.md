# Warp Specialization Research

## 概述

Warp Specialization 允许将一个 Warp 分成 Producer 和 Consumer 角色，实现计算与内存访问的重叠。

## 1. Producer-Consumer 模式

```
Warp 0 (Producer):  load/compute data
Warp 1 (Consumer):  use data from Producer
```

## 2. 关键 API

| API | 描述 |
|-----|------|
| __shfl_down_sync | Warp 内数据交换 |
| cp.async | 异步拷贝 |
| bar.sync | Barrier 同步 |

## 3. Warp 级同步原语

| 原语 | 功能 |
|------|------|
| Mutex | 互斥锁 |
| Barrier | 同步屏障 |
| Reduction | 归约操作 |
| Scan | 前缀和 |

## 4. 多级 Pipeline

```
Stage 1: Load A, B    (Producer)
Stage 2: MMA Compute  (Consumer)
Stage 3: Store D      (Producer)
```

## 5. TMA + Barrier 协同

```cuda
// TMA 异步拷贝
cp.async.ca.shared.global ...;
// Barrier 等待
bar.sync ...;
```

## 6. NCU 指标

| 指标 | 含义 |
|------|------|
| sm__warp_divergence_efficiency | Warp 分歧效率 |
| sm__average_active_warps_per_sm | 每SM活跃warp |

## 参考文献

- [CUDA Programming Guide - Warp](../ref/cuda_programming_guide.html)
- [Parallel Programming Guide](../ref/parallel_programming_guide.html)
