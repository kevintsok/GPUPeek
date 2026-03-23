# Atomic Operations Research

## 概述

原子操作研究，测试不同粒度和类型的原子操作性能。

## 1. 原子操作类型

| 类型 | 描述 |
|------|------|
| atomicAdd | 原子加法 |
| atomicMin/Max | 原子最小/最大 |
| atomicCAS | Compare-and-Swap |
| atomicAnd/Or/Xor | 位操作 |

## 2. 粒度级别

### Warp 级原子操作
- 同 warp 内先归约，再单次原子
- 最小化原子争用

### Block 级原子操作
- 同 block 内归约，再单次原子
- 中等争用

### Grid 级原子操作
- 所有线程直接原子
- 高争用环境

## 3. 性能考虑

| 级别 | 争用程度 | 性能 |
|------|----------|------|
| Warp 级 | 低 | 最高 |
| Block 级 | 中 | 中等 |
| Grid 级 | 高 | 最低 |

## 4. NCU 指标

| 指标 | 含义 |
|------|------|
| sm__average_active_warps_per_sm | 每SM活跃warp |
| sm__warp_issue_stalled_by_barrier.pct | 同步开销 |

## 参考文献

- [CUDA Programming Guide - Atomics](../ref/cuda_programming_guide.html)
