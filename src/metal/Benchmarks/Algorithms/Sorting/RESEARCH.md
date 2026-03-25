# Sorting Algorithms Research

## 概述

本专题研究GPU上的排序算法性能，包括Bitonic Sort、Radix Sort和Odd-Even Transposition Sort。

## 关键发现

### 排序算法性能

| 算法 | 性能 | 复杂度 | 备注 |
|------|------|--------|------|
| Odd-Even Transposition | ~0.0001 GOPS | O(n²) | 并行度低 |
| Bitonic Sort | 0.0001 GOPS | O(log² n) | kernel launch开销大 |
| Radix Sort | 0.017 GOPS | O(n*k) | k=digit数 |

### 排序算法对比

| 指标 | Bitonic | Radix | Odd-Even |
|------|---------|-------|----------|
| 并行度 | 高 | 高 | 中 |
| 比较次数 | O(log² n) | O(n*k) | O(n²) |
| 适合规模 | 中 | 大 | 小 |
| 整数排序 | 否 | **是** | 否 |

## 关键洞察

1. **GPU排序有很高的kernel launch开销** - 多次小kernel调用显著降低性能
2. **Bitonic Sort的kernel launch开销抵消了并行优势** - 实测极慢
3. **Radix Sort是GPU排序的最佳选择** - O(n*k)复杂度，适合整数
4. **小规模数据用CPU排序更快** - GPU适合大规模并行排序

## 优化策略

1. **减少kernel launch次数** - 使用单一kernel实现完整排序
2. **利用shared memory** - 减少全局内存访问
3. **批量处理** - 一次排序多个数据块
4. **考虑thrust库** - NVIDIA/CUDA的优化排序实现

## 相关专题

- [Scan/Reduction](../Scan/RESEARCH.md) - 并行扫描算法
- [Atomics](../../Synchronization/Atomics/RESEARCH.md) - 原子操作用于归约
