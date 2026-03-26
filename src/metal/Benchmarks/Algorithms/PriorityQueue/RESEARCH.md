# Priority Queue Research

## 概述

本专题研究GPU上优先级队列（Priority Queue）和堆（Heap）操作的性能。优先级队列是调度、路径规划和事件模拟中的核心数据结构。

## 关键发现

### 堆操作性能

| 操作 | 性能 | 说明 |
|------|------|------|
| Heap Push | ~10K-50K ops/s | 串行操作 |
| Heap Pop | ~10K-50K ops/s | 串行操作 |
| Bucket Sort | ~50-200 M/s | 批处理并行 |

### 堆操作复杂度

```
Binary Heap (Max-Heap):
        50
       /  \
     30    20
    /  \   / \
   10  15 18  25

Push: O(log n) - 向上冒泡
Pop:  O(log n) - 向下冒泡
```

### GPU堆操作限制

1. **串行性** - 堆操作有数据依赖，难以并行
2. **原子操作** - 需要原子操作维护堆大小
3. **批处理** - GPU通常用批处理替代单个堆操作

## 替代方案

### 1. Bucket Sort
```metal
// 批量优先级操作
uint bucket = val * num_buckets / 256;
uint offset = atomic_fetch_add(&counts[bucket], 1);
out[bucket * size + offset] = val;
```

### 2. Radix Sort
```metal
// 多轮基数排序
for (bit = 0; bit < 8; bit++) {
    // Count and reorder by bit
}
```

### 3. Top-K Selection
```metal
// 找最大K个元素
for each element:
    count how many are larger
    if count < K: store it
```

## 性能影响因子

1. **堆大小** - 操作复杂度 O(log n)
2. **原子争用** - 多线程竞争修改堆大小
3. **批处理大小** - 批量操作提高并行度
4. **数据依赖** - 堆操作难以SIMD并行

## 应用场景

1. **调度** - GPU任务调度器
2. **路径规划** - Dijkstra/A*算法
3. **事件模拟** - 时间队列
4. **Top-K** - 数据流统计

## 相关专题

- [Sorting](../Sorting/RESEARCH.md) - 堆排序算法
- [Atomics](../../Synchronization/Atomics/RESEARCH.md) - 原子操作实现
- [Scan](../Scan/RESEARCH.md) - 前缀和用于排序
