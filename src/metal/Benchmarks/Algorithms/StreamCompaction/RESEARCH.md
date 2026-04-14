# Stream Compaction Research

## 概述

本专题研究GPU上流压缩（Stream Compaction）算法的性能。流压缩用于过滤/选择满足条件的元素，是并行编程中的重要模式。

## 关键发现

### 流压缩性能对比

| 算法 | 性能 | 原子争用 | 适用场景 |
|------|------|---------|---------|
| Naive | 基准 | 高 | 小规模数据 |
| Tiled | 1.5-3x | 低 | 大规模数据 |
| Radix | 2-4x | 中 | 二进制过滤 |

### 流压缩模式

```
输入: [a0, a1, a2, a3, a4, a5, ...]
Predicate: [0, 1, 1, 0, 1, 0, ...]
          ↓
输出: [a1, a2, a4, ...]
```

### Apple M2 流压缩特性

1. **原子操作开销** - 每个满足条件的元素需要原子操作
2. **线程束级原语** - simd_any/all可用于predicate计算
3. **共享内存缓冲** - Tiled算法减少全局原子争用

## 优化策略

### 1. Naive算法
```metal
// 每个线程原子竞争
if (predicate(val)) {
    uint idx = atomic_fetch_add(&counter, 1);
    output[idx] = val;
}
```

### 2. Tiled算法
```metal
// 每个tile本地累积，最后合并
threadgroup uint localCount;
threadgroup float localOut[128];

// ... 本地累积 ...

// Tile leader一次性写入
if (localId == 0 && localCount > 0) {
    uint offset = atomic_fetch_add(&counter, localCount);
    for (i = 0; i < localCount; i++)
        output[offset + i] = localOut[i];
}
```

### 3. Radix分区
```metal
// 按位过滤，适合二进制属性
if ((val & (1 << bit)) != 0) {
    uint idx = atomic_fetch_add(&counter, 1);
    output[idx] = val;
}
```

## 性能影响因子

1. **Keep Ratio** - 满足条件的元素比例越低，原子争用越少
2. **Tile Size** - 决定本地缓冲大小和合并开销
3. **原子操作类型** - memory_order_relaxed最快
4. **Predicate复杂度** - 复杂predicate增加计算开销

## 相关专题

- [Scan](../Scan/RESEARCH.md) - 前缀和是压缩的基础
- [Atomics](../../Synchronization/Atomics/RESEARCH.md) - 原子操作性能
- [Predicate](../Predicate/RESEARCH.md) - 谓词操作
