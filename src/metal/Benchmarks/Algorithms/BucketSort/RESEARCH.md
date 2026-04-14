# Bucket Sort / Hash-Based Distribution Research

## 概述

本专题研究GPU上桶排序（Bucket Sort）和基于哈希的分布式排序算法。桶排序是一种分布排序（Distribution Sort），通过将数据分散到多个"桶"中来并行处理。

## 关键发现

### 算法流程

```
Bucket Sort 四个阶段:

Phase 1: Hash (哈希分配)
- 每个元素根据值映射到对应桶
- 使用原子操作计数

Phase 2: Scan (前缀和)
- 计算每个桶的起始偏移
- O(n) 扫描计数数组

Phase 3: Distribute (分发)
- 根据哈希值将元素分发到对应桶
- 使用原子操作保证位置正确

Phase 4: Local Sort (桶内排序)
- 对每个桶内元素单独排序
- 桶较小时可用插入排序
```

### 复杂度分析

| 阶段 | 复杂度 | 说明 |
|------|---------|------|
| Hash | O(n) | 并行哈希到桶 |
| Scan | O(n) | 串行扫描计数 |
| Distribute | O(n) | 并行分发 |
| Local Sort | O(Σni log ni) | 桶内排序 |
| **Total** | **O(n)** | 均匀分布时 |

### 与其他排序对比

| 算法 | 时间复杂度 | 空间 | 稳定性 |
|------|------------|------|--------|
| Bucket Sort | O(n) avg | O(n) | Yes |
| Radix Sort | O(nk) | O(n) | Yes |
| Merge Sort | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) avg | O(log n) | No |
| Bitonic Sort | O(n log² n) | O(n) | No |

## 原子操作优化

### 哈希分配中的原子操作
```metal
// 原子递增计数
atomic_fetch_add_explicit(&bucket_counts[bucket], 1, memory_order_relaxed);

// 原子交换位置
uint myOffset = atomic_fetch_add_explicit(&bucket_pos[bucket], 1, memory_order_relaxed);
out[offsets[bucket] + myOffset] = val;
```

### 争用优化
```cpp
// 减少原子争用的策略:
// 1. 增加桶数量减少争用
// 2. 使用local memory缓存再批量写入
// 3. 使用warp-level投票减少全局原子操作
```

## 应用场景

### 1. 浮点数排序
```cpp
// 均匀分布的浮点数[0,1)
float val = data[i];
uint bucket = uint(val * NUM_BUCKETS);
```

### 2. 整数分类
```cpp
// 统计分布/直方图
uint bin = key % NUM_BINS;
atomic_inc(&histogram[bin]);
```

### 3. 数据库操作
```cpp
// 分组聚合
GROUP BY hash(key) % NUM_BUCKETS
```

### 4. GPU排序管道
```cpp
// 大数据排序组合
if (n > THRESHOLD) {
    // Radix Sort for high bits
    // Bucket Sort for low bits
}
```

## 性能影响因子

### 1. 数据分布
```
均匀分布: O(n) 性能最佳
不均匀分布: 某些桶过大，退化到O(n²)
```

### 2. 桶数量
```
太多桶: 扫描开销增加
太少桶: 桶内排序成为瓶颈
最佳: 约等于 n^(1/2) 或 n^(2/3)
```

### 3. 原子操作争用
```
高争用: 严重影响性能
解决: 增加桶数、使用warp级聚合
```

### 4. 桶内排序算法
```
小桶(n<100): 插入排序O(n²)足够
大桶: 需要更好的排序算法
```

## Apple Metal 优化

### 1. SIMD组投票
```metal
// 使用simd_any减少原子争用
if (simd_any(need_this_bucket)) {
    // 协作分配
}
```

### 2. 共享内存缓冲
```metal
// 先收集到shared memory，再批量写入
threadgroup uint local_counts[256];
```

### 3. 分块处理
```metal
// 大数据分块避免内存不足
for (uint chunk = 0; chunk < num_chunks; chunk++) {
    process_chunk(chunk);
}
```

## 与Radix Sort对比

| 特性 | Bucket Sort | Radix Sort |
|------|-------------|------------|
| 时间复杂度 | O(n) avg | O(nk) |
| 空间 | O(n) | O(n) |
| 稳定性 | Yes | Yes |
| 分布式 | Yes | No |
| 原子操作 | 需要 | 不需要 |

## 相关专题

- [Sorting](./Sorting/RESEARCH.md) - 排序算法概述
- [RadixSort](./RadixSort/RESEARCH.md) - 基数排序
- [Histogram](./Histogram/RESEARCH.md) - 直方图统计
