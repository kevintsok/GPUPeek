# Database Operations and Parallel Aggregation Research

## 概述

本专题研究GPU上并行数据库操作（Database Operations）的性能。包括并行过滤（WHERE）、聚合（GROUP BY）、排名（RANK）和Top-K选择等操作。

## 关键发现

### 数据库操作分类

```
1. Filter (WHERE): 谓词-based selection
2. Aggregation (GROUP BY): 并行归约
3. Ranking (ORDER BY): 全局排序
4. Top-K: 选择最大/最小的K个元素
5. Join: 多表关联
```

### 性能数据

| 操作 | 性能 | 说明 |
|------|------|------|
| Filter (WHERE) | ~10-50 M/s | 取决于密度 |
| Aggregation (GROUP BY) | ~5-20 M/s | 原子操作争用 |
| Ranking | ~0.01 M/s | O(n²)复杂度 |
| Top-K | ~100 M/s | 简化版本 |

### 操作实现

#### Filter (WHERE)
```metal
kernel void db_filter(device uint* keys [[buffer(0)]],
             device uint* output [[buffer(1)]],
             device atomic_uint* count [[buffer(2)]],
             uint id [[thread_position_in_grid]]) {
    if (keys[id] > threshold) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        output[idx] = values[id];
    }
}
```

#### Aggregation (GROUP BY)
```metal
kernel void db_aggregate(device uint* keys [[buffer(0)]],
                device atomic_uint* buckets [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    uint bucket = keys[id] % num_buckets;
    atomic_fetch_add_explicit(&buckets[bucket], values[id], memory_order_relaxed);
}
```

## 应用场景

### 1. 数据分析
```
SELECT category, COUNT(*) FROM sales GROUP BY category;
SELECT * FROM events WHERE timestamp > t1;
```

### 2. ML特征工程
```
特征过滤: WHERE feature > threshold
特征聚合: GROUP BY user_id
```

### 3. ETL管道
```
并行数据清洗
分布式Join
```

### 4. 图数据库
```
顶点过滤
边聚合
```

## 性能优化策略

### 1. 减少原子争用
```cpp
// 改用共享内存聚合后再原子写入
threadgroup uint local_counts[64];
```

### 2. 过滤后聚合
```cpp
// 先过滤，再聚合，减少处理数据量
if (predicate[id]) {
    atomic_add(counts[bucket], 1);
}
```

### 3. 使用压缩索引
```cpp
// 对于排序数据，使用二分查找代替线性扫描
```

## 与传统数据库对比

| 方面 | GPU数据库 | CPU数据库 |
|------|----------|-----------|
| 吞吐量 | 10-100 M/s | 0.1-1 M/s |
| 延迟 | 高 | 低 |
| 并发 | 低 | 高 |
| 适合场景 | 批量分析 | 实时查询 |

## 相关专题

- [Histogram](./Histogram/RESEARCH.md) - 直方图统计
- [Scan](./Scan/RESEARCH.md) - 并行扫描
- [Sorting](./Sorting/RESEARCH.md) - 排序算法
- [PredicateMasking](./PredicateMasking/RESEARCH.md) - 谓词过滤
