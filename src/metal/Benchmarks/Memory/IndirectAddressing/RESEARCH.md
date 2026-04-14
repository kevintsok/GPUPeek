# Indirect Addressing (Scatter-Gather) Research

## 概述

本专题研究GPU上间接寻址（Indirect Addressing）和散聚（Scatter-Gather）操作的性能。间接寻址是指通过索引数组来访问数据，而不是直接按顺序访问。

## 关键发现

### 内存访问模式对比

| 模式 | 性能 | 说明 |
|------|------|------|
| Sequential | 1.0x (baseline) | 完全合并的内存访问 |
| Strided (stride=7) | ~0.7x | 跨步访问中等性能损失 |
| Random Gather | ~0.3-0.5x | 随机索引严重降低性能 |
| Random Scatter | ~0.2-0.4x | 写操作比读更慢 |

### Scatter-Gather 模式

```
Gather (读操作):
Thread 0: read data[indices[0]]  →  data[42]
Thread 1: read data[indices[1]]  →  data[17]
Thread 2: read data[indices[2]]  →  data[89]
...

Scatter (写操作):
Thread 0: write data[indices[0]] = val  →  data[42] = val
Thread 1: write data[indices[1]] = val  →  data[17] = val
Thread 2: write data[indices[2]] = val  →  data[89] = val
```

### 性能影响原因

1. **内存合并失效**: GPU内存系统优化顺序访问，随机索引无法合并
2. **缓存污染**: 随机访问模式导致缓存行利用率低
3. **Bank冲突**: 共享内存模式下可能产生bank冲突
4. **原子操作开销**: Scatter操作可能需要原子操作保证正确性

## 优化策略

### 1. 索引排序
```metal
// 在CPU或GPU上先排序索引
sort(indices);  // 排序后访问更连续
```

### 2. 批量合并
```metal
// 将多个小访问合并成批量操作
for (int i = 0; i < batchSize; i++) {
    gather(data, sortedIndices[i]);
}
```

### 3. 缓存优化
```metal
// 利用数据局部性
if (cacheHit) {
    val = cache[idx];
} else {
    val = data[idx];
    cache[idx] = val;
}
```

### 4. 展开合并
```metal
// 手动展开利用SIMD组能力
float4 v0 = data[indices[id * 4 + 0]];
float4 v1 = data[indices[id * 4 + 1]];
```

## 应用场景

1. **稀疏矩阵**: CSR/ELLPACK格式的矩阵向量乘法
2. **图算法**: BFS/DFS的邻接表遍历
3. **物理模拟**: 粒子系统的位置更新
4. **图像处理**: 稀疏采样和插值

## 相关专题

- [SparseMatrix](../Algorithms/SparseMatrix/RESEARCH.md) - 稀疏矩阵运算
- [Coalescing](../Memory/Coalescing/RESEARCH.md) - 内存合并
- [Graph](../Algorithms/Graph/RESEARCH.md) - 图算法
