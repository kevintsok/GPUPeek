# Hash Analysis Research

## 概述

本专题研究GPU上的哈希算法性能，包括哈希表查找、布隆过滤器（Bloom Filter）和哈希连接（Hash Join）。这些是数据库和数据处理中的核心操作。

## 关键发现

### 哈希操作性能对比

| 操作 | 性能 | 说明 |
|------|------|------|
| Hash Lookup | ~10-20 GE/s | 取决于冲突率 |
| Bloom Insert | ~5-10 GE/s | 多哈希开销 |
| Bloom Test | ~15-25 GE/s | 概率操作 |
| Hash Join | ~3-8 GE/s | 复杂匹配 |

### 哈希表方法对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| Chaining | 简单 | 额外内存/指针追逐 |
| Open Addressing | 缓存友好 | 探测开销 |
| Linear Probing | 最简单 | 聚集问题 |
| Quadratic Probing | 减少聚集 | 不保证找到 |

### Bloom Filter特性

```
Bloom Filter 结构:
- 位数组: M bits
- 哈希函数: K 个
- 假阳性率: (1 - e^(-kn/m))^k

Apple M2 优化:
- 原子操作 set bit
- 共享内存位数组
- 多哈希并行计算
```

## 优化策略

### 1. 哈希函数选择
```metal
// 快速哈希函数
inline uint hash(uint key, uint seed) {
    uint k = key;
    k *= 0xcc9e2d51u;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593u;
    return k ^ seed;
}
```

### 2. 开放寻址
```metal
// 线性探测
uint idx = hash(key, seed) & (tableSize - 1);
for (uint i = 0; i < tableSize; i++) {
    if (table[idx] == key || table[idx] == EMPTY) {
        return idx;
    }
    idx = (idx + 1) & (tableSize - 1);
}
```

### 3. Bloom Filter 多哈希
```metal
// 使用不同seed生成多个哈希
for (uint h = 0; h < numHashes; h++) {
    uint idx = hash(key, h * seed);
    bit_array[idx / 32] |= 1u << (idx % 32);
}
```

## 性能影响因子

1. **冲突率** - 高冲突降低性能
2. **哈希函数质量** - 分布均匀性影响
3. **缓存局部性** - 开放寻址优于链式
4. **假阳性率** - Bloom filter参数选择

## 相关专题

- [Atomics](../../Synchronization/Atomics/RESEARCH.md) - 原子操作实现
- [Memory Bandwidth](../Bandwidth/RESEARCH.md) - 哈希是内存密集型
- [Scan](../Scan/RESEARCH.md) - 前缀和用于哈希分组
