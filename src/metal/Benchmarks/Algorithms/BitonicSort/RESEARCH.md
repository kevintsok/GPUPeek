# Bitonic Sort (Parallel Sorting Network) Research

## 概述

本专题研究GPU上双调排序（Bitonic Sort）的性能。双调排序是一种并行排序网络（Sorting Network），具有确定性拓扑结构，不受数据分布影响，适合GPU并行执行。

## 关键发现

### 算法复杂度

| 算法 | 时间复杂度 | 空间复杂度 | 并行深度 |
|------|------------|------------|----------|
| Bitonic Sort | O(n log² n) | O(n) | O(log² n) |
| Merge Sort | O(n log n) | O(n) | O(log n) |
| Quick Sort | O(n log n) avg | O(log n) | O(log n) |
| Radix Sort | O(n log n) | O(n) | O(n/k) |

### Bitonic Sort 原理

```
Bitonic Sort 构建过程:

Stage 0 (k=1): 比较距离 1
  [0,1] [2,3] [4,5] ...  (log n steps)

Stage 1 (k=2): 比较距离 1,2
  [0,2] [1,3] [4,6] ...
  [0,1] [2,3] ...

Stage 2 (k=4): 比较距离 1,2,4
  [0,4] [1,5] [2,6] ...
  [0,2] [1,3] [4,6] ...
  [0,1] [2,3] [4,5] ...
```

### 核心操作

```metal
// Bitonic step kernel
uint ixj = id ^ j;  // XOR pattern找到配对索引
if (ixj > id) {
    bool asc = ((id & k) == 0);  // 升序或降序
    // 比较和交换
}
```

## 为什么适合GPU

### 1. 无分支
```metal
// 所有线程执行相同的代码路径
if (ixj > id) {  // 简单的数据相关，无分支分歧
    // 比较交换
}
```

### 2. 锁步执行
```metal
// 排序网络是确定性的
// 所有线程同步执行，无warp分歧
```

### 3. 可预测的内存访问
```metal
// 配对模式 (id ^ j) 是确定性的
// 有利于内存合并和缓存预取
```

## 排序网络 vs 比较排序

| 特性 | 排序网络 | 比较排序 |
|------|----------|----------|
| 最坏情况 | 固定 | 可能退化 |
| 数据依赖 | 无 | 分支预测 |
| GPU适配 | 完美 | 中等 |
| 输入大小 | 2的幂 | 任意 |

## 应用场景

### 1. GPU并行排序
```
最常见的GPU排序算法
常与Radix Sort结合使用
```

### 2. 硬件排序单元
```
网络硬件交换机
FPGA加速
```

### 3. 确定性排序
```
实时系统
需要保证最坏情况性能的场合
```

## 优化策略

### 1. 批量处理
```metal
// 一次排序多个数组
// 共享排序网络配置
```

### 2. 混合排序
```cpp
// 小数组用Bitonic，大数组用Radix + Bitonic
if (n < 1024) {
    bitonic_sort(data);
} else {
    radix_sort(data);
    bitonic_sort_final(data);
}
```

### 3. 共享内存优化
```metal
// 使用threadgroup存储中间结果
threadgroup float tile[256];
```

## Apple Metal 注意事项

1. **Threadgroup限制**: 32KB per group
2. **Power of 2要求**: 需要填充到2的幂
3. **内存带宽**: 大数组受限于内存带宽

## 性能影响因子

1. **数组大小**: 必须是2的幂
2. **Kernel启动开销**: log² n次kernel调用
3. **内存访问模式**: 配对访问可能不连续

## 相关专题

- [Sorting](./Sorting/RESEARCH.md) - 其他排序算法
- [RadixSort](./RadixSort/RESEARCH.md) - 基数排序
- [Scan](./Scan/RESEARCH.md) - 前缀和（用于归并）
