# Predicate and Thread Masking Analysis Research

## 概述

本专题研究GPU上谓词（Predicate）和线程掩码（Thread Masking）技术，用于在不规则计算模式中跳过无效工作。

## 关键发现

### 谓词概念

```
Predicate (谓词): 一个布尔值，表示元素是否应该被处理

线程掩码: 使用谓词有选择性地启用/禁用线程执行

应用场景:
- 过滤操作 (Filter)
- 条件计算
- 不规则数据结构遍历
- 稀疏矩阵操作
```

### 谓词计算方法

#### 1. 分支方法（可能导致分歧）
```metal
kernel void process_with_branch(device const uchar* predicate [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (predicate[id] == 1) {
        // 执行计算 - 但导致线程分歧
        out[id] = expensive_computation(id);
    } else {
        out[id] = 0.0f;  // 分支分歧
    }
}
```

#### 2. 压缩方法（无分歧）
```metal
kernel void process_compacted(device const float* in [[buffer(0)]],
                            device const uchar* predicate [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (predicate[id] == 1) {
        out[id] = expensive_computation(id);
    }
    // 不需要else分支
}
```

### 性能对比

| 方法 | 性能 | 分歧成本 | 适用场景 |
|------|------|----------|----------|
| 谓词计算 | ~10 M/s | 无 | 过滤判断 |
| 分支处理 | 可变 | 20-30% | 简单条件 |
| 压缩收集 | 取决于密度 | 无 | 高密度数据 |

### Warp级投票操作

```metal
// Apple Metal SIMD组投票
bool condition = in[id] > 0.5f;
// simd_any: 任一线程满足条件
// simd_all: 所有线程满足条件
```

## 实现细节

### 谓词直方图
```metal
kernel void predicate_histogram(device const float* in [[buffer(0)]],
                              device const uchar* predicate [[buffer(1)]],
                              device atomic_uint* histogram [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (predicate[id] == 1) {
        uint bin = uint(in[id] * 16.0f);
        atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
    }
}
```

### 前缀和压缩
```metal
kernel void compact_indices(device const uchar* predicate [[buffer(0)]],
                         device uint* indices [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    uint ps = 0;
    for (uint i = 0; i < id; i++) {
        ps += predicate[i];  // 计算前缀和
    }
    if (predicate[id] == 1) {
        indices[ps] = id;  // 收集有效索引
    }
}
```

## 应用场景

### 1. 图遍历
```cpp
// BFS中只处理活跃顶点
if (predicate[vertex] == 1) {
    process_vertex(vertex);
}
```

### 2. 稀疏矩阵
```cpp
// CSR格式：只处理非零元素
if (values[idx] != 0) {
    compute(values[idx]);
}
```

### 3. 条件计算
```cpp
// 物理模拟：只计算受力粒子
if (particle_mass > threshold) {
    apply_force(particle);
}
```

### 4. 过滤操作
```cpp
// 数据库过滤
if (record.value > predicate) {
    output[count++] = record;
}
```

## Apple Metal优化

### SIMD组投票函数
```metal
// Metal提供原生SIMD组投票
bool result = simd_any(condition);
bool result = simd_all(condition);
```

### 线程掩码策略
```metal
// 避免分歧：使用连续分支而非随机分支
if (thread_position < work_count) {
    process_work();
}
// 而不是
if (predicate[thread_position]) {
    process_work();
}
```

## 性能影响因子

1. **谓词密度**: 有效元素比例影响性能
2. **分歧成本**: SIMD组内分歧线程等待
3. **压缩开销**: 需要额外的前缀和计算
4. **原子操作**: 密集谓词导致原子争用

## 与NVIDIA对比

| 特性 | Apple Metal | NVIDIA CUDA |
|------|-------------|-------------|
| 谓词类型 | uchar | bool |
| Warp投票 | simd_any/all | __any/__all |
| 线程掩码 | 显式分支 | __activemask() |
| 性能 | 良好 | 优秀 |

## 相关专题

- [BranchDivergence](../Compute/BranchDivergence/RESEARCH.md) - 分支分歧分析
- [StreamCompaction](./StreamCompaction/RESEARCH.md) - 流压缩
- [Histogram](./Histogram/RESEARCH.md) - 直方图统计
- [Sorting](./Sorting/RESEARCH.md) - 排序算法
