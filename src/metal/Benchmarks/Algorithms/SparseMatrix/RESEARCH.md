# Sparse Matrix (SPMV) Research

## 概述

本专题研究GPU上稀疏矩阵向量乘法（Sparse Matrix-Vector Multiply, SPMV）的性能。SPMV是科学计算和图处理中的核心操作，使用CSR、ELLpack等格式存储稀疏矩阵。

## 关键发现

### SPMV性能对比

| 格式 | 性能 | 适用场景 |
|------|------|---------|
| CSR Naive | 基准 | 通用稀疏矩阵 |
| CSR Vectorized | 1.5-3x | 批量稀疏操作 |
| ELLPACK | 2-4x | 规则稀疏结构 |
| COO | ~CSR | 简单实现 |

### 稀疏矩阵格式对比

| 格式 | 存储效率 | 访问模式 | 适合场景 |
|------|---------|---------|---------|
| CSR | 高 | 不规则 | 通用 |
| ELLPACK | 中 | 规则 | 固定宽度行 |
| COO | 中 | 简单 | 构建阶段 |

### Apple M2 SPMV特性

1. **内存带宽受限** - SPMV是内存密集型操作
2. **不规则访问** - 列索引随机访问vector
3. **原子操作开销** - COO格式需要原子累加

## 优化策略

### 1. CSR格式向量化
```metal
// 每次处理4个非零元素
float4 vals = values[i / 4];
uint4 cols = column_indices[i / 4];
float4 vec_vals = float4(vector[cols.x], vector[cols.y],
                          vector[cols.z], vector[cols.w]);
sum += vals * vec_vals;
```

### 2. ELLPACK格式
```metal
// 固定宽度，适合规则稀疏结构
for (uint i = 0; i < max_nnz_per_row; i++) {
    uint col = column_indices[offset + i];
    if (col != UINT_MAX) {
        sum += values[offset + i] * vector[col];
    }
}
```

### 3. 格式选择策略
```
if (行非零元分布规则):
    使用 ELLPACK
else:
    使用 CSR
```

## 性能影响因子

1. **稀疏度** - 非零元素比例越低，内存访问越不规则
2. **行长度方差** - 方差大时CSR优于ELLPACK
3. **向量化** - 批量处理提高带宽利用率
4. **原子争用** - COO的原子累加可能成为瓶颈

## 相关专题

- [Graph](../Graph/RESEARCH.md) - 图算法使用SPMV
- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - SPMV内存受限
- [Vectorization](../../Compute/Vectorization/RESEARCH.md) - 向量化优化
