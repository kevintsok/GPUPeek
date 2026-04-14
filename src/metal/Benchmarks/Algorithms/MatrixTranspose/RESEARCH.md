# Matrix Transpose Research

## 概述

本专题研究GPU上矩阵转置（Matrix Transpose）的性能。矩阵转置是线性代数中的基本操作，在图像处理、深度学习和科学计算中广泛使用。

## 关键发现

### 矩阵转置性能对比

| 算法 | 性能 | 说明 |
|------|------|------|
| Naive | 基准 | 非合并访问 |
| Tiled | 1.5-3x | 改善合并访问 |
| Padded | 2-4x | 避免bank冲突 |
| Vectorized | 2-5x | 使用float4 |

### 转置特性分析

```
原始矩阵          转置后
[A00 A01 A02]    [A00 B00 C00]
[B00 B01 B02] -> [A01 B01 C01]
[C00 C01 C02]    [A02 B02 C02]

问题：列访问变成行访问，导致非合并内存访问
```

### Apple M2 转置优化

1. **共享内存分块** - 将矩阵分成tile减少全局内存访问
2. **Padding** - 避免共享内存bank冲突
3. **向量化** - 使用float4一次传输4个元素

## 优化策略

### 1. Naive转置（基线）
```metal
// 非合并访问 - 写入时列访问
out[col * rows + row] = in[row * cols + col];
```

### 2. Tiled转置
```metal
// 分块加载到共享内存
tile[tid.y * TILE_SIZE + tid.x] = in[gid.y * cols + gid.x];
threadgroup_barrier(flags::mem_threadgroup);
// 合并写入
out[out_gid.y * rows + out_gid.x] = tile[tid.x * TILE_SIZE + tid.y];
```

### 3. Padding避免Bank冲突
```metal
// 添加padding列
threadgroup float tile[(TILE_SIZE + PAD) * TILE_SIZE];
```

## 性能影响因子

1. **内存访问模式** - 行优先vs列优先访问
2. **Tile大小** - 影响共享内存利用率
3. **Bank冲突** - 跨步访问导致性能下降
4. **缓存效率** - 分块大小影响缓存命中率

## 相关专题

- [GEMM](../GEMM/RESEARCH.md) - 矩阵乘法中的转置操作
- [Memory Bandwidth](../Bandwidth/RESEARCH.md) - 转置是内存受限操作
- [BankConflict](../BankConflict/RESEARCH.md) - Padding减少bank冲突
