# Matrix Square (A × A^T) Research

## 概述

本专题研究GPU上矩阵平方运算（Matrix Square, A × A^T）的性能。这种运算常见于神经网络反向传播、协方差计算和图神经网络中。

## 关键发现

### 性能对比

| 操作 | 内存访问 | 性能 | 说明 |
|------|----------|------|------|
| GEMM (A × B) | 连续 | 1.0x | 基准性能 |
| Matrix Square (A × A^T) | 非连续 | 0.4-0.7x | 列访问导致性能下降 |

### 内存访问模式

```
Matrix Square C = A × A^T:

C[gid.y][gid.x] = Σ_k A[gid.y][k] × A[gid.x][k]

行内访问 (gid.y 固定):
- A[gid.y * K + k]  连续访问

列内访问 (gid.x 固定):
- A[gid.x * K + k]  对不同 gid.y 是非连续访问

问题: 当 gid.y ≠ gid.x 时，两个访问模式不同
```

### 应用场景

1. **神经网络反向传播** - 梯度计算中的自相关
2. **协方差矩阵** - 统计学中的 Var(X) = E[X²] - E[X]²
3. **图神经网络** - 邻接矩阵与自身转置相乘
4. **高斯过程** - 核矩阵计算

## 优化策略

### 1. 共享内存分块
```metal
// 将A的行列加载到共享内存
for (uint t = 0; t < (K + tileSize - 1) / tileSize; t++) {
    // Load tile
    tileA[lid.y * tileSize + k] = A[gid.y * K + globalK];
    tileA[lid.x * tileSize + k] = A[gid.x * K + globalK];
    threadgroup_barrier(mem_flags::mem_none);

    // Compute partial result
    for (uint k = 0; k < tileSize; k++) {
        sum += tileA[lid.y * tileSize + k] * tileA[lid.x * tileSize + k];
    }
}
```

### 2. 预转置
```cpp
// 预先转置A为A^T，这样A^T * A的访问就是连续的
transpose(A);  // A[i][j] -> A[j][i]
C = matmul(A_T, A);
```

### 3. 批量处理
```metal
// 一次处理多个行，减少线程间访问模式差异
for (uint row = 0; row < batchSize; row++) {
    // Process rows together for better coalescing
}
```

## 性能影响因子

1. **访问模式** - 非连续列访问是主要瓶颈
2. **矩阵维度** - K维度决定列访问步长
3. **缓存效率** - 共享内存可以缓解非连续访问
4. **数据类型** - FP16/FP32会影响访问模式效率

## 相关专题

- [GEMM](./GEMM/RESEARCH.md) - 通用矩阵乘法
- [MatrixTranspose](./MatrixTranspose/RESEARCH.md) - 矩阵转置
- [SparseMatrix](./SparseMatrix/RESEARCH.md) - 稀疏矩阵运算
