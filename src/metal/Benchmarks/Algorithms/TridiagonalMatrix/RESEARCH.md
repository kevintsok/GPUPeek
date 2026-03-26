# Tridiagonal Matrix Solver Research

## 概述

本专题研究GPU上三对角矩阵（Tridiagonal Matrix）求解器的性能。三对角矩阵求解是科学与工程计算中的核心操作，广泛应用于偏微分方程数值解、插值和电路仿真等领域。

## 关键发现

### 算法对比

| 算法 | 复杂度 | 并行性 | 备注 |
|------|---------|--------|------|
| Thomas | O(n) | 低 | 顺序前向-后向回代 |
| Cyclic Reduction | O(n log n) | 高 | 分治并行化 |
| PCR | O(log n) | 最高 | 需多轮同步 |

### Thomas 算法原理

```
三对角矩阵系统 Ax = d:

| b0 c0  0  0 ...           | | x0 |   | d0 |
| a1 b1 c1  0 ...           | | x1 |   | d1 |
|  0 a2 b2 c2 ...           | | x2 | = | d2 |
|  ...                      | | .. |   | .. |
|  0 ...  a{n-1} b{n-1} c{n-1} | |x{n-1}| |d{n-1}|
|  0 ...    0   an   bn     | | xn |   | dn |

Thomas算法 (前向消元后向回代):

Forward Sweep:
c'[0] = c[0] / b[0]
d'[0] = d[0] / b[0]
for i = 1 to n-1:
    denom = b[i] - a[i] * c'[i-1]
    c'[i] = c[i] / denom
    d'[i] = (d[i] - a[i] * d'[i-1]) / denom

Backward Substitution:
x[n] = (d[n] - a[n] * d'[n-1]) / (b[n] - a[n] * c'[n-1])
for i = n-1 to 0:
    x[i] = d'[i] - c'[i] * x[i+1]
```

## 应用场景

### 1. 偏微分方程离散化
```
1D热传导方程: ∂T/∂t = α ∂²T/∂x²

隐式格式 (Crank-Nicolson):
产生三对角矩阵系统
```

### 2. 三次样条插值
```
每个样条段由三次多项式定义
连接点处一阶、二阶导数连续
产生三对角矩阵求解
```

### 3. 电路仿真
```
电阻网络节点分析
每个节点电压与相邻节点相关
形成三对角/带状矩阵
```

## 优化策略

### 1. 批量求解
```metal
// 一次求解多个独立的三对角系统
for (uint sys = 0; sys < batchSize; sys++) {
    solve_tridiagonal(A[sys], b[sys], x[sys]);
}
```

### 2. 共享内存暂存
```metal
// 使用threadgroup存储中间结果
threadgroup float cp[256];
threadgroup float dp[256];
```

### 3. 分块循环展开
```metal
// 处理多个元素减少分支开销
for (uint i = id; i < size; i += blockSize) {
    process_element(i);
}
```

## GPU并行化挑战

1. **数据依赖** - Thomas算法的前向/后向扫描有依赖
2. **同步开销** - PCR需要log n轮同步
3. **内存访问** - 非连续访问模式

## 性能影响因子

1. **矩阵大小** - 大矩阵更值得并行化
2. **对角占优性** - 影响数值稳定性
3. **批量大小** - 多个系统可并行求解
4. **硬件支持** - M3的硬件调度可能有帮助

## 相关专题

- [Stencil](../Stencil/RESEARCH.md) - 模板计算（PDE离散化）
- [Scan](../Scan/RESEARCH.md) - 前缀和算法
- [GEMM](../GEMM/RESEARCH.md) - 矩阵乘法
