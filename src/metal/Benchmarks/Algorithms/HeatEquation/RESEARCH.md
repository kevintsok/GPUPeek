# Heat Equation / Jacobi Iteration Research

## 概述

本专题研究GPU上热方程（Heat Equation）的数值解法，重点是 Jacobi 迭代法求解二维离散拉普拉斯方程。这是偏微分方程（PDE）数值求解的经典案例。

## 关键发现

### 算法对比

| 算法 | 收敛速度 | 并行性 | 复杂度 |
|------|----------|--------|---------|
| Jacobi | 最慢 | 高 | 简单 |
| Gauss-Seidel | 中等 | 低 | 中等 |
| SOR | 最快 | 低 | 复杂 |
| Red-Black SOR | 快 | 高 | 中等 |

### 热方程离散化

```
2D热方程: ∂T/∂t = α ∇²T

空间离散化 (5点 stencil):
∇²T ≈ (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]) / dx²

时间迭代 (显式 Euler):
T^{n+1} = T^n + Δt * α * ∇²T
```

### Jacobi 迭代

```metal
// 每次迭代求解: Ax = b 的形式
// 对于热方程的稳态解: ∇²T = 0
// 迭代公式: T_new[i,j] = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])
```

## 应用场景

### 1. 热传导模拟
```
- 电子设备散热
- 建筑能耗分析
- 材料加工（焊接、锻造）
```

### 2. 流体力学
```
- 不可压缩 Navier-Stokes 方程的压力泊松方程
- 速度场散度消除
```

### 3. 图像处理
```
- 各向异性扩散滤波
- 边缘检测
- 图像平滑
```

### 4. 电气工程
```
- 电阻网络的电势分布
- 电容器的电场计算
```

## 优化策略

### 1. 共享内存分块
```metal
// 加载tile和halo cells到共享内存
uint localIdx = lid.y * (tileSize + 2) + lid.x;
tile[localIdx] = in[globalPos.y * size.x + globalPos.x];

// 同步后计算laplacian
threadgroup_barrier(mem_flags::mem_none);
float lap = tile[idx-N] + tile[idx+N] + tile[idx-1] + tile[idx+1] - 4*tile[idx];
```

### 2. SOR (逐次超松弛)
```cpp
// x_new = x_old + ω * (x_new_estimate - x_old)
// ω > 1: SOR (加速收敛)
// ω = 1: Gauss-Seidel
// ω < 1: under-relaxation
```

### 3. Red-Black 着色
```metal
// 红黑棋盘着色: (x + y) % 2
// 红色和黑色可以并行更新
// 保持收敛性的同时提高并行度
```

## 性能分析

### 内存访问模式
- **Stencil 宽度**: 5点 (上下左右中心)
- **内存访问**: 每次迭代 5 reads + 1 write per cell
- **带宽需求**: 高 - 热方程是内存绑定问题

### 并行效率限制
1. **数据依赖**: Gauss-Seidel 的顺序依赖
2. **边界条件**: 需要特殊处理边界
3. **收敛条件**: 显式方法的时间步长限制

## Apple Metal 优化

1. **Threadgroup 优化**: 减少全局内存访问
2. **SIMD 组**: 利用 32-thread SIMD-group 并行
3. **Unified Memory**: 直接访问 CPU/GPU 共享内存

## 相关专题

- [Stencil](./Stencil/RESEARCH.md) - 模板计算通用模式
- [TridiagonalMatrix](./TridiagonalMatrix/RESEARCH.md) - 另一种迭代求解器
- [GEMM](./GEMM/RESEARCH.md) - 矩阵运算
