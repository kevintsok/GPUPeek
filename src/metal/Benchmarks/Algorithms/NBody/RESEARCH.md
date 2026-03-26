# N-Body Simulation Research

## 概述

本专题研究GPU上N体模拟（N-Body Simulation）的性能。N体模拟计算N个粒子间的相互作用力，广泛应用于天体物理学（引力模拟）和分子动力学（分子模拟）领域。

## 关键发现

### N体模拟性能

| 粒子数 | 交互次数/帧 | 复杂度 | 性能 |
|--------|-------------|--------|------|
| 256 | 65,280 | O(n²) | ~0.5-1 GFLOPS |
| 512 | 261,632 | O(n²) | ~0.8-1.5 GFLOPS |
| 1024 | 1,046,528 | O(n²) | ~1-2 GFLOPS |

### 算法复杂度

```
N体模拟算法:

Naive O(n²):
for each body i:
    for each body j (j ≠ i):
        compute force F_ij

复杂度: O(n²) pairwise interactions
每帧交互数: n × (n-1) ≈ n²
```

### 优化策略

#### 1. 共享内存分块
```metal
// 分块处理以利用缓存局部性
for (uint tile = 0; tile < numTiles; tile++) {
    // Load tile into shared memory
    // Process tile
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

#### 2. Barnes-Hut算法 O(n log n)
```cpp
// 构建八叉树空间分区
// 对远距离粒子使用粗粒度近似
if (distance > theta * cell_size) {
    // Use cell center of mass
} else {
    // Recursively process children
}
```

#### 3. GPU并行化
```metal
// 每个线程处理一个粒子
kernel void nbody_parallel(device float4* pos [[buffer(0)]],
                          device float3* acc [[buffer(1)]]) {
    uint id = thread_position_in_grid;
    // Compute force on particle id from all others
}
```

## 物理公式

### 引力相互作用
```
F = G * m1 * m2 / r²

其中:
- G: 引力常数
- m1, m2: 粒子质量
- r: 粒子间距离

软化参数(softening)防止奇点:
r² → r² + ε²
```

### 积分方法
```
Velocity Verlet (能量守恒):
v(t+dt) = v(t) + a(t) * dt
x(t+dt) = x(t) + v(t+dt) * dt
```

## 应用场景

1. **天体物理学** - 星系形成、黑洞模拟
2. **分子动力学** - 蛋白质折叠、材料模拟
3. **流体力学** - SPH (平滑粒子流体动力学)
4. **图形学** - 粒子系统、布料模拟

## 性能影响因子

1. **粒子数量** - O(n²)复杂度，大规模模拟需优化
2. **交互范围** - 截断距离可减少计算量
3. **软化参数** - 影响精度和数值稳定性
4. **时间步长** - 小步长提高精度但增加计算量

## 相关专题

- [MonteCarlo](../MonteCarlo/RESEARCH.md) - 随机方法
- [Stencil](../Stencil/RESEARCH.md) - 模板计算
- [AccelerationStructures](../AccelerationStructures/RESEARCH.md) - 加速结构
