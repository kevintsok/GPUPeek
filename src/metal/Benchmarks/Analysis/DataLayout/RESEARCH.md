# SoA vs AoS Data Layout Analysis Research

## 概述

本专题研究GPU上数据结构布局（Data Layout）对缓存效率的影响。对比Structure of Arrays (SoA) 和 Array of Structures (AoS) 两种数据组织方式。

## 关键发现

### 数据布局概念

```
AoS (Array of Structures): 按对象交错存储
位置0: [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, mass]
位置1: [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, mass]
...

SoA (Structure of Arrays): 按字段连续存储
pos_x: [p0.x, p1.x, p2.x, ...]
pos_y: [p0.y, p1.y, p2.y, ...]
pos_z: [p0.z, p1.z, p2.z, ...]
vel_x: [v0.x, v1.x, v2.x, ...]
...

Hybrid: SoA + 对象分组
组0: [pos[0..255], vel[0..255], mass[0..255]]
组1: [pos[256..511], vel[256..511], mass[256..511]]
...
```

### 性能对比

| 布局 | 缓存效率 | 内存连续性 | 适用场景 |
|------|----------|------------|----------|
| AoS | 差 | 差(交错) | 少量对象，coalesced访问 |
| SoA | 优 | 好 | 大量对象，字段并行处理 |
| Hybrid | 中 | 中 | 平衡局部性和批处理 |

### 性能数据

| 配置 | AoS | SoA | Hybrid |
|------|-----|-----|--------|
| 1024粒子 | ~2-4x慢 | 最快 | 中等 |
| 4096粒子 | ~2-4x慢 | 最快 | 中等 |
| 16384粒子 | ~2-4x慢 | 最快 | 中等 |

## 实现细节

### AoS访问模式（缓存不友好）
```metal
kernel void aos_process(device float* data [[buffer(0)]], uint id) {
    // 跨步访问：id*7, id*7+1, id*7+2, ...
    float3 pos = float3(data[id * 7], data[id * 7 + 1], data[id * 7 + 2]);
    float3 vel = float3(data[id * 7 + 3], data[id * 7 + 4], data[id * 7 + 5]);
    float mass = data[id * 7 + 6];
}
```

### SoA访问模式（缓存友好）
```metal
kernel void soa_process(device float3* pos [[buffer(0)]], uint id) {
    // 连续访问：pos[id], vel[id], mass[id]
    float3 p = pos[id];
    float3 v = vel[id];
    float m = mass[id];
}
```

### Hybrid访问模式（平衡）
```metal
kernel void hybrid_process(device float* posX [[buffer(0)]], uint id) {
    uint group = id / 256;  // 每256个对象一组
    uint idx = id % 256;
    uint base = group * 256;
    float3 pos = float3(posX[base + idx], posY[base + idx], posZ[base + idx]);
}
```

## 应用场景

### 1. 粒子系统
```cpp
// 物理模拟：粒子属性独立访问
// SoA: 同时处理所有粒子的x坐标
// AoS: 需要跳跃访问每个粒子的所有字段
```

### 2. 有限元分析
```cpp
// 节点数据: pressure[], temperature[], velocity[]
// SoA更好：每次迭代只更新部分属性
```

### 3. 机器学习
```cpp
// 权重矩阵: weights[input][output]
// 批量数据: features[batch][feature]
// SoA适合特征并行处理
```

### 4. 图处理
```cpp
// 顶点属性: color[], distance[], parent[]
// 边列表: src[], dst[], weight[]
```

## 缓存行为分析

### AoS缓存污染
```
缓存行(64字节) = 16个float

AoS访问pos.x时加载:
[pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, mass, ?]
              ^^^^^^^^ 这些可能用不到

造成缓存污染，浪费带宽
```

### SoA缓存利用
```
SoA访问pos.x时加载:
[pos.x, pos.x, pos.x, pos.x, ...] x16个连续float

完全利用缓存行，无浪费
```

## Apple Metal优化

### 向量化加载
```metal
// Metal可以向量化加载连续数据
float4 posX = posXBuffer[id];  // 一次加载4个float
```

### 内存对齐
```metal
// SoA天然对齐到4字节边界
// AoS需要手动填充对齐
```

## 选择指南

| 场景 | 推荐布局 | 原因 |
|------|----------|------|
| 少量对象，高局部性 | AoS | 一次加载相关字段 |
| 大量对象，字段并行 | SoA | 最大化缓存利用 |
| 批处理，SIMD友好 | Hybrid | 平衡两者 |
| 稀疏访问 | SoA | 只加载需要的字段 |

## 相关专题

- [Cache](../Cache/RESEARCH.md) - 缓存行为分析
- [Bandwidth](../Memory/Bandwidth/RESEARCH.md) - 内存带宽
- [NBody](../NBody/RESEARCH.md) - 粒子系统模拟
- [MonteCarlo](../MonteCarlo/RESEARCH.md) - 蒙特卡洛模拟
