# Acceleration Structures (BVH) Research

## 概述

本专题研究GPU上加速结构（Bounding Volume Hierarchy, BVH）的性能。BVH是光线追踪和碰撞检测中的核心数据结构，用于加速光线与场景物体的求交测试。

## 关键发现

### 光线追踪性能对比

| 方法 | 复杂度 | 性能 |
|------|---------|------|
| Brute Force | O(rays × spheres) | ~1-5 MR/s |
| Ray-AABB | O(rays × nodes) | ~10-50 MR/s |
| BVH Traversal | O(rays × log(spheres)) | ~50-200 MR/s |

### BVH结构

```
BVH Node:
[min.x, min.y, min.z, max.x, max.y, max.z, left_idx, data]

Binary Tree:
       [Root AABB]
       /         \
  [Left AABB]  [Right AABB]
    /    \        /    \
 [Leaf] [Leaf]  [Leaf] [Leaf]
```

### Apple GPU光线追踪

1. **M3硬件支持** - Apple M3系列有硬件光线追踪单元
2. **软件模拟** - M1/M2需要软件实现
3. **统一内存** - 共享内存架构，光线追踪延迟低

## 加速结构类型

### 1. BVH (Bounding Volume Hierarchy)
```metal
// 光线-AABB求交
float3 t0 = (minB - orig) * invDir;
float3 t1 = (maxB - orig) * invDir;
float tNear = max(max(tmin.x, tmin.y), tmin.z);
float tFar = min(min(tmax.x, tmax.y), tmax.z);
```

### 2. SAH (Surface Area Heuristic)
```cpp
// 最优分割成本
cost = left_count * SAH(left) + right_count * SAH(right)
```

### 3. KD-Tree
```cpp
// 按轴分割空间
if (axis == 0) {
    // split by x
} else if (axis == 1) {
    // split by y
}
```

## 优化策略

### 1. 光线批处理
```metal
// 所有光线一起测试AABB
for (uint node = 0; node < maxNodes; node++) {
    // 测试当前节点
}
```

### 2. 栈遍历
```metal
// 深度优先遍历
stack[stackPtr++] = leftChild;
stack[stackPtr++] = rightChild;
```

### 3. 剪枝
```metal
if (tNear > tFar) continue;  // 未命中
if (tFar < 0.0f) continue;   // 在光线背后
```

## 性能影响因子

1. **场景复杂度** - 物体数量影响BVH深度
2. **光线数量** - 批处理提高吞吐量
3. **加速结构质量** - SAH vs 均匀分割
4. **硬件支持** - M3硬件光线追踪

## 应用场景

1. **光线追踪渲染** - 阴影、反射、折射
2. **碰撞检测** - 游戏物理引擎
3. **路径规划** - RRT算法
4. **射线投射** - UI交互点击检测

## 相关专题

- [Ray Tracing](../RayTracing/RESEARCH.md) - 光线追踪渲染
- [Physics](../Physics/RESEARCH.md) - 物理模拟
- [SpatialIndexing](../SpatialIndexing/RESEARCH.md) - 空间索引
