# Ray-Sphere Intersection Research

## 概述

本专题研究GPU上光线与球体求交测试（Ray-Sphere Intersection）的性能。这是光线追踪和碰撞检测中的核心基元操作。

## 关键发现

### 光线求交公式

```
光线方程: P(t) = origin + t * direction
球体方程: |P - center|² = radius²

代入得二次方程:
|b|²t² + 2(b·(O-C))t + |O-C|² - r² = 0

其中:
- b = direction
- O = ray origin
- C = sphere center
- r = sphere radius

判别式: disc = b² - c
- disc > 0: 两个交点
- disc = 0: 一个切点
- disc < 0: 无交点
```

### 性能数据

| 配置 | 性能 |
|------|------|
| 64K rays x 64 spheres | ~50-100 Mrays/s |
| 256K rays x 64 spheres | ~40-80 Mrays/s |
| 1M rays x 64 spheres | ~30-60 Mrays/s |

### 优化策略

#### 1. 提前退出
```metal
if (t > 0.001f && t < tMin) {
    tMin = t;
}
```

#### 2. BVH预检测
```metal
// 先测试AABB，命中再测试球体
float3 t0 = (minB - ro) * invDir;
float tNear = max(max(tmin.x, tmin.y), tmin.z);
if (tNear <= tFar) {
    // 测试实际球体
}
```

#### 3. SIMD批量处理
```metal
// 一次处理4个球体
float4 oc0 = ro - sc0.xyz;
// ... 类似处理oc1, oc2, oc3
```

## 应用场景

### 1. 光线追踪
```
- 阴影光线测试
- 反射/折射光线
- 环境光遮蔽
```

### 2. 碰撞检测
```
- 粒子系统碰撞
- 游戏物理引擎
- 机器人路径规划
```

### 3. 射线投射
```
- UI点击检测
- 选择高亮
- 地形高度查询
```

## 硬件加速

### Apple M3 硬件光线追踪
```
- 专用RT单元
- 盒状交叉测试硬件支持
- BVH遍历加速
```

### 与NVIDIA对比
| 方面 | Apple M3 | NVIDIA RTX |
|------|----------|------------|
| 硬件RT | 有 | 有 |
| 加速结构 | BVH | BVH + Octree |
| 峰值性能 | ~10 GRays/s | ~100 GRays/s |

## 算法变体

### 1. 光线-盒子求交
```metal
float3 t0 = (minB - ro) * invDir;
float3 t1 = (maxB - ro) * invDir;
float tNear = max(max(t0.x, t0.y), t0.z);
float tFar = min(min(t1.x, t1.y), t1.z);
```

### 2. 光线-平面求交
```metal
float t = dot(planeNormal, ro) + d) / dot(planeNormal, rd);
```

### 3. 光线-三角形求交 (Möller-Trumbore)
```metal
float3 e1 = v1 - v0;
float3 e2 = v2 - v0;
float3 p = cross(rd, e2);
float det = dot(e1, p);
```

## 性能影响因子

1. **场景复杂度** - 球体数量直接影响性能
2. **加速结构** - BVH可指数级减少测试
3. **光线数量** - 批处理提高吞吐量
4. **硬件支持** - M3硬件RT单元大幅加速

## 相关专题

- [AccelerationStructures](./AccelerationStructures/RESEARCH.md) - BVH加速结构
- [RayTracing](../AccelerationStructures/RESEARCH.md) - 光线追踪完整流程
- [NBody](./NBody/RESEARCH.md) - N体模拟（引力交互）
