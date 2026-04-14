# Image Processing Research

## 概述

本专题研究GPU上常见图像处理操作的性能，包括卷积（Box Blur、Gaussian Blur）、边缘检测（Sobel）和形态学操作（Dilate、Erode）。

## 关键发现

### 图像处理操作性能对比

| 操作 | 性能 | 说明 |
|------|------|------|
| Box Blur 3x3 | ~100-200 MP/s | 简单但有伪影 |
| Gaussian Blur 5x5 | ~150-300 MP/s | 可分离优化 |
| Sobel Edge | ~80-150 MP/s | 两遍操作 |
| Dilate 3x3 | ~100-200 MP/s | 内存受限 |

### 滤波器对比

```
Box Blur (3x3):
[1 1 1]   优点: 简单
[1 1 1]   缺点: 产生伪影
[1 1 1] / 9

Gaussian Blur (5x5):
[1 4 6 4 1]   优点: 平滑，无伪影
[4 16 24 16 4] 缺点: 计算量更大
[6 24 36 24 6]
[4 16 24 16 4]
[1 4 6 4 1] / 256
```

### Apple M2 图像处理特性

1. **Float4向量化** - RGBA四个通道同时处理
2. **共享内存** - 分块处理提高缓存命中率
3. **可分离卷积** - 2D卷积分解为两个1D卷积

## 优化策略

### 1. 可分离卷积
```metal
// 水平 pass
for (int dx = -2; dx <= 2; dx++) {
    sum += in[y * width + x + dx] * weights[dx + 2];
}

// 垂直 pass
for (int dy = -2; dy <= 2; dy++) {
    sum += temp[(y + dy) * width + x] * weights[dy + 2];
}

// 复杂度从 O(n²) 降到 O(2n)
```

### 2. 边界处理
```metal
// Clamp 避免分支
int2 idx = clamp(int2(gid) + int2(dx, dy), int2(0), int2(size) - 1);
```

### 3. 并行 bilateral filter
```metal
// 每个像素独立计算，适合GPU
float4 center = in[gid.y * width + gid.x];
float weight_sum = 0.0f;
float4 result = float4(0.0f);
```

## 性能影响因子

1. **滤波器大小** - 越大计算量越大
2. **可分离性** - 可分离操作可减少计算量
3. **内存访问模式** - 局部性影响缓存效率
4. **向量化程度** - Float4 同时处理4通道

## 相关专题

- [Convolution](../Convolution/RESEARCH.md) - 卷积神经网络基础
- [Stencil](../Stencil/RESEARCH.md) - 模板计算
- [Memory Bandwidth](../Bandwidth/RESEARCH.md) - 图像处理内存受限
