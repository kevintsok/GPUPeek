# Tensor Core Emulation (WMMA) Research

## 概述

本专题研究GPU上Warp Matrix Multiply Accumulate (WMMA) 操作，模拟Tensor Core的行为，分析Apple GPU在缺少原生Tensor Core情况下的矩阵乘法性能。

## 背景

### 什么是WMMA？

WMMA (Warp Matrix Multiply Accumulate) 是NVIDIA引入的API，允许在warp级别执行矩阵乘法操作。每个warp的32个线程协作执行16x16x16的矩阵乘法。

```
NVIDIA Tensor Core (原生支持):
- 16x16x16 FMA per cycle
- 8-16x speedup vs CUDA cores
- 支持FP16, BF16, FP64, INT8

Apple GPU (软件模拟):
- 32-thread SIMD groups
- 无原生tensor core
- WMMA通过共享内存和tiling实现
```

## 关键发现

### WMMA实现方式

| 实现 | 原理 | 性能 |
|------|------|------|
| Naive | O(n³) 三层循环 | 最慢 |
| Tiled | 使用threadgroup共享内存 | 显著提升 |
| SIMD Block | 32线程协作 | 良好 |
| Vectorized | float4向量化 | 最优 |
| FP16 Tiled | 半精度+tiling | 2x加速 |

### 性能对比

| 实现 | GFLOPS | 说明 |
|------|--------|------|
| Naive | ~0.1-0.5 | 基准 |
| Tiled | ~5-15 | 共享内存复用 |
| SIMD Block | ~8-20 | 32线程协作 |
| FP16 Tiled | ~10-30 | 半精度+tiling |

### Apple M2 vs NVIDIA

| 方面 | Apple M2 | NVIDIA A100 (Tensor Core) |
|------|----------|---------------------------|
| Tensor Core | 无 | 192个 |
| INT8 TOPS | N/A | 1248 |
| FP16 TFLOPS | ~3.5 | 312 |
| WMMA模拟 | 是 | 原生支持 |

## 实现细节

### Tiled WMMA
```metal
kernel void wmma_tiled(...) {
    threadgroup float As[TILE_SIZE * TILE_SIZE];
    threadgroup float Bs[TILE_SIZE * TILE_SIZE];

    for (uint block = 0; block < size; block += TILE_SIZE) {
        // Load tiles into shared memory
        As[lid.y * TILE_SIZE + lid.x] = a[gid.x * size + block + lid.x];
        Bs[lid.y * TILE_SIZE + lid.x] = b[(block + lid.y) * size + gid.y];
        threadgroup_barrier(mem_flags::mem_none);

        // Compute partial result
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[lid.y * TILE_SIZE + k] * Bs[k * TILE_SIZE + lid.x];
        }
    }
}
```

### FP16 WMMA
```metal
kernel void wmma_fp16_tiled(...) {
    threadgroup half As[TILE_SIZE * TILE_SIZE];
    threadgroup half Bs[TILE_SIZE * TILE_SIZE];

    // half precision = 2x bandwidth
    // FP32 accumulation for accuracy
    float sum = 0.0f;
    for (...) {
        sum += float(As[i]) * float(Bs[j]);
    }
}
```

## 应用场景

### 1. 深度学习推理
```cpp
// Apple M2上运行神经网络
// 使用FP16 WMMA模拟tensor core
// 适用于Core ML模型加速
```

### 2. 矩阵运算库
```cpp
// 实现BLAS风格操作
// 利用tiling和FP16优化
```

### 3. 科学计算
```cpp
// 需要矩阵乘法的科学应用
// 性能不如专用tensor core
```

## 优化策略

1. **使用FP16** - 内存带宽减半
2. **Tiling** - 共享内存复用数据
3. **Vectorization** - float4一次处理4元素
4. **异步执行** - 计算与内存传输重叠

## 限制

- Apple GPU没有原生Tensor Core
- WMMA是软件模拟，性能有限
- 适合小规模矩阵运算
- 大规模运算建议用GPU专用库

## 相关专题

- [GEMM](./GEMM/RESEARCH.md) - 基础矩阵乘法
- [MixedPrecisionGEMM](./MixedPrecisionGEMM/RESEARCH.md) - 混合精度
- [FP16](../Analysis/FP16/RESEARCH.md) - 半精度性能
