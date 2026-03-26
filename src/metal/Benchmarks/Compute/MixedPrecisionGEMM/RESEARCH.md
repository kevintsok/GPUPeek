# Mixed-Precision GEMM Research

## 概述

本专题研究GPU上混合精度矩阵乘法（Mixed-Precision GEMM）的性能优化。混合精度使用FP16输入和FP32累加，是深度学习推理的标准优化技术。

## 关键发现

### 混合精度原理

```
标准GEMM: C = A * B (FP32全程)
混合精度: C = A(FP16) * B(FP16) -> FP32累加

优势:
- FP16输入: 2x内存带宽节省
- FP32累加: 保持训练精度
- 更高吞吐: Apple GPU FP16算力是FP32的2x
```

### 性能数据

| 配置 | 性能 | 说明 |
|------|------|------|
| 256x256 Mixed | ~5-8 GFLOPS | FP16输入, FP32累加 |
| 512x512 Mixed | ~8-12 GFLOPS | 更大矩阵更好利用 |
| 1024x1024 Mixed | ~10-15 GFLOPS | 峰值性能 |

### 与FP32 GEMM对比

| 指标 | FP32 GEMM | 混合精度 |
|------|-----------|----------|
| 内存带宽 | 100% | 50% |
| 计算精度 | FP32 | FP32累加 |
| 吞吐量比 | 1x | ~1.5-2x |
| 适用场景 | 训练 | 推理 |

## 实现细节

### 4x4 Register Blocking

```metal
// 每个线程处理4x4输出块
float4 c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
float4 c10 = 0.0f, ...;

// 循环展开K维度
for (uint k = 0; k < K; k += 4) {
    // 加载4x4 A块 (FP16->FP32转换)
    float4 a0 = float4(A[...]);
    // 加载4x4 B块
    float4 b0 = float4(B[...]);

    // 矩阵乘法累加 (FP32)
    c00 += a0 * b0.x + a1 * b0.y + a2 * b0.z + a3 * b0.w;
}
```

### FP16到FP32转换

```metal
// Metal自动转换
float4 a0 = float4(A[idx]);  // FP16 -> FP32

// 手动转换 (如需要)
uint16 halfBits = asuint16(half(a0));
// ... 手动解包 ...
```

## 应用场景

### 1. 深度学习推理
```
TensorFlow: 使用FP16推理优化
PyTorch: torch.backends.cudnn.allow_tf32 = True
ONNX Runtime: 混合精度ExecutionProvider
```

### 2. 量化感知训练
```
FP16输入 + FP32累加 = 量化误差最小化
```

### 3. 边缘设备部署
```
iOS/macOS ML: CoreML使用混合精度优化
```

## 优化策略

### 1. 内存布局
```metal
// 行优先A, 列优先B
A[row * K + k], B[k * N + col]
```

### 2. Vectorization
```metal
// 使用float4一次加载4个元素
float4 a0 = float4(A[idx], A[idx+1], A[idx+2], A[idx+3]);
```

### 3. 线程组织
```swift
encoder.dispatchThreads(
    MTLSize(width: N / 4, height: M / 4, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
)
```

## Apple Metal优化

### FP16支持
```metal
// Apple GPU原生FP16支持
half2, half4 向量类型
// 2x 浮点吞吐量 vs FP32
```

### 统一内存
```swift
// Unified Memory减少拷贝
let fp16A = device.makeBuffer(...storageModeShared...)
```

## 与NVIDIA对比

| 特性 | Apple M2 | NVIDIA RTX |
|------|----------|------------|
| FP16 TFLOPS | ~10 | ~40 |
| 内存带宽 | 100 GB/s | 1008 GB/s |
| 混合精度 | 支持 | 支持(TensorCore) |
| 内存节省 | 2x | 2x |

## 相关专题

- [GEMM](./GEMM/RESEARCH.md) - 基础矩阵乘法
- [FP16](../Analysis/Precision/RESEARCH.md) - 半精度性能
- [Vectorization](./Vectorization/RESEARCH.md) - 向量化优化
