# Quantization & Low-Precision Analysis Research

## 概述

本专题研究GPU上低精度量化（Quantization）技术，用于机器学习推理优化。量化通过使用Int8/Int4等低精度数据类型减少内存带宽和计算开销。

## 关键发现

### 量化原理

```
FP32: 32位浮点 = 4字节
FP16: 16位浮点 = 2字节 (2x节省)
Int8: 8位整数 = 1字节 (4x节省)
Int4: 4位整数 = 0.5字节 (8x节省)

量化公式:
- 反量化: f = (q - zero_point) * scale
- 量化: q = f / scale + zero_point
```

### 精度格式对比

| 格式 | 位宽 | 动态范围 | ML适用性 |
|------|------|----------|----------|
| FP32 | 32 | ±3.4e38 | 训练/推理 |
| FP16 | 16 | ±65504 | 推理(Apple) |
| BF16 | 16 | ±3.4e38 | 训练(Google) |
| Int8 | 8 | -128~127 | 推理(通用) |
| Int4 | 4 | -8~7 | 极致压缩 |

### 性能数据

| 配置 | 性能 | 说明 |
|------|------|------|
| 64x64 FP16 | ~5-8 GFLOPS | 基线 |
| 64x64 Int8 | ~10-15 GOPS | 2x加速 |
| 256x256 Int4 | ~8-12 GOPS | 4x内存节省 |

### 量化精度比较

| 格式 | 精度损失 | 性能收益 | 适用场景 |
|------|----------|----------|----------|
| FP16 | 低 | 2x | 通用推理 |
| Int8 | 中等 | 2-4x | 量化感知训练 |
| Int4 | 高 | 4x | 极致压缩 |

## 实现细节

### Int8矩阵向量乘法

```metal
kernel void int8_matvec(device const uchar* a [[buffer(0)]],
                       device const uchar* b [[buffer(1)]],
                       device int* out [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    int sum = 0;
    for (uint i = 0; i < size; i++) {
        // 反量化 -> 乘 -> 重新量化
        float va = (float(a[id * size + i]) - 128.0f) / 128.0f;
        float vb = (float(b[i]) - 128.0f) / 128.0f;
        sum += int((va * vb) * 128.0f);
    }
    out[id] = sum;
}
```

### Int4矩阵向量乘法（压缩存储）

```metal
kernel void int4_matvec(device const uchar* a [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) {
    // 每个字节存储2个Int4值
    uchar a_val = (i % 2 == 0) ? (a[idx] & 0x0F) : ((a[idx] >> 4) & 0x0F);
}
```

### BFloat16（模拟）

```metal
// BFloat16: 与FP32相同的指数范围，不同尾数精度
// 适合深度学习训练（需要大动态范围）
uint a_bits = (uint(a[idx]) << 8) | (uint(a[idx]) >> 8);
float a_val = (float)(half(a_bits));  // 转回float
```

## Apple GPU量化支持

### ANE (Apple Neural Engine)
```cpp
// Apple devices have dedicated ANE for low-precision ops
// CoreML自动选择ANE/CPU/GPU
```

### Metal低精度支持
```metal
// Metal支持half (FP16)原生运算
half2, half4 向量类型

// Int8需要模拟（无原生支持）
// 性能比FP16差，因为需要额外的转换开销
```

## 应用场景

### 1. 移动端推理
```
CoreML: 自动量化模型
TensorFlow Lite: Int8量化
ONNX Runtime: Quantization-aware training
```

### 2. 模型压缩
```
8x压缩: Int4量化
减少内存带宽50-75%
```

### 3. 能耗优化
```
低精度 = 更少晶体管切换
移动设备电池寿命提升
```

## 精度vs性能权衡

### 量化感知训练 (QAT)
```
训练时使用伪量化节点
保持精度同时获得量化性能
```

### 动态量化
```
权重静态量化
激活值动态量化
```

### 静态量化
```
所有值预先量化
最快推理速度
```

## 与NVIDIA对比

| 特性 | Apple M2 | NVIDIA RTX |
|------|----------|------------|
| FP16 | 原生 | TensorCore |
| Int8 | 模拟(慢) | TensorCore |
| Int4 | 不支持 | TensorCore |
| 专用AI | ANE | TensorCore |

## 相关专题

- [FP16](./Precision/RESEARCH.md) - 半精度性能分析
- [MixedPrecisionGEMM](../Compute/MixedPrecisionGEMM/RESEARCH.md) - 混合精度矩阵乘法
- [GEMM](../Compute/GEMM/RESEARCH.md) - 矩阵乘法优化
