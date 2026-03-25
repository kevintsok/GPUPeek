# Vectorization Research

## 概述

本专题研究Apple M2 GPU上的向量类型(float2, float4, half2, half4)对内存带宽和计算性能的影响。

## 关键发现

### 向量化读取带宽

| 数据类型 | 带宽 | 相对标量 |
|----------|------|----------|
| Float (标量) | 0.80-0.92 GB/s | 1.0x |
| Float2 | ~1.2 GB/s | ~1.3x |
| Float4 | 3.56-3.79 GB/s | **~4x** |
| Half2 | ~1.5 GB/s | ~1.7x |
| Half4 | ~3.8 GB/s | **~4x** |

### 向量化性能对比

| 操作 | Float4 | Half4 | 说明 |
|------|--------|-------|------|
| 向量读取 | 0.17 GOPS | 0.19 GOPS | Half4略优 |
| 向量运算 | 0.17 GOPS | 0.19 GOPS | Half4更快 |

### Float vs Half 对比

| 指标 | Float4 | Half4 | 差异 |
|------|--------|-------|------|
| 带宽效率 | ~4x | ~4x | 相同 |
| 计算效率 | 1.0x | **1.12x** | Half4更快 |
| 精度 | 32位 | 16位 | Float更高 |

## 关键洞察

1. **Float4提供约4x加速** - 接近理论向量化收益
2. **Half4是最高效的向量格式** - 结合了带宽和计算效率
3. **向量化是内存受限Kernel的关键优化** - 充分利用内存带宽
4. **Half精度对ML推理足够** - 可接受精度损失换取性能

## 优化策略

1. **使用float4进行内存操作** - 读取和写入都使用float4
2. **优先使用half4** - 对于精度要求不高的场景
3. **对齐到向量边界** - 确保数据地址是向量大小的倍数
4. **利用Metal的向量化加载** - 使用texture或buffer的向量类型

## 相关专题

- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽基础
- [GEMM](../GEMM/RESEARCH.md) - 矩阵乘法中的向量化
- [Precision Analysis](../../Analysis/Precision/RESEARCH.md) - 数值精度分析
