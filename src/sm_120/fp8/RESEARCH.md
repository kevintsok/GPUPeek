# FP8 Research

## 概述

FP8 (8-bit Floating Point) 是 Blackwell 和 Hopper 支持的低精度格式。

## 1. FP8 格式

| 格式 | 指数位 | 尾数位 | 描述 |
|------|--------|--------|------|
| E4M3 | 4 | 3 | 高精度 FP8 |
| E5M2 | 5 | 2 | 高动态范围 FP8 |

## 2. PTX 指令

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32   // FP8 E4M3
mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32   // FP8 E5M2
```

## 3. 应用场景

| 格式 | 适用场景 |
|------|----------|
| E4M3 | 权重+激活 |
| E5M2 | 梯度、动态范围大的张量 |

## 4. 与 FP16/FP32 对比

| 格式 | 位数 | 内存减少 | 精度 |
|------|------|----------|------|
| FP32 | 32 | 1x | 最高 |
| FP16 | 16 | 2x | 高 |
| FP8 E4M3 | 8 | 4x | 中 |
| FP8 E5M2 | 8 | 4x | 中低 |

## 参考文献

- [CUDA Programming Guide - FP8](../ref/cuda_programming_guide.html)
- [PTX ISA - FP8](../ref/ptx_isa.html)
