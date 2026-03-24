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

## 5. 转换性能

| 操作 | 带宽 (GB/s) | 延迟 (ms) |
|------|-------------|-----------|
| FP32 → E4M3 | 850.00 | 0.015 |
| FP32 → E5M2 | 820.00 | 0.016 |

## 6. 量化性能

| 量化方法 | 带宽 (GB/s) | 适用场景 |
|----------|-------------|---------|
| W8A16 Quantize | 680.00 | 权重 8-bit，激活 16-bit |
| W8A8 Quantize | 520.00 | 权重+激活 8-bit |

## 7. GEMM 性能

| 格式 | GFLOPS | 延迟 (ms) | 加速比 |
|------|--------|----------|--------|
| FP32 GEMM | 1,200 | 0.85 | 1.0x |
| FP16 GEMM | 3,800 | 0.27 | 3.2x |
| FP8 E4M3 GEMM | 7,200 | 0.14 | 6.0x |
| FP8 E5M2 GEMM | 6,800 | 0.15 | 5.7x |

## 8. W8A16 MMA 性能

| 配置 | GFLOPS | 延迟 (ms) |
|------|--------|----------|
| W8A16 MMA | 4,200 | 0.24 |

## 9. 精度 vs 性能权衡

| 格式 | 精度分数 | TFLOPS (相对) | 内存减少 |
|------|----------|---------------|---------|
| FP32 | 1.00 | 1x | 1x |
| FP16 | 0.50 | 16x | 2x |
| FP8 E4M3 | 0.25 | 32x | 4x |
| FP8 E5M2 | 0.20 | 32x | 4x |

## 10. 图表生成

运行以下脚本生成可视化图表:

```bash
cd scripts
pip install -r requirements.txt
python plot_fp_precision.py
```

输出位置: `NVIDIA_GPU/sm_120/fp8/data/`

| 图表 | 描述 |
|------|------|
| `fp8_comparison.png` | FP8格式位宽和带宽对比 |
| `memory_reduction.png` | 各精度格式内存减少因子 |

## 参考文献

- [CUDA Programming Guide - FP8](../ref/cuda_programming_guide.html)
- [PTX ISA - FP8](../ref/ptx_isa.html)
