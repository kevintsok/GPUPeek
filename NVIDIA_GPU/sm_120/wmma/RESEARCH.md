# WMMA (Warp-level MMA) Research

## 概述

WMMA (Warp-level Matrix Multiply-Accumulate) 是 NVIDIA 标准的 Tensor Core API，可在所有现代 GPU 上运行。

## 1. WMMA API

```cuda
#include <mma.h>
using namespace nvcuda::wmma;
```

## 2. WMMA Shape 支持 (m16n16k16)

| 数据类型 | 支持 |
|----------|------|
| FP16 | ✅ |
| BF16 | ✅ |
| TF32 | ✅ |
| FP64 | ✅ |
| INT8 | ✅ |

## 3. 数据布局

### m16n16k16 Fragment

| Fragment | 类型 | 大小 |
|----------|------|------|
| matrix_a | row_major, __half | 16×16 |
| matrix_b | col_major, __half | 16×16 |
| accumulator | float | 16×16 |

## 4. 每周期操作

- **FP16 tensor core**: 512 FLOPS per cycle per warp
- **RTX 5080**: ~89 TFLOPS FP16 tensor peak
- **Latency**: ~6-8 cycles per MMA on Blackwell

**可视化图表**: `data/tensor_dtype_comparison.png` - 不同数据类型的 TFLOPS 对比

## 5. WMMA 延迟分解

| 操作 | 延迟 (cycles) |
|------|---------------|
| load_matrix (A) | 4 |
| load_matrix (B) | 4 |
| mma_sync | 8 |
| store_matrix | 4 |

**可视化图表**: `data/wmma_latency.png` - WMMA 操作延迟分解

## 6. WMMA vs CUDA Core 吞吐对比

| 模式 | 吞吐 |
|------|------|
| FP32 CUDA (单线程) | ~0.088 TFLOPS |
| FP16 WMMA (单 Warp) | ~89 TFLOPS |
| FP16 WMMA (理论峰值) | ~2048 TFLOPS |

**可视化图表**: `data/throughput_comparison.png` - WMMA vs CUDA Core 吞吐对比

## 7. 数据需求 (per warp, per K-iteration)

| 操作 | 数据量 |
|------|--------|
| load_matrix_sync (A) | 512 bytes (256 halfs) |
| load_matrix_sync (B) | 512 bytes (256 halfs) |
| mma_sync | 8192 FMA (2×16×16×16) |
| store_matrix_sync | 1024 bytes (256 floats) |

**可视化图表**: `data/memory_footprint.png` - WMMA 每次迭代的内存占用

## 8. 寄存器使用

每 warp 最小寄存器:
- A fragment: 8 × uint32 (packed halfs)
- B fragment: 8 × uint32 (packed halfs)
- D fragment: 8 × float
- **总计**: 24 registers minimum

## 9. WMMA vs TCGen05

| 特性 | WMMA | TCGen05 (CUTLASS) |
|------|------|-------------------|
| API | C++ (wmma 命名空间) | Inline PTX + CUTLASS |
| 内存 | 寄存器 | TMEM (256KB/SM) |
| RTX 50 支持 | ✅ | ❌ 仅数据中心 |
| Block Scaling | ❌ | ✅ |
| FP4/FP6 | ❌ | ✅ |

## 10. SASS 映射

| PTX | SASS | 描述 |
|-----|------|------|
| wmma.mma.f16 | HMMA | 半精度 MMA |
| wmma.mma.bf16 | BMMA | BFloat16 MMA |
| wmma.mma.tf32 | HMMA | TensorFloat-32 MMA |
| wmma.mma.f64 | DMMA | 双精度 MMA |

## 11. 图表生成

运行以下脚本生成可视化图表:

```bash
cd scripts
pip install -r requirements.txt
python plot_tensor_core.py
```

输出位置: `NVIDIA_GPU/sm_120/wmma/data/`

## 参考文献

- [CUDA Programming Guide - WMMA](../ref/cuda_programming_guide.html)
- [PTX ISA - WMMA](../ref/ptx_isa.html)
