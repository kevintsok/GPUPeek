# Deep Research

## 概述

深度研究测试，包括 L2 缓存、TMA、prefetch 等高级内存操作。

## 1. L2 缓存

### L2 工作集分析

| 数据大小 | 带宽 | 状态 |
|---------|------|------|
| 64 KB | ~123 GB/s | L1 缓存 |
| 1 MB | ~408 GB/s | L2 边界 |
| 4 MB | ~678 GB/s | L2 缓存 |
| 8 MB | ~748 GB/s | L2 缓存 |
| 16 MB | ~798 GB/s | L2 miss → DRAM |

**可视化图表**:
- `data/l2_throughput_vs_size.png` - L2 带宽 vs 数据尺寸曲线
- `data/l2_thrashing_vs_stride.png` - L2 Thrashing vs Stride 曲线

### L2 Thrashing

Stride > 8 导致带宽急剧下降，表明缓存行跨距访问效率低。

| Stride | 带宽 |
|--------|------|
| 1 | ~1234 GB/s |
| 16 | ~432 GB/s |
| 4096 | ~23 GB/s |

**分析**: Stride 增大会导致严重的 L2 cache thrashing

## 2. TMA (张量内存访问器)

### TMA 1D 拷贝

| 数据大小 | TMA 带宽 |
|---------|----------|
| 64 KB | ~7 GB/s |
| 1 MB | ~134 GB/s |
| 4 MB | ~431 GB/s |
| 16 MB | **~850 GB/s** |

### TMA 2D 拷贝

TMA 2D 带宽 ~626 GB/s

## 3. Prefetch

软件预取性能 ~251 GB/s

## 4. NCU 指标

| 指标 | 含义 |
|------|------|
| lts__tcs_hit_rate.pct | L2 缓存命中率 |
| dram__bytes.sum | 内存带宽 |
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU 利用率 |

## 图表生成

运行以下脚本生成可视化图表:

```bash
cd scripts
pip install -r requirements.txt
python plot_l2_cache_analysis.py
```

输出位置: `NVIDIA_GPU/sm_120/deep/data/`

## 参考文献

- [CUDA Programming Guide - L2 Cache](../ref/cuda_programming_guide.html)
