# Deep Research

## 概述

深度研究测试，包括 L2 缓存、TMA、prefetch 等高级内存操作。

## 1. L2 缓存

### L2 工作集分析 (RTX 5080 实测)

| 数据大小 | 带宽 | 状态 |
|---------|------|------|
| 64 KB | 102.74 GB/s | L2 fits |
| 1 MB | 354.85 GB/s | L2 borderline |
| 4 MB | 511.35 GB/s | L2 thrashing |
| 8 MB | 578.21 GB/s | L2 thrashing |
| 16 MB | 648.17 GB/s | L2 thrashing |
| 32 MB | 564.99 GB/s | L2 thrashing |

![L2 带宽 vs 数据尺寸](data/l2_throughput_vs_size.png)

### L2 Thrashing

Stride > 8 导致带宽急剧下降，表明缓存行跨距访问效率低。

| Stride | 带宽 |
|--------|------|
| 1 | 643.67 GB/s |
| 2 | 676.75 GB/s |
| 4 | 631.22 GB/s |
| 8 | 409.35 GB/s |
| 16 | 402.12 GB/s |
| 64 | 197.60 GB/s |
| 256 | 224.76 GB/s |
| 1024 | 236.30 GB/s |
| 4096 | 333.56 GB/s |

![L2 Thrashing vs Stride](data/l2_thrashing_vs_stride.png)

**分析**: Stride 增大会导致严重的 L2 cache thrashing

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
