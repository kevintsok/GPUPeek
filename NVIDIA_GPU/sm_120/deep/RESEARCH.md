# Deep Research

## 概述

深度研究测试，包括 L2 缓存、TMA、prefetch 等高级内存操作。

## 1. L2 缓存

### L2 工作集分析 (RTX 5080 实测)

| 数据大小 | 带宽 | 状态 |
|---------|------|------|
| 64 KB | 136.66 GB/s | L2 fits |
| 1 MB | 367.34 GB/s | L2 borderline |
| 4 MB | 677.53 GB/s | L2 thrashing |
| 8 MB | 740.78 GB/s | L2 thrashing |
| 16 MB | 772.16 GB/s | L2 thrashing |
| 32 MB | 772.16 GB/s | L2 thrashing |

![L2 带宽 vs 数据尺寸](data/l2_throughput_vs_size.png)

### L2 Thrashing

Stride > 8 导致带宽急剧下降，表明缓存行跨距访问效率低。

| Stride | 带宽 |
|--------|------|
| 1 | 739.74 GB/s |
| 2 | 605.92 GB/s |
| 4 | 566.30 GB/s |
| 8 | 420.93 GB/s |
| 16 | 403.85 GB/s |
| 64 | 203.34 GB/s |
| 256 | 226.21 GB/s |
| 1024 | 199.26 GB/s |
| 4096 | 380.82 GB/s |

![L2 Thrashing vs Stride](data/l2_thrashing_vs_stride.png)

**分析**: Stride 增大会导致严重的 L2 cache thrashing

## 4. NCU Profiling 分析

### FMA Throughput Kernel (FP16) NCU Profile

| 指标 | 值 | 说明 |
|------|-----|------|
| dram__bytes.sum | ~25-35 Mbyte | DRAM访问字节数 |
| l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum | 786,432 sectors | L1加载请求 |
| sm__pipe_fma_cycles_active.sum | ~1,573,500 cycles | FMA单元活跃周期 |

### 分析结论

| 指标 | 值 | 说明 |
|------|-----|------|
| DRAM访问 | ~25-35 MB/kernel | 内存带宽消耗 |
| L1加载 | 786,432 sectors | 恒定的加载请求量 |
| FMA活跃 | ~1.57M cycles/SM | 高计算密度 |

**L2 缓存瓶颈分析**:
- 实测 L2 带宽: 677-772 GB/s (4-32MB 工作集)
- L2 thrashing 导致带宽下降 (~740 GB/s peak)
- Stride > 8 时缓存效率急剧下降

### NCU 指标参考

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
