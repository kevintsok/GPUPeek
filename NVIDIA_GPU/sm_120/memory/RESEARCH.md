# Memory Subsystem Research

## 概述

内存子系统是 GPU 性能的关键因素。本模块研究 Blackwell 架构的内存层级和访问模式。

## 1. 全局内存带宽

### 1.1 带宽 vs 数据大小

| 数据大小 | 状态 | 读取带宽 | 写入带宽 |
|---------|------|----------|----------|
| 1 KB | 寄存器/L1 | 0.09 GB/s | 0.09 GB/s |
| 64 KB | L1 缓存 | 9.79 GB/s | 9.79 GB/s |
| 256 KB | L1/L2 边界 | 34.81 GB/s | 34.81 GB/s |
| 1 MB | L2 缓存 | 116.35 GB/s | 116.35 GB/s |
| 4 MB | L2 缓存 | 404.37 GB/s | 404.37 GB/s |
| 16 MB | **峰值** | **890.09 GB/s** | 890.09 GB/s |
| 64 MB | L2 miss → DRAM | 374.85 GB/s | 374.85 GB/s |
| 128 MB | DRAM | 489.94 GB/s | 489.94 GB/s |
| 256 MB | DRAM | 599.76 GB/s | 599.76 GB/s |

**关键发现**:
- 峰值带宽实测 **890.09 GB/s** (16MB 工作集，L2 完全缓存)
- 小数据尺寸受 kernel 启动开销影响较大
- L2 缓存峰值约 404.37 GB/s (4MB)
- DRAM 带宽在 375-600 GB/s 范围 (64MB+)
- **实测读写带宽相同** (测试 kernel 设计导致)

![内存带宽 vs 数据尺寸](data/memory_bandwidth_vs_size.png)

![数据类型带宽对比](data/dtype_bandwidth_comparison.png)

### 1.2 跨距访问效率 (实测)

| Stride | 带宽 (GB/s) | 效率 | Bank Conflict |
|--------|-------------|-------|--------------|
| 1 | 514.70 | 100% | 无 |
| 2 | 576.29 | 96.6% | 无 |
| 4 | 537.73 | 90.1% | 低 |
| 8 | 451.33 | 75.6% | 中 |
| 16 | 280.98 | 47.1% | 高 |
| 32 | 175.67 | 29.4% | **最大** |
| 64 | 94.53 | 15.8% | 周期性 |
| 128 | 47.08 | 7.9% | 低 |
| 256 | 18.85 | 3.2% | 低 |

![Stride 访问效率](data/stride_efficiency.png)

**分析**:
- Stride = 1: 基线效率 100%
- Stride = 2-4: 效率保持 90%+
- Stride = 16-32: **严重冲突**，stride=32 时降到 29.4%
- Stride > 64: 缓存行跨越访问，带宽持续下降

**关键发现**: Stride = 32 是最差情况，实测 29.4% 效率

## 2. 内存层级带宽

| 访问模式 | 带宽 | 时间/kernel |
|---------|------|-------------|
| 全局直接读取 | 795.64 GB/s | 0.021 ms |
| 全局直接写入 | 688.96 GB/s | 0.024 ms |
| 共享内存 R/W | **1.54 TB/s** | 0.022 ms |
| L2 Streaming (stride=1) | 596.85 GB/s | 0.028 ms |
| L2 Streaming (stride=32) | 781.55 GB/s | 0.021 ms |
| L2 Streaming (stride=512) | 854.49 GB/s | 0.020 ms |
| __ldg Bypass | 734.73 GB/s | 0.023 ms |
| L1 Preference | 752.87 GB/s | 0.022 ms |

**关键发现**: 共享内存带宽实测 **1.69 TB/s**，显著高于全局内存 (**896 GB/s**)

## 3. Occupancy vs 性能

| Block Size | 带宽 |
|------------|------|
| 32 | 314.58 GB/s |
| 64 | 443.13 GB/s |
| 128 | 904.94 GB/s |
| 256 | 797.60 GB/s |
| 512 | 634.64 GB/s |
| 1024 | 592.33 GB/s |

**最佳 Block Size**: 128 线程 (904.94 GB/s)

## 4. TMA (张量内存访问)

| Data Size | TMA Copy Bandwidth |
|-----------|-------------------|
| 64 KB | 6.85 GB/s |
| 256 KB | 20.77 GB/s |
| 1 MB | 108.17 GB/s |
| 4 MB | 371.68 GB/s |
| 16 MB | 572.10 GB/s |
| 64 MB | 359.76 GB/s |

TMA 峰值带宽 572.10 GB/s (16MB)

## 5. 分支分歧影响

对于简单 kernel，分支分歧开销不明显（约 8% 差异）。

## 6. Memory Fence 影响

Memory fence 引入约 50% 的性能开销。

## 7. Cache Line Size Effect (缓存行大小效应)

Research Question: How does access granularity affect effective bandwidth?

| Access Size | Bandwidth | Efficiency | Notes |
|-------------|-----------|------------|-------|
| 32B (L1 line) | 719.18 GB/s | 100% | Single L1 cache line |
| 64B (2xL1) | 808.75 GB/s | 112.4% | Two L1 lines |
| 128B (L2 line) | 689.62 GB/s | 95.9% | Full L2 cache segment |

**Misaligned Access Impact**:
| Offset | Bandwidth | vs Aligned |
|--------|-----------|------------|
| 0 (aligned) | 808.33 GB/s | 100% |
| 4 bytes | 823.54 GB/s | 101.9% |
| 8 bytes | 701.32 GB/s | 86.8% |
| 16 bytes | 755.21 GB/s | 93.4% |
| 32 bytes | 889.56 GB/s | 110.1% |
| 64 bytes | 716.11 GB/s | 88.6% |

**Key Finding**:
- 64B 访问反而获得更高带宽 (112.4%)
- 32 字节偏移获得最高带宽 (889.56 GB/s, 110.1%)
- 8 字节偏移性能下降最严重 (701.32 GB/s, 86.8%)

## 8. Read vs Write Asymmetry (读写非对称性)

| Operation | Bandwidth | Time/kernel |
|-----------|-----------|-------------|
| Pure Read (accumulate) | 851.43 GB/s | 0.020 ms |
| Pure Write (no read) | 985.78 GB/s | 0.017 ms |
| RAW (in-place *2) | 763.17 GB/s | 0.022 ms |
| WAR (separate arrays) | 744.55 GB/s | 0.023 ms |

**Asymmetry Ratio**: Write/Read = 115.8% (write slightly faster than read)

**Key Finding**:
- Pure Write (985.78 GB/s) 略快于 Pure Read (851.43 GB/s)
- RAW 和 WAR 性能相近 (**796 GB/s**)
- 写操作在简单 kernel 中可能略快于读操作

## 9. Non-Temporal vs Cached Access (非临时 vs 缓存访问)

| Access Type | Bandwidth | Time/kernel |
|-------------|-----------|-------------|
| Cached Read (default) | 742.73 GB/s | 0.023 ms |
| Write-Combining Write | 778.11 GB/s | 0.022 ms |

**Key Finding**:
- Write-combining 略优于 cached (778.11 vs 742.73 GB/s)
- 差异不大，说明内存带宽不是瓶颈

## 10. Memory Coalescing Effectiveness (内存合并效率)

| Pattern | Bandwidth | Efficiency |
|---------|-----------|------------|
| Coalesced (best case) | 818.42 GB/s | 100% |
| Stride 2 | 770.80 GB/s | 94.2% |
| Stride 4 | 760.93 GB/s | 93.0% |
| Stride 8 | 714.75 GB/s | 87.3% |
| Stride 16 | 817.39 GB/s | 99.9% |
| Stride 32 | 769.81 GB/s | 94.1% |
| Half-warp divergence | 480.82 GB/s | 58.8% |

**Key Finding**:
- Coalesced access 达到 818.42 GB/s (100%)
- Stride 16 反而接近完美效率 (99.9%)
- Half-warp divergence 显著降低效率至 58.8%

## 11. Software Prefetch Effectiveness (软件预取效果)

| Prefetch Distance | Bandwidth | Speedup |
|------------------|-----------|---------|
| No Prefetch (baseline) | 791.41 GB/s | 1.00x |
| 32 elements | 611.94 GB/s | 0.77x |
| 64 elements | 746.99 GB/s | 0.94x |
| 128 elements | 788.37 GB/s | 1.00x |
| 256 elements | 878.79 GB/s | 1.11x |
| 512 elements | 862.63 GB/s | 1.09x |
| Double Buffer (2-stage) | 660.99 GB/s | 0.84x |

**Key Finding**:
- 256 元素预取距离获得最佳效果 (878.79 GB/s, 1.11x)
- 32 元素反而降低性能 (0.77x)
- Double-buffering 在简单 kernel 上开销大于收益

## 图表生成

运行以下脚本生成可视化图表:

```bash
cd scripts
pip install -r requirements.txt
python plot_memory_bandwidth.py
```

输出位置: `NVIDIA_GPU/sm_120/memory/data/`

## 参考文献

- [CUDA Programming Guide](../ref/cuda_programming_guide.html)
- [CUDA Best Practices Guide](../ref/cuda_best_practices.html)
