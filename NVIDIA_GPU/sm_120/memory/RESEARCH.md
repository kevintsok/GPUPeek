# Memory Subsystem Research

## 概述

内存子系统是 GPU 性能的关键因素。本模块研究 Blackwell 架构的内存层级和访问模式。

## 1. 全局内存带宽

### 1.1 带宽 vs 数据大小

| 数据大小 | 状态 | 读取带宽 | 写入带宽 |
|---------|------|----------|----------|
| 1 KB | 寄存器/L1 | ~95 GB/s | ~92 GB/s |
| 64 KB | L1 缓存 | ~285 GB/s | ~278 GB/s |
| 256 KB | L1/L2 边界 | ~420 GB/s | ~415 GB/s |
| 1 MB | L2 缓存 | ~580 GB/s | ~565 GB/s |
| 4 MB | L2 缓存 | ~745 GB/s | ~730 GB/s |
| 16 MB | **峰值** | **~811 GB/s** | ~798 GB/s |
| 64 MB | L2 miss → DRAM | ~765 GB/s | ~740 GB/s |
| 128 MB | DRAM | ~720 GB/s | ~705 GB/s |
| 256 MB | DRAM | ~680 GB/s | ~665 GB/s |

**关键发现**:
- 峰值带宽约 **811 GB/s** (16MB 工作集，L2 完全缓存)
- L1 缓存带宽约 285 GB/s (64KB)
- L2 缓存带宽约 745 GB/s (4MB)
- DRAM 带宽约 680-765 GB/s (64MB+)
- **写入带宽略低于读取带宽** (约 3-5% 差异)

![内存带宽 vs 数据尺寸](data/memory_bandwidth_vs_size.png)

![数据类型带宽对比](data/dtype_bandwidth_comparison.png)

### 1.2 跨距访问效率 (Read vs Write)

| Stride | 读取效率 | 写入效率 | Bank Conflict |
|--------|----------|----------|--------------|
| 1 | 100% | 100% | 无 |
| 2 | 98% | 97% | 无 |
| 4 | 95% | 94% | 低 |
| 8 | 88% | 85% | 中 |
| 16 | 72% | 68% | 高 |
| 32 | 45% | 42% | **最大** |
| 64 | 32% | 30% | 周期性 |
| 128 | 18% | 16% | 低 |

![Stride 访问效率](data/stride_efficiency.png)

**分析**:
- Stride = 1-2: 无明显冲突，效率接近 100%
- Stride = 4-8: 低中度冲突，效率开始下降
- Stride = 16-32: **严重冲突**，特别是 stride=32 时降到 35-45%
- Stride > 64: 缓存行跨越访问，带宽持续下降

**关键发现**: Stride = 32 是最差情况，与 32-bank 架构直接相关

## 2. 内存层级带宽

| 访问模式 | 带宽 |
|---------|------|
| 全局直接读取 | ~811 GB/s |
| 全局直接写入 | ~821 GB/s |
| 共享内存 R/W | **~1.50 TB/s** |
| L2 Streaming | ~767 GB/s |
| __ldg Bypass | ~822 GB/s |

**关键发现**: 共享内存带宽约 1.5 TB/s，显著高于全局内存

## 3. Occupancy vs 性能

| Block Size | 带宽 |
|------------|------|
| 32 | ~300 GB/s |
| 64 | ~450 GB/s |
| 128 | ~800 GB/s |
| 256 | ~880 GB/s |
| **512** | **~900 GB/s** |
| 1024 | ~610 GB/s |

**最佳 Block Size**: 256-512 线程

## 4. TMA (张量内存访问)

TMA 峰值带宽约 850 GB/s (16MB)

## 5. 分支分歧影响

对于简单 kernel，分支分歧开销不明显（约 8% 差异）。

## 6. Memory Fence 影响

Memory fence 引入约 50% 的性能开销。

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
