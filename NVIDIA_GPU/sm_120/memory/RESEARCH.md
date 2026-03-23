# Memory Subsystem Research

## 概述

内存子系统是 GPU 性能的关键因素。本模块研究 Blackwell 架构的内存层级和访问模式。

## 1. 全局内存带宽

### 1.1 带宽 vs 数据大小

| 数据大小 | 状态 | 带宽 |
|---------|------|------|
| 64 KB | L1 缓存 | ~32 GB/s |
| 1 MB | L1/L2 边界 | ~74 GB/s |
| 4 MB | L2 缓存 | ~296 GB/s |
| 16 MB | 峰值 | ~643 GB/s |
| 256 MB | 恢复 | ~615 GB/s |

**关键发现**: 峰值带宽约 640-820 GB/s (16MB 工作集)

**可视化图表**:
- `data/memory_bandwidth_vs_size.png` - 带宽 vs 尺寸曲线
- `data/dtype_bandwidth_comparison.png` - 不同数据类型带宽对比

### 1.2 跨距访问效率

| Stride | 带宽效率 |
|--------|----------|
| 1 | 100% |
| 2 | 86% |
| 4 | 86% |
| 8 | 80% |
| 16 | 62% |
| 32 | 35% |
| 64 | 23% |
| 128 | 11% |

**分析**: Stride > 16 后带宽急剧下降

**可视化图表**: `data/stride_efficiency.png` - Stride 访问效率柱状图

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
- [NVIDIA Memory Documentation](../ref/memory_guide.html)
