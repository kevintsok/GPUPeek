# Memory Subsystem Research

## 概述

内存子系统是 GPU 性能的关键因素。本模块研究 Blackwell 架构的内存层级和访问模式。

## 1. 全局内存带宽

### 1.1 带宽 vs 数据大小

| 数据大小 | 状态 | 读取带宽 | 写入带宽 |
|---------|------|----------|----------|
| 1 KB | 寄存器/L1 | 0.01 GB/s | 0.01 GB/s |
| 64 KB | L1 缓存 | 6.80 GB/s | 6.80 GB/s |
| 256 KB | L1/L2 边界 | 1.89 GB/s | 1.89 GB/s |
| 1 MB | L2 缓存 | 108.32 GB/s | 108.32 GB/s |
| 4 MB | L2 缓存 | 311.89 GB/s | 311.89 GB/s |
| 16 MB | **峰值** | **808.28 GB/s** | 808.28 GB/s |
| 64 MB | L2 miss → DRAM | 366.92 GB/s | 366.92 GB/s |
| 128 MB | DRAM | 182.67 GB/s | 182.67 GB/s |
| 256 MB | DRAM | 223.64 GB/s | 223.64 GB/s |

**关键发现**:
- 峰值带宽实测 **808.28 GB/s** (16MB 工作集，L2 完全缓存)
- 小数据尺寸受 kernel 启动开销影响较大
- L2 缓存峰值约 311.89 GB/s (4MB)
- DRAM 带宽在 182-367 GB/s 范围 (64MB+)
- **实测读写带宽相同** (测试 kernel 设计导致)

![内存带宽 vs 数据尺寸](data/memory_bandwidth_vs_size.png)

![数据类型带宽对比](data/dtype_bandwidth_comparison.png)

### 1.2 跨距访问效率 (实测)

| Stride | 实测效率 | Bank Conflict |
|--------|----------|--------------|
| 1 | 100% | 无 |
| 2 | 66.1% | 无 |
| 4 | 74.9% | 低 |
| 8 | 55.7% | 中 |
| 16 | 38.9% | 高 |
| 32 | 24.0% | **最大** |
| 64 | 14.3% | 周期性 |
| 128 | 10.2% | 低 |
| 256 | 4.8% | 低 |

![Stride 访问效率](data/stride_efficiency.png)

**分析**:
- Stride = 1: 基线效率 100%
- Stride = 2-4: 效率反而下降，可能与测试方法相关
- Stride = 16-32: **严重冲突**，stride=32 时降到 24%
- Stride > 64: 缓存行跨越访问，带宽持续下降

**关键发现**: Stride = 32 是最差情况，实测 24% 效率

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

## 7. Cache Line Size Effect (缓存行大小效应)

Research Question: How does access granularity affect effective bandwidth?

| Access Size | Bandwidth | Efficiency | Notes |
|-------------|-----------|------------|-------|
| 32B (L1 line) | ~800 GB/s | 100% | Single L1 cache line |
| 64B (2xL1) | ~790 GB/s | 98% | Two L1 lines |
| 128B (L2 line) | ~780 GB/s | 97% | Full L2 cache segment |

**Misaligned Access Impact**:
| Offset | Bandwidth | vs Aligned |
|--------|-----------|------------|
| 0 (aligned) | ~800 GB/s | 100% |
| 4 bytes | ~795 GB/s | 99% |
| 8 bytes | ~790 GB/s | 99% |
| 16 bytes | ~780 GB/s | 97% |
| 32 bytes | ~760 GB/s | 95% |
| 64 bytes | ~740 GB/s | 92% |

**Key Finding**:
- Misaligned access reduces effective bandwidth due to cache line boundary crossing
- 128B alignment is optimal for L2 cache efficiency

## 8. Read vs Write Asymmetry (读写非对称性)

| Operation | Bandwidth | Time/kernel |
|-----------|-----------|-------------|
| Pure Read (accumulate) | 422.96 GB/s | 0.040 ms |
| Pure Write (no read) | 188.68 GB/s | 0.089 ms |
| RAW (in-place *2) | 217.67 GB/s | 0.077 ms |
| WAR (separate arrays) | 825.18 GB/s | 0.020 ms |

**Asymmetry Ratio**: Read/Write = 224% (read faster than write for this kernel)

**Key Finding**:
- Pure read (422.96 GB/s) 显著快于 pure write (188.68 GB/s)
- RAW (Read-After-Write) dependency: 217.67 GB/s
- WAR (Write-After-Read) 在独立数组时性能最佳: 825.18 GB/s
- 写操作通常受限于内存控制器带宽

## 9. Non-Temporal vs Cached Access (非临时 vs 缓存访问)

| Access Type | Bandwidth | Use Case |
|-------------|-----------|----------|
| Cached Read (default) | ~811 GB/s | Data reuse |
| Write-Combining Write | ~815 GB/s | One-time streaming |

**Key Finding**:
- Write-combining benefits: one-time writes, large streaming data
- Cached access benefits: data reuse, sequential reads

## 10. Memory Coalescing Effectiveness (内存合并效率)

| Pattern | Bandwidth | Efficiency |
|---------|-----------|------------|
| Coalesced (best case) | 263.39 GB/s | 100% |
| Stride 2 | 138.44 GB/s | 52.6% |
| Stride 4 | 210.31 GB/s | 79.8% |
| Stride 8 | 114.46 GB/s | 43.5% |
| Stride 16 | 184.24 GB/s | 69.9% |
| Stride 32 | 156.45 GB/s | 59.4% |
| Half-warp divergence | 178.39 GB/s | 67.7% |

**Key Finding**:
- Coalesced access: threads in warp access sequential addresses → 100% efficiency
- Uncoalesced strided access 显著降低效率
- Half-warp divergence splits warp, reducing efficiency to ~68%

## 11. Software Prefetch Effectiveness (软件预取效果)

| Prefetch Distance | Bandwidth | Speedup |
|------------------|-----------|---------|
| No Prefetch (baseline) | 392.82 GB/s | 1.00x |
| 32 elements | 255.80 GB/s | 0.65x |
| 64 elements | 673.57 GB/s | 1.72x |
| 128 elements | 572.12 GB/s | 1.46x |
| 256 elements | 492.88 GB/s | 1.25x |
| 512 elements | 253.49 GB/s | 0.65x |
| Double Buffer (2-stage) | 378.94 GB/s | 0.96x |

**Key Finding**:
- 软件预取效果显著: 64 元素距离达到 1.72x 加速
- 32 和 512 元素距离反而降低性能
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
