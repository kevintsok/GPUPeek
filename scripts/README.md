# GPUPeek Chart Generation Scripts

## 概述

本目录包含用于生成 GPU 性能图表的 Python 脚本。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 脚本列表

| 脚本 | 用途 | 输出目录 |
|------|------|----------|
| `plot_l2_cache_analysis.py` | L2 缓存分析图表 | `sm_120/deep/data/` |
| `plot_memory_bandwidth.py` | 内存带宽分析图表 | `sm_120/memory/data/` |
| `plot_cuda_core_throughput.py` | CUDA Core 算力图表 | `sm_120/cuda_core/data/` |
| `plot_redux_sync.py` | Redux.sync 性能图表 | `sm_120/redux_sync/data/` |
| `plot_barrier_sync.py` | Barrier 同步图表 | `sm_120/barrier/data/` |
| `plot_atomic_ops.py` | 原子操作图表 | `sm_120/atomic/data/` |
| `plot_tensor_core.py` | Tensor Core 图表 | `sm_120/wmma/data/` |
| `plot_warp_specialize.py` | Warp Specialization 图表 | `sm_120/warp_specialize/data/` |
| `plot_dp4a.py` | DP4A 图表 | `sm_120/dp4a/data/` |
| `plot_mbarrier.py` | MBarrier 图表 | `sm_120/mbarrier/data/` |
| `plot_tensor_mem.py` | Tensor Mem 图表 | `sm_120/tensor_mem/data/` |
| `plot_unified_memory.py` | Unified Memory 图表 | `sm_120/unified_memory/data/` |
| `plot_multi_stream.py` | Multi-Stream 图表 | `sm_120/multi_stream/data/` |
| `plot_cuda_graph.py` | CUDA Graph 图表 | `sm_120/cuda_graph/data/` |
| `plot_fp_precision.py` | FP8/FP4/FP6 精度图表 | `sm_120/fp8/data/`, `sm_120/fp4_fp6/data/` |
| `plot_cooperative_groups.py` | Cooperative Groups 图表 | `sm_120/cooperative_groups/data/` |

## 使用方法

```bash
# 生成所有图表
python plot_l2_cache_analysis.py
python plot_memory_bandwidth.py
python plot_cuda_core_throughput.py
python plot_redux_sync.py
python plot_barrier_sync.py
python plot_atomic_ops.py
python plot_tensor_core.py
python plot_multi_stream.py
python plot_cuda_graph.py
python plot_fp_precision.py
python plot_cooperative_groups.py
python plot_warp_specialize.py
python plot_dp4a.py
python plot_mbarrier.py
python plot_tensor_mem.py
python plot_unified_memory.py
```

## 输出格式

- **PNG 图表**: 高分辨率图表文件
- **CSV 数据**: 原始数据便于复现和进一步分析

## 图表命名规范

- `throughput_vs_size.png` - 吞吐 vs 尺寸
- `latency_vs_size.png` - 延迟 vs 尺寸
- `efficiency_*.png` - 效率对比
- `comparison_*.png` - 多系列对比
