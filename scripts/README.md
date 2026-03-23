# GPUPeek Chart Generation Scripts

## 概述

本目录包含用于生成 GPU 性能图表的 Python 脚本。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 脚本列表

| 脚本 | 用途 | 输出 |
|------|------|------|
| `plot_l2_cache_analysis.py` | L2 缓存分析图表 | `sm_120/deep/data/*.png` |
| `plot_memory_bandwidth.py` | 内存带宽分析图表 | `sm_120/memory/data/*.png` |

## 使用方法

```bash
# 生成 L2 缓存分析图表
python plot_l2_cache_analysis.py

# 生成内存带宽分析图表
python plot_memory_bandwidth.py
```

## 输出格式

- **PNG 图表**: 高分辨率图表文件
- **CSV 数据**: 原始数据便于复现和进一步分析

## 图表命名规范

- `throughput_vs_size.png` - 吞吐 vs 尺寸
- `latency_vs_size.png` - 延迟 vs 尺寸
- `efficiency_*.png` - 效率对比
- `comparison_*.png` - 多系列对比
