# Deep Research Module

## 概述

深度研究测试，包括 L2 缓存、TMA、prefetch 等高级内存操作。

## 文件

- `deep_research_kernel.cu` - 深度研究内核
- `deep_research_benchmarks.cu` - 深度研究基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe deep
```

## 测试内容

| 测试项 | 描述 |
|--------|------|
| L2 工作集分析 | 不同数据大小的 L2 性能 |
| L2 Thrashing | 跨距访问对 L2 的影响 |
| TMA 2D 拷贝 | 张量内存访问器 2D 操作 |
| Prefetch | 软件预取性能 |

## NCU 指标

| 指标 | 含义 |
|------|------|
| lts__tcs_hit_rate.pct | L2 缓存命中率 |
| dram__bytes.sum | 内存带宽 |

## NCU 分析

```bash
# L2 命中率分析
ncu --set full --metrics lts__tcs_hit_rate.pct ./build/gpupeek.exe deep

# GPU 利用率
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./build/gpupeek.exe deep
```
