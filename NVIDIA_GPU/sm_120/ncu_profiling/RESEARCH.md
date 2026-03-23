# NCU Profiling Research

## 概述

NCU (NVIDIA Nsight Compute) 是 GPU 性能分析工具。

## 1. 基本使用

```bash
ncu --set full ./build/gpupeek.exe <test>
```

## 2. 关键指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU 利用率 | 越高越好 |
| sm__pipe_fp32_cycles_active.pct | FP32 利用率 | 越高越好 |
| sm__pipe_tensor_cycles_active.pct | Tensor Core 利用率 | 越高越好 |
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall | 越低越好 |
| dram__bytes.sum | 内存带宽 | 参考 |
| lts__tcs_hit_rate.pct | L2 缓存命中率 | 越高越好 |

## 3. 常用命令

```bash
# 完整指标集
ncu --set full ./build/gpupeek.exe <test>

# 指定指标
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./build/gpupeek.exe <test>

# Kernel 级分析
ncu --set full --kernels-by-compute ./build/gpupeek.exe <test>
```

## 4. 指标分类

### 计算指标
- sm__pipe_fp32_cycles_active.pct
- sm__pipe_fp64_cycles_active.pct
- sm__pipe_tensor_cycles_active.pct

### 内存指标
- dram__bytes.sum
- lts__tcs_hit_rate.pct
- sm__throughput.avg.pct_of_peak_sustainedTesla

### 同步指标
- sm__warp_issue_stalled_by_barrier.pct
- sm__average_active_warps_per_sm

## 参考文献

- [NCU Documentation](../ref/ncu_guide.html)
