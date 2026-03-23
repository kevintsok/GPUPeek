# NCU Profiling 研究 Module

## 概述

NCU 性能分析工具研究，指标收集和内核分析。

## 文件

- `ncu_profiling_kernel.cu` - Profiling 内核
- `ncu_profiling_benchmarks.cu` - Profiling 基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe ncu
```

## NCU 关键指标

| 指标 | 含义 |
|------|------|
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU 利用率 |
| sm__pipe_fp32_cycles_active.pct | FP32 利用率 |
| sm__pipe_tensor_cycles_active.pct | Tensor Core 利用率 |
| dram__bytes.sum | 内存带宽 |
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall |

## NCU 分析

```bash
# 完整指标集
ncu --set full ./build/gpupeek.exe ncu

# 指定指标
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./build/gpupeek.exe ncu
```
