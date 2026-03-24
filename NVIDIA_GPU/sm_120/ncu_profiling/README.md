# NCU Profiling 研究 Module

## 概述

NCU 性能分析工具研究，指标收集和内核分析。

## 独立编译和运行

```bash
cd NVIDIA_GPU/sm_120/ncu_profiling
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_ncu_profiling [元素数量]
```

## 文件

- `ncu_profiling_kernel.cu` - Profiling 内核
- `ncu_profiling_benchmarks.cu` - Profiling 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

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
ncu --set full ./gpupeek_ncu_profiling

# 指定指标
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek_ncu_profiling
```
