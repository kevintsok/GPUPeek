# CUDA Core 算力研究 Module

## 概述

CUDA Core 算力测试，研究不同数据类型的计算性能和指令吞吐量。

## 独立编译和运行

```bash
cd src/sm_120/cuda_core
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_cuda_core [元素数量]
```

## 文件

- `cuda_core_kernels.cu` - CUDA Core 算力内核
- `cuda_core_benchmarks.cu` - CUDA Core 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## 测试内容

| 测试项 | 描述 |
|--------|------|
| FP32 性能 | FP32 FMA 吞吐量和延迟 |
| FP64 性能 | 双精度浮点性能 |
| FP16 性能 | 半精度浮点性能 |
| INT32 性能 | 整数运算性能 |
| 向量指令 | float2/float4/double2 向量运算 |
| 超越函数 | sin/cos/exp/log 延迟和吞吐量 |

## NCU 指标

| 指标 | 含义 |
|------|------|
| sm__pipe_fp32_cycles_active.pct | FP32 计算单元利用率 |
| sm__pipe_fp64_cycles_active.pct | FP64 计算单元利用率 |
| sm__average_execution_latency | 平均执行延迟 |

## NCU 分析

```bash
# FP32 利用率
ncu --set full --metrics sm__pipe_fp32_cycles_active.pct ./gpupeek_cuda_core

# 指令级分析
ncu --set full --kernels-by-compute ./gpupeek_cuda_core
```
