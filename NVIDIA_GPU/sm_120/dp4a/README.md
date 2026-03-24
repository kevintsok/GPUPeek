# DP4A 研究 Module

## 概述

DP4A (Dot Product of 4 Bytes Accumulated) 是 INT8 矩阵乘法指令。

## 独立编译和运行

```bash
cd NVIDIA_GPU/sm_120/dp4a
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_dp4a [元素数量]
```

## 文件

- `dp4a_research_kernel.cu` - DP4A 内核
- `dp4a_research_benchmarks.cu` - DP4A 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## DP4A 指令

```ptx
dp4a.s32.s8 result, a, b, acc;
```

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_dp4a
```
