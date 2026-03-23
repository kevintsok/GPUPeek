# Cooperative Groups 研究 Module

## 概述

Cooperative Groups API 研究，包括线程组协作和同步。

## 独立编译和运行

```bash
cd src/sm_120/cooperative_groups
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_cooperative_groups [元素数量]
```

## 文件

- `cooperative_groups_research_kernel.cu` - CG 内核
- `cooperative_groups_research_benchmarks.cu` - CG 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_cooperative_groups
```
