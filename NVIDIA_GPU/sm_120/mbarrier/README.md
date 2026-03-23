# MBarrier 研究 Module

## 概述

MBarrier (Multi-Block Barrier) 研究，Hopper 架构的增强同步机制。

## 独立编译和运行

```bash
cd src/sm_120/mbarrier
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_mbarrier [元素数量]
```

## 文件

- `mbarrier_research_kernel.cu` - MBarrier 内核
- `mbarrier_research_benchmarks.cu` - MBarrier 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_mbarrier
```
