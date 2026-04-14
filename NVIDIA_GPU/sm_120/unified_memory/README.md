# Unified Memory 研究 Module

## 概述

Unified Memory (统一内存) 研究，Page fault、prefetch、page migration。

## 独立编译和运行

```bash
cd NVIDIA_GPU/sm_120/unified_memory
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_unified_memory [元素数量]
```

## 文件

- `unified_memory_research_kernel.cu` - 统一内存内核
- `unified_memory_research_benchmarks.cu` - 统一内存基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## 测试内容

| 测试项 | 描述 |
|--------|------|
| Page Fault | 页面错误分析 |
| Prefetch | 预取策略 |
| Page Migration | 页面迁移 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_unified_memory
```
