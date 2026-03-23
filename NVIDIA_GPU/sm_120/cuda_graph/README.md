# CUDA Graph 研究 Module

## 概述

CUDA Graph API 研究，图捕获、实例化和启动优化。

## 独立编译和运行

```bash
cd src/sm_120/cuda_graph
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_cuda_graph [元素数量]
```

## 文件

- `cuda_graph_research_kernel.cu` - CUDA Graph 内核
- `cuda_graph_research_benchmarks.cu` - CUDA Graph 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## 测试内容

| 测试项 | 描述 |
|--------|------|
| Graph Capture | 内核图捕获 |
| Graph Instantiate | 图实例化 |
| Graph Launch | 图启动性能 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_cuda_graph
```
