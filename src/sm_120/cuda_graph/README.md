# CUDA Graph 研究 Module

## 概述

CUDA Graph API 研究，图捕获、实例化和启动优化。

## 文件

- `cuda_graph_research_kernel.cu` - CUDA Graph 内核
- `cuda_graph_research_benchmarks.cu` - CUDA Graph 基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe graph
```

## 测试内容

| 测试项 | 描述 |
|--------|------|
| Graph Capture | 内核图捕获 |
| Graph Instantiate | 图实例化 |
| Graph Launch | 图启动性能 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./build/gpupeek.exe graph
```
