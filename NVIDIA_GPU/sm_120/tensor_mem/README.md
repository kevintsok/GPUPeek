# Tensor Memory 研究 Module

## 概述

张量内存操作研究，包括 LDMATRIX、STMATRIX、cp.async 等指令。

## 独立编译和运行

```bash
cd src/sm_120/tensor_mem
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_tensor_mem [元素数量]
```

## 文件

- `tensor_mem_research_kernel.cu` - 张量内存内核
- `tensor_mem_research_benchmarks.cu` - 张量内存基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## 测试内容

| 测试项 | 描述 |
|--------|------|
| LDMATRIX | 矩阵加载指令 |
| STMATRIX | 矩阵存储指令 |
| cp.async | 异步拷贝指令 |
| cp.async.bulk | 批量异步拷贝 |

## LDMATRIX 变体

| 指令 | 描述 |
|------|------|
| ldmatrix.sync.aligned.m8n8.x1 | 8x8 tile, 1 矩阵 |
| ldmatrix.sync.aligned.m8n8.x2 | 8x8 tile, 2 矩阵 |
| ldmatrix.sync.aligned.m8n8.x4 | 8x8 tile, 4 矩阵 |

## NCU 分析

```bash
# SASS 指令分析
ncu --set full --kernels-by-compute ./gpupeek_tensor_mem
```
