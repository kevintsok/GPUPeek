# Tensor Memory 研究 Module

## 概述

张量内存操作研究，包括 LDMATRIX、STMATRIX、cp.async 等指令。

## 文件

- `tensor_mem_research_kernel.cu` - 张量内存内核
- `tensor_mem_research_benchmarks.cu` - 张量内存基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe tensor_mem
```

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
ncu --set full --kernels-by-compute ./build/gpupeek.exe tensor_mem
```
