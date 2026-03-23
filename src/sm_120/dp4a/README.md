# DP4A 研究 Module

## 概述

DP4A (Dot Product of 4 Bytes Accumulated) 是 INT8 矩阵乘法指令。

## 文件

- `dp4a_research_kernel.cu` - DP4A 内核
- `dp4a_research_benchmarks.cu` - DP4A 基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe dp4a
```

## DP4A 指令

```ptx
dp4a.s32.s8 result, a, b, acc;
```

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./build/gpupeek.exe dp4a
```
