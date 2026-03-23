# FP8 研究 Module

## 概述

FP8 (8-bit Floating Point) 研究，支持 E4M3 和 E5M2 格式。

## 文件

- `fp8_research_kernel.cu` - FP8 内核
- `fp8_research_benchmarks.cu` - FP8 基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe fp8
```

## FP8 格式

| 格式 | 指数位 | 尾数位 | 描述 |
|------|--------|--------|------|
| E4M3 | 4 | 3 | 高精度 FP8 |
| E5M2 | 5 | 2 | 高动态范围 FP8 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./build/gpupeek.exe fp8
```
