# FP4/FP6 研究 Module

## 概述

FP4/FP6 低精度研究，Blackwell 支持的极低精度格式。

## 文件

- `fp4_fp6_research_kernel.cu` - FP4/FP6 内核
- `fp4_fp6_research_benchmarks.cu` - FP4/FP6 基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe fp4
```

## FP4/FP6 格式

| 格式 | 位数 | 指数位 | 尾数位 |
|------|------|--------|--------|
| FP4 (e2m1) | 4 | 2 | 1 |
| FP6 (e2m3) | 6 | 2 | 3 |
| FP6 (e3m2) | 6 | 3 | 2 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./build/gpupeek.exe fp4
```
