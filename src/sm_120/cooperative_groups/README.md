# Cooperative Groups 研究 Module

## 概述

Cooperative Groups API 研究，包括线程组协作和同步。

## 文件

- `cooperative_groups_research_kernel.cu` - CG 内核
- `cooperative_groups_research_benchmarks.cu` - CG 基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe coop
```

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./build/gpupeek.exe coop
```
