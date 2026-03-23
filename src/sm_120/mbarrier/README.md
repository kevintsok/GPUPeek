# MBarrier 研究 Module

## 概述

MBarrier (Multi-Block Barrier) 研究，Hopper 架构的增强同步机制。

## 文件

- `mbarrier_research_kernel.cu` - MBarrier 内核
- `mbarrier_research_benchmarks.cu` - MBarrier 基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe mbarrier
```

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./build/gpupeek.exe mbarrier
```
