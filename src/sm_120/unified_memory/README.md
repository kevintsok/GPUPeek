# Unified Memory 研究 Module

## 概述

Unified Memory (统一内存) 研究，Page fault、prefetch、page migration。

## 文件

- `unified_memory_research_kernel.cu` - 统一内存内核
- `unified_memory_research_benchmarks.cu` - 统一内存基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe unified
```

## 测试内容

| 测试项 | 描述 |
|--------|------|
| Page Fault | 页面错误分析 |
| Prefetch | 预取策略 |
| Page Migration | 页面迁移 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./build/gpupeek.exe unified
```
