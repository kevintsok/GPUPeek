# Multi-Stream 研究 Module

## 概述

Multi-Stream 并发执行研究，流依赖、重叠执行、流优先级。

## 文件

- `multi_stream_research_kernel.cu` - 多流内核
- `multi_stream_research_benchmarks.cu` - 多流基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe multi_stream
```

## 测试内容

| 测试项 | 描述 |
|--------|------|
| 流依赖 | 依赖流的同步 |
| 重叠执行 | 计算与拷贝重叠 |
| 流优先级 | 优先级调度 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./build/gpupeek.exe multi_stream
```
