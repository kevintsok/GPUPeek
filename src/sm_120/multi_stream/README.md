# Multi-Stream 研究 Module

## 概述

Multi-Stream 并发执行研究，流依赖、重叠执行、流优先级。

## 独立编译和运行

```bash
cd src/sm_120/multi_stream
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_multi_stream [元素数量]
```

## 文件

- `multi_stream_research_kernel.cu` - 多流内核
- `multi_stream_research_benchmarks.cu` - 多流基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## 测试内容

| 测试项 | 描述 |
|--------|------|
| 流依赖 | 依赖流的同步 |
| 重叠执行 | 计算与拷贝重叠 |
| 流优先级 | 优先级调度 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_multi_stream
```
