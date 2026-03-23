# FP8 研究 Module

## 概述

FP8 (8-bit Floating Point) 是 Blackwell 和 Hopper 支持的低精度格式。

## 独立编译和运行

```bash
cd src/sm_120/fp8
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_fp8 [元素数量]
```

## 文件

- `fp8_research_kernel.cu` - FP8 内核
- `fp8_research_benchmarks.cu` - FP8 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## FP8 格式

| 格式 | 指数位 | 尾数位 | 描述 |
|------|--------|--------|------|
| E4M3 | 4 | 3 | 高精度 FP8 |
| E5M2 | 5 | 2 | 高动态范围 FP8 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_fp8
```
