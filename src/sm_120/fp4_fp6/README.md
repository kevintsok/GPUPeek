# FP4/FP6 研究 Module

## 概述

FP4/FP6 低精度研究，Blackwell 支持的极低精度格式。

## 独立编译和运行

```bash
cd src/sm_120/fp4_fp6
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_fp4_fp6 [元素数量]
```

## 文件

- `fp4_fp6_research_kernel.cu` - FP4/FP6 内核
- `fp4_fp6_research_benchmarks.cu` - FP4/FP6 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## FP4/FP6 格式

| 格式 | 位数 | 指数位 | 尾数位 |
|------|------|--------|--------|
| FP4 (e2m1) | 4 | 2 | 1 |
| FP6 (e2m3) | 6 | 2 | 3 |
| FP6 (e3m2) | 6 | 3 | 2 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_fp4_fp6
```
