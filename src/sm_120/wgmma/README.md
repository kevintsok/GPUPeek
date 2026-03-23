# WGMMA 研究 Module

## 概述

WGMMA (Warpgroup MMA) 是 Hopper 架构的异步 MMA 指令。**注意: Blackwell 不支持 WGMMA。**

## 警告

**WGMMA 仅在 Hopper (sm_90) 上支持。Blackwell (sm_120) 不支持 WGMMA。**

此模块主要用于研究和学习，实际运行会在不支持的 GPU 上失败。

## 独立编译和运行

```bash
cd src/sm_120/wgmma
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_wgmma [元素数量]
```

## 文件

- `wgmma_research_kernel.cu` - WGMMA 内核
- `wgmma_research_benchmarks.cu` - WGMMA 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## WGMMA Shape (Hopper)

| Shape | FP16 | BF16 | TF32 | FP64 | INT8 |
|-------|------|------|------|------|------|
| m64nNk16 | Yes | Yes | Yes | Yes | Yes |
| m64nNk8 | Yes | Yes | Yes | - | Yes |
| m64nNk32 | Yes | Yes | - | - | Yes |
| m64nNk256 | Yes | - | - | - | - |

**N = K / 16**

## NCU 分析

```bash
# 指令分析
ncu --set full --kernels-by-compute ./gpupeek_wgmma
```
