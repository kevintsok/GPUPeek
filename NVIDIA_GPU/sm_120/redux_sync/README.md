# Redux.sync 研究 Module

## 概述

Redux.sync 指令研究，单指令完成 warp 级归约。

## 独立编译和运行

```bash
cd NVIDIA_GPU/sm_120/redux_sync
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_redux_sync [元素数量]
```

## 文件

- `redux_sync_research_kernel.cu` - Redux 内核
- `redux_sync_research_benchmarks.cu` - Redux 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## Redux.sync 支持的操作

| 操作 | 描述 |
|------|------|
| ADD | 加法归约 |
| MIN | 最小值归约 |
| MAX | 最大值归约 |
| AND | 按位与归约 |
| OR | 按位或归约 |
| XOR | 按位异或归约 |

## NCU 分析

```bash
ncu --set full --kernels-by-compute ./gpupeek_redux_sync
```
