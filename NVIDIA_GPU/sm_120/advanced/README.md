# Advanced Research Module

## 概述

高级研究测试，包括 Constant Memory、Bank Conflict 分析等。

## 独立编译和运行

```bash
cd NVIDIA_GPU/sm_120/advanced
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_advanced [元素数量]
```

## 文件

- `advanced_research_kernel.cu` - 高级研究内核
- `advanced_research_benchmarks.cu` - 高级研究基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## 测试内容

| 测试项 | 描述 |
|--------|------|
| Constant Memory | 常量内存访问性能 |
| Bank Conflict 分析 | 共享内存银行冲突 |
| Atomic Operations | 原子操作性能 |
| Memory Fence | 内存栅栏影响 |

## NCU 分析

```bash
# 共享内存银行冲突
ncu --set full --metrics sm__shared_bank_conflict_throughput ./gpupeek_advanced
```
