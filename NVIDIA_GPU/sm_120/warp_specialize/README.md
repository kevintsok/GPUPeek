# Warp Specialization 研究 Module

## 概述

Warp Specialization 与 Producer-Consumer 模式研究。

## 独立编译和运行

```bash
cd src/sm_120/warp_specialize
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_warp_specialize [元素数量]
```

## 文件

- `warp_specialize_kernels.cu` - Warp 特化内核
- `warp_specialize_benchmarks.cu` - Warp 特化基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## 测试内容

| 测试项 | 描述 |
|--------|------|
| Warp Specialization 基础 | 2-warp producer/consumer |
| TMA + Barrier 协同 | Async Copy + Barrier 同步 |
| 多级 Pipeline | 3-stage 流水线 (load/compute/store) |
| Block Specialization | 半block=producer，另半=consumer |
| Warp级同步原语 | Mutex/Barrier/Reduction/Scan |

## 关键 API

| API | 描述 |
|-----|------|
| __shfl_down_sync | Warp 内数据交换 |
| cp.async | 异步拷贝 |
| bar.sync | Barrier 同步 |

## NCU 分析

```bash
# Warp 分歧效率
ncu --set full --metrics sm__warp_divergence_efficiency ./gpupeek_warp_specialize
```
