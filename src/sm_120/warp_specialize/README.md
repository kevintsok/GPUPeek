# Warp Specialization 研究 Module

## 概述

Warp Specialization 与 Producer-Consumer 模式研究。

## 文件

- `warp_specialize_kernels.cu` - Warp 特化内核
- `warp_specialize_benchmarks.cu` - Warp 特化基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe warp
```

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
ncu --set full --metrics sm__warp_divergence_efficiency ./build/gpupeek.exe warp
```
