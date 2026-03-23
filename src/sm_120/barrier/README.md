# Barrier 同步研究 Module

## 概述

Barrier 同步机制研究，测量 __syncthreads() 开销和 barrier stall 分析。

## 文件

- `barrier_kernels.cu` - Barrier 同步内核
- `barrier_benchmarks.cu` - Barrier 基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe barrier
```

## 测试内容

| 测试项 | 描述 |
|--------|------|
| __syncthreads() 开销 | 最小同步开销测量 |
| bar.sync 指令分析 | barrier stall 分析 |
| Block Size vs Barrier | 32/64/128/256/512/1024 效率 |
| 多Block同步 | grid级flag同步模式 |

## NCU 指标

| 指标 | 含义 |
|------|------|
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall 比例 |
| sm__average_active_warps_per_sm | 每SM活跃warp |

## NCU 分析

```bash
# Barrier stall 分析
ncu --set full --metrics sm__warp_issue_stalled_by_barrier.pct ./build/gpupeek.exe barrier

# Warp 活跃度
ncu --set full --metrics sm__average_active_warps_per_sm ./build/gpupeek.exe barrier
```
