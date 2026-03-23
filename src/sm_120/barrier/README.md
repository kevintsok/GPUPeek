# Barrier 同步研究 Module

## 概述

Barrier 同步机制研究，测量 __syncthreads() 开销和 barrier stall 分析。

## 独立编译和运行

```bash
cd src/sm_120/barrier
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_barrier [元素数量]
```

## 文件

- `barrier_kernels.cu` - Barrier 同步内核
- `barrier_benchmarks.cu` - Barrier 基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

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
ncu --set full --metrics sm__warp_issue_stalled_by_barrier.pct ./gpupeek_barrier

# Warp 活跃度
ncu --set full --metrics sm__average_active_warps_per_sm ./gpupeek_barrier
```
