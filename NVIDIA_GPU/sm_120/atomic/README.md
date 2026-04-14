# Atomic 原子操作研究 Module

## 概述

原子操作深入研究，测试不同粒度和类型的原子操作性能。

## 独立编译和运行

```bash
cd NVIDIA_GPU/sm_120/atomic
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
./gpupeek_atomic [元素数量]
```

## 文件

- `atomic_kernels.cu` - 原子操作内核
- `atomic_benchmarks.cu` - 原子操作基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

## 测试内容

| 测试项 | 描述 |
|--------|------|
| Warp级原子操作 | 同warp内归约后单次原子 |
| Block级原子操作 | 同block归约后单次原子 |
| Grid级原子操作 | 所有线程直接原子(高竞争) |
| 原子操作对比 | atomicAdd vs CAS vs Min/Max |

## 原子操作类型

| 类型 | 描述 |
|------|------|
| atomicAdd | 原子加法 |
| atomicMin/Max | 原子最小/最大 |
| atomicCAS | Compare-and-Swap |
| atomicAnd/Or/Xor | 位操作 |

## NCU 指标

| 指标 | 含义 |
|------|------|
| sm__average_active_warps_per_sm | 每SM活跃warp数 |
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall |

## NCU 分析

```bash
# 原子操作冲突分析
ncu --set full --metrics sm__average_active_warps_per_sm ./gpupeek_atomic
```
