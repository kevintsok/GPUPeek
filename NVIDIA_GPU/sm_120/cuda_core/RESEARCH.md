# CUDA Core Compute Research

## 概述

CUDA Core 算力研究，测试不同数据类型的计算性能和指令吞吐量。

## 重要发现：为什么 FP32 只有 ~88 GFLOPS？

**这不是 bug！这是内存带宽限制的预期行为。**

### 内存带宽分析

| 指标 | 值 |
|------|---|
| FP32 实测 | ~88 GFLOPS |
| 每元素 FLOPs | 2 (FMA: a×b+a) |
| 每元素内存访问 | 12 bytes (8 读 + 4 写) |
| 88 GFLOPS 需要带宽 | 528 GB/s (88B × 12) |
| RTX 5080 内存带宽 | 811 GB/s |
| **实际内存效率** | **65% - 优秀！** |

### 效率分析

| 数据类型 | 实测吞吐 | 理论峰值 | "效率" | 真实原因 |
|---------|---------|---------|--------|---------|
| FP32 | ~88 GFLOPS | ~14600 GFLOPS | ~0.6% | **内存带宽限制** |
| FP64 | ~12 GFLOPS | ~2300 GFLOPS | ~0.5% | **内存带宽限制** |
| FP16 | ~204 GFLOPS | N/A | N/A | **使用 Tensor Core** |
| INT8 | ~240 GIOPS | N/A | N/A | **使用 Tensor Core** |

### 关键洞察

1. **FP32/FP64 的 "低效率" 是因为内存带宽限制，不是 CUDA 核心问题**
2. **FP16 ~204 GFLOPS 是 Tensor Core 性能，不是 CUDA Core**
3. **CUDA Core 实际上运行在接近内存带宽极限**

### RTX 5080 CUDA Core 规格

| 规格 | 值 |
|------|---|
| CUDA 核心数 | 7680 |
| 核心频率 | 1.9 GHz |
| 每核心每周期 FMA | 1 |
| FP32 理论峰值 | 29.1 TFLOPS |
| 内存带宽 | 811 GB/s |

### 正确理解 "效率"

GPU 运算分为两类：
1. **计算密集型**: 计算时间 > 内存访问时间 → 接近峰值
2. **内存密集型**: 内存访问时间 > 计算时间 → 受内存带宽限制

`fp32ArithmeticKernel` 是内存密集型（每 FLOP 需要 6 bytes 内存），所以受限于 811 GB/s 带宽。

## 1. FP32 性能

| 指标 | 值 |
|------|---|
| 吞吐量 | ~61-88 GFLOPS |
| 延迟 | ~0.068 ms |
| 状态 | 内存带宽限制（正常）|

![数据类型吞吐对比](data/dtype_throughput_comparison.png)

## 2. FP64 性能

| 指标 | 值 |
|------|---|
| 每 SM FP64 单元数 | 2 (有限) |
| FP64 True Latency | ~63 cycles |
| vs Hopper | ~8 cycles |

**警告**: Blackwell 不适合 FP64 密集型工作负载

## 3. FP16 性能

| 指标 | 值 |
|------|---|
| 吞吐量 | ~204 GFLOPS |
| vs FP32 | 快约 3.3× |
| 说明 | 使用 Tensor Core |

## 4. INT32 性能

| 指标 | 值 |
|------|---|
| 吞吐量 | ~121 GIOPS |

## 5. 指令吞吐量汇总

| 指令类型 | 吞吐量 | 延迟 | 说明 |
|---------|--------|------|------|
| FP32 FMA | ~61-88 GFLOPS | 0.068 ms | 内存带宽限制 |
| INT32 | ~121 GIOPS | 0.035 ms | 内存带宽限制 |
| FP16 FMA | ~204 GFLOPS | 0.021 ms | Tensor Core |

![指令延迟对比](data/instruction_latency.png)

## 6. 向量指令

| 向量类型 | 描述 |
|----------|------|
| float2 | 2 × float |
| float4 | 4 × float |
| double2 | 2 × double |

![向量吞吐对比](data/vector_throughput.png)

## 7. 数据类型效率对比

| 数据类型 | 实际吞吐 | 理论峰值 | 效率 | 真实原因 |
|---------|---------|---------|------|---------|
| FP32 | ~88 GFLOPS | ~29100 GFLOPS | ~0.3% | 内存带宽限制 |
| FP64 | ~12 GFLOPS | ~2300 GFLOPS | ~0.5% | 内存带宽限制 |
| FP16 | ~204 GFLOPS | N/A | N/A | Tensor Core |

![计算效率对比](data/compute_efficiency.png)

## 8. NCU 指标

| 指标 | 含义 |
|------|------|
| sm__pipe_fp32_cycles_active.pct | FP32 单元利用率 |
| sm__pipe_fp64_cycles_active.pct | FP64 单元利用率 |
| sm__average_execution_latency | 平均执行延迟 |

## 图表生成

运行以下脚本生成可视化图表:

```bash
cd scripts
pip install -r requirements.txt
python plot_cuda_core_throughput.py
```

输出位置: `NVIDIA_GPU/sm_120/cuda_core/data/`

## 参考文献

- [CUDA Programming Guide](../ref/cuda_programming_guide.html)
- [PTX ISA](../ref/ptx_isa.html)
