# CUDA Core Compute Research

## 概述

CUDA Core 算力研究，测试不同数据类型的计算性能和指令吞吐量。

## 1. FP32 性能

| 指标 | 值 |
|------|---|
| 吞吐量 | ~61-88 GFLOPS |
| 延迟 | ~0.068 ms |

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

## 4. INT32 性能

| 指标 | 值 |
|------|---|
| 吞吐量 | ~121 GIOPS |

## 5. 指令吞吐量汇总

| 指令类型 | 吞吐量 | 延迟 |
|---------|--------|------|
| FP32 FMA | ~61-88 GFLOPS | 0.068 ms |
| INT32 | ~121 GIOPS | 0.035 ms |
| FP16 FMA | ~204 GFLOPS | 0.021 ms |

## 6. 向量指令

| 向量类型 | 描述 |
|----------|------|
| float2 | 2 × float |
| float4 | 4 × float |
| double2 | 2 × double |

## 7. NCU 指标

| 指标 | 含义 |
|------|------|
| sm__pipe_fp32_cycles_active.pct | FP32 单元利用率 |
| sm__pipe_fp64_cycles_active.pct | FP64 单元利用率 |
| sm__average_execution_latency | 平均执行延迟 |

## 参考文献

- [CUDA Programming Guide](../ref/cuda_programming_guide.html)
- [PTX ISA](../ref/ptx_isa.html)
