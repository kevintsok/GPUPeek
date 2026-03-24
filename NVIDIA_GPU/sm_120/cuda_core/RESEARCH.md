# CUDA Core Compute Research

## 概述

CUDA Core 算力研究，测试不同数据类型的计算性能和指令吞吐量。

## 1. FP32 性能

| 指标 | 值 |
|------|---|
| 吞吐量 | ~61-88 GFLOPS |
| 延迟 | ~0.068 ms |

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

![指令延迟对比](data/instruction_latency.png)

## 6. 向量指令

| 向量类型 | 描述 |
|----------|------|
| float2 | 2 × float |
| float4 | 4 × float |
| double2 | 2 × double |

![向量吞吐对比](data/vector_throughput.png)

## 7. 数据类型效率对比

| 数据类型 | 实际吞吐 | 理论峰值 | 效率 |
|---------|---------|---------|------|
| FP32 | ~88 GFLOPS | ~1024 GFLOPS | ~8.6% |
| FP64 | ~12 GFLOPS | ~32 GFLOPS | ~37.5% |
| FP16 | ~204 GFLOPS | ~2048 GFLOPS | ~10% |
| INT8 | ~240 GIOPS | ~2048 GIOPS | ~12% |

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
