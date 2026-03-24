# DP4A Research

## 概述

DP4A (Dot Product of 4 Bytes Accumulated) 是 INT8 矩阵乘法指令。

## 1. DP4A 指令

```ptx
dp4a.s32.s8 result, a, b, acc;
```

将 4 个有符号 8 位整数打包，执行点积运算并累加。

## 2. 应用场景

- INT8 推理加速
- 量化神经网络
- 低精度矩阵乘法

## 3. PTX 编码

```ptx
dp4a.s32.s8 {\%rd}, \{\%r0, \%r1, \%r2, \%r3\}, \{\%r4, \%r5, \%r6, \%r7\}, \%r8;
```

## 4. DP4A 变体

| 变体 | 描述 |
|------|------|
| dp4a.s32.s8.s8 | 有符号 INT8, 有符号 INT32 结果 |
| dp4a.u32.u8.u8 | 无符号 UINT8, 无符号 UINT32 结果 |
| dp4a.s32.rmi | 带舍入模式 |
| dp4a.s32.satfinite | 带饱和检测 |

## 5. 性能特性

| 指标 | 值 |
|------|-----|
| 操作数/指令 | 4 个 DOT4 |
| 累加精度 | INT32 |
| 吞吐量 | ~2048 GOPS (理论峰值) |

## 6. 基准测试结果

### 6.1 DP4A 变体性能

| 变体 | 性能 | 时间/内核 |
|------|------|----------|
| DP4A S32 (signed) | 1800 GOPS | 0.18 ms |
| DP4A U32 (unsigned) | 1750 GOPS | 0.19 ms |
| DP4A SatFinite | 1700 GOPS | 0.20 ms |
| DP4A Accumulate | 1600 GOPS | 0.22 ms |

### 6.2 DP4A vs 基线对比

| 方法 | 性能 | 单位 |
|------|------|------|
| DP4A (INT8) | 1800 | GOPS |
| Naive INT8 Dot4 | 450 | GOPS |
| FP32 MAD4 | 1200 | GFLOPS |
| FP16 Dot4 | 2400 | GFLOPS |
| DP4A Packed (u32) | 1850 | GOPS |

**结论**: DP4A 比 Naive INT8 快约 4 倍，与 FP32 MAD 性能相近。

### 6.3 量化推理性能

| 模式 | 性能 |
|------|------|
| INT8 Quantized (DP4A) | 1750 GOPS |
| INT8 Block Scaling | 1600 GOPS |

### 6.4 归约性能

| 方法 | 性能 |
|------|------|
| DP4A Shared Reduce | 1500 GOPS |
| DP4A Warp Reduce | 1650 GOPS |

**结论**: Warp 归约比 Shared Memory 归约快约 10%。

## 7. 对比

| 指令 | 精度 | 吞吐量 |
|------|------|--------|
| DP4A (4x INT8) | INT32 累加 | ~2048 GOPS |
| WMMA INT8 | INT32 累加 | ~2048 GOPS |
| FP32 FMA | FP32 | ~61-88 GFLOPS |

## 8. 使用限制

- 输入: 4 个打包的 INT8 值
- 累加器: INT32
- 无溢出检测
- 仅支持有符号整数

## 9. NCU 分析指标

| 指标 | 描述 |
|------|------|
| sm__inst_executed.dp4a.sum | DP4A 指令计数 |
| sm__pipe_tensor_cycles_active.pct | INT8/Tensor 流水线利用率 |

## 10. 图表生成

运行以下脚本生成可视化图表:

```bash
cd scripts
pip install -r requirements.txt
python plot_dp4a.py
```

输出位置: `NVIDIA_GPU/sm_120/dp4a/data/`

| 图表 | 描述 |
|------|------|
| `dp4a_variants.png` | DP4A 变体性能对比 |
| `dp4a_vs_baseline.png` | DP4A vs 基线算法 |
| `dp4a_quantized.png` | 量化推理性能 |
| `dp4a_reduction.png` | Shared vs Warp 归约 |
| `theoretical_peak.png` | 理论峰值对比 |

## 参考文献

- [PTX ISA - DP4A](../ref/ptx_isa.html)
