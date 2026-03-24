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

## 4. 性能特性

| 指标 | 值 |
|------|-----|
| 操作数/指令 | 4 个 DOT4 |
| 累加精度 | INT32 |
| 吞吐量 | ~2048 GOPS (理论峰值) |

## 5. 对比

| 指令 | 精度 | 吞吐量 |
|------|------|--------|
| DP4A (4x INT8) | INT32 累加 | ~2048 GOPS |
| WMMA INT8 | INT32 累加 | ~2048 GOPS |
| FP32 FMA | FP32 | ~61-88 GFLOPS |

## 6. 使用限制

- 输入: 4 个打包的 INT8 值
- 累加器: INT32
- 无溢出检测
- 仅支持有符号整数

## 参考文献

- [PTX ISA - DP4A](../ref/ptx_isa.html)
