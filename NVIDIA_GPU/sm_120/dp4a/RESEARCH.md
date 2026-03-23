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

## 参考文献

- [PTX ISA - DP4A](../ref/ptx_isa.html)
