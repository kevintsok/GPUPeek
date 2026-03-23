# Redux.sync Research

## 概述

Redux.sync 在单条指令内完成 warp 级归约。

## 1. vs Shuffle 循环

| 方法 | 指令数 | 延迟 |
|------|--------|------|
| Shuffle 循环 | log2(32) = 5 次 | 较高 |
| **Redux.sync** | **1 条指令** | **最低** |

## 2. 支持的操作

| 操作 | 描述 |
|------|------|
| ADD | 加法归约 |
| MIN | 最小值归约 |
| MAX | 最大值归约 |
| AND | 按位与归约 |
| OR | 按位或归约 |
| XOR | 按位异或归约 |

## 3. PTX 指令

```ptx
redux.sync.cta.add.s32 %r0, %r1;
redux.sync.cta.min.s32 %r0, %r1;
```

## 4. CUDA C++ API

```cuda
int result = __redux_sync_sync(operand, REDUCE_OP_ADD);
```

## 参考文献

- [PTX ISA - Redux](../ref/ptx_isa.html)
