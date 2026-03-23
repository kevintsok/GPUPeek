# WGMMA (Warpgroup MMA) Research

## 警告

**WGMMA 仅在 Hopper (sm_90) 上支持。Blackwell (sm_120) 不支持 WGMMA。**

此模块主要用于研究和学习目的。

## 1. WGMMA 概述

WGMMA 是 Hopper 架构引入的异步 Warpgroup MMA 指令。

## 2. WGMMA Shape (Hopper)

| Shape | FP16 | BF16 | TF32 | FP64 | INT8 |
|-------|------|------|------|------|------|
| m64nNk16 | Yes | Yes | Yes | Yes | Yes |
| m64nNk8 | Yes | Yes | Yes | - | Yes |
| m64nNk32 | Yes | Yes | - | - | Yes |
| m64nNk256 | Yes | - | - | - | - |

**N = K / 16**

## 3. 与 WMMA 的区别

| 特性 | WMMA | WGMMA |
|------|------|-------|
| 架构 | 通用 | Hopper+ |
| 同步 | sync | async |
| Shape | m16n16k16 | m64nNk16 |
| Warp Group | 1 warp | 4 warps |

## 4. PTX 指令

```ptx
wgmma.mma_async.sync.aligned.m64nNk16 ...;
wgmma.wait_group 0;
```

## 参考文献

- [CUDA Programming Guide - WGMMA](../ref/cuda_programming_guide.html)
- [PTX ISA - WGMMA](../ref/ptx_isa.html)
