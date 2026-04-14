# Roofline Model Research

## 概述

本专题研究Apple M2 GPU的Roofline模型，分析计算受限(kernel-bound)和内存受限(memory-bound)操作的性能特征。

## Roofline模型

Roofline模型将GPU性能分为两个区域：
1. **Memory Bound Region** - 算术强度(AI) < 交叉点
2. **Compute Bound Region** - 算术强度(AI) > 交叉点

### Apple M2 参数

| 参数 | 值 |
|------|-----|
| 理论内存带宽 | 100 GB/s |
| 实测带宽 | ~2 GB/s |
| 峰值计算 | ~12 GFLOPS |
| **交叉点** | **~6 FLOP/byte** |

## 关键发现

### 算术强度 vs 性能

| Kernel类型 | 算术强度 | 性能 | 受限类型 |
|-----------|----------|------|----------|
| Memory Bound | 1 FLOP/byte | ~0.8 GB/s | 内存 |
| Stencil (5pt) | 5 FLOP/byte | ~0.8 GB/s | 内存 |
| GEMM 16x16 | ~30 FLOP/byte | ~9 GFLOPS | 计算 |
| Compute Bound | 256 FLOP/byte | ~0.5 GFLOPS | 计算 |

### Roofline图表

```
Performance (GFLOPS)
     ^
12   |      * Compute Bound
     |     / 
     |    /  
 6   |   /  * GEMM
     |  /   
 2   | / * Stencil
     |/ * Memory Bound
     +---------------------> AI (FLOP/byte)
      1    6    30   256
```

## 关键洞察

1. **Apple M2大多处于内存受限** - 统一内存架构导致带宽共享
2. **交叉点约6 FLOP/byte** - 高于这个值才达到峰值计算
3. **内存带宽是主要瓶颈** - 实测仅~2 GB/s (理论100 GB/s的2%)
4. **优化内存访问至关重要** - 对大多数kernel影响更大

## 优化策略

### 内存受限Kernel
1. **提高算术强度** - 增加计算减少内存访问
2. **优化内存模式** - 合并访问、避免跨步
3. **使用共享内存** - 减少全局内存访问

### 计算受限Kernel
1. **提高并行度** - 增加occupancy
2. **使用更高效的指令** - FMA替代mul+add
3. **减少分支分歧** - 提高SIMD效率

## 相关专题

- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽
- [GEMM](../../Compute/GEMM/RESEARCH.md) - 矩阵乘法
- [Occupancy](../../Analysis/Occupancy/RESEARCH.md) - 占用率分析
