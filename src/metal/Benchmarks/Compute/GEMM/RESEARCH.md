# GEMM (Matrix Multiply) Research

## 概述

本专题研究Apple M2 GPU上的矩阵乘法(GEMM)性能，包括朴素算法、共享内存分块和寄存器分块优化。

## 关键发现

### FP32 GEMM 性能

| 实现 | 256³ | 512³ | 1024³ |
|------|------|------|-------|
| Naive | 4.30 GFLOPS | - | - |
| Tiled (16x16) | 9.11 GFLOPS | 2.1x | 2.1x |
| Register-Blocked 4x4 | 13.14 GOPS | - | 21.89 GFLOPS |

### FP16 GEMM 性能

| 实现 | 256³ | 512³ | 1024³ |
|------|------|------|-------|
| FP16 Naive | 4.88 GFLOPS | - | - |
| FP16 Tiled | 14.98 GFLOPS | **5.92x** | **3.07x** |

### Tile Size 优化

| Tile Size | Performance | Notes |
|-----------|-------------|-------|
| 8x8 | 7.28 GFLOPS | 小tile，barrier开销大 |
| 16x16 | 7.46 GFLOPS | **最优选择** |
| 32x32 | - | 超过共享内存限制 |

## 关键洞察

1. **共享内存分块是GEMM最重要的优化** - 可实现2-5x加速
2. **16x16 tile是最优选择** - 平衡共享内存利用率和barrier开销
3. **FP16比FP32快** - 在tiled实现下，FP16可达14.98 GFLOPS
4. **寄存器分块最高效** - 4x4 blocking可达21.89 GFLOPS
5. **Apple M2 GEMM受限于统一内存带宽** - 计算能力远未饱和

## 优化策略

1. **使用共享内存分块** - 将A和B的tile加载到shared，减少全局内存访问
2. **选择合适的tile大小** - 16x16是Apple M2的最优选择
3. **利用float4向量化** - 寄存器分块使用float4提高带宽利用率
4. **避免bank conflict** - 使用(row*TILE+k)和(k*TILE+col)访问模式

## 相关专题

- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽基础
- [Bank Conflict](../../Memory/BankConflict/RESEARCH.md) - 共享内存bank冲突
- [Vectorization](../Vectorization/RESEARCH.md) - 向量化技术
