# Memory Bandwidth Research

## 概述

本专题研究Apple M2 GPU的内存带宽特性，包括写入、读取、向量化操作等。

## 测试环境

- GPU: Apple M2 (8-core GPU, Family Apple 7)
- 内存: 统一内存 LPDDR5
- 理论带宽: 100 GB/s

## 关键发现

### 带宽 vs Buffer Size (饱和点分析)

| Buffer Size | Write BW | Read BW | Float4Read | 状态 |
|-------------|----------|---------|------------|------|
| 64KB | 0.05 GB/s | 0.05 GB/s | 0.05 GB/s | L1缓存 |
| 256KB | 0.17-0.19 GB/s | 0.15-0.17 GB/s | 0.17 GB/s | L2缓存 |
| 1MB | 0.50-0.54 GB/s | 0.39-0.43 GB/s | 0.56 GB/s | 过渡区 |
| 8MB | 1.31-1.65 GB/s | 0.79-0.87 GB/s | 2.18 GB/s | 接近饱和 |
| 64MB | 1.87-2.11 GB/s | 0.95-1.04 GB/s | 3.56 GB/s | 饱和 |
| 256MB | 1.70-2.28 GB/s | 1.00-1.05 GB/s | 3.79 GB/s | 峰值 |

### 高性能写入技术

| 技术 | 带宽 | 提升 |
|------|------|------|
| 普通写入 | 1.37-1.81 GB/s | 基准 |
| **Burst Write (16元素/线程)** | **6.17 GB/s** | **3-4x提升** |
| WriteCombine (16元素/线程) | 2.05-2.11 GB/s | 1.1-1.2x提升 |
| Combined Write+Read | 4.03-4.18 GB/s | 2x双工 |

### 向量化效果

| 数据类型 | 带宽 | 相对标量 |
|----------|------|----------|
| Float (标量) | 0.80-0.92 GB/s | 1.0x |
| Float2 | ~1.2 GB/s | ~1.3x |
| Float4 | 3.56-3.79 GB/s | **3.9x** |

## 关键洞察

1. **Burst Write是最重要的优化** - 每个线程写16个连续元素可达到6.17 GB/s，比普通写入快3-4倍
2. **Float4向量化读取** - 3.56-3.79 GB/s，比标量读取快约4倍
3. **Combined Write+Read双工** - 4.03-4.18 GB/s，同时读写可接近饱和带宽
4. 统一内存的读写带宽不对称 - 写入比读取快近2倍
5. **带宽饱和点** - 在64MB达到饱和，超过后带宽稳定

## 优化建议

1. **始终使用Burst Write** - 每线程写入16+连续元素
2. **使用Float4向量化** - 读取时使用float4类型
3. **合并读写操作** - 在同一kernel中同时读写
4. **避免小buffer** - 小于64KB的buffer利用不足缓存

## 相关专题

- [Memory Coalescing](../Coalescing/RESEARCH.md) - 内存合并访问模式
- [Bank Conflict](../BankConflict/RESEARCH.md) - 共享内存bank冲突
- [Cache Analysis](../../Analysis/Cache/RESEARCH.md) - 缓存行为分析
