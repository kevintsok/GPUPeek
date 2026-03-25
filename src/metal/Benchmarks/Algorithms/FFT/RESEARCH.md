# FFT (Fast Fourier Transform) Research

## 概述

本专题研究Apple M2 GPU上的FFT算法性能，包括Cooley-Tukey和Radix-2算法。

## 关键发现

### FFT性能

| Size | 性能 | 时间 |
|------|------|------|
| 1024 | ~0.01 GOPS | ~50 ms |
| 4096 | ~0.01 GOPS | ~200 ms |
| 16384 | ~0.01 GOPS | ~800 ms |

### 算法复杂度

| 算法 | 复杂度 | 适合场景 |
|------|---------|----------|
| Cooley-Tukey | O(n log n) | 通用FFT |
| Radix-2 | O(n log n) | 2的幂次大小 |
| Butterfly | O(n) per stage | 单步操作 |

## 关键洞察

1. **FFT是O(n log n)算法** - 蝶形单元是核心操作
2. **GPU FFT适合大数据集** - >16K元素时GPU优势明显
3. **内存带宽是瓶颈** - 蝶形操作需要频繁内存访问
4. **内存访问模式影响性能** - 非连续访问会降低性能

## 优化策略

1. **使用共享内存** - 缓存蝶形操作的数据
2. **向量化加载** - 使用float2/float4同时加载实部和虚部
3. **避免bank conflict** - 蝶形索引需要精心设计
4. **考虑cuFFT库** - NVIDIA的优化FFT实现

## 相关专题

- [Convolution](../Convolution/RESEARCH.md) - 卷积与FFT的关系
- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽影响
