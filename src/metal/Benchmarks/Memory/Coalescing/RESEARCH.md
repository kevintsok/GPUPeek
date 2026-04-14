# Memory Coalescing Research

## 概述

本专题研究内存访问模式对GPU性能的影响，特别是合并(coalesced)访问与非合并(non-coalesced)访问的差异。

## 关键发现

### 访问模式对比

| 访问模式 | 带宽 | 相对性能 |
|----------|------|----------|
| Coalesced (顺序) | 0.75 GB/s | 1.0x (基准) |
| Strided (stride=2) | 0.33 GB/s | 0.44x |
| Strided (stride=8) | 0.14 GB/s | 0.19x |
| Random | 0.05 GB/s | 0.07x |

### 性能提升

**合并访问带来的性能提升:**

| 对比 | 加速比 |
|------|--------|
| Coalesced vs Stride-2 | 2.3x |
| Coalesced vs Stride-8 | 5.3x |
| Coalesced vs Random | 15x |

## 关键洞察

1. **合并访问是GPU性能的关键** - 顺序访问可实现5.3x加速
2. **跨步访问显著降低性能** - stride越大，浪费的内存带宽越多
3. **随机访问最慢** - 无局部性，每次访问都是独立的主存访问
4. **内存访问模式比计算更重要** - 在Apple M2统一内存架构下尤其如此

## 优化策略

1. **重排数据布局** - 确保相邻线程访问相邻内存
2. **使用共享内存** - 先合并读取到shared，再分散到各个线程
3. **避免随机写入** - 使用原子操作或先收集再批量写入
4. **结构体数组 vs 数组结构体** - SoA比AoS更有利于合并访问

## 相关专题

- [Memory Bandwidth](../Bandwidth/RESEARCH.md) - 内存带宽基础
- [Bank Conflict](../BankConflict/RESEARCH.md) - 共享内存bank冲突
- [Cache Analysis](../../Analysis/Cache/RESEARCH.md) - 缓存行为分析
