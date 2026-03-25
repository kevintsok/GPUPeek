# Parallel Scan Research

## 概述

本专题研究GPU上的并行扫描算法，包括Kogge-Stone和Hillis-Steele算法。

## 关键发现

### Scan算法性能

| 算法 | 性能 | 复杂度 |
|------|------|--------|
| Kogge-Stone | 0.375 GOPS | O(log n)延迟最优 |
| Hillis-Steele | 0.311 GOPS | O(n log n)工作最优 |

## 关键洞察

1. **Kogge-Stone更快** - 延迟最优算法
2. **SIMD shuffle高效** - Warp级别扫描
3. **适合GPU的算法** - 并行度高
4. **应用广泛** - 排序、直方图、前缀和

## 相关专题

- [WarpPrimitives](../../Synchronization/WarpPrimitives/RESEARCH.md) - SIMD原语
