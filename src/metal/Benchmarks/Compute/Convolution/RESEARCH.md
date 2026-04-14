# Convolution Research

## 概述

本专题研究GPU上的卷积运算性能，特别是3x3卷积在CNN中的应用。

## 关键发现

### 3x3 Convolution性能

| 方法 | 性能 | 说明 |
|------|------|------|
| Naive | 0.47 GOPS | 9次内存读取/像素 |

## 关键洞察

1. **卷积是内存密集型** - 每像素9次读取
2. **共享内存优化有效** - 减少全局内存访问
3. **3x3是CNN常用** - 几乎所有现代CNN都使用

## 相关专题

- [GEMM](../GEMM/RESEARCH.md) - 矩阵乘法
- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽
