# Graph Algorithms Research

## 概述

本专题研究GPU上的图算法性能，特别是BFS（广度优先搜索）等常见图遍历操作。

## 关键发现

### BFS性能

| 规模 | 性能 | 说明 |
|------|------|------|
| 65K顶点, 256K边 | ~0.040 GOPS | 受限于随机访问 |

## 关键洞察

1. **图算法受限于随机内存访问** - 不规则访问模式
2. **Frontier-based方法有效** - 管理并行度
3. **原子操作用于visited标记** - 防止重复访问
4. **适合GPU的场景** - 大规模并行、规则图结构

## 相关专题

- [Memory Coalescing](../../Memory/Coalescing/RESEARCH.md) - 内存合并
- [Atomics](../../Synchronization/Atomics/RESEARCH.md) - 原子操作
