# SIMD Group (Warp) Primitives Research

## 概述

本专题研究Apple M2 GPU上SIMD组的 primitives 操作，包括vote、shuffle、broadcast和prefix sum等 warp-level 通信机制。

## 背景

Apple GPU使用32线程的SIMD组（类似于NVIDIA的warp），支持多种硬件原生操作：
- SIMD宽度: 32线程
- 线程组大小: 最大1024线程
- SIMD组内所有线程同时执行相同指令

## 关键发现

### SIMD 操作性能

| 操作 | 性能 | 说明 |
|------|------|------|
| SIMD Vote Any | ~0.02 GOPS | 全线程投票任一 |
| SIMD Vote All | ~0.02 GOPS | 全线程投票所有 |
| SIMD Shuffle | ~0.02 GOPS | Lane数据交换 |
| SIMD Shuffle XOR | ~0.02 GOPS | 蝶形模式 |
| SIMD Broadcast | ~0.02 GOPS | 单值广播 |
| SIMD Prefix Sum | ~0.02 GOPS | O(log n)扫描 |

### Shuffle 模式

| Shuffle类型 | 用途 | 模式 |
|------------|------|------|
| shuffle_down | 从大索引获取数据 | i→i+offset |
| shuffle_up | 从小索引获取数据 | i→i-offset |
| shuffle_xor | 蝶形交换 | i→i^mask |
| broadcast | 广播单值 | lane→all |

### XOR Shuffle优化

XOR shuffle是 reductions 的最佳选择：
- Mask 16, 8, 4, 2, 1 的递进模式
- 每个step只交换配对线程
- 总共5步完成32线程 reduction

## 关键洞察

1. **SIMD primitives是硬件原生支持** - 极快的 warp-level 操作
2. **Vote操作几乎无开销** - 用于条件分支同步
3. **Shuffle实现高效数据交换** - 无需共享内存
4. **XOR shuffle是reduction最优** - 蝶形模式减少通信
5. **Broadcast用于热点数据共享** - 一个线程读，所有线程用

## 优化策略

1. **使用simd_shuffle_xor做reduction** - 比共享内存快
2. **避免顺序通信模式** - shuffle_down 31次很慢
3. **Broadcast热点数据** - 一次读，多次用
4. **利用prefix sum做扫描** - O(n) → O(n/log n)

## 相关专题

- [Barriers](Barriers/RESEARCH.md) - 同步屏障
- [Atomics](Atomics/RESEARCH.md) - 原子操作
- [Reduction](../Algorithms/Reduction/RESEARCH.md) - 并行归约算法
