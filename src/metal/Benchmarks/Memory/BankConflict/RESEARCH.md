# Bank Conflict Research

## 概述

本专题研究GPU共享内存(shared memory)的bank conflict问题及其对性能的影响。

## 背景

Apple M2 GPU的共享内存被组织成多个bank，每个bank可以同时被不同线程访问。但当多个线程访问同一个bank的不同地址时，就会发生bank conflict，导致串行化访问。

## 关键发现

### 访问模式对比

| 访问模式 | 性能 | 相对基准 |
|----------|------|----------|
| Sequential | 0.53 GOPS | 1.0x (基准) |
| Strided (conflict) | 0.28 GOPS | 0.53x |
| Broadcast | ~0.50 GOPS | ~0.95x |

### Bank Conflict成本

**Strided访问导致的性能损失:**
- 顺序访问: 0.53 GOPS
- 跨步访问: 0.28 GOPS
- **性能下降: 1.9x**

## 关键洞察

1. **Bank conflict产生1.8-1.9x性能损失** - 跨步访问让多个线程访问同一bank
2. **顺序访问无conflict** - 每个线程访问不同的bank
3. **Broadcast很高效** - 当数据被所有线程需要时，一个线程读，其他线程直接使用
4. **Apple GPU共享内存有32个bank**

## 优化策略

1. **避免跨步访问** - 使用 (thread_id + offset) % num_threads 而非 thread_id * stride
2. **使用padding** - 在共享内存数组中添加空隙避免冲突
3. **利用broadcast** - 热点数据使用single-thread读取 + barrier + broadcast
4. **共享内存分块** - 将大数组分成小块分别处理

## 相关专题

- [Memory Bandwidth](../Bandwidth/RESEARCH.md) - 内存带宽基础
- [Memory Coalescing](../Coalescing/RESEARCH.md) - 全局内存合并访问
- [Threadgroup Memory](../Threadgroup/RESEARCH.md) - 共享内存使用模式
