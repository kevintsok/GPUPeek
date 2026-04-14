# Atomic Operations Research

## 概述

本专题研究Apple M2 GPU上的原子操作性能，包括atomic_add、atomic_min、atomic_max和compare-and-swap (CAS)。

## 关键发现

### 原子操作性能

| 操作 | 性能 | 备注 |
|------|------|------|
| Non-Atomic Increment | ~0.57 GOPS | 基准 |
| Atomic Fetch Add (无争用) | ~0.04 GOPS | 每个线程不同地址 |
| Atomic Fetch Add (高争用) | ~0.016 GOPS | 所有线程同一地址 |
| Atomic Fetch Min | ~0.036 GOPS | 原子最小值 |
| Atomic Fetch Max | ~0.038 GOPS | 原子最大值 |
| Atomic CAS | ~0.012 GOPS | 最慢，需要重试 |

### 争用(Contention)影响

**高争用下的性能下降:**
- 无争用: 0.04 GOPS
- 高争用: 0.016 GOPS
- **下降: 2.5x**

## 关键洞察

1. **原子操作有显著开销** - 比非原子操作慢10-50倍
2. **争用导致性能下降** - 多个线程竞争同一地址会串行化执行
3. **CAS最慢** - 因为需要重试机制，多次内存操作
4. **Fetch Add/Min/Max类似** - 都需要读-修改-写回

## 优化策略

1. **减少争用** - 让每个线程操作不同的原子地址
2. **使用本地聚合** - 先在threadgroup内聚合，再做原子操作
3. **避免CAS** - 尽可能使用Fetch Add等简单操作
4. **使用共享内存** - 先收集再一次性原子更新

## 相关专题

- [Barriers](../Barriers/RESEARCH.md) - 线程组同步
- [Memory Coalescing](../../Memory/Coalescing/RESEARCH.md) - 内存访问优化
