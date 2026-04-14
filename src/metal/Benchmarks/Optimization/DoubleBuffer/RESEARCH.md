# Double Buffering Research

## 概述

本专题研究Double Buffering（双缓冲）和Triple Buffering（三缓冲）技术，用于流水线操作中的读写重叠。

## 双缓冲原理

双缓冲使用两个缓冲区交替读写：
- 阶段1: 读取Buffer A，处理，写入Buffer B
- 阶段2: 读取Buffer B，处理，写入Buffer A
- 两个阶段可以并行（读取和写入同时进行）

## 关键发现

### 缓冲区对比

| 配置 | 性能 | 说明 |
|------|------|------|
| Single Buffer | ~0.018 GOPS | 基准，无重叠 |
| Double Buffer | ~0.018 GOPS | 同步等待无差异 |
| Triple Buffer | ~0.018 GOPS | 同样无差异 |

### 双缓冲条件

双缓冲只有在**异步执行**时才有优势：
- 同步等待(waitUntilCompleted)会阻塞，无法重叠
- 使用completion handler才能真正实现读写并行

## 关键洞察

1. **同步执行时双缓冲无优势** - CPU等待导致无法重叠
2. **异步执行是双缓冲的前提** - 使用completion handler
3. **Triple buffer增加流水线深度** - 适合多阶段处理
4. **适用场景** - 迭代算法、图像处理、视频编解码

## 优化策略

1. **使用异步API** - makeComandBuffer + commit + addCompletedHandler
2. **分离读写队列** - CPU处理和GPU执行并行
3. **增加buffer数量** - 多缓冲增加流水线深度
4. **避免同步等待** - 用通知机制替代waitUntilCompleted

## 相关专题

- [Kernel Fusion](../KernelFusion/RESEARCH.md) - 内核融合
- [Command Buffer](../CommandBuffer/RESEARCH.md) - 命令缓冲优化
- [Barriers](../../Synchronization/Barriers/RESEARCH.md) - 同步开销
