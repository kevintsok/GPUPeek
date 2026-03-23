# Multi-Stream Research

## 概述

Multi-Stream 并发执行实现计算与内存操作重叠。

## 1. Stream 基本概念

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 不同流执行
kernel1<<<..., stream1>>>(...);
kernel2<<<..., stream2>>>(...);
```

## 2. 流依赖

```cuda
cudaEvent_t event;
cudaEventCreate(&event);
kernel1<<<..., stream1>>>(...);
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<..., stream2>>>(...);
```

## 3. 重叠执行

| 配置 | 重叠效果 |
|------|----------|
| 计算 + 拷贝 | 高重叠 |
| 计算 + 计算 | 中重叠 |
| 同一流 | 无重叠 |

## 4. 流优先级

```cuda
cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
```

## 参考文献

- [CUDA Programming Guide - Streams](../ref/cuda_programming_guide.html)
