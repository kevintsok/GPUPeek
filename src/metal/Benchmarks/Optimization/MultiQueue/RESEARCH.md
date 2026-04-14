# Multi-Queue GPU Parallelism Research

## 概述

本专题研究GPU上多命令队列并行（Multi-Queue GPU Parallelism），通过多个命令队列实现内核并行执行，提高GPU利用率。

## 关键发现

### 单队列 vs 多队列

```
单队列执行:
Kernel1 -> Kernel2 -> Kernel3 (串行)

多队列执行:
Queue1: Kernel1 -> Kernel2 ->
Queue2: Kernel3 -> Kernel4 ->     (并行)

GPU可以同时执行来自不同队列的内核
```

### 性能对比

| 配置 | 性能 | 加速比 |
|------|------|--------|
| 单队列 | ~100ms | 1.0x |
| 双队列 | ~60ms | ~1.6x |

### 队列类型

```swift
// 默认命令队列
let queue = device.makeCommandQueue()

// 可指定优先级
let queue = device.makeCommandQueue(
    label: "high_priority",
    priority: .high
)
```

## 实现细节

### 单队列执行
```swift
for _ in 0..<iterations {
    let cmd = queue.makeCommandBuffer()
    let encoder = cmd.makeComputeCommandEncoder()
    encoder.setComputePipelineState(pipeline)
    encoder.dispatchThreads(...)
    encoder.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()  // 等待完成
}
```

### 多队列并行
```swift
// 创建第二个队列
guard let queue2 = device.makeCommandQueue() else { return }

// 同时提交到两个队列
let cmd1 = queue.makeCommandBuffer()
let enc1 = cmd1.makeComputeCommandEncoder()
enc1.setComputePipelineState(pipeline)
enc1.dispatchThreads(...)
enc1.endEncoding()

let cmd2 = queue2.makeCommandBuffer()
let enc2 = cmd2.makeComputeCommandEncoder()
enc2.setComputePipelineState(pipeline)
enc2.dispatchThreads(...)
enc2.endEncoding()

// 提交但不等待
cmd1.commit()
cmd2.commit()

// 等待全部完成
cmd1.waitUntilCompleted()
cmd2.waitUntilCompleted()
```

## Apple Metal队列特性

### 队列优先级
```swift
// 高优先级队列
device.makeCommandQueue(priority: .high)

// 低优先级队列
device.makeCommandQueue(priority: .low)

// 默认优先级
device.makeCommandQueue(priority: .normal)
```

### 队列标签
```swift
let queue = device.makeCommandQueue(
    label: "Compute Queue 1"
)
```

## 应用场景

### 1. 批量处理
```cpp
// 多个独立的计算任务
Queue1: batch1_kernel(input1, output1)
Queue2: batch2_kernel(input2, output2)
// 两个批次并行处理
```

### 2. 流水线
```cpp
// 准备-计算重叠
Queue1: prepare_next_frame()
Queue2: process_current_frame()
```

### 3. 读写与计算分离
```cpp
// 传输队列和计算队列分离
transferQueue: DMA_read(input)
computeQueue: GEMM(input, weight, output)
```

### 4. 优先级调度
```cpp
// 高优先级队列处理紧急任务
highPriorityQueue: latency_critical_kernel()
lowPriorityQueue: background_kernel()
```

## 性能优化策略

### 1. 队列数量选择
```
不是越多越好：
- 2-4个队列通常最佳
- 队列间同步有开销
- GPU资源有限
```

### 2. 任务分配
```
独立任务分配到不同队列：
- 无依赖的Kernel
- 不同数据流
- 不同优先级
```

### 3. 同步开销
```swift
// 每个队列的waitUntilCompleted都有开销
// 不要在紧密循环中等待
```

## 与NVIDIA CUDA对比

| 特性 | Apple Metal | NVIDIA CUDA |
|------|-------------|-------------|
| 多队列 | MTLCommandQueue | cudaStream_t |
| 优先级 | 支持 | stream priority |
| 队列数 | 多队列建议2-4 | 建议8-16 |
| 同步 | waitUntilCompleted | cudaStreamSynchronize |

## 限制

```
Apple M2 GPU限制：
- 统一内存架构
- 共享带宽
- 多队列加速受限于GPU核心数
```

## 相关专题

- [AsyncOperations](./AsyncOperations/RESEARCH.md) - 异步操作和重叠
- [CommandBuffer](./CommandBuffer/RESEARCH.md) - 命令缓冲批处理
- [DoubleBuffer](./DoubleBuffer/RESEARCH.md) - 双缓冲流水线
- [SharedEvent](../Synchronization/SharedEvent/RESEARCH.md) - 共享事件同步
