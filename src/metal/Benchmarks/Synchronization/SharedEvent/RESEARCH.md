# Shared Event Synchronization Research

## 概述

本专题研究GPU上共享事件（MTLSharedEvent）同步机制，实现GPU到CPU和GPU到GPU的同步通信。

## 关键发现

### Shared Event概念

```
MTLSharedEvent: GPU和CPU之间共享的同步原语

用途:
- GPU通知CPU某个操作完成
- CPU等待GPU信号
- 多GPU/多队列之间的同步
```

### 核心API

| API | 说明 |
|-----|------|
| makeSharedEvent() | 创建共享事件 |
| encodeSignalEvent(event, value) | GPU编码信号 |
| waitUntilSignaledValue:value:timeout: | CPU等待信号 |
| getCompletedValue() | 查询事件状态(不阻塞) |

### 工作流程

```
GPU:                          CPU:
-----                         -----
encodeSignalEvent(event, 1)   waitUntilSignaledValue(1)
     |                              |
     v                              |
  (command buffer)             (blocks until)
     |                              |
     v                              |
  completes                     notified
```

## 实现细节

### GPU编码信号
```swift
guard let cmd = queue.makeCommandBuffer(),
      let encoder = cmd.makeComputeCommandEncoder() else { return }
encoder.setComputePipelineState(pipeline)
encoder.dispatchThreads(...)
encoder.endEncoding()

// 命令缓冲完成时信号事件
cmd.encodeSignalEvent(sharedEvent, value: 1)
cmd.commit()
```

### CPU等待
```swift
let start = getTimeNanos()
let signaled = sharedEvent.wait(
    untilSignaledValue: 1,
    timeoutMS: 10_000  // 10秒超时
)
let end = getTimeNanos()
let waitTime = Double(end - start) / 1e6  // 毫秒
```

### CPU轮询(不阻塞)
```swift
let value = sharedEvent.getCompletedValue()
if (value >= 1) {
    // 事件已完成
}
```

## 应用场景

### 1. 批处理完成通知
```cpp
// 提交多个命令缓冲
for (int i = 0; i < batchSize; i++) {
    cmd[i].encodeSignalEvent(event, value: i + 1);
    cmd[i].commit();
}

// CPU等待所有批次完成
event.wait(untilSignaledValue: batchSize, timeoutMS: ...);
```

### 2. 流水线同步
```cpp
// CPU准备下一帧数据
while (processing) {
    // 等待GPU完成当前帧
    event.wait(...);

    // 准备下一帧数据
    prepareNextFrame();

    // GPU处理下一帧
    submitNextFrame();
}
```

### 3. 多队列同步
```cpp
// 队列1处理完成
cmd1.encodeSignalEvent(event, value: 1);
cmd1.commit();

// 队列2等待队列1完成
event.wait(untilSignaledValue: 1, timeoutMS: ...);
cmd2.commit();
```

## 性能特性

| 操作 | 延迟 | 说明 |
|------|------|------|
| GPU信号 | < 0.5 μs | 命令缓冲完成后触发 |
| CPU等待 | 0.1-1 ms | 取决于GPU负载 |
| 轮询 | 0.001 ms | getCompletedValue |

## 与其他同步机制对比

| 机制 | 方向 | 阻塞 | 精度 |
|------|------|------|------|
| MTLSharedEvent | GPU→CPU | 可选 | 命令缓冲级 |
| dispatch_semaphore | CPU→CPU | 是 | 线程级 |
| threadgroup_barrier | GPU→GPU | 是 | 线程组级 |
| MTLCommandBuffer | CPU→GPU | 否 | 提交级 |

## Apple Metal独有特性

```swift
// MTLSharedEvent是Apple独有，NVIDIA使用cudaEvent
// 优势：与Metal命令缓冲天然集成
// 劣势：不可跨平台
```

## 最佳实践

1. **避免频繁等待**: GPU到CPU切换有开销
2. **使用超时**: 防止无限等待
3. **批量信号**: 减少同步次数
4. **轮询替代等待**: 对延迟不敏感的场景用getCompletedValue

## 相关专题

- [Atomics](./Atomics/RESEARCH.md) - 原子操作
- [Barriers](./Barriers/RESEARCH.md) - 线程组屏障
- [AsyncOperations](../Optimization/AsyncOperations/RESEARCH.md) - 异步操作
- [CommandBuffer](../Optimization/CommandBuffer/RESEARCH.md) - 命令缓冲批处理
