# Shader Compilation and Kernel Launch Overhead Research

## 概述

本专题研究GPU着色器编译（Shader Compilation）和内核启动（Kernel Launch）的开销，分析编译时间与执行时间的关系，以及 dispatch 调度的固定开销。

## 关键发现

### 开销分解

| 阶段 | 时间 | 说明 |
|------|------|------|
| makeLibrary | ~100-500 ms | 着色器源码编译 |
| makeComputePipelineState | ~10-50 ms | 管道状态创建 |
| dispatch encode | ~0.5-5 μs | 调度编码 |
| kernel execution | ~10-1000 μs | 内核执行 |

### 编译开销

```swift
// 运行时编译（昂贵）
let compileStart = getTimeNanos()
let library = try device.makeLibrary(source: shaderSource, options: nil)
let compileTime = getElapsedSeconds(start: compileStart, end: compileEnd)
// ~100-500ms 取决于shader复杂度
```

### 管道状态创建

```swift
// 每个kernel函数需要创建一次
let function = library.makeFunction(name: "kernel_name")
let pipeline = try device.makeComputePipelineState(function: function)
// ~10-50ms
```

### 调度编码开销

```swift
let encodeStart = getTimeNanos()
encoder.setComputePipelineState(pipelineState)
encoder.setBuffer(buffer, offset: 0, index: 0)
encoder.dispatchThreads(...)
encoder.endEncoding()
let encodeTime = getElapsedSeconds(start: encodeStart, end: encodeEnd)
// ~0.5-5 μs
```

## 性能数据

### 开销占比分析

| 配置 | 编码开销 | 实际执行 | 开销占比 |
|------|----------|----------|----------|
| 小kernel (1K元素) | ~5 μs | ~10 μs | 33% |
| 中kernel (64K元素) | ~5 μs | ~100 μs | 5% |
| 大kernel (1M元素) | ~5 μs | ~1000 μs | 0.5% |

### 批处理效率

| 批大小 | 总时间 | 每kernel | 加速比 |
|--------|--------|----------|--------|
| 1 | 100 μs | 100 μs | 1.0x |
| 4 | 120 μs | 30 μs | 3.3x |
| 16 | 180 μs | 11.25 μs | 8.9x |
| 64 | 350 μs | 5.5 μs | 18x |

## 优化策略

### 1. 预编译Shader

```swift
// 应用启动时编译，而不是运行时
let library = try device.makeLibrary(source: shaderSource, options: nil)
// 保存library重复使用
```

### 2. 缓存Pipeline State

```swift
// 每个kernel只创建一次pipeline state
let pipeline = try device.makeComputePipelineState(function: function)
// 缓存并重用
```

### 3. 批量调度

```swift
// 减少每次调度的固定开销
encoder.dispatchThreads(...)
encoder.dispatchThreads(...)  // 同一个command buffer
encoder.dispatchThreads(...)
encoder.endEncoding()
```

### 4. 异步编译

```swift
// 使用makeLibrary不阻塞主线程
DispatchQueue.global().async {
    let library = try device.makeLibrary(source: shaderSource, options: nil)
    // 编译完成后通知
}
```

## 编译时间估算

```
Shader复杂度 -> 编译时间:
- 简单kernel (< 100行): ~50-100ms
- 中等kernel (~500行): ~200-300ms
- 复杂kernel (> 1000行): ~500ms+
- 模板/元编程shader: ~1s+
```

## Apple Metal vs NVIDIA

| 方面 | Apple Metal | NVIDIA CUDA |
|------|-------------|-------------|
| JIT编译 | 是 | 是 |
| 预编译 | 支持 | cuModuleLoad |
| 编译缓存 | 部分 | NVRTC缓存 |
| 首次调用延迟 | 高 | 中 |

## 应用场景

### 1. 长期运行应用
```swift
// 启动时预编译所有shader
// 运行时无编译开销
```

### 2. 短时批处理
```swift
// 批量处理减少调度开销
for batch in batches {
    encoder.dispatchThreads(...)
}
```

### 3. 动态Shader生成
```swift
// 运行时生成shader需要考虑编译延迟
// 建议使用预编译 + 参数化kernel
```

## 最佳实践

1. **应用启动时编译** - 避免运行时延迟
2. **缓存Pipeline State** - 每个kernel只创建一次
3. **批量提交** - 减少调度开销
4. **使用简单Shader** - 减少编译时间
5. **避免运行时生成Shader** - 除非绝对必要

## 相关专题

- [CommandBuffer](./CommandBuffer/RESEARCH.md) - 命令缓冲批处理
- [KernelFusion](./KernelFusion/RESEARCH.md) - 内核融合
- [AsyncOperations](./AsyncOperations/RESEARCH.md) - 异步操作
