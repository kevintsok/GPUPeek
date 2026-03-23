# Apple Metal GPU Research

## 概述

本目录用于研究Apple M系列芯片的GPU特性和性能，使用Metal API进行基准测试。

## Apple M系列GPU架构

### 芯片概览

| 芯片 | GPU核心 | 统一内存 | 带宽 | 备注 |
|------|---------|----------|------|------|
| M1 | 8核 | 最高16GB | 68.25 GB/s | 第一代Apple Silicon |
| M1 Pro | 16核 | 最高32GB | 204 GB/s | 增强型GPU |
| M1 Max | 32核 | 最高64GB | 408 GB/s | 专业级 |
| M2 | 10核 | 最高24GB | 100 GB/s | 第二代 |
| M2 Pro | 19核 | 最高32GB | 200 GB/s | |
| M2 Max | 38核 | 最高96GB | 400 GB/s | |
| M3 | 10核 | 最高24GB | 100 GB/s | 新一代架构 |
| M3 Pro | 14/18核 | 最高36GB | 150-200 GB/s | |
| M3 Max | 40核 | 最高128GB | 300-400 GB/s | |
| M4 | 10核 | 最高32GB | 120 GB/s | 最新代 |
| M4 Pro | 20核 | 最高64GB | 273 GB/s | |
| M4 Max | 40核 | 最高128GB | 546 GB/s | |

### 架构特性

1. **统一内存架构 (Unified Memory)**
   - CPU和GPU共享同一内存池
   - 无需显式内存拷贝
   - 支持内存压缩技术
   - 延迟统一内存访问 (LATENCY NULL)

2. **Metal性能Shaders**
   - 支持GPU渲染
   - 并行计算任务
   - 整数和浮点混合计算
   - 硬件级光线追踪 (M3+)

3. **内存带宽**
   - 统一内存带宽因芯片而异
   - M1: 68.25 GB/s
   - M2: 100 GB/s
   - M3: 100 GB/s
   - M4 Max: 546 GB/s

4. **SIMD/并行计算**
   - SIMD组操作
   - 线程组栅栏
   - 原子操作支持
   - 纹理和缓冲区支持

## 研究计划

### Phase 1: 基础带宽测试
- [x] 内存带宽基准测试
- [x] 计算吞吐量测试
- [ ] 延迟测试

### Phase 2: 高级特性
- [ ] 统一内存访问模式
- [ ] 线程组性能
- [ ] 原子操作性能

### Phase 3: 架构对比
- [ ] 与NVIDIA GPU对比
- [ ] 与AMD GPU对比
- [ ] 性能效率分析

## 测试环境

- **设备**: Apple M2 (MacBook Air)
- **操作系统**: macOS
- **Swift版本**: 6.1.2
- **Metal版本**: Apple 7+ GPU Family

## 基准测试结果

### Apple M2 实测结果

```
=== Apple Metal GPU Info ===
Device Name: Apple M2
Unified Memory: Yes (Shared with CPU)
Max Threadgroup Memory: 32 KB
Max Threads Per Threadgroup: 1024
GPU Family: Apple 7+
```

### 内存带宽测试

| 测试类型 | 实测带宽 | 理论带宽 | 利用率 | 备注 |
|---------|---------|---------|--------|------|
| Memory Copy | 1.03 GB/s | 100 GB/s | ~1% | 256MB x 100次 |
| Memory Set | 1.87 GB/s | 100 GB/s | ~2% | 内存写入 |
| Vector Add | 1.81 GB/s | 100 GB/s | ~2% | 2读1写模式 |

**注意**: 实测带宽远低于理论带宽，主要原因是：
1. Swift调用Metal API的额外开销
2. 每次kernel launch的开销（未使用command buffer batching）
3. 同步等待`waitUntilCompleted()`的阻塞开销

### 计算吞吐量

| 操作类型 | 实测性能 | 理论性能 | 备注 |
|---------|---------|---------|------|
| FP32 MatMul (512x512x512) | 4.04 GFLOPS | ~1000+ GFLOPS | 10次迭代平均 |
| 三角函数 (sin+cos+tan) | 0.56 GOPS | - | 8M元素 x 20次 |

## 研究发现

### 1. 统一内存特性
- M2的统一内存带宽为100 GB/s
- 实际应用中由于API开销，利用率较低
- 统一内存消除了CPU-GPU数据传输延迟

### 2. Metal API特性
- Metal Shader使用`metal::`命名空间函数（如`metal::sin`, `metal::cos`）
- `threadgroup_barrier`用于线程组同步
- GPU Family 7+ 是M2支持的最高功能集

### 3. 性能特点
- Apple M2 GPU理论性能约3.5 TFLOPS (FP32)
- 实际benchmark性能受API开销影响较大
- 批处理和command buffer优化可显著提升性能

## 代码文件

| 文件 | 说明 |
|------|------|
| `Package.swift` | Swift Package Manager配置 |
| `Sources/MetalBenchmark/main.swift` | 主程序和Metal Shader代码 |
| `bandwidth_test.metal` | 带宽测试内核（备用） |
| `compute_test.metal` | 计算测试内核（备用） |

## 运行基准测试

```bash
cd src/metal
swift build --configuration release
swift run --configuration release
```

## 参考资料

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Apple GPU Architecture](https://developer.apple.com/documentation/metal/metal_feature_set_tables)
