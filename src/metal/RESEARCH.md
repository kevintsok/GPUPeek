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

### Phase 1: 基准测试优化
- [x] 内存带宽基准测试（初步）
- [x] Command Buffer Batching优化
- [x] 三缓冲(Triple Buffering)优化
- [x] 异步执行优化
- [x] 计算吞吐量测试（优化版）

### Phase 2: 内存子系统研究
- [ ] 统一内存访问模式
- [ ] 线程组性能
- [ ] 原子操作性能
- [ ] LATENCY NULL技术研究

### Phase 3: 架构特性研究
- [ ] Apple GPU微架构分析
- [ ] TBDR (Tile-Based Deferred Rendering) 特性
- [ ] 内存压缩技术

### Phase 4: 架构对比
- [ ] 与NVIDIA GPU对比
- [ ] 与AMD GPU对比
- [ ] 性能效率分析

## 测试环境

- **设备**: Apple M2 (MacBook Air)
- **操作系统**: macOS Darwin 25.3.0
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
ReadWriteTextureSupport: MTLReadWriteTextureTier2
```

### Phase 1: 优化后带宽测试结果

| 测试类型 | 实测带宽 | 理论带宽 | 利用率 | 优化技术 |
|---------|---------|---------|--------|----------|
| Memory Copy (Batched) | 0.99 GB/s | 100 GB/s | ~1% | 10x批量调度 |
| Memory Copy (Triple Buffer) | 1.07 GB/s | 100 GB/s | ~1% | 三缓冲 |
| Memory Copy (Async) | 0.93 GB/s | 100 GB/s | ~1% | 异步回调 |
| Vector Add (Batched) | 1.88 GB/s | 100 GB/s | ~2% | simd_float4向量化 |
| Threadgroup Reduction | 0.98 GB/s | 100 GB/s | ~1% | 32KB共享内存 |

### Phase 1: 优化后计算吞吐量

| 操作类型 | 规模 | 实测性能 | 备注 |
|---------|------|---------|------|
| FP32 MatMul | 1024x1024x1024 | 4.62 GFLOPS | 朴素实现 |
| FP16 MatMul | 1024x1024x1024 | 4.93 GFLOPS | 半精度 |

## 关键发现

### 1. API开销不是瓶颈

通过实施三种优化技术（批量处理、三缓冲、异步执行），带宽没有显著变化（都在~1 GB/s）。这表明：
- **内核启动开销不是瓶颈**
- **CPU-GPU同步开销不是瓶颈**
- **瓶颈在GPU内存访问级别**

### 2. 统一内存架构开销

Apple M2的统一内存架构与独立GPU有本质区别：
- CPU和GPU共享内存带宽
- Apple实现内存压缩技术
- 可能存在系统级虚拟化开销
- 实际带宽利用率约1%

### 3. 向量化有效

使用`simd_float4`（16字节）相对于标量操作提升了~50%带宽（1.88 vs 1.03 GB/s）。

### 4. 矩阵乘法性能

- FP32: 4.62 GFLOPS (1024³矩阵)
- FP16: 4.93 GFLOPS (1024³矩阵)
- 半精度略高于单精度（可能因内存访问减少）

## 研究发现

### 统一内存特性
- M2的统一内存带宽为100 GB/s（理论）
- 实测带宽仅~1-2 GB/s（~1%利用率）
- 统一内存消除了CPU-GPU数据传输延迟
- 但共享带宽意味着CPU会与GPU争抢内存

### Metal API特性
- Metal Shader使用`metal::`命名空间函数
- `threadgroup_barrier`用于线程组同步
- GPU Family 7+ 是M2支持的最高功能集
- 使用`simd_float4`等向量类型可提升性能

### Apple GPU架构假设
- TBDR（瓦片式延迟渲染）可能影响计算内核性能
- Apple可能使用内存压缩影响测量精度
- 电源管理可能导致持续负载节流

## 代码文件

| 文件 | 说明 |
|------|------|
| `Package.swift` | Swift Package Manager配置 |
| `Sources/MetalBenchmark/main.swift` | 优化后的主程序和Metal Shader |
| `bandwidth_test.metal` | 带宽测试内核（备用） |
| `compute_test.metal` | 计算测试内核（备用） |
| `docs/` | 研究报告目录 |

## 运行基准测试

```bash
cd src/metal
swift build --configuration release
swift run --configuration release
```

## 研究报告

- [Phase 1 Report (EN)](docs/Phase1_Benchmark_Optimization_Report.md)
- [Phase 1 Report (CN)](docs/Phase1_Benchmark_Optimization_Report_CN.md)

## 参考资料

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Apple GPU Architecture](https://developer.apple.com/documentation/metal/metal_feature_set_tables)
- [WWDC20: Harness Apple GPUs with Metal](https://developer.apple.com/videos/play/wwdc2020/10602/)
- [WWDC20: GPU Performance Counters](https://developer.apple.com/videos/play/wwdc2020/10603/)
