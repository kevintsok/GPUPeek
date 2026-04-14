# Apple Metal GPU Optimization Summary

## 概述

本文档汇总Apple M2 GPU所有基准测试的关键发现，提供可操作的优化建议。

## 性能天花板 (实测)

| 指标 | 实测值 | 理论值 | 利用率 |
|------|--------|--------|--------|
| 峰值内存带宽 | ~2 GB/s | 100 GB/s | ~2% |
| 峰值计算 (FP32) | ~12 GFLOPS | 未知 | N/A |
| 峰值GEMM (FP16 tiled) | ~15 GFLOPS | 未知 | N/A |
| 突发写入 (16元素/线程) | ~6 GB/s | 100 GB/s | ~6% |

## 关键发现汇总

### 内存访问 (Memory Access)

| 优化项 | 效果 | 优先级 |
|--------|------|--------|
| 内存合并访问 | 5.3x 加速 | P0 |
| 突发写入 (16/线程) | 3-4x 加速 | P0 |
| Float4向量化 | 2x 加速 | P1 |
| 避免跨步访问 | 2x 加速 | P1 |

### 计算优化 (Compute)

| 优化项 | 效果 | 优先级 |
|--------|------|--------|
| FP16半精度 | 2x 加速 | P1 |
| GEMM分块tiling | 2-5x 加速 | P1 |
| 4x4寄存器分块 | 5x 加速 | P1 |
| FMA指令 | 1.5x 加速 | P2 |

### 同步优化 (Synchronization)

| 优化项 | 效果 | 优先级 |
|--------|------|--------|
| 命令缓冲批处理 | 1.88x 加速 | P1 |
| 内核融合 | 2.36x 加速 | P1 |
| 异步执行 | 1.65x 加速 | P2 |
| 多队列 | 0.76x (变慢!) | P3 |

## 优化优先级清单

### P0 - 必须做 (最高影响)

1. **确保顺序内存访问 (合并访问)**
```metal
// ❌ 错误：跨步访问
uint idx = id * stride % size;
value = data[idx];

// ✅ 正确：连续访问
value = data[id];
```

2. **使用Float4/Half4向量化**
```metal
// ❌ 慢：单元素访问
float val = data[id];

// ✅ 快：4元素向量化
float4 vals = data[id / 4];
```

3. **突发写入 (每线程多个元素)**
```metal
// 写入16个元素而不是1个
for (uint i = 0; i < 16; i++) {
    out[id * 16 + i] = compute();
}
```

### P1 - 应该做 (高影响)

4. **FP16用于ML/推理**
```swift
config.computeUnits = .all // CPU + GPU + ANE
// 或使用FP16纹理
```

5. **GEMM使用共享内存分块**
```metal
// 将矩阵块加载到共享内存
threadgroup_barrier(mem_flags::mem_none);
// 在共享内存中计算
```

6. **融合多个kernel为一个**
```metal
// 融合前：两个kernel
kernel1(out);
kernel2(out);

// 融合后：一个kernel
fused_kernel(out);
```

7. **批处理命令缓冲**
```swift
// 收集多个kernel后一起提交
for cmd in cmdBuffers {
    cmd.commit()
}
```

### P2 - 可以做 (中等影响)

8. **调整threadgroup大小 (256+)**
```swift
encoder.dispatchThreads(
    MTLSize(width: width, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
)
```

9. **4x4寄存器分块 (大型矩阵)**
```metal
// 每个线程处理4x4块
float4 row0 = A[i + 0 * lda + k * 4 ...];
```

## 按应用场景的优化建议

### 机器学习推理
1. 使用FP16精度
2. 利用ANE (CoreML: .aneOnly)
3. 批量处理输入
4. 使用量化 (Int8/Int4)

### 矩阵乘法 (GEMM)
1. 共享内存分块 (32KB限制)
2. 4x4寄存器分块
3. FP16累加
4. 批量矩阵乘法

### 图像处理
1. 使用Metal Performance Shaders
2. 纹理缓存优化
3. 逐帧流水线
4. 双缓冲

### 通用GPU计算
1. 合并内存访问
2. 突发写入
3. 减少原子操作
4. 避免分支分歧

## 架构注意事项

### Apple M2 GPU规格
| 特性 | 值 |
|------|-----|
| SIMD宽度 | 32线程 |
| Threadgroup内存 | 32 KB |
| 统一内存 | 是 (与CPU共享) |
| GPU Family | Apple 7 |
| 峰值计算 | ~12 GFLOPS |

### 统一内存的影响
- 带宽与CPU共享
- 有效带宽 ~2 GB/s
- 需要优化内存访问模式
- 不适合内存密集型操作

## 性能问题排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 带宽低 | 非合并访问 | 重排线程访问模式 |
| GPU慢 | 内存瓶颈 | 使用共享内存分块 |
| 波动大 | 分支分歧 | 使用predication |
| 启动慢 | 编译开销 | 预编译shader |
| 占用率低 | threadgroup太大 | 减小threadgroup |

## 相关专题

- [Roofline](./Roofline/RESEARCH.md) - Roofline模型分析
- [GEMM](../../Compute/GEMM/RESEARCH.md) - 矩阵乘法优化
- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽
- [ThreadgroupMemory](../../Memory/ThreadgroupMemory/RESEARCH.md) - 共享内存优化
