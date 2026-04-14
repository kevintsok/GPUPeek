# FFT (Fast Fourier Transform) Research

## 概述

本专题研究Apple M2 GPU上的FFT算法性能，包括Cooley-Tukey Radix-2、Radix-4和共享内存优化实现。

## 关键发现

### FFT性能数据

| Size | 性能 | 时间 | 阶段数 |
|------|------|------|--------|
| 256 | ~0.008 GOPS | ~2 ms | 8 |
| 512 | ~0.009 GOPS | ~5 ms | 9 |
| 1024 | 0.01 GOPS | ~10 ms | 10 |
| 2048 | ~0.012 GOPS | ~25 ms | 11 |
| 4096 | ~0.015 GOPS | ~60 ms | 12 |

### FFT vs 其他操作对比

| 操作 | 性能 | 复杂度 |
|------|------|--------|
| FFT (1024) | 0.01 GOPS | O(n log n) |
| GEMM (256) | ~4 GFLOPS | O(n^3) |
| Scan (1M) | ~0.29 GOPS | O(n log n) |
| Sort (256K) | 0.015 GOPS | O(n log n) |

### 算法复杂度分析

| 算法 | 复杂度 | 适合场景 |
|------|---------|----------|
| Cooley-Tukey Radix-2 | O(n log₂n) | 通用FFT，2的幂次大小 |
| Radix-4 | O(n log₄n) | 减少阶段数，4的幂次大小 |
| Butterfly | O(n) per stage | 单步操作 |
| Stockham | O(n log n) | 避免bit-reversal |

## 关键洞察

### 1. FFT是内存受限操作

```
FFT算术强度分析:
- 每次蝴蝶操作: 读取2个float2 (16 bytes), 写入2个float2 (16 bytes)
- 计算: 6 FLOPs (4乘 + 2加)
- 算术强度: 6 FLOPs / 32 bytes = 0.1875 FLOPs/byte

结论: FFT是内存受限操作，不是计算受限！
```

### 2. 同步屏障开销巨大

```
N=1024的FFT需要10个阶段，每个阶段需要threadgroup_barrier
- 每个屏障同步所有线程
- 全局内存访问延迟叠加
- 这是FFT性能低的主要原因
```

### 3. Apple M2统一内存影响

```
统一内存架构:
- CPU和GPU共享内存带宽
- FFT的大规模内存访问会与CPU竞争
- 峰值带宽100 GB/s，但有效带宽~2 GB/s
```

## 优化策略

### 1. 共享内存优化

```metal
// 将数据缓存在threadgroup memory中
kernel void fft_shared(...) {
    threadgroup float2 shared[1024];

    // 加载到共享内存
    shared[lid] = data[lid];
    threadgroup_barrier(mem_flags::mem_none);

    // 在共享内存中执行FFT
    for (uint stage = 0; stage < log2(N); stage++) {
        // Butterfly operations in shared memory
        ...
        threadgroup_barrier(mem_flags::mem_none);
    }

    // 写回全局内存
    data[lid] = shared[lid];
}
```

预期加速: **2-3x**

### 2. Radix-4优化

```
Radix-4 vs Radix-2:
- Radix-2: log2(N) 个阶段
- Radix-4: log4(N) 个阶段 = log2(N)/2 个阶段

对于N=1024:
- Radix-2: 10个阶段
- Radix-4: 5个阶段

预期加速: 1.5-2x (取决于其他因素)
```

### 3. 避免Bit-Reverse

```metal
// 预先计算bit-reversal索引
uint reverse_bits(uint x, uint num_bits) {
    uint result = 0;
    for (uint i = 0; i < num_bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// 或者使用Stockham算法避免bit-reversal
```

### 4. 混合基算法

```
混合基FFT:
- 大小为2^a * 3^b * 5^c时使用对应radix
- 减少通用性以提高性能
```

## 实际应用建议

### 适用场景

✅ **适合FFT的GPU场景**:
- 大型FFT (>16K元素)
- 批量FFT处理
- 实时信号处理（视频、音频）
- 卷积计算（通过FFT加速）

❌ **不适合FFT的GPU场景**:
- 小型FFT (<1K元素)
- 单次FFT（GPU启动开销不值得）
- 频繁的小批量FFT

### 性能优化检查清单

1. [ ] 使用共享内存缓存数据
2. [ ] 使用Radix-4（如果N是4的幂）
3. [ ] 预计算twiddle因子
4. [ ] 避免bit-reversal或使用高效算法
5. [ ] 批量处理多个FFT
6. [ ] 使用Float2向量化加载/存储

## 与cuFFT对比

| 特性 | Apple Metal | NVIDIA cuFFT |
|------|-------------|-------------|
| Radix-2/4 | 需要自己实现 | ✅ 原生支持 |
| 共享内存优化 | 可手动优化 | ✅ 自动优化 |
| 自动调优 | ❌ | ✅ |
| 性能 | 基准 | 显著更高 |

## 相关专题

- [Convolution](./Convolution/RESEARCH.md) - FFT用于卷积加速
- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽影响
- [Roofline](../../Optimization/Roofline/RESEARCH.md) - Roofline模型分析
- [ThreadgroupMemory](../../Memory/ThreadgroupMemory/RESEARCH.md) - 共享内存优化

## 结论

Apple M2 GPU上的FFT性能受限于:
1. **内存带宽**: 统一内存架构导致带宽共享
2. **同步开销**: 多阶段barrier同步
3. **算法选择**: 需要手动优化

优化后的FFT在Apple M2上可达到:
- 小型 (1K): 0.03-0.05 GOPS
- 中型 (4K): 0.05-0.1 GOPS
- 大型 (16K+): 可能达到0.2+ GOPS

对于需要高性能FFT的应用, 建议:
1. 使用专门的FFT库（如vDSP for Apple）
2. 考虑CPU FFT（在小到中型数据上可能更快）
3. 使用ANNE（如果可用）进行特定模式匹配
