# Branch Divergence Research

## 概述

本专题研究GPU上分支发散（Branch Divergence）的性能影响。Apple GPU使用SIMD-group（32线程一组），当同一SIMD组内的线程执行不同分支路径时，会产生发散损失。

## 关键发现

### 分支发散性能对比

| 分支类型 | 性能 | 开销 |
|---------|------|------|
| Converged (收敛) | 基准 | 0% |
| Even/Odd Divergence | ~70% | ~30% |
| Quarter-warp (8 threads) | ~75% | ~25% |
| Branchless Select | ~95% | ~5% |
| Divergent Loop | ~50-60% | ~40-50% |

### SIMD组执行模型

Apple M2 GPU按32线程一组（SIMD-group）执行指令：
- 同一组的线程必须执行相同指令
- 不同分支路径会被串行化执行
- 未执行的路径结果被丢弃

### 发散模式分析

| 模式 | 说明 | 效率损失 |
|------|------|---------|
| All-same | 所有线程同路 | 无 |
| Half-split | 16+16发散 | ~30% |
| Quarter-split | 8+24发散 | ~25% |
| Individual | 每线程不同 | ~97% (几乎串行) |

## 优化策略

### 1. 避免发散分支
```metal
// 低效：发散分支
if (lane < 16) {
    out = val * 2.0f;
} else {
    out = val * 0.5f;
}

// 高效：使用select (branchless)
float2 result = select(float2(val * 0.5f), float2(val * 2.0f), float2(lane < 16));
out = result[0];
```

### 2. 合并相同路径线程
```metal
// 重组线程使相同路径的线程在相邻位置
uint new_id = (id / 32) * 32 + (id % 16) * 2 + (id % 2);
```

### 3. 循环展开考虑
```metal
// Divergent loop: different iterations
for (uint i = 0; i < 4 + (lane & 15); i++) { ... }

// Converged: all threads same iterations
for (uint i = 0; i < 16; i++) { ... }
```

## Apple Metal 特定

### SIMD Group大小
- Apple Metal: 32线程为一组
- NVIDIA CUDA Warp: 32线程
- 对应概念：SIMD-group = Warp

### 避免发散的patterns
1. **Predication** - 使用mask提前计算，避免实际分支
2. **Branchless** - 使用select/ternary替代if-else
3. **Thread reordering** - 重新排列线程使相同路径相邻

## 相关专题

- [WarpPrimitives](../../Synchronization/WarpPrimitives/RESEARCH.md) - SIMD组通信原语
- [Memory Coalescing](../../Memory/Coalescing/RESEARCH.md) - 内存访问模式与线程排列
- [Occupancy](../../Analysis/Occupancy/RESEARCH.md) - 占用率与发散的关系
