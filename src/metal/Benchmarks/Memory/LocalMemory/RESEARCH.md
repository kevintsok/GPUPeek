# Local Memory Copy Research

## 概述

本专题研究GPU上本地内存（Threadgroup/Shared Memory）的访问特性和全局内存到本地内存的拷贝性能。本地内存是GPU上最快可编程访问的内存，位于每个GPU核心附近。

## 关键发现

### 内存层级性能

| 内存类型 | 延迟 | 带宽 | 备注 |
|----------|------|------|------|
| Register | ~1 cycle | 极高 | 最快，但数量有限 |
| Threadgroup (Local) | ~10-20 cycles | ~1-2 TB/s | 32KB per threadgroup |
| Global | ~200-400 cycles | ~100 GB/s | Unified Memory |

### 拷贝模式对比

```
Global → Global (Baseline):
- 最简单，但无优化
- 性能取决于合并访问

Global → Shared → Global:
- 额外拷贝开销
- 适合需要多次访问同一数据的场景
- 需要barrier同步

Block-strided:
- 线程处理连续数据块
- 改善合并但增加计算
```

## 优化策略

### 1. 合并访问
```metal
// 连续线程访问连续内存
for (uint i = lid; i < size; i += threadgroupSize) {
    tile[lid] = global[i];  // Coalesced
}
```

### 2. 避免Barrier冲突
```metal
// mem_none vs mem_threadgroup
threadgroup_barrier(mem_flags::mem_none);  // 仅同步同threadgroup
threadgroup_barrier(mem_flags::mem_threadgroup);  // 同步所有线程
```

### 3. 双缓冲
```metal
// 交替使用两个buffer实现计算与拷贝重叠
if (bufferIdx == 0) {
    process(tileA);
    load(tileB);
} else {
    process(tileB);
    load(tileA);
}
```

### 4. 向量化拷贝
```metal
// 使用float4一次拷贝4个元素
float4 val = global[id];
tile[lid] = val;
```

## Apple Metal本地内存特性

1. **大小**: 32KB per threadgroup (Apple GPU)
2. **延迟**: 约10-20 cycles
3. **带宽**: 约1-2 TB/s (取决于核心频率)
4. **用途**:
   - 矩阵乘法tiling
   - 数据重排
   - 临时变量存储

## 应用场景

1. **GEMM Tiling** - 16x16或32x32 tile
2. **Stencil计算** - 共享数据供多次使用
3. **排序算法** - 块内排序
4. **Graph algorithms** - 邻接表缓存

## 性能影响因子

1. **数据重用次数** - 重用越多，local memory越值得
2. **Barrier频率** - 每次同步都有开销
3. **Threadgroup大小** - 影响Occupancy
4. **Bank冲突** - 32 banks，访问同bank会串行

## 相关专题

- [GEMM](../Compute/GEMM/RESEARCH.md) - 矩阵乘法tiling
- [BankConflict](../Memory/BankConflict/RESEARCH.md) - 共享内存bank冲突
- [Stencil](../Algorithms/Stencil/RESEARCH.md) - 模板计算
