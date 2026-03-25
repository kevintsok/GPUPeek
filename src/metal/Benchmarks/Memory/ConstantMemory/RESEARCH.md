# Constant Memory Research

## 概述

本专题研究GPU上常量内存（Constant Memory）的性能特性。常量内存是专用只读内存，GPU对其进行缓存优化以支持广播读取模式。

## 关键发现

### 常量内存访问模式性能对比

| 模式 | 性能 | 说明 |
|------|------|------|
| Broadcast | 最高 | 所有线程读同一值 |
| Sequential | 高 | 顺序访问利用缓存线 |
| Strided | 中 | 有 stride 的访问 |
| Scattered | 低 | 随机访问效率低 |

### Apple M2 GPU 常量内存特性

| 特性 | 值 |
|------|-----|
| 常量缓存大小 | ~32KB |
| 缓存行大小 | 32 bytes |
| 广播支持 | 是 |
| 原子操作 | 不支持 |

### 常量内存 vs 全局内存

| 指标 | 常量内存 | 全局内存 |
|------|---------|---------|
| 读取速度 | 快（缓存命中） | 慢 |
| 写入 | 只读 | 可写 |
| 广播 | 优化 | 普通 |
| 线程分歧 | 惩罚严重 | 正常 |

## 访问模式分析

### 1. Broadcast（最优）
```metal
// 所有线程读取相同常量
constant float4& sharedVal = ...;
out[id] = sharedVal[0] + dev[id];
```
- 一次缓存读取，所有线程共享
- 适合不变的数据（系数、偏移量）

### 2. Sequential（良好）
```metal
// 线程读取相邻常量
out[id] = cst[id] * dev[id];
```
- 缓存线获取，多线程利用
- 适合查表操作

### 3. Scattered（差）
```metal
// 线程读取不同位置
out[id] = cst[id % 1024] * dev[id];
```
- 缓存未命中频繁
- 每次读取可能触发新缓存行加载

## 优化策略

### 1. 利用广播读取
```metal
// 将共享系数放在常量内存
kernel void process(constant float4& coeff [[buffer(1)]], ...) {
    float4 val = coeff[0];  // 广播到所有线程
    ...
}
```

### 2. 避免线程分歧
```metal
// 不好：线程读取不同常量
if (thread_id < 10) {
    val = cst[thread_id];  // 分歧
}

// 好：所有线程使用相同索引
val = cst[common_index];  // 广播
```

### 3. 结构体打包
```metal
// 使用 float4 而非 float
constant float4& params [[buffer(0)]];  // 一次读取 4 个值
```

## 性能影响因子

1. **访问模式** - 广播 > 顺序 >> 散射
2. **线程分歧** - 同一 warp 内线程读不同常量会失败
3. **常量大小** - 超过缓存大小时性能下降
4. **数据复用** - 常量被重复使用越多收益越大

## 相关专题

- [Memory Bandwidth](../Bandwidth/RESEARCH.md) - 内存带宽对比
- [Coalescing](../Coalescing/RESEARCH.md) - 全局内存合并访问
- [Cache](../Cache/RESEARCH.md) - 缓存层级结构
