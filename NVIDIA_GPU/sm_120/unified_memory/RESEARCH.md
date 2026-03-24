# Unified Memory Research

## 概述

Unified Memory (统一内存) 提供单一的指针空间，简化 CPU-GPU 内存管理。

## 1. 特性

| 特性 | 描述 |
|------|------|
| Page Fault | 按需页面迁移 |
| Prefetch | 预取数据到指定设备 |
| Migration | 自动页面迁移 |

## 2. 分配

```cuda
cudaMallocManaged(&ptr, size);
// 现在可以在 CPU 或 GPU 上访问
```

## 3. Prefetch

```cuda
cudaMemPrefetchAsync(ptr, size, deviceId, stream);
```

## 4. 页面迁移

| 事件 | 行为 |
|------|------|
| CPU 访问 | 页面迁移到 CPU |
| GPU 访问 | 页面迁移到 GPU |
| 访问冲突 | 迁移回发起方 |

## 5. 性能考虑

- 顺序访问模式更好
- 避免过度迁移
- 使用提示性 API

## 6. 性能数据

| 操作 | 延迟/带宽 | 描述 |
|------|----------|------|
| 页面迁移 | ~1-5 μs/页 | 取决于数据局部性 |
| GPU 预取 | ~100-800 GB/s | 取决于工作集大小 |
| Page Fault | ~5-20 μs | 首次访问开销 |

## 7. 最佳实践

1. **使用 cudaMemPrefetchAsync**: 提前将数据迁移到目标设备
2. **批量访问**: 将多次小访问合并为连续大访问
3. **访问局部性**: 保持良好的空间和时间局部性
4. **避免频繁跨设备访问**: CPU-GPU 迁移开销较大

## 8. 适用场景

| 场景 | 建议 |
|------|------|
| 简单数据交换 | 使用 cudaMemcpy |
| 复杂算法迭代 | Unified Memory + Prefetch |
| 大型数据结构 | 分块预取策略 |

## 参考文献

- [CUDA Programming Guide - Unified Memory](../ref/cuda_programming_guide.html)
