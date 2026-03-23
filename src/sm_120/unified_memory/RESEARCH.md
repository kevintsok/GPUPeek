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

## 参考文献

- [CUDA Programming Guide - Unified Memory](../ref/cuda_programming_guide.html)
