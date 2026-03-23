# Tensor Memory Operations Research

## 概述

张量内存操作研究，包括 LDMATRIX、STMATRIX、cp.async 等指令。

## 1. LDMATRIX

矩阵加载指令，Warp 级操作。

### 变体

| 指令 | 描述 | 每线程元素 |
|------|------|-----------|
| ldmatrix.sync.aligned.m8n8.x1 | 8x8, 1 矩阵 | 2 |
| ldmatrix.sync.aligned.m8n8.x2 | 8x8, 2 矩阵 | 4 |
| ldmatrix.sync.aligned.m8n8.x4 | 8x8, 4 矩阵 | 8 |
| ldmatrix.sync.aligned.m16n8.k1 | 16x8 tile | varies |

### 关键特性
- Warp 级操作 (32 线程协作)
- 转置布局 (MMA 友好)
- 需要 16 字节对齐

## 2. STMATRIX

矩阵存储指令。

| 指令 | 描述 |
|------|------|
| stmatrix.sync.aligned.m8n8.x1 | 8x8, 1 矩阵 |
| stmatrix.sync.aligned.m8n8.x2 | 8x8, 2 矩阵 |
| stmatrix.sync.aligned.m8n8.x4 | 8x8, 4 矩阵 |

## 3. cp.async

异步拷贝指令。

```ptx
cp.async.ca    // cache-atomic
cp.async.commit_group  // 提交异步组
cp.async.wait_group n  // 等待 n 个组
```

## 4. cp.async.bulk

批量异步拷贝。

```ptx
cp.async.bulk
cp.async.bulk.commit_group
cp.reduce.async.bulk.add  // 拷贝+求和
```

## 5. 张量内存性能汇总

| 操作 | 优势 | 适用场景 |
|------|------|----------|
| LDMATRIX | Warp 协作、转置 | MMA A/B 加载 |
| STMATRIX | Warp 协作 | MMA 结果存储 |
| cp.async | 延迟隐藏 | 计算/拷贝重叠 |
| TMA | 大块 2D 传输 | 大矩阵分块 |

## 参考文献

- [CUDA Programming Guide - LDMATRIX](../ref/cuda_programming_guide.html)
- [PTX ISA - Matrix](../ref/ptx_isa.html)
