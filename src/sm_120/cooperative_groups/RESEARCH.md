# Cooperative Groups Research

## 概述

Cooperative Groups API 支持灵活的线程组协作。

## 1. 线程组类型

| 类型 | 描述 |
|------|------|
| thread_block | CTA 级别 (最多 1024 线程) |
| thread_block_group | Warp 级别 (32 线程) |
| grid_group | Grid 级别 |
| multi_grid_group | 多 GPU |

## 2. 基本 API

```cuda
thread_block block = this_thread_block();
block.sync();  // 同步

grid_group grid = this_grid();
grid.sync();  // Grid 同步
```

## 3. Warp 级协作

```cuda
thread_block_group warp = this_thread_block().group(16);  // 指定 warp
warp.sync();
```

## 4. 归约操作

```cuda
collective_sum<int> sum(this_thread_block());
```

## 参考文献

- [Cooperative Groups API](../ref/cooperative_groups.html)
