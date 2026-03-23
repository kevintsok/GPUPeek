# Barrier Synchronization Research

## 概述

Barrier 同步机制研究，测量 __syncthreads() 开销和 barrier stall。

## 1. __syncthreads()

```cuda
__syncthreads();
```

所有线程在屏障点同步，确保所有线程到达后再继续。

## 2. 开销因素

| 因素 | 影响 |
|------|------|
| Block 大小 | 越大开销越大 |
| 线程分歧 | 分歧越大等待越长 |
| 共享内存访问 | 可能导致 stall |

## 3. Block Size vs 效率

| Block Size | 效率 |
|------------|------|
| 32 | 高 |
| 64 | 高 |
| 128 | 中 |
| 256 | 中低 |
| 512 | 低 |
| 1024 | 最低 |

**建议**: 根据计算密度选择合适的 Block Size

## 4. bar.sync 指令

PTX 中的 barrier 指令:
```ptx
bar.sync 0;  // CTA barrier
bar.red.popc.gpu.s32 ...;  // reduction barrier
```

## 5. NCU 指标

| 指标 | 含义 |
|------|------|
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall 比例 |
| sm__average_active_warps_per_sm | 每SM活跃warp |

## 参考文献

- [CUDA Programming Guide - Synchronization](../ref/cuda_programming_guide.html)
