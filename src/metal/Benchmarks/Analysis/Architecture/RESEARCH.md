# GPU Architecture Research

## 概述

本专题记录Apple M2 GPU的架构规格和实测数据。

## Apple M2 GPU规格

| 规格 | 值 | 说明 |
|------|-----|------|
| GPU型号 | Apple M2 | 8核GPU (7核可用) |
| Threadgroup内存 | 32 KB | 最大共享内存限制 |
| 最大Buffer大小 | 8.88 GB | MaxBufferLength |
| 最大Working Set | 11.84 GB | RecommendedMaxWorkingSetSize |
| 统一内存 | 是 | CPU/GPU共享内存 |
| SIMD宽度 | 32 | 固定值 |
| GPU Family | Apple 7 | M2所属GPU家族 |

## 关键洞察

1. **统一内存架构** - CPU和GPU共享内存
2. **32KB共享内存限制** - Threadgroup内存上限
3. **SIMD宽度固定32** - 与NVIDIA warp相同
4. **GPU Family 7** - M2属于Apple GPU Family 7

## 相关专题

- [Occupancy](../Occupancy/RESEARCH.md) - 占用率分析
