# Multi-Stream Research

## 概述

Multi-Stream 并发执行实现计算与内存操作重叠。

## 1. Stream 基本概念

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 不同流执行
kernel1<<<..., stream1>>>(...);
kernel2<<<..., stream2>>>(...);
```

## 2. 流依赖

```cuda
cudaEvent_t event;
cudaEventCreate(&event);
kernel1<<<..., stream1>>>(...);
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<..., stream2>>>(...);
```

## 3. 重叠执行

| 配置 | 带宽 | 重叠效果 |
|------|------|----------|
| Single Stream | ~450 GB/s | 无重叠 |
| 2 Streams (Compute + Copy) | ~820 GB/s | 高重叠 |
| 4 Streams (Compute + Copy) | ~950 GB/s | 最佳重叠 |
| 2 Streams (Compute + Compute) | ~680 GB/s | 中重叠 |
| 8 Streams (Mixed) | ~880 GB/s | 良好重叠 |

**可视化图表**: `data/stream_overlap.png`

## 4. 流优先级

```cuda
cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
```

## 5. 最佳实践

1. **使用 streams 重叠计算和内存操作**: 带宽提升可达 2x
2. **避免过度使用 streams**: 资源竞争反而降低性能
3. **使用 cudaEvent 精确控制依赖**: 避免不必要的等待
4. **考虑流优先级**: 高优先级流可获得更多资源

## 6. 图表生成

运行以下脚本生成可视化图表:

```bash
cd scripts
pip install -r requirements.txt
python plot_multi_stream.py
```

输出位置: `NVIDIA_GPU/sm_120/multi_stream/data/`

## 参考文献

- [CUDA Programming Guide - Streams](../ref/cuda_programming_guide.html)
