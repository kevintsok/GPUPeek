# Apple Metal GPU Benchmarks

本目录包含Apple M2 GPU的各项专题基准测试，每个子目录都有独立的Benchmark代码和对应的研究报告。

## 目录结构

```
Benchmarks/
├── Memory/
│   ├── Bandwidth/          # 内存带宽测试
│   ├── Coalescing/        # 内存合并访问
│   ├── BankConflict/       # 共享内存bank冲突
│   └── LatencyHiding/      # 内存延迟隐藏
├── Compute/
│   ├── GEMM/               # 矩阵乘法
│   ├── Convolution/        # 卷积运算
│   ├── Vectorization/      # 向量化优化
│   ├── FP64/               # 双精度测试
│   └── InstructionMix/     # 指令吞吐量
├── Synchronization/
│   ├── Atomics/            # 原子操作
│   ├── Barriers/           # 同步屏障
│   └── WarpPrimitives/     # SIMD组操作
├── Algorithms/
│   ├── Sorting/            # 排序算法
│   ├── FFT/                # 快速傅里叶变换
│   ├── Graph/              # 图算法
│   ├── Scan/               # 并行扫描
│   ├── Histogram/          # 直方图
│   └── Stencil/            # 模板计算
├── Analysis/
│   ├── Occupancy/          # 占用率分析
│   ├── Cache/              # 缓存行为
│   ├── Precision/          # 数值精度
│   ├── Texture/            # 纹理性能
│   └── Architecture/        # 架构查询
└── Optimization/
    ├── KernelFusion/       # 内核融合
    ├── CommandBuffer/      # 命令缓冲批处理
    ├── DoubleBuffer/       # 双缓冲
    └── Roofline/           # Roofline模型
```

## 运行基准测试

主程序位于 `../Sources/MetalBenchmark/main.swift`，包含所有84个基准测试专题：

```bash
cd ../Sources/MetalBenchmark
swift build
swift run MetalBenchmark
```

## 研究报告与日志

每个benchmark目录下包含：

| 文件 | 说明 |
|------|------|
| `Benchmark.swift` | 独立基准测试代码 |
| `RESEARCH.md` | 详细研究报告和分析 |
| `LOG.txt` | 基准测试运行日志 |

日志文件 `_logs/full_run.log` 包含完整的84个专题运行输出。

## 关键发现汇总

### 内存优化最重要

1. **Burst Write** - 16元素/线程可达到6.17 GB/s (3-4x提升)
2. **Float4向量化** - 读取可达3.79 GB/s (4x提升)
3. **内存合并** - 合并访问比非合并快5.3x

### 计算优化

1. **GEMM Tiled** - 共享内存分块可实现2-5x加速
2. **FP16** - 比FP32快约2x（tiled实现）
3. **Half4** - 最高效的向量格式

### 同步开销

1. **Barrier** - 固定开销约4.8μs
2. **Kernel Launch** - 约0.5μs
3. **Atomic** - 高争用时性能下降2.5x

## 与主文档的关系

主研究报告 `../RESEARCH.md` 包含所有84个专题的完整数据，本目录的 `RESEARCH.md` 文件是从主文档中提取的各专题详细内容。
