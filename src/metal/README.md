# Apple Metal GPU Benchmark

Apple M系列芯片GPU基准测试，使用Metal API进行深度研究。

## 研究进度

| 阶段 | 主题 | 状态 | 关键发现 |
|------|------|------|----------|
| Phase 1 | 基准测试优化 | ✅ 完成 | API开销非瓶颈，带宽~1 GB/s |
| Phase 2 | 内存子系统 | ✅ 完成 | 跨步访问慢2.3x，写入比读取快 |
| Phase 3 | 计算吞吐量 | ✅ 完成 | 分块MatMul达9.11 GFLOPS |
| Phase 4 | 并行计算特性 | ✅ 完成 | 线程组效率高，原子操作可扩展 |
| Phase 5 | 架构深入分析 | ✅ 完成 | 随机访问慢27x，内存延迟可流水线化 |
| Phase 6 | 跨架构对比 | ✅ 完成 | Apple vs NVIDIA设计哲学差异 |

## 目录结构

```
metal/
├── Phase1_Benchmark_Optimization/      # 基准测试优化
├── Phase2_Memory_Subsystem/          # 内存子系统
├── Phase3_Compute_Throughput/         # 计算吞吐量
├── Phase4_Parallel_Computing/        # 并行计算特性
├── Phase5_Architecture_Deep_Dive/     # 架构深入分析
├── Phase6_Cross_Architecture_Comparison/ # 跨架构对比
├── Sources/                           # Swift Package源码
├── ref/                               # 参考文档
├── Package.swift                      # Swift Package配置
├── RESEARCH.md                        # 统一研究报告
└── README.md                          # 本文件
```

## 快速开始

```bash
cd src/metal

# 编译并运行
swift build --configuration release
swift run --configuration release
```

## 关键发现摘要

### 内存架构
- **统一内存带宽**: 理论100 GB/s，实测~1-2 GB/s (~1%利用率)
- **写入优化**: 写入比读取快2.9倍 (Apple写合并)
- **访问模式**: 顺序访问比随机访问快27倍

### 计算性能
- **FP32 MatMul (分块)**: 9.11 GFLOPS
- **FP16 vs FP32**: 向量操作快3x，计算操作相近
- **线程组大小**: 256+最优

### 架构对比
- **Apple M2**: 统一内存，25W，效率优先
- **NVIDIA RTX 4090**: 独立显存，450W，吞吐量优先

## 参考资料

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Metal Shading Language Specification](ref/Metal_Shading_Language_Specification.pdf)
