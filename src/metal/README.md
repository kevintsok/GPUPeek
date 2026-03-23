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

## 硬件支持

| 芯片 | GPU核心 | 统一内存带宽 | 状态 |
|------|---------|-------------|------|
| M1 | 8核 | 68.25 GB/s | 待测试 |
| M2 | 10核 | 100 GB/s | 已测试 |
| M3 | 10核 | 100 GB/s | 待测试 |
| M4 | 10核 | 120 GB/s | 待测试 |
| M2 Max | 38核 | 400 GB/s | 待测试 |
| M3 Max | 40核 | 400 GB/s | 待测试 |
| M4 Max | 40核 | 546 GB/s | 待测试 |

## 关键发现

### 内存架构
- **统一内存带宽**: 理论100 GB/s，实测~1-2 GB/s (~1%利用率)
- **写入优化**: 写入比读取快2.9倍 (Apple写合并)
- **访问模式**: 顺序访问比随机访问快27倍
- **内存压缩**: 影响极小，难以测量

### 计算性能
- **FP32 MatMul (分块)**: 9.11 GFLOPS
- **FP16 vs FP32**: 仅快5%，无明显张量单元优势
- **FMA**: 0.22 GFLOPS (内存受限)
- **线程组大小**: 64-1024对性能影响极小

### 并行特性
- **原子操作**: 0.016-0.57 GOPS (随争用扩展)
- **线程分歧**: 10-15%性能变化
- **屏障开销**: 单次4.8μs，流水线后89ns

## 快速开始

### 前置要求

- macOS 13.0+
- Xcode Command Line Tools
- Swift 6.0+

```bash
# 检查工具
swift --version
xcode-select -p
```

### 编译运行

```bash
cd src/metal

# Debug模式编译
swift build

# Release模式编译（推荐，更快）
swift build --configuration release

# 运行
swift run --configuration release
```

## 研究报告

完整报告位于 `docs/` 目录：

| 报告 | 内容 |
|------|------|
| Phase 1 Report | API优化、带宽测试结果 |
| Phase 2 Report | 内存访问模式、原子操作 |
| Phase 3 Report | 矩阵乘法、计算吞吐量 |
| Phase 4 Report | 线程组、SIMD、屏障性能 |
| Phase 5 Report | 架构深入分析、内存特性 |

## 项目结构

```
metal/
├── README.md                      # 本文件
├── RESEARCH.md                     # 研究文档和发现
├── Package.swift                  # Swift Package配置
├── Sources/
│   └── MetalBenchmark/
│       └── main.swift             # 主程序和Metal Shader (Phase 5)
├── docs/                          # 研究报告
│   ├── Phase1_*.md               # 英文报告
│   └── Phase*_*.md               # 中文报告
├── ref/                          # 参考文档
│   └── Metal_Shading_Language_Specification.pdf
└── bandwidth_test.metal           # 备用内核
```

## 运行结果示例

```
Apple Metal GPU Benchmark - Phase 5: Architecture Deep Dive
======================================

=== Apple Metal GPU Info ===
Device Name: Apple M2
GPU Family: Apple 7+

--- Access Patterns ---
Sequential Access: 0.88 GB/s (baseline)
Random Access: 0.03 GB/s (27x slower)

--- Bandwidth Stress ---
Read: 0.62 GB/s
Write: 1.80 GB/s

--- Memory Latency ---
Single access: 61.27 ns/op
Pipelined: 0.45 ns/op
```

## 性能优化建议

### 内存访问
- ✅ 使用顺序访问模式
- ✅ 使用float4向量化
- ✅ 批量内存操作
- ❌ 避免随机访问
- ❌ 避免小内存操作

### 计算优化
- ✅ 使用分块算法利用共享内存
- ✅ 平衡计算与内存操作
- ✅ 使用FMA融合操作
- ❌ 不要假设FP16有显著优势

### 并行编程
- ✅ 使用256线程组作为基准
- ✅ 分布原子操作减少争用
- ✅ 流水线屏障分摊开销
- ❌ 避免单热点原子

## Apple vs NVIDIA 架构对比

| 指标 | Apple M2 | NVIDIA RTX 4090 | 备注 |
|------|----------|-----------------|------|
| 内存带宽 | 100 GB/s | 1008 GB/s | 10x差距 |
| 实测带宽 | ~1-2 GB/s | ~800 GB/s | 统一内存开销 |
| FP32 MatMul | 9.11 GFLOPS | ~1000+ GFLOPS | 100x差距 |
| 内存类型 | 统一内存 | GDDR6X | 架构本质不同 |
| 能效 | 高 | 中 | Apple优势 |

> 关键洞察：Apple M2的统一内存架构消除了CPU-GPU数据传输，但共享带宽导致实际吞吐量远低于独立GPU。

## 参考资料

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Metal Shader Language Specification](ref/Metal_Shading_Language_Specification.pdf)
- [WWDC20: Harness Apple GPUs with Metal](https://developer.apple.com/videos/play/wwdc2020/10602/)
- [WWDC20: GPU Performance Counters](https://developer.apple.com/videos/play/wwdc2020/10603/)
