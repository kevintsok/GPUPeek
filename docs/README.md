# GPUPeek 文档目录

## 研究报告

### 综合研究报告

| 文件 | 语言 | 描述 |
|------|------|------|
| [RESEARCH_REPORT_EN.md](RESEARCH_REPORT_EN.md) | English | Comprehensive GPU Research Report |
| [RESEARCH_REPORT_ZH.md](RESEARCH_REPORT_ZH.md) | 中文 | GPU 综合研究报告 |

### 架构特定研究

| 文件 | 架构 | 描述 |
|------|------|------|
| [../src/sm_120/RESEARCH.md](../src/sm_120/RESEARCH.md) | SM 12.0 (Blackwell) | RTX 5080 详细研究数据 |

## 报告内容概览

### English Version (RESEARCH_REPORT_EN.md)

1. **Executive Summary** - Overview of GPUPeek framework
2. **GPU Hardware Specifications** - RTX 5080 specs and Blackwell features
3. **Memory Subsystem Analysis** - Bandwidth, hierarchy, L2 cache
4. **CUDA Core Arithmetic Research** - FP64/FP32/FP16/INT8 throughput
5. **Atomic Operations Deep Research** - Warp/Block/Grid level atomics
6. **Barrier Synchronization Research** - __syncthreads() overhead
7. **Warp Specialization Patterns** - Producer/consumer pipelines
8. **NCU Profiling Metrics** - Key metrics reference
9. **Key Findings and Recommendations** - Optimization guidance
10. **Future Research Directions** - Planned investigations

### 中文版 (RESEARCH_REPORT_ZH.md)

1. **执行摘要** - GPUPeek 框架概述
2. **GPU 硬件规格** - RTX 5080 规格和 Blackwell 特性
3. **内存子系统分析** - 带宽、层级、L2 缓存
4. **CUDA 核心算力研究** - FP64/FP32/FP16/INT8 吞吐量
5. **原子操作深入研究** - Warp/Block/Grid 级原子操作
6. **屏障同步研究** - __syncthreads() 开销分析
7. **Warp 特化模式** - 生产者/消费者流水线
8. **NCU 性能分析指标** - 关键指标参考
9. **关键发现与建议** - 优化指导
10. **未来研究方向** - 计划中的研究

## 快速链接

- **项目主页**: https://github.com/kevintsok/GPUPeek
- **问题反馈**: https://github.com/kevintsok/GPUPeek/issues
- **CUDA 文档**: https://docs.nvidia.com/cuda/

## 相关文档

- [CUDA Programming Guide](../ref/cuda_programming_guide.html) - NVIDIA 官方编程指南
- [PTX ISA](../ref/ptx_isa.html) - PTX 指令集架构
- [Blackwell Guide](../ref/blackwell_guide.html) - Blackwell 兼容性指南
