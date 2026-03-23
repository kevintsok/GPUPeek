# GPUPeek - CUDA GPU Benchmark Framework

## 项目概述

GPUPeek是一个用于探索GPU机制和性能指标的CUDA基准测试框架。

## 项目结构

```
GPUPeek/
├── CMakeLists.txt
├── CLAUDE.md              # 项目规则和约定
├── README.md
├── ref/                   # NVIDIA官方文档
├── docs/                  # 研究报告
└── src/
    ├── common/            # 通用代码
    ├── generic/           # 通用内核（所有GPU可用）
    ├── metal/             # Apple Metal GPU (M系列) 研究代码
    └── sm_120/            # SM 12.0 (Blackwell) 特定代码
```

## 重要约定

### 研究文档要求

**⚠️ 重要：所有GPU架构研究的数据和基准测试结果必须记录到相应的文档中，这是强制要求！**

1. **每架构Research文件**: 每个`src/sm_XX/`目录必须包含`RESEARCH.md`文件
   - 硬件规格（GPU型号、计算能力、SM数量、内存等）
   - 基准测试结果（所有性能指标及分析）
   - 架构特性（独特能力和特性）
   - 与前代架构对比（如果有）
   - 测试环境（CUDA版本、驱动、操作系统、编译选项）
   - **每次研究循环后，必须更新此文件记录新数据和发现**

2. **最终研究报告**: 项目达到重要里程碑时，撰写综合研究报告
   - 输出位置: `docs/RESEARCH_REPORT.md`
   - 内容: 汇总所有架构研究发现，提供跨架构性能对比、关键洞察和建议

### 研究循环要求

在每次研究循环中：
1. 运行基准测试并收集数据
2. **必须将数据记录到 `src/sm_120/RESEARCH.md`**
3. 分析结果并更新文档中的发现
4. 提交代码时同时提交更新的文档

## 架构支持

- `metal/` - Apple Metal (M1/M2/M3/M4系列)
- `sm_120/` - Blackwell (RTX 5080, RTX 5070等)
- `sm_90/` - Ada Lovelace (RTX 4090, RTX 4080等)
- `sm_80/` - Ampere (RTX 3090, A100等)
- `sm_70/` - Volta/Vega (V100等)

## 构建和运行

```bash
# CMake构建
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release

# 运行
./build/gpupeek        # 所有基准测试
./build/gpupeek generic # 仅通用基准测试
./build/gpupeek arch    # 仅架构特定基准测试
```
