# GPUPeek - CUDA GPU Benchmark Framework

## 项目概述

GPUPeek是一个用于探索GPU机制和性能指标的CUDA基准测试框架。

## 项目结构

```
GPUPeek/
├── CMakeLists.txt
├── CLAUDE.md              # 项目规则和约定
├── README.md
├── docs/                  # 研究报告
├── include/               # 头文件
├── NVIDIA_GPU/           # NVIDIA GPU 相关代码
│   ├── ref/             # NVIDIA 官方文档
│   ├── common/           # 通用代码
│   ├── generic/          # 通用内核（所有GPU可用）
│   └── sm_120/          # SM 12.0 (Blackwell) 特定代码
│       ├── memory/      # 内存研究（独立可编译）
│       ├── wmma/        # WMMA 研究
│       ├── cuda_core/    # CUDA Core 算力研究
│       ├── atomic/       # 原子操作研究
│       ├── barrier/      # Barrier 同步研究
│       └── [其他模块]...  # 更多研究模块
└── APPLE_GPU/            # Apple GPU 代码（未来扩展）
```

## 重要约定

### 研究文档要求

**⚠️ 重要：所有GPU架构研究的数据和基准测试结果必须记录到相应的文档中，这是强制要求！**

1. **每模块Research文件**: 每个研究模块目录必须包含`RESEARCH.md`文件
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
2. **必须将数据记录到对应模块的 `RESEARCH.md`**
3. 分析结果并更新文档中的发现
4. 提交代码时同时提交更新的文档

## 架构支持

- `NVIDIA_GPU/sm_120/` - Blackwell (RTX 5080, RTX 5070等)
- `NVIDIA_GPU/sm_90/` - Ada Lovelace (RTX 4090, RTX 4080等)
- `NVIDIA_GPU/sm_80/` - Ampere (RTX 3090, A100等)
- `NVIDIA_GPU/sm_70/` - Volta/Vega (V100等)

## 独立模块构建

每个研究模块可以独立编译和运行：

```bash
# 进入模块目录
cd NVIDIA_GPU/sm_120/memory

# 创建构建目录
mkdir -p build && cd build

# 配置和编译
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release

# 运行
./gpupeek_memory [元素数量]
```

## 全局构建（可选）

```bash
# CMake构建
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
```
