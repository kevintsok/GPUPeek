# GPUPeek - CUDA GPU Benchmark Framework

## 项目概述

GPUPeek是一个用于探索GPU机制和性能指标的CUDA基准测试框架。

## 标准目录结构（复用于所有架构和GPU厂商）

```
GPUPeek/
├── CMakeLists.txt              # 全局构建配置
├── README.md                   # 项目概览和快速开始
├── CLAUDE.md                   # 项目规则和约定
├── docs/                       # 研究报告
├── NVIDIA_GPU/                 # NVIDIA GPU 代码
│   ├── common/                 # 共享工具（main.cu, gpu_info, timer）
│   ├── generic/                # 通用内核（所有NVIDIA GPU可用）
│   ├── ref/                    # NVIDIA 官方文档
│   ├── COMPARISON.md           # 跨代GPU对比 (EN)
│   ├── COMPARISON_CN.md        # 跨代GPU对比 (CN)
│   └── sm_120/                 # SM 12.0 (Blackwell)
│       ├── arch.cu             # 架构信息
│       ├── arch_kernels.cu     # 架构专用内核
│       ├── benchmarks.cu       # Benchmark 运行器
│       └── [module]/           # 研究模块（独立可编译）
│           ├── CMakeLists.txt  # 模块构建配置
│           ├── main.cu         # 模块入口
│           ├── README.md       # 操作指南
│           ├── RESEARCH.md     # 研究成果和教学材料
│           └── *_kernel.cu     # 内核源码
└── APPLE_GPU/                  # Apple GPU 代码（未来扩展）
    └── ...
```

### 研究模块标准模板

每个研究模块是完全独立可编译的目录，包含：

| 文件 | 用途 |
|------|------|
| `CMakeLists.txt` | 独立编译配置 |
| `main.cu` | 模块入口点 |
| `README.md` | 操作指南（如何编译运行） |
| `RESEARCH.md` | 教学材料和研究成果 |
| `*_kernel.cu` | 内核源码 |
| `*_benchmarks.cu` | benchmark函数 |

**复用规则**：新加架构（如 `sm_90/`）或新厂商（如 `APPLE_GPU/`）时，直接复用此目录结构。

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

### Benchmark 设计规范

**⚠️ 重要：所有 benchmark 必须包含尺寸扫描和指令变种对比，并生成可视化图表！**

1. **尺寸扫描 (Size Sweep)**：
   - 测试多个不同数据尺寸，覆盖关键拐点
   - 例如：1KB, 64KB, 1MB, 4MB, 16MB, 64MB, 256MB
   - 尺寸应覆盖 L1/L2/DRAM 等内存层级边界

2. **指令变种 (Instruction Variants)**：
   - 对比多种实现或指令类型
   - 例如：FP32 vs FP64 vs FP16 vs BF16
   - 例如：shuffle vs redux.sync vs butterfly

3. **生成图表**：
   - 工具：Python + matplotlib/seaborn
   - 格式：PNG/SVG 输出到模块目录
   - 同时保存 CSV 原始数据到 `data/` 子目录
   - 图表规范：
     - X轴：数据尺寸或问题规模
     - Y轴：TFLOPS / 吞吐带宽 / 延迟
     - 多系列：不同指令变种用不同颜色/线型标注
     - 必须有图例 (legend)、坐标轴标签、标题

4. **Benchmark 输出结构**：
   ```
   [module]/
   ├── data/
   │   ├── benchmark_results.csv   # 原始数据
   │   └── ...
   ├── throughput_vs_size.png     # 吞吐 vs 尺寸图
   ├── latency_vs_size.png       # 延迟 vs 尺寸图
   └── ...                       # 其他图表
   ```

5. **图表生成脚本**：
   - 在 `scripts/` 目录创建 Python 脚本
   - 脚本命名：`plot_[module]_[test].py`
   - 输出到对应模块目录

**示例 benchmark 设计**：
```
Benchmark: Warp Reduction Performance
├── Size sweep: 32, 64, 128, 256, 512, 1024 elements
├── Variants: shuffle_add, butterfly_add, redux_add
└── Output: throughput_vs_elements.png, latency_vs_elements.png
```

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

---

## 研究方法论

本项目定义了一套可复用的 GPU 研究方法论，适用于任何 NVIDIA GPU 架构。

### 研究循环流程

每次研究循环包含以下 4 个步骤：

```
┌─────────────────────────────────────────────────────────────┐
│  1. 探索 (Explore)                                        │
│     - 阅读现有代码和文档，理解当前研究状态                    │
│     - 识别可以深入研究的专题                                 │
│     - 查阅 NVIDIA 官方文档 (PTX ISA, CUDA Programming Guide)│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 计划与执行 (Plan & Execute)                           │
│     - 制定研究计划，选择具体专题                             │
│     - 实现 kernel 或 benchmark                              │
│     - 编译并运行测试，收集数据                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 记录 (Document)                                       │
│     - 将数据记录到 RESEARCH.md                             │
│     - 生成图表 (PNG) 和原始数据 (CSV)                       │
│     - 更新关键发现和分析                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 提交 (Commit)                                         │
│     - 检查代码质量和文档完整性                               │
│     - 更新 README 如果需要                                 │
│     - git commit 并 push 到 GitHub                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
                          回到步骤 1
```

### 内核实现规范

#### 1. 基础 kernel 模板

```cuda
// [instruction]_research_kernel.cu
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// [Instruction Name] Research
// =============================================================================
//
// PTX ISA: Section reference
// SASS: Instruction name
//
// Key concepts:
// - What the instruction does
// - Important constraints or requirements
// =============================================================================

// Basic kernel template
template <typename T>
__global__ void [instruction]_basic_kernel(const T* __restrict__ input,
                                            T* __restrict__ output,
                                            size_t size) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;

    size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    size_t start = bid * per_block;
    size_t end = min(start + per_block, size);

    for (size_t i = start + tid; i < end; i += block_size) {
        // Kernel logic here
        output[i] = input[i];
    }
}

// Size sweep kernel - tests multiple data sizes
template <typename T>
__global__ void [instruction]_size_sweep_kernel(const T* __restrict__ input,
                                                  T* __restrict__ output,
                                                  size_t* sizes,
                                                  size_t num_sizes) {
    // Size-dependent behavior test
}

// Variant kernel - different implementation or instruction variant
template <typename T>
__global__ void [instruction]_variant_kernel(const T* __restrict__ input,
                                             T* __restrict__ output,
                                             size_t size) {
    // Alternative implementation
}
```

#### 2. Benchmark 函数模板

```cuda
// [instruction]_research_benchmarks.cu
#include "timer.h"

void run[Instruction]Tests() {
    printf("\n=== [Instruction] Tests ===\n");

    const size_t N = 1 << 20;  // Default 1M elements
    const int iterations = 100;

    // Allocate memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    GPUTimer timer;

    // Test 1: Basic
    timer.start();
    for (int i = 0; i < iterations; i++) {
        [instruction]_basic_kernel<<<gridDim, blockDim>>>(d_input, d_output, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("Basic: %.2f GB/s\n", bytes * iterations / timer.elapsed_ms() / 1e6);

    // Test 2: Size sweep
    printf("\n--- Size Sweep ---\n");
    size_t sizes[] = {1<<10, 1<<12, 1<<14, 1<<16, 1<<20, 1<<24};
    for (auto size : sizes) {
        // Run benchmark at this size
    }

    cudaFree(d_input);
    cudaFree(d_output);
}
```

#### 3. Size Sweep 基准测试模式

```cuda
// Size sweep benchmark - runs at multiple data sizes
// Outputs CSV format for easy parsing
void runSizeSweepBenchmark() {
    printf("size_bytes,bandwidth_gb_s,latency_ms\n");

    size_t sizes[] = {
        1 << 10,   // 1KB
        1 << 12,   // 4KB
        1 << 14,   // 16KB
        1 << 16,   // 64KB
        1 << 18,   // 256KB
        1 << 20,   // 1MB
        1 << 22,   // 4MB
        1 << 24,   // 16MB
        1 << 26,   // 64MB
        1 << 28,   // 256MB
    };

    for (size_t N : sizes) {
        // Run benchmark
        printf("%zu,%.2f,%.3f\n", N * sizeof(float), bandwidth, latency);
    }
}
```

### 文档记录规范

#### RESEARCH.md 必须包含

```markdown
# [模块名称] Research

## 概述
- 研究目标
- 关键发现摘要

## 硬件规格
- GPU型号
- 计算能力 (SM 版本)
- SM 数量
- 内存大小和带宽
- Warp Size

## 基准测试结果

### 1. [测试类别名]
| 变体 | 带宽/吞吐 | 延迟 | 条件 |
|------|-----------|------|------|
| variant1 | 123 GB/s | 0.5 ms | 1MB数据 |

### 2. Size Sweep 结果
| Size | 性能指标 | ... |
|------|----------|------|
| 1KB | ... | ... |

## 架构特性
- 重要观察
- 指令限制
- 特殊行为

## 与前代架构对比
| 特性 | 本架构 | 前代架构 |
|------|--------|----------|
| ... | ... | ... |

## 测试环境
- CUDA 版本
- 驱动版本
- 操作系统
- 编译选项
```

### 数据文件结构

```
[module]/
├── data/
│   ├── benchmark_results.csv     # 原始数据
│   ├── size_sweep.csv          # Size sweep 数据
│   └── [specific_test].csv     # 特定测试数据
├── charts/
│   ├── bandwidth_vs_size.png   # 带宽 vs 尺寸
│   ├── latency_vs_size.png     # 延迟 vs 尺寸
│   └── comparison.png          # 对比图
└── RESEARCH.md                  # 研究报告
```

### 图表生成脚本规范

```python
#!/usr/bin/env python3
"""
[Module] Chart Generator
=====================
Generates charts for [module] benchmark results.

Usage:
    python3 plot_[module].py

Output:
    ../NVIDIA_GPU/sm_XX/[module]/data/*.png
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
DATA = [
    # (size, value1, value2, ...)
]

def plot_bandwidth_vs_size():
    """Generate bandwidth vs data size chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = [d[0] for d in DATA]
    values = [d[1] for d in DATA]

    ax.plot(sizes, values, 'o-', label='Method')
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Bandwidth vs Data Size\n[GPU Name]')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../NVIDIA_GPU/sm_XX/[module]/data/bandwidth_vs_size.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    plot_bandwidth_vs_size()
```

---

## 在新 GPU 上启动研究

### 1. 克隆项目

```bash
git clone https://github.com/kevintsok/GPUPeek.git
cd GPUPeek
```

### 2. 创建新架构目录

```bash
# 复制模板目录结构
cp -r NVIDIA_GPU/sm_120 NVIDIA_GPU/sm_XX

# 重命名目录中的特定文件
cd NVIDIA_GPU/sm_XX
mv sm_120_arch sm_XX_arch  # 或创建新的
```

### 3. 识别目标 GPU

```bash
# 运行 GPU 信息工具
./common/gpu_info  # 如果存在

# 或使用 nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

### 4. 更新架构信息

编辑 `NVIDIA_GPU/sm_XX/arch.cu`:
```cuda
// 架构信息
const char* getArchitectureName() { return "sm_XX"; }
const char* getGPUName() { return "Your GPU Name"; }
int getComputeCapabilityMajor() { return XX; }
int getComputeCapabilityMinor() { return Y; }
```

### 5. 开始研究循环

```bash
# 编译一个已知模块测试构建环境
cd NVIDIA_GPU/sm_XX/memory
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX
cmake --build . --config Release

# 运行测试
./gpupeek_memory

# 验证结果合理后，开始新模块研究
```

### 6. 编译问题排查

| 问题 | 解决方案 |
|------|----------|
| nvcc not found | 设置 CUDA PATH: `export PATH=/usr/local/cuda/bin:$PATH` |
| cl.exe not found (Windows) | 使用 VS Developer Command Prompt 或设置 MSVC PATH |
| CMake can't find CUDA | 指定 `-DCUDAToolkit_ROOT=/path/to/cuda` |
| Architecture not supported | 检查 GPU .compute capability vs CMake 设置 |

### 7. 研究优先级建议

对于新 GPU，建议按以下顺序研究：

1. **memory** - 内存带宽和缓存层级 (最基础)
2. **cuda_core** - 基础计算吞吐 (FP32/FP64/INT)
3. **wmma** - Tensor Core MMA 性能
4. **shuffle/redux_sync** - Warp 级原语
5. **barrier** - 同步开销
6. **atomic** - 原子操作
7. **tensor_mem** - LDMATRIX/STMATRIX/cp.async

---

## 关键 PTX/SASS 参考

### 常用指令映射

| PTX | SASS | 说明 |
|-----|------|------|
| `shfl.sync` | SHFL | Warp shuffle |
| `redux.sync` | RRED | Warp reduction sync |
| `ldmatrix` | LDMATRIX | 矩阵加载 |
| `stmatrix` | STMATRIX | 矩阵存储 |
| `cp.async` | CP.ASYNC | 异步拷贝 |
| `mma.sync` | HMMA/IMMA | MMA 计算 |
| `bar.sync` | BAR | 屏障同步 |
| `membar` | MEMBAR | 内存屏障 |

### Inline PTX 注意事项

1. **指针约束**: 64位系统用 `l` 约束而非 `r`
   ```cuda
   // 正确
   asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
       : "=l"(dst), "=l"(src) : "l"(dst), "l"(src));

   // 错误 (32位约束用于64位指针)
   asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
       : "=r"(dst), "=l"(src) : "r"(dst), "l"(src));  // 编译警告/错误
   ```

2. **WMMA fragment**: 只支持特定 shape (16x16x16, 8x8x16 等)
   ```cuda
   // 正确
   fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;

   // 错误 (8x8x8 不存在)
   fragment<matrix_a, 8, 8, 8, __half, row_major> frag_a;  // 编译错误
   ```

3. **wait_cnt**: 内存ordering需要正确设置
   ```cuda
   asm volatile("waitcnt 0;" : : );  // 等待所有pending loads
   ```

---

## 常见问题排查

| 症状 | 可能原因 | 解决方案 |
|------|---------|----------|
| `illegal memory access` | 数组越界或未对齐访问 | 检查边界条件，使用 `__ldg()` 对齐加载 |
| GPU 重启/挂起 | Inline PTX 错误或无限循环 | 检查 asm 语句约束，减少 block 大小 |
| 带宽异常低 | kernel 启动开销主导数据太小 | 使用更大的数据尺寸或更多迭代 |
| `cudaErrorUnknown` | GPU 不兼容或驱动问题 | 更新驱动，检查 GPU  compute mode |
