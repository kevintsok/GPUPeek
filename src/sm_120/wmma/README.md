# WMMA (Warp-level MMA) Research Module

## 概述

WMMA (Warp-level Matrix Multiply-Accumulate) 是标准的 CUDA Tensor Core API，可在所有现代 NVIDIA GPU 上运行。

## 文件

- `wmma_mma_kernel.cu` - WMMA 内核 (含 cycle counting)
- `wmma_mma_benchmarks.cu` - WMMA 基准测试
- `wmma_test_kernel.cu` - WMMA 测试内核
- `wmma_test_benchmarks.cu` - WMMA 测试基准
- `mma_research_kernel.cu` - MMA 研究内核
- `mma_research_benchmarks.cu` - MMA 研究基准

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
# 运行所有 WMMA 测试
./build/gpupeek.exe wmma

# 运行 MMA 研究测试
./build/gpupeek.exe mma
```

## WMMA Shape 支持 (m16n16k16)

| 数据类型 | 支持 |
|----------|------|
| FP16 | ✅ |
| BF16 | ✅ |
| TF32 | ✅ |
| FP64 | ✅ |
| INT8 | ✅ |

## 核心指标

| 指标 | 含义 |
|------|------|
| sm__pipe_tensor_cycles_active.pct | Tensor Core 利用率 |
| sm__inst_executed.sum | 执行指令数 |
| dram__bytes.sum | 内存带宽 |

## NCU 分析

```bash
# Tensor Core 利用率
ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./build/gpupeek.exe wmma

# MMA 指令吞吐量
ncu --set full --metrics sm__inst_executed.mma.sum ./build/gpupeek.exe wmma
```

## 数据需求 (m16n16k16 per warp)

| 操作 | 数据量 |
|------|--------|
| load_matrix_sync (A) | 512 bytes (256 halfs) |
| load_matrix_sync (B) | 512 bytes (256 halfs) |
| mma_sync | 256 FMA ops |
| store_matrix_sync | 1024 bytes (256 floats) |
