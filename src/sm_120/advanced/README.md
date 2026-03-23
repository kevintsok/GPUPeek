# Advanced Research Module

## 概述

高级研究测试，包括 Constant Memory、Bank Conflict 分析等。

## 文件

- `advanced_research_kernel.cu` - 高级研究内核
- `advanced_research_benchmarks.cu` - 高级研究基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
./build/gpupeek.exe advanced
```

## 测试内容

| 测试项 | 描述 |
|--------|------|
| Constant Memory | 常量内存访问性能 |
| Bank Conflict 分析 | 共享内存银行冲突 |
| Atomic Operations | 原子操作性能 |
| Memory Fence | 内存栅栏影响 |

## NCU 分析

```bash
# 共享内存银行冲突
ncu --set full --metrics sm__shared_bank_conflict_throughput ./build/gpupeek.exe advanced
```
