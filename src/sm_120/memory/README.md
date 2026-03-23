# Memory Research Module

## 概述

内存子系统研究测试，包括全局内存带宽、L1/L2缓存性能、共享内存、跨距访问等。

## 文件

- `memory_research_kernel.cu` - 内存操作内核
- `memory_research_benchmarks.cu` - 内存基准测试

## 编译

```bash
cd D:/Projects/dissecting-sm110
cmake --build build --config Release
```

## 运行

```bash
# 运行内存研究测试
./build/gpupeek.exe memory
```

## 测试内容

| 测试项 | 描述 |
|--------|------|
| 全局内存带宽 | 不同数据大小的顺序读写带宽 |
| L1/L2缓存带宽 | 分层内存访问性能 |
| 共享内存性能 | 共享内存读写及广播 |
| 跨距访问效率 | Stride访问对带宽的影响 |
| TMA拷贝 | 张量内存访问器性能 |
| PCIe带宽 | 主机与设备间传输带宽 |

## NCU 分析

```bash
# 内存带宽分析
ncu --set full --metrics dram__bytes.sum ./build/gpupeek.exe memory

# 详细指标
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./build/gpupeek.exe memory
```
