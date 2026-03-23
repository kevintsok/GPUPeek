# Memory Research Module

## 概述

内存子系统研究测试，包括全局内存带宽、L1/L2缓存性能、共享内存、跨距访问等。

## 独立编译和运行

```bash
# 1. 创建构建目录
cd src/sm_120/memory
mkdir -p build && cd build

# 2. 配置项目
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90

# 3. 编译
cmake --build . --config Release

# 4. 运行
./gpupeek_memory [元素数量]
# 例如: ./gpupeek_memory 1048576
```

## 文件

- `memory_research_kernel.cu` - 内存操作内核
- `memory_research_benchmarks.cu` - 内存基准测试
- `main.cu` - 主程序入口
- `CMakeLists.txt` - 构建配置

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
ncu --set full --metrics dram__bytes.sum ./gpupeek_memory

# 详细指标
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek_memory
```
