# SM 12.0 (Blackwell) GPU 架构研究报告

## 1. 目标GPU硬件规格

| 参数 | 值 |
|------|-----|
| GPU型号 | NVIDIA GeForce RTX 5080 Laptop GPU |
| 架构代号 | Blackwell |
| Compute Capability | 12.0 |
| SM数量 | 60 |
| 每SM核心数 | 128 |
| 总核心数 | 7680 |
| 全局内存 | 15.92 GB |
| 每Block共享内存 | 48 KB |
| 每Block最大线程数 | 1024 |
| 每SM最大线程数 | 1536 |
| 每Block最大寄存器数 | 65536 |
| Warp Size | 32 |

## 2. 内存子系统

### L2 Cache
- **大小**: 5 MB (估计值，RTX 5080 Laptop)
- **用途**: 高速缓存，存储频繁访问的数据

### 内存层级带宽基准测试结果 (2026-03-22 更新)

#### 专题1: Global Memory 带宽 vs 数据Size

| 数据大小 | 顺序读取 | 顺序写入 | 读写混合 |
|---------|---------|---------|---------|
| 1 KB | 0.00 GB/s | 0.00 GB/s | 0.00 GB/s |
| 64 KB | 7.25 GB/s | 7.25 GB/s | 7.25 GB/s |
| 256 KB | 32.39 GB/s | 32.39 GB/s | 32.39 GB/s |
| 1 MB | 73.97 GB/s | 73.97 GB/s | 73.97 GB/s |
| 4 MB | 296.36 GB/s | 296.36 GB/s | 296.36 GB/s |
| 16 MB | 643.02 GB/s | 643.02 GB/s | 643.02 GB/s |
| 64 MB | 376.08 GB/s | 376.08 GB/s | 376.08 GB/s |
| 128 MB | 502.44 GB/s | 502.44 GB/s | 502.44 GB/s |
| 256 MB | 614.93 GB/s | 614.93 GB/s | 614.93 GB/s |

**分析**:
- 带宽随数据大小增加而增长，在16MB时达到第一个峰值(~643 GB/s)
- 64MB时带宽下降(~376 GB/s)，可能是因为L2缓存失效
- 128-256MB后带宽回升，表明内存控制器有更大的有效缓存窗口
- 峰值带宽约 640-820 GB/s

#### 专题2: Global -> L1 -> L2 显存层级带宽

| 访问模式 | 带宽 | 备注 |
|---------|------|------|
| Global 直接读取 | 810.89 GB/s | 基线读取 |
| Global 直接写入 | 820.60 GB/s | 基线写入 |
| Shared Memory 往返 | **1.50 TB/s** | L1级带宽 |
| L2 Streaming (stride=1) | 766.78 GB/s | L2缓存命中 |
| L2 Streaming (stride=1024) | 795.17 GB/s | 跨距访问仍高效 |
| __ldg Bypass | 822.43 GB/s | 绕过缓存 |
| L1 Preference | 780.32 GB/s | 寄存器优化 |

**分析**:
- Shared Memory (L1) 带宽达 **1.50 TB/s**，远高于全局内存
- L2 Streaming 跨距访问对带宽影响较小，保持在 770-795 GB/s
- __ldg 绕过缓存的带宽略高于普通读取

#### 专题3: TMA 拷贝性能

| 数据大小 | TMA 1D拷贝 | cudaMemcpy | 加速比 |
|---------|-----------|------------|--------|
| 64 KB | 6.88 GB/s | 6.88 GB/s | 0.99x |
| 256 KB | 33.97 GB/s | 33.97 GB/s | 0.97x |
| 1 MB | 133.87 GB/s | 133.87 GB/s | 1.13x |
| 4 MB | 431.20 GB/s | 431.20 GB/s | 1.04x |
| 16 MB | 850.07 GB/s | 850.07 GB/s | 0.72x |
| 64 MB | 382.15 GB/s | 382.15 GB/s | 1.06x |
| 128 MB | 373.07 GB/s | 373.07 GB/s | 0.99x |

**2D 拷贝测试 (1024x1024, pitch=2048)**:
| 方式 | 带宽 |
|------|------|
| TMA 2D | 626.36 GB/s |
| cudaMemcpy2D | 704.31 GB/s |

**分析**:
- TMA 1D拷贝在小数据量时效率较低
- 16MB时达到峰值850 GB/s
- cudaMemcpy2D 在2D拷贝场景下仍优于自定义kernel

#### 专题4: 内存访问Pattern对性能的影响

**顺序访问基准**: 822.37 GB/s

**跨距访问效率下降**:
| Stride | 带宽 | 效率 |
|--------|------|------|
| 2 | 582.20 GB/s | 86.0% |
| 4 | 581.24 GB/s | 85.9% |
| 8 | 544.01 GB/s | 80.4% |
| 16 | 421.00 GB/s | 62.2% |
| 32 | 239.11 GB/s | 35.3% |
| 64 | 154.13 GB/s | 22.8% |
| 128 | 76.88 GB/s | 11.4% |
| 256 | 39.62 GB/s | 5.9% |

**广播写入**: 1.30 TB/s (所有线程写相同值)

**Reduction模式**:
- 读取带宽: 332.91 GB/s
- 写入带宽: 1.30 GB/s

**不同数据类型带宽**:
| 类型 | 大小 | 带宽 |
|------|------|------|
| float | 4 B | 878.19 GB/s |
| int | 4 B | 882.25 GB/s |
| double | 8 B | 468.73 GB/s |
| half (FP16) | 2 B | 410.20 GB/s |

**分析**:
- Stride超过16后带宽急剧下降
- 广播写入达到1.30 TB/s，接近Shared Memory带宽
- float/int带宽相近(~880 GB/s)，double下降约47%，half(FP16)下降约53%

### 早期内存带宽测试结果 (旧测试)

| 测试项目 | 带宽 | 备注 |
|---------|------|------|
| Sequential Read | 3.84 GB/s | 读取模式 |
| Sequential Write | 366.53 GB/s | 写入模式 |
| Read-Modify-Write | 402.18 GB/s | 读-改-写模式 |

**分析**: Sequential Read 带宽显著低于写入带宽，这可能是由于读取操作需要额外的内存传输开销或者测试模式导致的。

## 2.1 L2 缓存深入分析 (2026-03-22 更新)

### L2 Working Set vs Bandwidth

| 数据大小 | 带宽 | 状态 |
|---------|------|------|
| 64 KB | 123.09 GB/s | L2 likely fits |
| 1 MB | 407.66 GB/s | L2 borderline |
| 4 MB | 678.20 GB/s | L2 thrashing |
| 8 MB | 747.53 GB/s | L2 thrashing |
| 16 MB | 797.97 GB/s | L2 thrashing |

**分析**: 64KB数据主要在L1缓存，1MB开始触及L2，4MB以上完全L2 miss，带宽上升至~800 GB/s（DRAM速度）

### L2 Thrashing Test (跨距访问)

| Stride | 带宽 |
|--------|------|
| 1 | 729.19 GB/s |
| 2 | 713.45 GB/s |
| 4 | 679.30 GB/s |
| 8 | 418.62 GB/s |
| 16 | 402.67 GB/s |
| 64 | 218.89 GB/s |
| 256 | 226.36 GB/s |
| 1024 | 244.46 GB/s |
| 4096 | 406.78 GB/s |

**分析**: Stride 8 开始带宽急剧下降，表明缓存行跨距访问效率低

## 2.2 Warp级操作详细分析 (2026-03-22 更新)

### Warp Shuffle 操作

| 操作 | 性能 |
|------|------|
| Shuffle Reduce | 0.022 ms/kernel (747.46 GB/s) |
| Butterfly Reduce | 0.023 ms/kernel (730.21 GB/s) |

### Warp Vote/Ballot

| 操作 | 性能 |
|------|------|
| Ballot Sync | 0.020 ms/kernel |

### Memory Fence 影响

| 配置 | 带宽 | 时间/kernel |
|------|------|-----------|
| No Fence | 793.25 GB/s | 0.021 ms |
| With Fence | 536.38 GB/s | 0.031 ms |

**分析**: Memory fence 引入约 50% 的性能开销

## 2.3 指令吞吐量分析 (2026-03-22 更新)

| 指令类型 | 吞吐量 | 延迟 |
|---------|--------|------|
| FP32 FMA | 61.55 GFLOPS | 0.068 ms |
| INT32 Arith | 121.52 GIOPS | 0.035 ms |
| FP16 FMA | 204.19 GFLOPS | 0.021 ms |

**分析**:
- FP16 (半精度) 吞吐量约是 FP32 的 3.3 倍
- INT32 整数运算吞吐量高于 FP32
- 这些是标量运算的吞吐量，Tensor Core 可以达到更高

## 2.4 Occupancy 深入分析 (2026-03-22 更新)

### Block Size vs 性能

| BlockSize | 寄存器使用 | 带宽 | 时间/kernel |
|-----------|---------|------|-----------|
| 32 | ~4 | 292-300 GB/s | ~57 us |
| 64 | ~4 | 374-453 GB/s | ~35-45 us |
| 128 | ~4 | 674-890 GB/s | ~18-25 us |
| 256 | ~4 | 802-876 GB/s | ~19-21 us |
| 512 | ~4 | 828-900 GB/s | ~18-21 us |
| 1024 | ~4 | 589-628 GB/s | ~26-29 us |

**分析**:
- BlockSize=128-512 是最佳性能区间
- BlockSize 过小(32)导致资源利用不足
- BlockSize 过大(1024)由于共享内存限制导致性能下降
- RTX 5080 最佳 blockSize 约为 256-512

### Shared Memory 使用影响

| BlockSize | 带宽 |
|-----------|------|
| 32 | 346-347 GB/s |
| 64 | 472-476 GB/s |
| 128 | 684-890 GB/s |
| 256 | 802-846 GB/s |
| 512 | 765-840 GB/s |
| 1024 | 617-628 GB/s |

**分析**: Shared Memory 访问在 blockSize=256-512 时性能最佳

## 2.5 PCIe 带宽分析 (2026-03-22 更新)

### Host-Device 传输带宽

| 传输类型 | 带宽 | 每次传输时间 |
|---------|------|------------|
| Pageable H2D | 47-49 GB/s | 2.7-2.8 ms |
| Pinned H2D | 47-52 GB/s | 2.5-2.8 ms |
| Pageable D2H | 34-36 GB/s | 3.6-3.8 ms |
| Pinned D2H | 34-36 GB/s | 3.7-3.8 ms |
| D2D | 336-361 GB/s | 0.37-0.40 ms |

**分析**:
- H2D (写入GPU) 比 D2H (从GPU读取) 快约 30%
- Pinned 内存比 Pageable 内存带宽略高
- D2D (设备内拷贝) 带宽远高于 PCIe
- RTX 5080 Laptop PCIe 实际带宽约 35-50 GB/s

## 2.6 Branch Divergence 分析 (2026-03-22 更新)

| 分支类型 | 带宽 | 时间/kernel |
|---------|------|-----------|
| 无分歧 | 746-761 GB/s | 0.021 ms |
| 高分歧 | 796-810 GB/s | 0.021 ms |

**分析**:
- 高分歧的性能反而略好（可能是因为其他因素）
- 分歧开销不明显可能是因为测试 kernel 本身计算量较小
- 实际应用中分歧的代价会更大

## 3. 计算吞吐量

### FP32 计算

## 3. 计算吞吐量

### FP32 计算

| 测试项目 | 吞吐量 | 备注 |
|---------|--------|------|
| FP32 FMA (Fused Multiply-Add) | 88.55 GFLOPS | 融合乘加运算 |

### INT32 计算

| 测试项目 | 吞吐量 | 备注 |
|---------|--------|------|
| INT32 Arithmetic | 106.38 GIOPS | 32位整数运算 |

## 4. Warp级操作

### 通用Warp操作 (Generic)

| 测试项目 | 性能 | 备注 |
|---------|------|------|
| Warp Shuffle | 305.59 GB/s | Warp内数据混洗 |
| Warp Reduction | 0.015 ms/ kernel | Warp归约操作 |
| Warp Vote | 0.015 ms/ kernel | Warp投票操作 |

### SM 12.0 增强型Warp操作

| 测试项目 | 性能 | 备注 |
|---------|------|------|
| Enhanced Shuffle | 418.49 GB/s | 增强型Warp混洗 |

**分析**: Blackwell的Enhanced Shuffle比通用Warp Shuffle性能提升约37% (418.49 vs 305.59 GB/s)

## 5. SM 12.0 特有操作

### 内存操作

| 测试项目 | 性能 | 备注 |
|---------|------|------|
| Async Copy | 422.69 GB/s | 异步拷贝操作 |
| L2 Streaming | 316.46 GB/s | L2缓存流式访问 |
| Register Bandwidth | 298.96 GB/s | 寄存器带宽 |
| Software Prefetch | 251.10 GB/s | 软件预取 |
| Reduced Precision | 303.11 GB/s | 半精度(FP16)操作 |

### 性能分析

1. **Async Copy (422.69 GB/s)**: 最高性能，表明Blackwell的异步拷贝引擎效率很高
2. **L2 Streaming (316.46 GB/s)**: 良好的缓存流式访问性能
3. **Software Prefetch (251.10 GB/s)**: 相比直接访问，带宽较低但可预测性更好
4. **Register Bandwidth (298.96 GB/s)**: 寄存器级操作的带宽表现

## 6. 架构特性分析

### Blackwell (SM 12.0) 关键特性

1. **Enhanced Warp Shuffle**: 相比前代架构，Shuffle操作有显著性能提升
2. **Async Copy Engine**: 独立的异步拷贝引擎，可与计算并行执行
3. **L2 Cache Streaming**: 优化了流式访问模式的缓存性能
4. **Reduced Precision Support**: 原生支持FP16加速推理任务

### 与前代架构对比 (推测)

| 特性 | Blackwell (SM 12.0) | Ada (SM 9.0) | Ampere (SM 8.0) |
|------|---------------------|--------------|-----------------|
| Warp Shuffle | 418.49 GB/s | ~300 GB/s | ~250 GB/s |
| L2 Streaming | 316.46 GB/s | - | - |
| Async Copy | 422.69 GB/s | - | - |

## 7. 测试环境

- **CUDA Toolkit**: 13.0
- **驱动版本**: 595.79
- **操作系统**: Windows 11
- **编译选项**: -O3 -arch=sm_90 --use_fast_math

## 8. 后续研究方向

1. **Tensor Core性能**: 测试矩阵乘法和AI推理性能
2. **RT Core性能**: 光线追踪操作测试
3. **PTX汇编优化**: 使用Inline PTX深入优化关键路径
4. **多GPU扩展**: NVLink互联带宽测试
5. **功耗分析**: 性能/瓦特效率研究

## 9. 参考文档

- [CUDA Programming Guide](../ref/cuda_programming_guide.html)
- [PTX ISA](../ref/ptx_isa.html)
- [Inline PTX Assembly](../ref/inline_ptx.html)
- [Blackwell Compatibility Guide](../ref/blackwell_guide.html)

## 10. 综合研究计划 (2026-03-22 更新)

### 专题A: CUDA Core 算力研究 (`./gpupeek cuda`)

| 测试项 | 描述 | NCU指标 |
|--------|------|---------|
| A.1 数据类型吞吐量 | FP64/FP32/FP16/BF16/INT8/INT32 | sm__pipe_fp32_cycles_active |
| A.2 指令延迟vs吞吐量 | 依赖FMA链 vs 独立FMA | sm__average_execution_latency |
| A.3 向量指令 | float2/float4/double2向量运算 | sm__inst_executed.sum |
| A.4 超越函数 | sin/cos/exp/log延迟和吞吐量 | sm__pipe_fp64_cycles_active |
| A.5 混合精度 | FP32输入→FP16计算→FP32输出 | sm__pipe_tensor_cycles_active |

**测试命令**:
```bash
./build/gpupeek.exe cuda
```

### 专题B: Atomic 深入研究 (`./gpupeek atomic`)

| 测试项 | 描述 | 关键指标 |
|--------|------|---------|
| B.1 Warp级原子操作 | 同warp内归约后单次原子 | atomic contention |
| B.2 Block级原子操作 | 同block归约后单次原子 | atomic throughput |
| B.3 Grid级原子操作 | 所有线程直接原子(高竞争) | atomic latency |
| B.4 原子操作对比 | atomicAdd vs CAS vs Min/Max | cycles |

**测试命令**:
```bash
./build/gpupeek.exe atomic
```

### 专题C: Barrier 同步研究 (`./gpupeek barrier`)

| 测试项 | 描述 | NCU指标 |
|--------|------|---------|
| C.1 __syncthreads()开销 | 最小同步开销测量 | clock64计时 |
| C.2 bar.sync指令分析 | barrier stall分析 | sm__warp_issue_stalled_by_barrier |
| C.3 Block Size vs Barrier | 32/64/128/256/512/1024效率 | sm__average_active_warps |
| C.4 多Block同步 | grid级flag同步模式(低效警告) | spin-wait分析 |

**测试命令**:
```bash
./build/gpupeek.exe barrier
```

### 专题D: Warp Specialization 与 Producer-Consumer (`./gpupeek warp`)

| 测试项 | 描述 | 关键API |
|--------|------|---------|
| D.1 Warp Specialization基础 | 2-warp producer/consumer | __shfl_down_sync |
| D.2 TMA + Barrier协同 | Async Copy + Barrier同步 | cp.async, bar.sync |
| D.3 多级Pipeline | 3-stage流水线(load/compute/store) | __syncthreads |
| D.4 Block Specialization | 半block=producer，另半=consumer | threadIdx.x判断 |
| D.5 Warp级同步原语 | Mutex/Barrier/Reduction/Scan | warp vote/ballot |

**测试命令**:
```bash
./build/gpupeek.exe warp
```

### 专题E: 高级专题

| 测试项 | 描述 | 状态 |
|--------|------|------|
| E.1 Multi-Stream并发 | 流依赖、重叠执行、优先级 | 待实现 |
| E.2 Unified Memory | Page fault、prefetch、page migration | 待实现 |
| E.3 Occupancy深入分析 | 精确occupancy边界测试 | 已实现 |
| E.4 CUDA Graph | Graph capture、instantiate、launch | 待实现 |

### NCU 关键指标参考

| 指标 | 含义 | 用途 |
|------|------|------|
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU利用率 | 越高越好 |
| dram__bytes.sum | 内存带宽 | 内存操作测试 |
| sm__pipe_fp32_cycles_active.pct | FP32计算单元利用率 | 算力测试 |
| sm__warp_issue_stalled_by_barrier.pct | Barrier stall | 同步开销 |
| sm__warp_divergence_efficiency | Warp分歧效率 | 分歧测试 |
| sm__average_active_warps_per_sm | 每SM平均活跃warp | Occupancy指标 |

### 运行所有研究测试

```bash
# 运行所有benchmark
./build/gpupeek.exe all

# 运行特定研究专题
./build/gpupeek.exe memory    # 内存研究
./build/gpupeek.exe deep      # 深度研究
./build/gpupeek.exe advanced  # 高级研究
./build/gpupeek.exe cuda      # CUDA Core算力
./build/gpupeek.exe atomic    # Atomic原子操作
./build/gpupeek.exe barrier   # Barrier同步
./build/gpupeek.exe warp      # Warp特化

# NCU profiling
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./build/gpupeek.exe cuda
```
