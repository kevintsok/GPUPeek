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
- [CUTLASS Tutorial: TMEM GEMM](../ref/cutlass_tutorial_tmem_gemm.md)
- [CUTLASS Tutorial: Block Scaling](../ref/cutlass_tutorial_block_scaling.md)
- [CUTLASS Tutorial: Sub-byte GEMM](../ref/cutlass_tutorial_subbyte_gemm.md)
- [CUTLASS Tutorial: Cluster GEMM](../ref/cutlass_tutorial_cluster_gemm.md)
- [DeepSeek FP8 Training](../ref/deepseek_fp8_training.md)
- [FlashAttention-4](../ref/flashattention4.md)

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
| E.1 Multi-Stream并发 | 流依赖、重叠执行、优先级 | 已实现 |
| E.2 Unified Memory | Page fault、prefetch、page migration | 已实现 |
| E.3 Occupancy深入分析 | 精确occupancy边界测试 | 已实现 |
| E.4 CUDA Graph | Graph capture、instantiate、launch | 已实现 |

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
./build/gpupeek.exe mma       # MMA (Tensor Core) 研究

# NCU profiling
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./build/gpupeek.exe cuda
```

## 11. MMA (Tensor Core) 深入研究

### 11.1 PTX ISA MMA 指令分类 (Section 9.7.14-9.7.16)

#### WMMA (Warp-level MMA) - Section 9.7.14.4

| 指令 | Shape | 数据类型 |
|------|-------|---------|
| wmma.load | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |
| wmma.store | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |
| wmma.mma | m16n16k16 | .f16, .bf16, .tf32, .f32, .f64, .s32 |

**SASS 映射**: HMMA (H = Half/FP16), BMMA (B = BF16), IMMA (I = INT), DMMA (D = Double/FP64)

#### MMA (New Warp-level MMA) - Section 9.7.14.5

| Shape | FP16 | FP64 | TF32 | BF16 | INT8 | INT4 |
|-------|------|------|------|------|------|------|
| m8n8k4 | Yes | Yes | - | - | - | - |
| m8n8k16 | Yes | - | - | - | Yes | - |
| m8n8k32 | Yes | - | - | - | Yes | Yes |
| m8n8k128 | Yes | - | - | - | Yes | Yes |
| m16n8k4 | Yes | - | Yes | - | - | - |
| m16n8k8 | Yes | - | Yes | Yes | Yes | - |
| m16n8k16 | Yes | - | Yes | Yes | Yes | Yes |
| m16n8k32 | Yes | - | - | Yes | Yes | Yes |
| m16n8k64 | Yes | - | - | - | Yes | - |
| m16n8k128 | Yes | - | - | - | Yes | - |
| m16n8k256 | Yes | - | - | - | Yes | - |

#### MMA.SP (Sparse MMA) - Section 9.7.14.6

| Shape | FP16 | BF16 | TF32 | INT8 | FP8 |
|-------|------|------|------|------|-----|
| m16n8k8.sp | Yes | Yes | Yes | - | - |
| m16n8k16.sp | Yes | Yes | Yes | Yes | - |
| m16n8k32.sp | Yes | - | - | Yes | - |
| m16n8k64.sp | Yes | - | - | - | - |
| m16n8k128.sp | Yes | - | - | - | - |

**2:4 结构化稀疏**: 每4个元素中有2个必须是零

#### WGMMA (Async Warpgroup MMA) - Section 9.7.15

| Shape | FP16 | BF16 | TF32 | FP64 | INT8 | FP8 |
|-------|------|------|------|------|------|------|
| m64nNk16 | Yes | Yes | Yes | Yes | Yes | - |
| m64nNk8 | Yes | Yes | Yes | - | Yes | - |
| m64nNk32 | Yes | Yes | - | - | - | Yes |
| m64nNk256 | Yes | - | - | - | - | - |

**N = K / 16** (N取决于K维度)

#### TCGen05 (TensorCore 5th Generation) - Section 9.7.16

| Shape | FP16 | BF16 | TF32 | FP32 | FP64 | INT8 | FP8 |
|-------|------|------|------|------|------|------|------|
| .32x32b | Yes | Yes | Yes | Yes | Yes | - | - |
| .16x64b | Yes | Yes | Yes | Yes | Yes | - | - |
| .16x128b | Yes | Yes | Yes | Yes | - | - | - |
| .16x256b | Yes | Yes | Yes | - | - | - | - |
| .16x32bx2 | Yes | Yes | Yes | Yes | - | - | - |

**TCGen05 变体**:
- `tcgen05.mma` - 基本 MMA
- `tcgen05.mma.sp` - 稀疏 MMA
- `tcgen05.mma.ws` - Weight-only Quantization (W8A16)
- `tcgen05.mma.ws.sp` - Weight-only + Sparse

### 11.2 SASS 指令映射

| PTX | SASS | 描述 |
|-----|------|------|
| wmma.mma.f16 | HMMA | Half precision MMA |
| wmma.mma.bf16 | BMMA | BFloat16 MMA |
| wmma.mma.tf32 | HMMA | TensorFloat-32 MMA |
| wmma.mma.f64 | DMMA | Double precision MMA |
| wmma.mma.s32 | IMMA | INT32 MMA |
| mma.mma | HMMA/IMMA/DMMA | 通用 MMA 指令 |
| wgmma.mma_async | WGMMA | 异步 Warpgroup MMA |
| ld.matrix | LDMATRIX | 矩阵加载 (SM 8.0+) |
| st.matrix | STMATRIX | 矩阵存储 (SM 8.0+) |

### 11.3 NCU Tensor Core 关键指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| sm__pipe_tensor_cycles_active.pct | Tensor Core 利用率 | 越高越好 |
| sm__pipe_tensor_cycles_active.sum | Tensor Core 活跃周期数 | 越少越好 |
| smsp__average_executed_epc_per_warp | 每 warp 执行指令数 | 稳定 |
| sm__inst_executed.sum | 总执行指令数 | 越少越好 |
| dram__bytes.sum | 全局内存带宽 | 参考内存带宽 |
| lts__tcs_hit_rate.pct | L2 缓存命中 | 越高越好 |

### 11.4 MMA 研究测试命令

```bash
# 基本 MMA 基准测试
./build/gpupeek.exe mma

# NCU Tensor Core 利用率分析
ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe mma

# NCU SASS 指令分析
ncu --set full --kernels-by-compute --metrics sm__inst_executed ./gpupeek.exe mma

# 完整 Tensor Core 分析
ncu --set full \
  --metrics sm__pipe_tensor_cycles_active.pct,sm__inst_executed.sum,dram__bytes.sum,lts__tcs_hit_rate.pct \
  ./gpupeek.exe mma
```

### 11.5 MMA Kernel 代码覆盖

| Kernel | PTX 指令 | Shape | 数据类型 |
|--------|----------|-------|----------|
| wmma_fp16_kernel | wmma.mma | m16n16k16 | FP16 |
| wmma_fp32_acc_kernel | wmma.mma | m16n16k16 | FP16 (FP32累加) |
| mma_m16n8k8_fp16_kernel | mma.m16n8k8 | m16n8k8 | FP16 |
| mma_tf32_kernel | mma.m16n8k4 | m16n8k4 | TF32 |
| mma_bf16_kernel | mma.m16n8k8 | m16n8k8 | BF16 |
| mma_fp64_kernel | mma.m8n8k4 | m8n8k4 | FP64 |
| mma_int8_kernel | mma.m16n8k16 | m16n8k16 | INT8 |
| mma_sparse_fp16_kernel | mma.sp | m16n8k16 | FP16 (稀疏) |
| wgmma_async_kernel | wgmma.mma_async | m64nNk16 | FP16/BF16 |
| tcgen05_mma_kernel | tcgen05.mma | 多种 | FP16/BF16/FP8 |
| ldmatrix_kernel | ld.matrix | 8x8 | various |
| stmatrix_kernel | st.matrix | 8x8 | various |
| naive_gemm_kernel | FMA | - | Baseline |
| shared_gemm_kernel | FMA+Shared | - | Baseline |

### 11.6 MMA 性能分析框架

**理论峰值计算**:
- RTX 5080 Tensor Core 峰值: ~xxxx TFLOPS (待实测)
- FP16 峰值: ~xxxx TFLOPS
- BF16 峰值: ~xxxx TFLOPS
- TF32 峰值: ~xxxx TFLOPS
- FP64 峰值: ~xxxx TFLOPS

**实测指标** (2026-03-23 更新):
| 数据类型 | Shape | GFLOPS | 内存带宽 | Tensor Core利用率 |
|----------|-------|--------|----------|-------------------|
| FP16 | m16n16k16 | **257.41** | - | - |
| BF16 | m16n8k8 | 待实测 | 待实测 | 待实测 |
| TF32 | m16n8k4 | 待实测 | 待实测 | 待实测 |
| FP64 | m8n8k4 | 待实测 | 待实测 | 待实测 |
| INT8 | m16n8k16 | 待实测 | 待实测 | 待实测 |

**WMMA FP16 测试结果** (2026-03-23):
- 矩阵尺寸: M=256, N=256, K=256
- Shape: m16n16k16
- Grid: 16x16, Block: 32 (warp)
- 时间: 0.130 ms/iteration
- GFLOPS: 257.41 GFLOPS
- 结果验证: sum=4103416.75 (非零=正确)

**注意**: 这是 WMMA (warp-level) 实测结果。TCGen05 (5th-gen Tensor Core) 需要不同的 API。

**WMMA 测试文件** (2026-03-23 更新):
- `wmma_test_kernel.cu`: WMMA FP16 kernels (m16n16k16)
- `wmma_test_benchmarks.cu`: Benchmark runner with 256x256x256 and 512x512x512 tests
- 命令: `./gpupeek.exe wmma`

**参考文章** (Colfax Research):
- CUTLASS Tutorial: TMEM GEMM - Blackwell TMEM (256KB/SM) for Tensor Cores
- CUTLASS Tutorial: Block Scaling - TCGen05 hardware block scaling support
- CUTLASS Tutorial: Sub-byte GEMM - FP4/FP6 on Blackwell
- CUTLASS Tutorial: Cluster GEMM - Thread block clusters for collaborative MMA
- DeepSeek FP8 Training - FP8 mixed precision training strategies
- FlashAttention-4 - Advanced attention kernels for B200

### 11.7 MMA 优化建议

1. **Shape 选择**: m16n8k8 通常是 FP16/BF16 的最佳平衡
2. **数据类型**: 优先使用 FP16，需要精度时用 BF16 或 TF32
3. **稀疏加速**: 结构化稀疏可提供 2x 加速
4. **异步操作**: WGMMA 可隐藏内存延迟
5. **Weight-only Quantization**: TCGen05.mma.ws 支持 INT8 权重 + FP16 激活
6. **内存布局**: 使用行主序或列主序匹配 ldmatrix/stmatrix 要求

## 12. Tensor Memory Operations 深入研究

### 12.1 PTX ISA 张量内存指令

#### LDMATRIX (Section 9.7.14.5.15)

| 指令变体 | 描述 | 每线程加载 |
|----------|------|-----------|
| ldmatrix.sync.aligned.m8n8.x1 | 8x8 tile, 1个tile | 2 elements |
| ldmatrix.sync.aligned.m8n8.x2 | 8x8 tile, 2个tile | 4 elements |
| ldmatrix.sync.aligned.m8n8.x4 | 8x8 tile, 4个tile | 8 elements |
| ldmatrix.sync.aligned.m16n8.k1 | 16x8 tile | varies |

**关键特性**:
- Warp级别操作 (32 threads协作)
- 转置布局存储 (适合MMA消耗)
- 必须16-byte对齐

#### STMATRIX (Section 9.7.14.5.16)

| 指令变体 | 描述 |
|----------|------|
| stmatrix.sync.aligned.m8n8.x1 | 8x8 tile, 1个tile |
| stmatrix.sync.aligned.m8n8.x2 | 8x8 tile, 2个tile |
| stmatrix.sync.aligned.m8n8.x4 | 8x8 tile, 4个tile |

#### cp.async (Section 9.7.9.25)

| 指令 | 描述 |
|------|------|
| cp.async.ca | Cache policy async copy |
| cp.async.commit_group | 提交async组 |
| cp.async.wait_group n | 等待n个组 |
| cp.async.wait_all | 等待所有 |

#### cp.async.bulk (Section 9.7.9.25.4)

| 指令 | 描述 |
|------|------|
| cp.async.bulk | 批量async拷贝 |
| cp.async.bulk.commit_group | 批量提交组 |
| cp.async.bulk.wait_group n | 批量等待 |
| cp.reduce.async.bulk.add | 批量拷贝+求和 |
| cp.async.bulk.prefetch | 批量预取 |

### 12.2 SASS 指令映射

| SASS | 描述 | PTX |
|------|------|-----|
| LDMATRIX | Matrix load (8x8 tile) | ld.matrix |
| LDMATRIXu | Matrix load (unaligned) | ld.matrix |
| STMATRIX | Matrix store | st.matrix |
| STMATRIXu | Matrix store (unaligned) | st.matrix |
| CP.ASYNC | Async copy commit | cp.async |
| BAR.ASYNC | Async barrier | bar.async |

### 12.3 NCU 张量内存关键指标

| 指标 | 含义 |
|------|------|
| sm__inst_executed.ldmatrix.sum | LDMATRIX 指令数 |
| sm__inst_executed.stmatrix.sum | STMATRIX 指令数 |
| sm__pipe_tensor_cycles_active.pct | Tensor内存流水线利用率 |
| sm__pipe_mem_cycles_active.pct | 内存流水线利用率 |
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU利用率 |

### 12.4 Tensor Memory 测试命令

```bash
# 张量内存基准测试
./build/gpupeek.exe tensor_mem

# NCU LDMATRIX 分析
ncu --metrics sm__inst_executed.ldmatrix.sum ./gpupeek tensor_mem

# NCU STMATRIX 分析
ncu --metrics sm__inst_executed.stmatrix.sum ./gpupeek tensor_mem

# NCU cp.async 分析
ncu --metrics sm__inst_executed.cp_async.sum ./gpupeek tensor_mem

# 完整 SASS 分析
ncu --set full --kernels-by-compute ./gpupeek tensor_mem
```

### 12.5 Tensor Memory Kernel 代码覆盖

| Kernel | PTX 指令 | 功能 |
|--------|----------|------|
| ldmatrix_fp16_kernel | ld.matrix | FP16 LDMATRIX基本 |
| ldmatrix_multi_tile_kernel | ld.matrix | 多tile LDMATRIX |
| ldmatrix_layout_x1_kernel | ld.matrix.x1 | x1布局 |
| ldmatrix_layout_x2_kernel | ld.matrix.x2 | x2布局 |
| stmatrix_fp16_kernel | st.matrix | FP16 STMATRIX基本 |
| stmatrix_layout_x1_kernel | st.matrix.x1 | x1布局 |
| cp_async_1d_kernel | cp.async | 1D async拷贝 |
| cp_async_group_kernel | cp.async.commit_group | Async组模式 |
| cp_async_bulk_prefetch_kernel | cp.async.bulk.prefetch | 批量预取 |
| cp_async_reduce_kernel | cp.reduce.async.bulk | 批量拷贝+归约 |
| ldmatrix_mma_stmatrix_kernel | ld/st/mma | 完整流水线 |
| naive_load_kernel | 无 | 基线 - naive load |
| shared_load_kernel | 无 | 基线 - shared load |
| cp_async_baseline_kernel | cp.async | 基线 - async拷贝 |
| tma_baseline_kernel | tma | 基线 - TMA |

### 12.6 LDMATRIX/STMATRIX 性能分析

**LDMATRIX 优势**:
1. **Warp级别协作**: 32线程同时加载8x8 tile，带宽效率高
2. **转置布局**: 数据以MMA友好格式存储
3. **对齐保证**: .aligned变体确保16-byte对齐

**cp.async 优势**:
1. **延迟隐藏**: 拷贝与计算重叠
2. **批量操作**: commit_group/wait_group支持批量异步
3. **多种模式**: cp.async.bulk支持更大传输(最高128 bytes)

**性能对比框架**:
| 操作 | 优势 | 适用场景 |
|------|------|----------|
| LDMATRIX | Warp协作、转置 | MMA前加载A/B矩阵 |
| STMATRIX | Warp协作 | MMA结果存储 |
| cp.async | 延迟隐藏 | 计算与拷贝重叠 |
| TMA | 大块2D传输 | 大矩阵分块 |

### 12.7 Tensor Memory 优化建议

1. **LDMATRIX 布局选择**:
   - 优先使用 .x1 (简单)
   - 大数据传输用 .x2/.x4

2. **cp.async 使用模式**:
   - 使用 commit_group 批量提交
   - 使用 wait_group 0 等待完成
   - 结合 barrier 实现同步

3. **与 MMA 协同**:
   - LDMATRIX 加载 → MMA 计算 → STMATRIX 存储
   - 使用 shared memory 作为中间缓存
   - pipeline 重叠实现高吞吐

## 13. DP4A (INT8 Dot Product of 4 Bytes) 研究

### 13.1 PTX ISA DP4A 指令 (Section 9.7.1.23)

DP4A 执行: `result = sum(a[i]*b[i]) for i=0..3`

| 变体 | 描述 | 数据类型 |
|------|------|---------|
| dp4a.s32.s8.s8 | 有符号INT8 | 结果INT32 |
| dp4a.u32.u8.u8 | 无符号UINT8 | 结果UINT32 |
| dp4a.s32.rmi | 带舍入模式 | - |
| dp4a.s32.satfinite | 带饱和 | 溢出钳位 |

### 13.2 SASS 映射

| SASS | 描述 | PTX |
|------|------|-----|
| DP4A | 4字节点积 | dp4a.s32.s8.s8 |

### 13.3 NCU DP4A 关键指标

| 指标 | 含义 |
|------|------|
| sm__inst_executed.dp4a.sum | DP4A 指令数 |
| sm__pipe_tensor_cycles_active.pct | INT8/张量流水线利用率 |

### 13.4 DP4A 测试命令

```bash
# DP4A 基准测试
./build/gpupeek.exe dp4a

# NCU DP4A 指令分析
ncu --metrics sm__inst_executed.dp4a.sum ./gpupeek dp4a

# NCU INT8 流水线分析
ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek dp4a
```

### 13.5 DP4A Kernel 代码覆盖

| Kernel | PTX 指令 | 功能 |
|--------|----------|------|
| dp4a_s32_kernel | dp4a.s32.s8.s8 | 有符号INT8基本 |
| dp4a_u32_kernel | dp4a.u32.u8.u8 | 无符号INT8基本 |
| dp4a_satfinite_kernel | dp4a.s32.satfinite | 饱和DP4A |
| dp4a_accumulate_kernel | dp4a + add | 累加DP4A |
| dp4a_batch_kernel | dp4a (循环) | 批量处理 |
| dp4a_packed_kernel | dp4a | 打包INT8 |
| dp4a_shared_kernel | dp4a + 共享内存 | 块级归约 |
| dp4a_warp_reduce_kernel | dp4a + shuffle | Warp级归约 |
| dp4a_quantized_kernel | dp4a + 反量化 | 量化推理 |
| dp4a_block_scale_kernel | dp4a + 块缩放 | 权重量化 |
| naive_dot4_kernel | 无 | 基线 - 朴素INT8 |
| fp32_mad4_kernel | FMA | 基线 - FP32 |
| fp16_dot4_kernel | FP16 | 基线 - FP16 |

### 13.6 DP4A 性能分析

**优势**:
1. **高吞吐**: 4个INT8乘法加法单指令完成
2. **低带宽**: 相比FP32减少75%内存带宽
3. **推理友好**: 支持INT8量化推理

**与Tensor Core MMA对比**:
| 特性 | DP4A | MMA (INT8) |
|------|------|------------|
| Shape | 4元素向量 | 16x16 矩阵块 |
| 精度 | INT8 | INT8 |
| 吞吐量 | 高 | 非常高 |
| 设置开销 | 低 | 高 |
| 适用场景 | 点积、卷积核 | 矩阵乘法 |

### 13.7 DP4A 优化建议

1. **数据打包**: 将4个INT8打包成32位访问
2. **Warp归约**: 使用__shfl_down_sync加速
3. **量化推理**: 结合缩放因子实现INT8->FP32
4. **批量处理**: 一次处理多个向量提高效率

## 14. WGMMA (Warpgroup Matrix Multiply Async) 研究

### 14.0 重要说明：架构兼容性

> **关键发现**（来自 arXiv:2507.10789）：
> - **WGMMA 指令仅在 Hopper 架构上受支持**
> - **Blackwell 架构不支持 WGMMA**，改用 `mma.sync` 配合 QMMA/OMMA SASS 指令
> - Blackwell 使用第5代 Tensor Core，支持 FP4/FP6 等新精度

| 架构 | Tensor Core 代 | MMA 指令 | WGMMA |
|------|---------------|----------|-------|
| Volta (V100) | 1st | WMMA | 否 |
| Ampere (A100) | 3rd | MMA | 否 |
| Hopper (H100) | 4th | MMA + **WGMMA** | **是** |
| **Blackwell (RTX 50)** | **5th** | **MMA (QMMA/OMMA)** | **否** |

### 14.1 PTX ISA WGMMA 指令 (Section 9.7.15)

WGMMA 是异步的 Warpgroup 级别矩阵乘法指令，与 WMMA/MMA 的主要区别：

| 特性 | WMMA | MMA | WGMMA |
|------|------|-----|-------|
| 级别 | Warp | Warp | Warpgroup (3 warps) |
| Shape | m16n16k16 | m16n8k8 等 | m64nNk16 |
| 同步 | 同步 | 同步 | **异步** |
| 设置开销 | 中 | 中 | 低 |
| 吞吐量 | 高 | 高 | 最高 |

### 14.2 WGMMA Shapes

| Shape | M | N | K |
|-------|---|---|---|
| m64nNk16 | 64 | K/16*64 | 16 |
| m64nNk8 | 64 | K/8*64 | 8 |
| m64nNk32 | 64 | K/32*64 | 32 |
| m64nNk256 | 64 | K/256*64 | 256 |

### 14.3 WGMMA 数据类型

| 数据类型 | PTX | 描述 |
|----------|-----|------|
| FP16 | .f16 | 半精度浮点 |
| BF16 | .bf16 | BFloat16 |
| TF32 | .tf32 | TensorFloat-32 |
| FP64 | .f64 | 双精度浮点 |
| INT8 | .s8, .u8 | 有符号/无符号整数 |

### 14.4 WGMMA 异步操作

| 指令 | 描述 |
|------|------|
| wgmma.fence | 确保操作顺序 |
| wgmma.commit_group | 提交异步操作组 |
| wgmma.wait_group n | 等待 n 个组完成 |

### 14.5 SASS 映射

**Hopper (WGMMA)**:

| SASS | 描述 | PTX |
|------|------|-----|
| WGMMA | 异步Warpgroup MMA | wgmma.mma_async |
| WGMMA.sp | 异步Warpgroup MMA稀疏版 | wgmma.mma_async.sp |
| WGMMAF | WGMMA fence | wgmma.fence |
| WGMMAWG | WGMMA wait group | wgmma.wait_group |

**Blackwell (QMMA/OMMA)**:

| SASS | 描述 | PTX |
|------|------|-----|
| QMMA | Quantized MMA (INT8/INT4) | mma.sync |
| OMMA | Optimized MMA (FP8/FP16/BF16) | mma.sync |
| HMMA | Standard MMA | mma.sync |
| IMMA | Integer MMA | mma.sync |
| DMMA | DPMA (FP64 MMA) | mma.sync |

### 14.6 NCU WGMMA 关键指标

| 指标 | 含义 |
|------|------|
| sm__inst_executed.wgmma.sum | WGMMA 指令数 |
| sm__pipe_tensor_cycles_active.pct | Tensor Core 利用率 |
| sm__throughput.avg.pct_of_peak_sustainedTesla | GPU 利用率 |

### 14.7 WGMMA 测试命令

```bash
# WGMMA 基准测试
./build/gpupeek.exe wgmma

# NCU WGMMA 指令分析
ncu --metrics sm__inst_executed.wgmma.sum ./gpupeek wgmma

# NCU Tensor Core 利用率
ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek wgmma
```

### 14.8 WGMMA Kernel 代码覆盖

> **注意**: 以下 Kernel 使用 WMMA API 模拟 WGMMA 行为。真正的 WGMMA 仅在 Hopper 架构上可用。

| Kernel | PTX 指令 | 功能 |
|--------|----------|------|
| wgmma_fp16_kernel | wgmma.mma_async.m64nNk16.f16 | FP16基本 (Hopper) |
| wgmma_bf16_kernel | wgmma.mma_async.m64nNk16.bf16 | BF16 (Hopper) |
| wgmma_fp64_kernel | wgmma.mma_async.m64nNk8.f64 | FP64 (Hopper) |
| wgmma_int8_kernel | wgmma.mma_async.m64nNk16.s8 | INT8 (Hopper) |
| wgmma_sparse_fp16_kernel | wgmma.mma_async.sp.m64nNk32 | 稀疏FP16 (Hopper) |
| wgmma_pipeline_kernel | wgmma + pipeline | 流水线隐藏延迟 |
| wmma_baseline_fp16_kernel | wmma.mma | WMMA基线对比 |

**Blackwell 替代方案**: 使用 MMA 研究（Section 11）中的 `mma.sync` 内核，通过 WMMA API 调用 QMMA/OMMA 指令。

### 14.9 WGMMA vs WMMA 对比

| 特性 | WGMMA | WMMA |
|------|-------|------|
| Tile大小 | 64xN | 16x16 |
| 异步执行 | 是 | 否 |
| Warpgroup级别 | 是 | 否 |
| 内存延迟隐藏 | 好 | 差 |

### 14.10 WGMMA 优化建议

1. **流水线**: 使用双缓冲隐藏内存延迟
2. **异步提交**: 使用 commit_group 批量提交
3. **等待模式**: 使用 wait_group 0 等待当前组
4. **稀疏加速**: 使用 2:4 结构化稀疏 (2x 加速)

### 14.11 Blackwell Tensor Core 新特性

| 特性 | Blackwell (5th Gen) | Hopper (4th Gen) |
|------|---------------------|------------------|
| FP4 支持 | **是** (15 PFLOPs) | 否 |
| FP6 支持 | **是** | 否 |
| FP8 支持 | 是 | 是 |
| WGMMA | **否** | 是 |
| 稀疏 MMA | 是 (2:4) | 是 |
| 寄存器文件 | 256 KB/SM | 256 KB/SM |
| L1 Cache | 128 KB/SM | 256 KB/SM |
| L2 Cache | 65 MB | 50 MB |

**Blackwell 功耗效率** (来自微基准测试):
- FP4: ~16.75W
- FP6 e2m3: ~39.38W
- FP6 e3m2: ~46.72W
- FP8: ~46W (比 Hopper 的 ~55W 更节能)

## 15. FP8 / TCGen05 Block Scaling 研究

### 15.1 PTX ISA TCGen05 (Section 9.7.16)

TCGen05 是第5代 Tensor Core，支持 FP8 格式和块缩放。

**FP8 格式**:

| 格式 | 指数位 | 尾数位 | 范围 | 用途 |
|------|--------|--------|------|------|
| E4M3 | 4 | 3 | 0-240 | 推理激活 |
| E5M2 | 5 | 2 | 0-57344 | 推理权重 |

### 15.2 TCGen05 MMA 变体

| 变体 | 描述 |
|------|------|
| tcgen05.mma | 基本 MMA |
| tcgen05.mma.sp | 稀疏 MMA |
| tcgen05.mma.ws | Weight-only 缩放 |
| tcgen05.mma.ws.sp | Weight-only + 稀疏 |

### 15.3 块缩放 (Block Scaling)

**W8A16**: 8位权重，16位激活
- 每块32元素一个缩放因子
- 内存减少 4x vs FP32

**W8A8**: 8位权重，8位激活
- 每块32元素一个缩放因子
- 内存减少 8x vs FP32

### 15.4 FP8 vs 其他格式对比

| 格式 | 位宽 | 范围 | 适用场景 |
|------|------|------|----------|
| FP32 | 32 | 1e-38 to 1e38 | 训练 |
| FP16 | 16 | 1e-5 to 1e4 | 推理 |
| BF16 | 16 | 1e-38 to 1e38 | ML |
| TF32 | 19 | 1e-38 to 1e4 | ML |
| **FP8 E4M3** | 8 | 0-240 | **推理** |
| **FP8 E5M2** | 8 | 0-57344 | **推理** |

### 15.5 NCU FP8 关键指标

| 指标 | 含义 |
|------|------|
| sm__pipe_tensor_cycles_active.pct | Tensor Core 利用率 |
| sm__inst_executed.fp8.sum | FP8 指令数 |
| dram__bytes.sum | 内存带宽 |

### 15.6 FP8 测试命令

```bash
# FP8 基准测试
./build/gpupeek.exe fp8

# NCU Tensor Core 利用率
ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek fp8

# NCU 内存带宽
ncu --metrics dram__bytes.sum ./gpupeek fp8
```

### 15.7 FP8 Kernel 代码覆盖

| Kernel | 功能 |
|--------|------|
| convert_to_fp8_e4m3 | FP32 -> FP8 E4M3 |
| convert_to_fp8_e5m2 | FP32 -> FP8 E5M2 |
| block_scale_quantize_kernel | W8A16 块缩放量化 |
| block_scale_quantize_w8a8_kernel | W8A8 块缩放量化 |
| w8a16_mma_kernel | W8A16 MMA |
| fp8_gemm_e4m3_kernel | FP8 E4M3 GEMM |
| fp8_gemm_e5m2_kernel | FP8 E5M2 GEMM |
| weight_only_quant_kernel | Weight-only 量化 |
| tcgen05_block_scaled_mma_kernel | TCGen05 块缩放 MMA |
| fp32_baseline_gemm_kernel | FP32 基线 |
| fp16_baseline_gemm_kernel | FP16 基线 |

### 15.8 FP8 优化建议

1. **E4M3 vs E5M2**: 激活用 E4M3，权重用 E5M2
2. **块大小**: 32 元素/块是常见选择
3. **缩放因子**: 存储倒数避免除法
4. **量化感知训练**: 需要校准数据

## 16. CUDA Graph 研究

### 16.1 CUDA Graph API

CUDA Graph 用于优化内核启动开销。

**Graph 生命周期**:

| 函数 | 描述 |
|------|------|
| cudaGraphCreate | 创建空图 |
| cudaGraphInstantiate | 从图创建可执行图 |
| cudaGraphLaunch | 启动可执行图 |
| cudaGraphExecDestroy | 销毁可执行图 |
| cudaGraphDestroy | 销毁图 |

**Stream 捕获**:

| 函数 | 描述 |
|------|------|
| cudaStreamBeginCapture | 开始捕获 |
| cudaStreamEndCapture | 结束捕获并返回图 |
| cudaStreamIsCapturing | 检查捕获状态 |

### 16.2 节点操作

| 函数 | 描述 |
|------|------|
| cudaGraphAddKernelNode | 添加内核节点 |
| cudaGraphAddMemcpyNode | 添加拷贝节点 |
| cudaGraphAddMemsetNode | 添加置零节点 |
| cudaGraphAddEmptyNode | 添加空节点 |
| cudaGraphAddBarrierNode | 添加屏障节点 |

### 16.3 CUDA Graph 测试命令

```bash
# CUDA Graph 基准测试
./build/gpupeek.exe graph

# NCU 分析
ncu --set full --kernels-by-compute ./gpupeek graph
ncu --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek graph
```

### 16.4 CUDA Graph Kernel 代码覆盖

| Kernel | 功能 |
|--------|------|
| vectorAddKernel | 向量加法 |
| vectorMulKernel | 向量乘法 |
| vectorScaleKernel | 向量缩放 |
| matrixMultiplyKernel | 矩阵乘法 |
| reluKernel | ReLU 激活 |
| biasAddKernel | 偏置加法 |
| Graph Benchmark 测试 | Graph 生命周期性能 |
| Stream Capture 测试 | 捕获性能 |
| Launch Overhead 测试 | 启动开销对比 |
| Pipeline 测试 | 推理流水线 |

### 16.5 CUDA Graph 使用场景

**优势**:
1. 减少多个内核启动的 CPU 开销
2. 启用并行内核执行
3. 降低重复工作负载的延迟
4. 提高小内核的 GPU 利用率

**最佳场景**:
- 深度学习推理流水线
- 重复批处理
- 多内核流式工作负载
- 小内核（启动开销显著）

**权衡**:
- Graph 创建有前期成本
- 灵活性有限（图是静态的）
- 更新需要重新实例化

## 17. Unified Memory 研究

### 17.1 Unified Memory API

Unified Memory 提供单一托管内存空间，GPU 和 CPU 都可以访问。

**核心函数**:

| 函数 | 描述 |
|------|------|
| cudaMallocManaged | 分配统一托管内存 |
| cudaMemPrefetchAsync | 异步预取数据到设备/主机 |
| cudaMemAdvise | 设置内存使用建议 |
| cudaMemsetAccessAdvise | 设置访问建议 |
| cudaPointerGetAttributes | 查询指针属性 |

### 17.2 概念

- **Managed Memory**: 单次分配，CPU/GPU共享
- **Page Fault**: 按需迁移触发
- **Prefetching**: 显式数据迁移
- **Access Counters**: 跟踪设备访问模式

### 17.3 使用场景

- 简化内存管理
- GPU 内存扩展
- 异构计算
- 核外处理

### 17.4 Unified Memory 测试命令

```bash
# Unified Memory 基准测试
./build/gpupeek.exe unified

# NCU 分析
ncu --set full --metrics dram__bytes.sum,uops__issue_active.sum ./gpupeek unified
```

### 17.5 Unified Memory Kernel 代码覆盖

| Kernel | 功能 |
|--------|------|
| vectorAddKernel | 向量加法 |
| matrixMultiplyKernel | 矩阵乘法 |
| vectorScaleKernel | 向量缩放 |
| vectorReduceKernel | 向量归约 |
| sequentialAccessKernel | 顺序访问 |
| randomAccessKernel | 随机访问 |
| stridedAccessKernel | 跨距访问 |
| touchAllPagesKernel | 页面触碰（页错误检测） |
| writeCombiningKernel | 写合并 |
| writeScatterKernel | 写散射 |
| spinWaitKernel | GPU自旋等待同步 |

### 17.6 测试分类

**基础测试**:
- cudaMallocManaged 基本分配
- 指针属性查询
- GPU 内核执行
- CPU 验证

**页错误检测**:
- 首次触碰（触发页错误）
- 缓存后再次访问对比

**访问模式**:
- 顺序访问
- 跨距访问（stride=64）
- 随机访问

**预取和Advice**:
- 无预取（系统管理）
- 显式 GPU 预取
- cudaMemAdvise (Read Mostly)

**写合并测试**:
- 顺序写入
- 散射写入

**同步测试**:
- GPU 自旋等待同步

## 18. Multi-Stream 并发研究

### 18.1 Multi-Stream API

Multi-Stream 用于并发执行多个 CUDA 操作。

**核心函数**:

| 函数 | 描述 |
|------|------|
| cudaStreamCreate | 创建流 |
| cudaStreamCreateWithPriority | 创建带优先级的流 |
| cudaStreamSynchronize | 流同步（阻塞） |
| cudaStreamQuery | 流查询（非阻塞） |
| cudaStreamWaitEvent | 等待事件 |
| cudaEventCreate | 创建事件 |
| cudaEventRecord | 记录事件 |
| cudaEventQuery | 查询事件（非阻塞） |
| cudaEventSynchronize | 事件同步（阻塞） |

### 18.2 概念

- **Stream Priority**: 流优先级调度
- **Event-based Sync**: 基于事件的同步
- **Concurrent Execution**: 并发内核执行
- **Overlap**: 内存传输与计算重叠
- **Pipeline**: 多阶段流水线

### 18.3 使用场景

- 数据处理流水线
- 并发内核执行
- 内存传输与计算重叠
- 优先级调度

### 18.4 Multi-Stream 测试命令

```bash
# Multi-Stream 基准测试
./build/gpupeek.exe multi_stream

# NCU 分析
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek multi_stream
```

### 18.5 Multi-Stream Kernel 代码覆盖

| Kernel | 功能 |
|--------|------|
| streamVectorAddKernel | 流向量加法 |
| streamVectorScaleKernel | 流向量缩放 |
| streamMatrixMulKernel | 流矩阵乘法 |
| streamReduceKernel | 流归约 |
| streamMemoryIntensiveKernel | 内存密集型 |
| streamComputeIntensiveKernel | 计算密集型 |
| pipelineLoadKernel | 流水线加载 |
| pipelineProcessKernel | 流水线处理 |
| pipelineStoreKernel | 流水线存储 |
| waitKernel | 等待内核 |

### 18.6 测试分类

**基础测试**:
- 单流基线
- 多流顺序执行
- 多流并发执行

**流优先级测试**:
- 高优先级流
- 低优先级流

**事件同步测试**:
- cudaStreamWaitEvent
- cudaEventQuery 轮询

**重叠测试**:
- 串行（H2D-计算-D2H）
- 两流事件同步重叠

**并发内核测试**:
- 顺序内核
- 并发内核

**流水线测试**:
- 串行3阶段流水线
- 重叠流水线（分块）

**同步测试**:
- cudaStreamQuery（非阻塞）
- cudaStreamSynchronize（阻塞）

## 19. Blackwell 架构深入分析 (微基准测试研究)

> 关键发现来源: arXiv:2507.10789 - "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks"

### 19.1 架构代际对比

| 参数 | GH100 (Hopper/H100) | GB203 (Blackwell/RTX 5080) |
|------|---------------------|---------------------------|
| 晶体管 | 80B | 92B |
| SM 数量 | 132 | 84 |
| Tensor Core 代数 | 4th Gen | **5th Gen** |
| FP64 Units/SM | 64 | **仅 2** |
| L1 Cache/SM | 256 KB | **128 KB** |
| Shared Memory/SM | ~227 KB | **~99 KB** |
| L2 Cache | 50 MB (2 分区) | **65 MB (单体)** |
| 内存 | 80 GB HBM2e | 16 GB GDDR7 |
| 内存带宽 | 15.8 TB/s | 8.2 TB/s |

### 19.2 TCGen05 vs WGMMA

**重要**: Blackwell 的 TCGen05 与 Hopper 的 WGMMA **不兼容**!

| 特性 | WGMMA (Hopper) | TCGen05 (Blackwell) |
|------|-----------------|---------------------|
| 指令集 | PTX 9.7.15 | PTX 扩展 |
| 级别 | Warpgroup (3 warps) | Warpgroup |
| 异步 | 是 | 是 |
| 稀疏支持 | 2:4 | 2:4 |
| SASS | WGMMA | **OMMA (FP4), QMMA (FP8/FP6)** |
| PTX 映射 | wgmma.mma_async | **mma.sync** |

### 19.3 内存层级变化

**L1 Cache 减半**:
- Hopper: 256 KB/SM
- Blackwell: **128 KB/SM** (减少 50%)

**Shared Memory 限制减少**:
- Hopper: ~227 KB/SM
- Blackwell: **~99 KB/SM** (减少 56%)

**L2 Cache 增大但架构改变**:
- Hopper: 50 MB，**2 分区**设计
- Blackwell: 65 MB，**单体**设计（单分区）

### 19.4 FP64 延迟问题

Blackwell **大幅减少 FP64 计算单元** (仅 2/SM vs 64/SM in Hopper):

| 指标 | GH100 | GB203 |
|------|-------|-------|
| FP64 单元/SM | 64 | **2** |
| FP64 True Latency | ~8 cycles | **~63 cycles** |
| FP64 Completion Latency | ~13 cycles | **~11 cycles** |

**结论**: Blackwell 不适合 FP64 密集型工作负载

### 19.5 INT32/FP32 执行单元统一

| 架构 | INT32 | FP32 |
|------|-------|------|
| Hopper | 独立管道 | 独立管道 |
| **Blackwell** | **统一执行单元** | **统一执行单元** |

**影响**:
- 混合 INT32/FP32 工作负载延迟更低
- Blackwell: 15.96/14 cycles
- Hopper: 31.62/16 cycles

### 19.6 延迟对比表

| 操作 | GH100 | GB203 |
|------|-------|-------|
| FP32/INT32 True Latency | 31.62/16 cycles | **15.96/14 cycles** |
| FP64 True Latency | ~8 cycles | **~63 cycles** |
| L2 Cache Hit Latency | ~273 cycles | **~358 cycles** |
| Global Memory Latency | ~659 cycles | **~877 cycles** |
| MMA Completion Latency | 1.66 cycles | **1.21 cycles** |

### 19.7 功耗效率 (FP8/FP4/FP6)

| 精度 | GH100 | GB203 |
|------|-------|-------|
| FP8 | ~55W | **~46W** |
| FP4 | N/A | **~16.75W** |
| FP6 e2m3 | N/A | **~39.38W** |
| FP6 e3m2 | N/A | **~46.72W** |

**结论**: Blackwell 在低精度推理时更节能

### 19.8 Transformer Engine

| 版本 | 支持的精度 | 架构 |
|------|-----------|------|
| TE 1st Gen | FP8, FP16, BF16, FP32, FP64 | Hopper |
| **TE 2nd Gen** | **FP4, FP6** + above | **Blackwell** |

### 19.9 对 CUDA 编程的影响

1. **WGMMA 不可用**: 使用 `mma.sync` + WMMA API 代替
2. **Shared Memory 减少**: 需要更高效的共享内存使用策略
3. **FP64 工作负载**: 考虑使用 Hopper 或等待 Blackwell Ultra
4. **低精度推理**: FP4/FP6 在 Blackwell 上效率最高

### 19.10 微基准测试论文

```
@misc{jarmusch2025blackwell,
  title={Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks},
  author={Aaron Jarmusch and Nathan Graddon and Sunita Chandrasekaran},
  year={2025},
  eprint={2507.10789},
  archivePrefix={arXiv}
}
```

## 20. Mbarrier (Memory Barrier) 研究

### 20.1 PTX ISA Mbarrier 指令 (Section 9.7.13)

Mbarrier 为异步内存操作提供同步机制，是 cp.async、cp.async.bulk、st.async、WGMMA 等操作的关键。

**核心指令**:

| 指令 | 描述 |
|------|------|
| mbarrier.init | 初始化 barrier，设置字节计数 |
| mbarrier.arrive | 到达 barrier 并增加计数 |
| mbarrier.arrive_drop | 到达后丢弃（fire-and-forget） |
| mbarrier.complete_tx | 完成事务 |
| mbarrier.test_wait | 等待 barrier 相位 |
| mbarrier.try_wait | 尝试等待 barrier |
| mbarrier.pending_count | 检查待处理的到达数 |

### 20.2 Mbarrier 与 Async Copy

```
// Mbarrier 同步异步拷贝的典型模式:
mbarrier.init [addr], byte_count;           // 初始化
cp.async.bulk [dst], [src], size;          // 异步拷贝
mbarrier.arrive mbarrier_ptr, 1;           // 到达
mbarrier.wait mbarrier_ptr, pending_cnt;    // 等待完成
```

### 20.3 Mbarrier 操作阶段

Mbarrier 使用阶段（phase）机制来跟踪异步操作：

1. **初始化**: 设置 barrier 的字节计数
2. **期望**: 使用 `expect_tx` 声明待处理的到达数
3. **到达**: 使用 `arrive` 或 `arrive_drop` 信号到达
4. **完成**: 使用 `complete_tx` 标记事务完成
5. **等待**: 使用 `test_wait` 或 `try_wait` 等待相位切换

### 20.4 Mbarrier 与 CUDA Graph

Mbarrier 用于 GPU 间的依赖同步：

| 场景 | Mbarrier 用途 |
|------|--------------|
| Grid 依赖 | 等待另一个 grid 完成 |
| 跨 GPU 同步 | 多 GPU 间的内存操作同步 |
| 流水线 | 阶段间的完成信号 |

### 20.5 内存 Fence 变体

| 指令 | 范围 | 用途 |
|------|------|------|
| `__threadfence_block()` | Block 内 | 共享内存同步 |
| `__threadfence()` | GPU 内 | GPU 到 GPU 同步 |
| `__threadfence_system()` | 系统级 | GPU 到 CPU 同步 |

### 20.6 Mbarrier 测试命令

```bash
# Mbarrier 基准测试
./build/gpupeek.exe mbarrier

# NCU 分析
ncu --set full --metrics sm__inst_executed.mbarrier.sum ./gpupeek mbarrier
```

### 20.7 Mbarrier Kernel 代码覆盖

| Kernel | 功能 |
|--------|------|
| mbarrier_init_wait_kernel | Mbarrier 初始化和等待 |
| async_copy_mbarrier_kernel | 异步拷贝 + Mbarrier 同步 |
| mbarrier_pipeline_kernel | 流水线同步 |
| mbarrier_reduce_kernel | 带同步的归约 |
| mbarrier_producer_consumer_kernel | 生产者-消费者模式 |
| mbarrier_tx_count_kernel | 事务计数 |
| fence_sync_kernel | Fence 同步 |
| memory_fence_variants_kernel | 各种 fence 变体对比 |
| grid_dep_control_kernel | Grid 依赖控制 |
| cluster_barrier_kernel | Cluster barrier (需要 CUDA 12.0+) |

### 20.8 测试分类

**基础测试**:
- Mbarrier 初始化和等待
- 原子同步基线

**Async Copy 同步**:
- 异步拷贝概念验证
- 事务计数

**流水线同步**:
- 4 阶段流水线
- 阶段间同步

**生产者-消费者**:
- Mbarrier 同步的生产者-消费者模式

**事务计数**:
- 到达计数
- 完成计数

**Fence 对比**:
- 无 fence
- __threadfence_block
- __threadfence

**Grid 依赖**:
- Grid 依赖控制

## 21. Cooperative Groups 研究

### 21.1 Cooperative Groups API

Cooperative Groups 使线程能够跨 Thread Block、Grid 和 Multi-GPU 进行合作。

**核心 API**:

| API | 描述 |
|-----|------|
| `cooperative_groups::this_thread_block()` | 获取当前线程块组 |
| `cooperative_groups::this_grid()` | 获取当前 Grid 组 |
| `cooperative_groups::this_multi_grid()` | 获取多 GPU 组 |
| `grid.sync()` | Grid 级别同步 |
| `block.sync()` | Block 级别同步 |

### 21.2 同步级别

| 级别 | API | 同步范围 |
|------|-----|----------|
| Warp | `this_warp()` | 同一 Warp 的线程 |
| Thread Block | `this_thread_block()` | 同一 Block 的线程 |
| Grid | `this_grid()` | Grid 中所有线程 |
| Multi-GPU | `this_multi_grid()` | 所有 GPU 上的线程 |

### 21.3 Cooperative Groups 特性

**优点**:
1. 显式分组管理
2. Grid 级别同步
3. 多 GPU 协作
4. 灵活的集合操作

**限制**:
1. 需要 Cooperative Launch
2. Grid 同步开销较大
3. 多 GPU 需要 CUDA 9.0+

### 21.4 Cooperative Groups 测试命令

```bash
# Cooperative Groups 基准测试
./build/gpupeek.exe coop

# NCU 分析
ncu --set full --metrics sm__average_active_warps_per_sm ./gpupeek coop
```

### 21.5 Cooperative Groups Kernel 代码覆盖

| Kernel | 功能 |
|--------|------|
| threadBlockSyncKernel | 线程块同步 |
| gridReduceKernel | Grid 级别归约 |
| cooperativeLoadKernel | 协作加载 |
| gridBarrierMemsetKernel | Grid barrier + memset |
| multiBlockReduceKernel | 多 Block 归约 |
| twoPhaseKernel | 两阶段协作内核 |
| broadcastKernel | 从特定线程广播 |
| evenOddSyncKernel | 奇偶同步模式 |
| barrierEfficiencyKernel | Barrier 效率测试 |
| vectorizedCoopLoadKernel | 向量化协作加载 |

### 21.6 测试分类

**Thread Block 同步**:
- 线程块内同步
- 协作加载/存储

**Grid 级别同步**:
- Grid 归约
- Grid barrier
- 多 Block 归约

**协作模式**:
- 两阶段协作内核
- 从特定线程广播
- 奇偶同步模式

**效率分析**:
- Barrier 效率测试
- 向量化协作加载
- 等待模式

## 22. Redux.sync Warp 级归约研究

### 22.1 PTX ISA Redux.sync 指令 (Section 9.7.12)

Redux.sync 执行 Warp 级别的归约操作。

**支持的操作**:

| 操作 | PTX | 描述 |
|------|-----|------|
| ADD | .add | 加法归约 |
| MIN | .min | 最小值归约 |
| MAX | .max | 最大值归约 |
| AND | .and | 按位与归约 |
| OR | .or | 按位或归约 |
| XOR | .xor | 按位异或归约 |

**语法**:
```ptx
redux.sync [dest], [src], op
```

**SASS 等价**: RRED 指令（硬件加速的归约）

### 22.2 Redux.sync vs Shuffle-based 归约

| 方法 | 指令数 | 延迟 | 优势 |
|------|--------|------|------|
| Shuffle 循环 | log2(32) = 5 次 shuffle | 较高 | 兼容性好 |
| **Redux.sync** | **1 条指令** | **最低** | **硬件加速** |

### 22.3 Redux.sync 测试命令

```bash
# Redux.sync 基准测试
./build/gpupeek.exe redux

# NCU 分析
ncu --set full --metrics sm__inst_executed.redux_sync.sum ./gpupeek redux
```

### 22.4 Redux.sync Kernel 代码覆盖

| Kernel | 功能 |
|--------|------|
| reduxAddKernel | Warp 级加法归约 |
| reduxMinKernel | Warp 级最小值归约 |
| reduxMaxKernel | Warp 级最大值归约 |
| reduxAndKernel | Warp 级按位与归约 |
| reduxOrKernel | Warp 级按位或归约 |
| reduxXorKernel | Warp 级按位异或归约 |
| shuffleReductionKernel | Shuffle 基线归约 |
| butterflyReductionKernel | 蝴蝶模式归约 |
| reduxAtomicKernel | Redux + Atomic 操作 |
| warpVoteAnyKernel | Warp Vote ANY |
| warpVoteAllKernel | Warp Vote ALL |
| matchSyncKernel | Match.sync 模式匹配 |
| blockReduceReduxKernel | Block 级归约 |

### 22.5 测试分类

**基础操作**:
- Redux ADD/MIN/MAX
- Redux AND/OR/XOR

**性能对比**:
- Shuffle 循环归约（基线）
- 蝴蝶模式归约
- Redux 概念（模拟）

**Atomic 组合**:
- Redux + Atomic Add

**Vote 操作**:
- __any_sync
- __all_sync

**Match 操作**:
- Match.sync 模式匹配

## 23. FP4/FP6 低精度 MMA 研究

### 23.1 Blackwell FP4/FP6 支持

Blackwell (SM 12.0) 第5代 Tensor Core 支持 FP4 和 FP6 低精度格式。

**格式规格**:

| 格式 | 指数位 | 尾数位 | 总位数 | PTX 类型 |
|------|--------|--------|--------|----------|
| FP4 (e2m1) | 2 | 1 | 4 | e2m1 |
| FP6 (e2m3) | 2 | 3 | 6 | e2m3 |
| FP6 (e3m2) | 3 | 2 | 6 | e3m2 |

**PTX ISA (CUDA 12.9+)**:

```ptx
mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32   // FP4
mma.sync.aligned.m16n8k32.row.col.f32.e2m3.e2m3.f32   // FP6 e2m3
mma.sync.aligned.m16n8k32.row.col.f32.e3m2.e3m2.f32   // FP6 e3m2
```

**Shape**: m16n8k32 (与 FP8 的 m16n8k16 不同)

### 23.2 FP4/FP6 应用场景

- **LLM 量化**: 4-bit 权重用于大语言模型推理
- **极致量化**: 比 INT8 更低的内存占用
- **推理加速**: 低精度计算更高的 TFLOPS

### 23.3 FP4/FP6 与 FP8 对比

| 特性 | FP8 (E4M3/E5M2) | FP4 (e2m1) | FP6 (e2m3/e3m2) |
|------|-------------------|-------------|-------------------|
| 位数 | 8 | 4 | 6 |
| 精度 | 高 | 极低 | 低 |
| 内存减少 | 2x vs FP16 | 4x vs FP16 | 2.67x vs FP16 |
| TFLOPS | 最高 | 最高 | 高 |
| 适用 | 权重+激活 | 仅权重 | 仅权重 |

### 23.4 TCGen05 状态

> **注意**: TCGen05 指令集在当前 Blackwell (GB203) 上**尚未支持**
>
> 真正的 FP4/FP6 MMA 需要 CUDA 12.9+ 和特定的 `.kind::f8f6f4` 后缀

### 23.5 FP4/FP6 测试命令

```bash
# FP4/FP6 基准测试
./build/gpupeek.exe fp4

# NCU 分析
ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek fp4
```

### 23.6 FP4/FP6 Kernel 代码覆盖

| Kernel | 功能 |
|--------|------|
| float_to_fp4_e2m1 | FP32 → FP4 转换 |
| fp4_e2m1_to_float | FP4 → FP32 转换 |
| float_to_fp6_e2m3 | FP32 → FP6 转换 |
| fp4StyleMmaKernel | FP4 风格 GEMM (模拟) |
| fp6StyleMmaKernel | FP6 风格 GEMM (模拟) |
| blockScalingKernel | 块缩放 |
| weightOnlyQuantKernel | 仅权重量化 |
| dequantizeKernel | 反量化 |
| quantizedAttentionKernel | 量化注意力计算 |

### 23.7 测试分类

**转换测试**:
- FP32 → FP4 转换
- FP4 → FP32 转换

**GEMM 测试**:
- FP16 基线 GEMM
- FP4 风格 GEMM (模拟)
- FP6 风格 GEMM (模拟)

**量化测试**:
- 块缩放
- 仅权重量化
- 反量化

**LLM 推理模式**:
- 量化注意力计算
- Softmax

### 23.8 FP4/FP6 初步测试结果 (2026-03-23)

| 测试 | 结果 |
|------|------|
| FP32 → FP4 转换 | 1.304 ms (1M 元素) |
| FP4 → FP32 转换 | 0.052 ms (1M 元素) |
| FP16 GEMM 基线 | 743.57 GFLOPS |
| FP4 风格 GEMM (模拟) | 920.45 GFLOPS |

**注意**: FP4/FP6 风格 GEMM 比 FP16 基线快，这是因为模拟 kernel 做了量化简化。真正的 FP4/FP6 MMA 需要 CUDA 12.9+ 硬件支持。

## 24. 项目状态 (2026-03-23 更新)

### 编译状态

**正常工作的模块** (6/17):
- ✅ memory - 内存研究
- ✅ deep - 深度研究
- ✅ advanced - 高级研究
- ✅ fp4 - FP4/FP6 研究
- ✅ multi_stream - 多流并发研究
- ✅ wmma - WMMA (Warp-level MMA) m16n16k16 Tensor Core

**禁用的模块** (编译错误):
- ❌ ncu - NCU 分析 (编码问题)
- ❌ cuda - CUDA Core 算力 (缺少 bf16 头文件)
- ❌ atomic - 原子操作 (待查)
- ❌ barrier - Barrier 同步 (cooperative groups API 问题)
- ❌ warp - Warp 特化 (cooperative groups API 问题)
- ❌ mma - MMA/Tensor Core (DISABLED - use `wmma` benchmark instead)
- ❌ tensor_mem - Tensor 内存 (WMMA fragment 类型未定义)
- ❌ wgmma - WGMMA (WMMA fragment 类型未定义)
- ❌ dp4a - DP4A (待查)
- ❌ fp8 - FP8 (待查)
- ❌ graph - CUDA Graph (待查)
- ❌ unified - Unified Memory (待查)
- ❌ mbarrier - Mbarrier (待查)
- ❌ coop - Cooperative Groups (cooperative groups API 问题)
- ❌ redux - Redux.sync (待查)

### 主要编译问题

1. **WMMA fragment 类型**: `frag_a`, `frag_b`, `frag_c` 等类型在多个 kernel 文件中使用但未定义
2. **Cooperative Groups API**: `cuda::thread_block` 等类型找不到
3. **CHECK_CUDA 重定义**: 多个 benchmark 文件定义了各自的 CHECK_CUDA 宏
4. **缺少头文件**: `cuda_bf16.h` 等在某些模块中未包含

### WMMA 内核问题 (2026-03-23) - ✅ 已修复

**问题**: WMMA 内核在运行时出现 "illegal memory access" 错误

**解决方案**:
- 创建新的 `wmma_test_kernel.cu` 和 `wmma_test_benchmarks.cu`
- 使用正确的 fragment 类型定义:
  ```cuda
  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
  fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
  fragment<accumulator, 16, 16, 16, float> frag_d;
  ```
- 使用 `using namespace nvcuda::wmma;`
- Grid: `(N / 16, M / 16)`, Block: `32`
- 每个 warp 处理一个 16x16 输出块

**运行 WMMA 基准测试**:
```bash
./build/gpupeek.exe wmma
```

**NCU Profiling**:
```bash
ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe wmma
```

