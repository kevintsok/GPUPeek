# Bank Conflict Research Module

## 概述

本模块深入研究 NVIDIA GPU 共享内存 bank conflict（存储体冲突）行为，测量不同访问模式下的性能影响。

## 背景

### Blackwell (SM 12.0) Bank 配置
- **Bank 数量**: 32 banks
- **Bank 宽度**: 4 bytes (32 bits)
- **Bank 寻址**: `address % 32` 确定 bank
- **存储体冲突**: 多个线程在同一周期访问同一个 bank

### 关键概念
| 概念 | 描述 |
|------|------|
| Bank Conflict | 多线程同时访问同一 bank，导致串行化 |
| Broadcast | 所有线程读同一地址（无冲突） |
| Double Pump | 一条指令访问同一 bank 两次 |
| Padding | 通过填充避免 bank 映射冲突 |

## 编译和运行

### 编译

```bash
cd NVIDIA_GPU/sm_120/bank_conflict
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build . --config Release
```

### 运行

```bash
./gpupeek_bank_conflict
```

## 测试内容

### 1. 基本 Stride 测试
测量 stride = 1, 2, 4, 8, 16, 32, 64, 128 时的 bank conflict 性能。

### 2. 读写冲突对比
区分 strided read 和 strided write 的冲突开销。

### 3. Broadcast 效率
测试所有线程访问同一地址的广播效率。

### 4. Padding 策略
测试 padding = 0, 1, 2 时的 bank conflict 缓解效果。

### 5. Warp 数量分析
对比 1 warp, 2 warps, 4 warps, full block 的 bank conflict 表现。

### 6. 数据类型分析
float32, float64, float16 的 bank conflict 行为差异。

### 7. 矩阵转置
经典 bank conflict 场景：转置时有/无 padding 的性能对比。

### 8. Size Sweep
不同数据尺寸下的 bank conflict 行为。

## 核心发现

### Stride vs Bandwidth (实测数据)

| Stride | 带宽 (GB/s) | 相对性能 | Bank Conflict |
|--------|-------------|----------|--------------|
| 1 | 729.41 | 100.0% | 无 |
| 2 | 736.80 | 101.0% | 无 |
| 4 | 489.65 | 67.1% | 低 |
| 8 | 285.56 | 39.1% | 高 |
| 16 | 285.94 | 39.2% | 严重 |
| 32 | 266.00 | 36.5% | **最大** |
| 64 | 464.22 | 63.6% | 周期性 |
| 128 | 669.90 | 91.8% | 低 |

**Stride = 32 时性能降至 36.5%**，证实了最大 bank conflict。

### Padding 效果 (Stride=32)

| Padding | 存储倍数 | 带宽 (GB/s) | 相对性能 |
|---------|----------|-------------|----------|
| 0 | 1x | 332.14 | 100.0% |
| 1 | 2x | 351.45 | 105.8% |
| 2 | 3x | 339.98 | 102.4% |

Padding 带来约 5.8% 的提升，但不如理论预期显著。

### 经典优化: 矩阵转置

| 实现 | 带宽 (GB/s) | 相对性能 |
|------|-------------|----------|
| 转置 + Padding (33列) | 251.3 | 135.1% |
| 转置 无 Padding | 186.1 | 100.0% |

**Padding 带来 35.1% 的提升**，这是 bank conflict 优化的经典案例。

### Broadcast 效率

| 访问模式 | 带宽 (GB/s) | 说明 |
|----------|-------------|------|
| Broadcast (同地址) | 1101.75 | 无冲突，硬件广播 |
| Strided (stride=32) | 287.8 | 串行化冲突 |

**Broadcast 优势: 3.8x**

## 数据文件

CSV 数据保存在 `data/` 目录：
- `bank_conflict_stride_data.csv` - Stride 测试数据
- `bank_conflict_padding_data.csv` - Padding 测试数据

## NCU 分析建议

使用以下指标分析 bank conflict:

```bash
ncu --set full ./gpupeek_bank_conflict
```

关键指标:
- `sm__pipe_shared_cycles_active.pct` - 共享内存带宽利用率
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` - 全局加载
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` - 全局存储

## 最佳实践

1. **避免 stride = 32**: 所有线程访问同一 bank，性能降至 36%
2. **使用 padding**: 矩阵转置时 35%+ 性能提升
3. **利用 broadcast**: 多线程读同地址无冲突
4. **选择合适数据类型**: double 冲突模式更优
