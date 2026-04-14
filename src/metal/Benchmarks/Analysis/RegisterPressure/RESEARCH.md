# Register Pressure Research

## 概述

本专题研究GPU上寄存器压力（Register Pressure）对性能的影响。GPU有有限的寄存器文件，当一个线程使用的寄存器超过可用数量时，会发生寄存器溢出（spilling）到共享内存或全局内存，导致性能下降。

## 关键发现

### 寄存器压力性能对比

| 配置 | 寄存器数 | 性能 | 说明 |
|------|---------|------|------|
| Low | 1 | 基准 | 最小压力 |
| Medium | 4 | ~100% | 无明显损失 |
| High | 8 | ~95-100% | 轻微压力 |
| Very High | 16 | ~70-90% | 可能溢出 |

### Apple M2 GPU寄存器特性

| 特性 | 值 |
|------|-----|
| 每线程寄存器上限 | 128 (估计) |
| 每SIMD组寄存器 | 32KB (估计) |
| 溢出代价 | 10-50x 性能损失 |

### 寄存器压力与Occupancy关系

```
高寄存器使用
    ↓
低occupancy (每SM少线程)
    ↓
高每个线程计算资源
    ↓
可能性能提升(如果计算密集)

vs

低寄存器使用
    ↓
高occupancy (每SM多线程)
    ↓
更好内存延迟隐藏
    ↓
内存密集型应用性能提升
```

## 优化策略

### 1. 减少寄存器使用
```metal
// 高压力：多个中间变量
float a = in[id];
float b = a * 2.0f;
float c = b + a;
float d = c * 0.5f;
out = d + 1.0f;

// 低压力：重用变量
float val = in[id];
val = val * 2.0f;
val = val + in[id];
val = val * 0.5f;
val = val + 1.0f;
out = val;
```

### 2. 循环体优化
```metal
// 高压力：循环内大量临时变量
for (uint i = 0; i < 32; i++) {
    float v0 = ...;
    float v1 = ...;
    float v2 = ...;
    float v3 = ...;
    sum += v0 + v1 + v2 + v3;
}

// 低压力：最小化循环内变量
for (uint i = 0; i < 32; i++) {
    sum += in[(id + i) % size];
}
```

### 3. 寄存器 vs 共享内存权衡
- 计算密集型：用寄存器存储中间结果
- 内存密集型：用共享内存缓存，减少寄存器使用

## 性能影响因子

1. **寄存器数量** - 越多可同时保持越多数据
2. **Occupancy** - 高occupancy掩盖内存延迟
3. **溢出代价** - 溢出到内存导致10-50x损失
4. **指令类型** - 算术指令使用寄存器多

## 相关专题

- [Occupancy](../Occupancy/RESEARCH.md) - 占用率与寄存器关系
- [BankConflict](../BankConflict/RESEARCH.md) - 共享内存冲突
- [GEMM](../GEMM/RESEARCH.md) - 寄存器分块优化
