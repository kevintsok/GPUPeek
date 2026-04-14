# Monte Carlo / Random Number Generation Research

## 概述

本专题研究GPU上随机数生成（PRNG）和蒙特卡洛模拟的性能。随机数生成是科学计算、金融模拟和机器学习中的基础操作。

## 关键发现

### PRNG性能对比

| 算法 | 性能 | 质量 | 说明 |
|------|------|------|------|
| XOR-shift | ~5-10 GR/s | 中等 | 最快 |
| LCG | ~3-5 GR/s | 较低 | 最简单 |
| PCG | ~2-4 GR/s | 高 | 最佳质量 |
| Hash-based | ~1-2 GR/s | 高 | 最灵活 |

### 随机数生成器类型

```
1. 线性同余生成器 (LCG)
   Xn+1 = (a*Xn + c) mod m
   优点: 简单
   缺点: 质量较低

2. XOR-shift
   s ^= s << a
   s ^= s >> b
   s ^= s << c
   优点: 快速
   缺点: 有短周期

3. PCG (Permuted Congruential Generator)
   组合LCG + 输出函数
   优点: 高质量
   缺点: 较慢
```

### 蒙特卡洛方法

```
Pi 估算:
- 在单位正方形内随机投点
- 计算落在单位圆内的比例
- Pi ≈ 4 * (inside / total)

收敛速度: O(1/sqrt(n))
```

## 优化策略

### 1. 批量生成
```metal
// 每个线程生成自己的随机数
uint s = seed[0];
s ^= s << 13;
s ^= s >> 17;
s ^= s << 5;
output[id] = s;
seed[0] = s;
```

### 2. Box-Muller变换
```metal
// 均匀分布转高斯分布
float z0 = sqrt(-2.0f * log(x1)) * cos(2.0f * M_PI_F * x2);
```

### 3. 避免原子操作
```metal
// Monte Carlo中每个线程独立生成
float px = x[id];
float py = y[id];
// 不要用原子操作更新共享计数器
```

## 性能影响因子

1. **PRNG算法复杂度** - XOR-shift > LCG > PCG
2. **原子操作** - Monte Carlo中的计数器更新
3. **内存带宽** - 大量随机数生成
4. **线程束效率** - 独立随机数生成

## 应用场景

1. **金融模拟** - 期权定价、风险分析
2. **科学计算** - 统计物理、量子蒙特卡洛
3. **机器学习** - 随机梯度下降、Dropout
4. **游戏** - 噪声生成、随机地形

## 相关专题

- [Atomics](../../Synchronization/Atomics/RESEARCH.md) - 原子操作性能
- [Scan](../Scan/RESEARCH.md) - 并行归约
- [InstructionMix](../../Compute/InstructionMix/RESEARCH.md) - 指令混合
