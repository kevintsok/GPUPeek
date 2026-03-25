# FP64 (Double Precision) Research

## 概述

本专题研究Apple M2 GPU的FP64（双精度浮点）支持情况。由于Apple M2采用统一内存架构，其GPU并不原生支持FP64计算。

## 关键发现

### FP64支持情况

| GPU | FP64支持 | 说明 |
|-----|---------|------|
| Apple M1 | ❌ 不支持 | 无FP64硬件单元 |
| Apple M2 | ❌ 不支持 | 无FP64硬件单元 |
| Apple M3 Pro | ❌ 不支持 | 统一内存架构 |
| NVIDIA RTX 4090 | ✅ 支持 | 专用FP64单元 |

### 性能对比（理论值）

| 精度 | Apple M2 | NVIDIA RTX 4090 |
|------|----------|-----------------|
| FP32 | 3.5 TFLOPS | 82.6 TFLOPS |
| FP64 | N/A | 51.0 TFLOPS |
| FP64比率 | N/A | 61.7% |

### Apple M2限制原因

1. **统一内存架构** - M系列芯片使用统一内存，GPU与CPU共享内存控制器
2. **功耗优化** - FP64单元面积大，Apple选择不使用
3. **目标场景** - Apple设备主要面向移动和ML，非科学计算

## 替代方案

### 使用FP32模拟FP64
```metal
// 使用两个FP32模拟FP64
struct DoubleFloat {
    float hi;
    float lo;
};

// Double-double算法
float twoSum(float a, float b, thread float& err) {
    float s = a + b;
    float z = s - a;
    err = b - z;
    return s;
}
```

### 应用场景建议

| 场景 | 推荐方案 |
|------|---------|
| 机器学习 | FP16/FP32已足够 |
| 游戏渲染 | FP32精度足够 |
| 科学计算 | 考虑NVIDIA GPU |
| 图像处理 | FP16/FP32足够 |

## 验证方法

```bash
# 检查GPU是否支持FP64
MTLGPUFamily.apple7 = M2 chip
# 实际测试：FP64操作会触发编译警告或错误
```

## 相关专题

- [FP32/FP16 Precision](../Precision/RESEARCH.md) - 精度分析
- [GEMM](../GEMM/RESEARCH.md) - 矩阵乘法性能
