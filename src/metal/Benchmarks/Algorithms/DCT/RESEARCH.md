# DCT (Discrete Cosine Transform) Research

## 概述

本专题研究GPU上离散余弦变换（DCT）的性能。DCT是JPEG图像压缩和视频编码中的核心算法，广泛应用于多媒体处理领域。

## 关键发现

### DCT性能对比

| 算法 | 复杂度 | 性能 | 说明 |
|------|---------|------|------|
| Naive DCT | O(N²) | ~0.01 GFLOPS | 简单但慢 |
| Butterfly DCT | O(N log N) | ~0.5-1 GFLOPS | 实际可用 |
| 2D DCT | O(N² log N) | ~5-20 MP/s | 行列分离 |

### DCT算法对比

```
Naive DCT:
for k in 0..N:
    for n in 0..N:
        y[k] += x[n] * cos(pi * k * (n + 0.5) / N)
复杂度: O(N²) per output

Butterfly DCT:
- 利用DCT对称性
- 类似FFT的蝶形结构
- 复杂度: O(N log N)
```

### DCT应用

1. **JPEG压缩** - 8x8块DCT
2. **视频编码** - H.264/H.265使用DCT
3. **信号处理** - 频域分析
4. **图像滤波** - 频域卷积

## 优化策略

### 1. 行列分离2D DCT
```metal
// Row pass
dct_1d_on_each_row();

// Column pass
dct_1d_on_each_column();
```

### 2. 蝶形优化
```metal
// 利用对称性减少计算
y0 = even + odd;
y1 = (even - odd) * twiddle;
```

### 3. 查表法
```metal
// 预计算cos值
constant float cos_table[256] = {...};
sum += in[n] * cos_table[k * n % 256];
```

## 性能影响因子

1. **DCT大小** - 8x8是JPEG标准块大小
2. **算法复杂度** - O(N²) vs O(N log N)
3. **内存带宽** - 2D DCT需要多次内存访问
4. **蝶形效率** - 依赖SIMD组效率

## 与FFT的关系

| 特性 | DCT | FFT |
|------|-----|-----|
| 输入 | 实数 | 复数 |
| 系数 | 仅余弦 | 正弦+余弦 |
| 应用 | 压缩 | 通用频谱 |
| 效率 | 类似 | 类似 |

## 相关专题

- [FFT](../FFT/RESEARCH.md) - 快速傅里叶变换
- [ImageProcessing](../ImageProcessing/RESEARCH.md) - 图像处理
- [Compression](../Compression/RESEARCH.md) - 数据压缩
