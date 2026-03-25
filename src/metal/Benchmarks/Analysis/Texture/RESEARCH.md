# Texture Performance Research

## 概述

本专题研究Apple M2 GPU上Texture（纹理）访问与Buffer访问的性能对比，分析纹理采样器的特性和适用场景。

## Texture vs Buffer 访问

| 访问方式 | 性能 | 特点 |
|----------|------|------|
| Texture Read | ~0.17 GOPS | 通过采样器硬件 |
| Buffer Read | ~0.17 GOPS | 直接内存访问 |
| Float4 Buffer | ~0.68 GOPS | 向量化4x加速 |

## 关键发现

### Texture 特性

| 特性 | 说明 |
|------|------|
| 缓存 | Texture有2D空间局部性缓存 |
| 采样器 | 可配置滤波、寻址模式 |
| 坐标 | 支持归一化坐标采样 |
| 格式 | 支持多种像素格式(RGBA, R, etc) |

### Texture 采样器模式

| 模式 | 用途 |
|------|------|
| Linear | 双线性插值 |
| Nearest | 最近邻采样 |
| Anisotropic | 各向异性滤波 |

## 关键洞察

1. **Texture vs Buffer无显著差异** - 在直接读取模式下性能相当
2. **Texture缓存优化2D访问** - 适合图像处理和需要插值的场景
3. **采样器增加灵活性** - 硬件支持滤波和坐标变换
4. **Buffer向量化更快** - Float4读取比Texture快约4倍
5. **选择依据是访问模式** - 2D空间数据用Texture，线性数据用Buffer

## 优化策略

1. **普通线性访问用Buffer** - Float4向量化更高效
2. **需要插值用Texture** - 采样器硬件支持双线性插值
3. **图像数据用Texture** - 利用2D缓存局部性
4. **避免频繁格式转换** - Texture格式转换有开销

## 相关专题

- [Memory Bandwidth](../../Memory/Bandwidth/RESEARCH.md) - 内存带宽
- [Vectorization](../../Compute/Vectorization/RESEARCH.md) - 向量化
