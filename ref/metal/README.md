# Apple Metal 参考文档

本目录存放Apple Metal GPU编程和架构相关的官方参考资料。

## 已下载的文档

| 文件 | 描述 | 大小 |
|------|-------|------|
| `Metal_Shading_Language_Specification.pdf` | Metal着色语言规范 (v1.2+) | ~11.7MB |

## 官方在线文档

### Metal 框架

| 文档 | URL |
|------|-----|
| Metal 框架概览 | https://developer.apple.com/documentation/metal |
| Metal 编程指南 | https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Introduction/Introduction.html |
| Metal 最佳实践指南 | https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html |

### Metal 性能优化

| 文档 | URL |
|------|-----|
| GPU性能优化 (Xcode) | https://developer.apple.com/documentation/xcode/optimizing-gpu-performance |
| Metal调试工具 | https://developer.apple.com/documentation/metal/frame_capture_debugging_tools |
| WWDC20: GPU性能计数器 | https://developer.apple.com/videos/play/wwdc2020/10603/ |
| WWDC20: Harness Apple GPUs with Metal | https://developer.apple.com/videos/play/wwdc2020/10602/ |

### Metal Performance Shaders (MPS)

| 文档 | URL |
|------|-----|
| MPS框架概览 | https://developer.apple.com/documentation/metalperformanceshaders/ |
| MPS核函数 | https://developer.apple.com/documentation/metalperformanceshaders/metalperformanceshaders_structures |
| MPS神经网络图 | https://developer.apple.com/documentation/metalperformanceshaders/mpsnngraph |
| WWDC24: 机器学习训练 | https://developer.apple.com/videos/play/wwdc2024/10160 |

### Metal 着色语言

| 文档 | URL |
|------|-----|
| Metal着色语言规范 (PDF) | https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf |
| Metal By Example (教程) | https://metalbyexample.com |

### GPU 架构

| 文档 | URL |
|------|-----|
| Apple GPU功能集 | https://developer.apple.com/documentation/metal/metal_feature_set_tables |
| GPU Family 支持 | https://developer.apple.com/documentation/metal/selecting_a_gpu_to_render_to |

### 工具和调试

| 文档 | URL |
|------|-----|
| Metal调试工具 | https://developer.apple.com/documentation/metal/frame_capture_debugging_tools |
| GPU frame capture | https://developer.apple.com/documentation/metal/frame_capture_debugging_tools/analyzing_draw_command_and_compute_dispatch_performance_with_gpu_counters |
| 纹理资源调试 | https://developer.apple.com/documentation/metal/frame_capture_debugging_tools/debugging_texture_resource_loading |

## 重要概念

### 1. 统一内存架构 (Unified Memory)
Apple GPU与CPU共享内存，无需显式拷贝：
- 带宽: M1 68GB/s, M2 100GB/s, M4 Max 546GB/s
- 优势: 零拷贝延迟，简化编程模型

### 2. TBDR (Tile-Based Deferred Rendering)
Apple GPU使用瓦片式延迟渲染：
- 将画面分成小瓦片处理
- 大幅减少带宽消耗
- 支持隐式表面消除

### 3. GPU Family
| Family | 代表芯片 | 特性 |
|--------|---------|------|
| Apple 7+ | M2, M3 | 最新特性 |
| Apple 6 | M1 | 基础特性 |
| Apple 5 | A13 | 纹理查询改进 |

### 4. Memory Storage Mode
| 模式 | 说明 |
|------|------|
| Shared | CPU和GPU共享（统一内存） |
| Private | GPU专用 |
| Managed | 协同管理 |

## 第三方教程

- [Metal By Example](https://metalbyexample.com) - 入门教程
- [Ray Wenderlich Metal Tutorial](https://www.raywenderlich.com/7475-metal-tutorial-getting-started)
- [WWDC视频](https://developer.apple.com/videos) - 历年WWDC Metal相关视频

## 维护说明

- 定期检查Apple开发者网站获取最新文档
- Metal Shading Language Specification PDF会随Xcode更新
- 关注WWDC新技术发布
