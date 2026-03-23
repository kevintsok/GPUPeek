# Apple Metal GPU Benchmark

Apple M系列芯片GPU基准测试，使用Metal API。

## 硬件支持

| 芯片 | GPU核心 | 统一内存带宽 | 状态 |
|------|---------|-------------|------|
| M1 | 8核 | 68.25 GB/s | 待测试 |
| M2 | 10核 | 100 GB/s | 已测试 |
| M3 | 10核 | 100 GB/s | 待测试 |
| M4 | 10核 | 120 GB/s | 待测试 |
| M2 Max | 38核 | 400 GB/s | 待测试 |
| M3 Max | 40核 | 400 GB/s | 待测试 |
| M4 Max | 40核 | 546 GB/s | 待测试 |

## 快速开始

### 前置要求

- macOS 13.0+
- Xcode Command Line Tools
- Swift 6.0+

```bash
# 检查工具
swift --version
xcode-select -p
```

### 编译运行

```bash
cd src/metal

# Debug模式编译
swift build

# Release模式编译（推荐，更快）
swift build --configuration release

# 运行
swift run --configuration release
```

### 运行结果示例

```
Apple Metal GPU Benchmark
======================================

=== Apple Metal GPU Info ===
Device Name: Apple M2
Unified Memory: Yes (Shared with CPU)
Max Threadgroup Memory: 32 KB
Max Threads Per Threadgroup: 1024
GPU Family: Apple 7+

Shader compilation: SUCCESS

=== Memory Copy Bandwidth Test ===
Buffer Size: 256.00 MB
Iterations: 100
Total Time: 26118.688 ms
Bandwidth: 1.03 GB/s

=== FP32 Matrix Multiply Test ===
Matrix Size: 512x512x512
Iterations: 10
Time: 665.203 ms
Performance: 4.04 GFLOPS

Benchmark completed.
```

## 测试项目

### 内存带宽测试

| 测试 | 描述 |
|------|------|
| Memory Copy | 256MB缓冲区拷贝带宽 |
| Memory Set | 内存写入带宽 |
| Vector Add | 2读1写模式带宽 |

### 计算吞吐量测试

| 测试 | 描述 |
|------|------|
| FP32 MatMul | 512x512x512 矩阵乘法 |
| Trig Functions | sin + cos + tan 性能 |

## 项目结构

```
metal/
├── README.md                      # 本文件
├── RESEARCH.md                     # 研究文档和发现
├── Package.swift                  # Swift Package配置
├── Sources/
│   └── MetalBenchmark/
│       └── main.swift             # 主程序和Metal Shader
├── bandwidth_test.metal           # 带宽测试内核（备用）
└── compute_test.metal            # 计算测试内核（备用）
```

## 添加新测试

在 `Sources/MetalBenchmark/main.swift` 中添加：

### 1. 添加Shader代码

```metal
// 在 shaderSource 字符串中添加kernel
kernel void my_kernel(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    output[id] = input[id] * 2.0f;
}
```

### 2. 添加测试函数

```swift
func testMyKernel(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    guard let pipeline = library.makeFunction(name: "my_kernel"),
          let computePipeline = try? device.makeComputePipelineState(function: pipeline) else {
        return
    }

    // 测试代码...
}
```

### 3. 在main()中调用

```swift
try testMyKernel(device: device, queue: queue, library: library)
```

## 性能优化建议

1. **批处理Command Buffer**：减少kernel launch开销
2. **使用Metal Performance Shaders**：Apple优化的核心操作
3. **合理设置threadsPerThreadgroup**：根据GPU调整
4. **避免同步等待**：使用completion handler异步处理

## 与NVIDIA GPU对比

| 指标 | Apple M2 | NVIDIA RTX 4090 |
|------|----------|-----------------|
| 内存带宽 | 100 GB/s | 1008 GB/s |
| FP32性能 | ~3.5 TFLOPS | ~82.6 TFLOPS |
| 内存类型 | 统一内存 | GDDR6X |
| API | Metal | CUDA |

> 注意：Apple GPU优势在于统一内存架构和能效比

## 参考资料

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Metal Shader Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
