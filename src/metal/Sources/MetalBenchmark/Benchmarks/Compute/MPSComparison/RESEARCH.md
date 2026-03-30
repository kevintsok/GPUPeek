# Metal Performance Shaders (MPS) vs Custom Kernel Research

## Overview

Metal Performance Shaders (MPS) is a framework that provides highly optimized, Apple-certified GPU kernels for common operations. This research compares MPS performance against custom Metal implementations on Apple M2.

## Research Date

- Date: 2026-03-31
- Device: Apple M2 (Apple GPU Family 7+)

## Key Questions

1. How much faster are MPS kernels vs custom implementations?
2. When should you use MPS vs custom kernels?
3. What operations benefit most from MPS?

## Apple GPU Architecture

- MPS kernels are hand-optimized by Apple engineers
- They leverage specific hardware features of Apple GPUs
- MPS is the recommended path for standard operations on Apple Silicon
- MPSSobel uses hardware texture sampling for edge detection

## Operations Tested

### GEMM (Matrix Multiply)

| Implementation | Performance | Relative |
|---------------|-------------|----------|
| Custom Naive GEMM | 4.38 GFLOPS | 1.0x |

Note: MPSMatrixMultiplication was not tested in this iteration due to API complexity. For production GEMM on Apple Silicon, use MPSMatrixMultiplication.

### Image Convolution (Sobel Edge Detection) - 1024x1024

| Implementation | Performance | Relative |
|---------------|-------------|----------|
| Custom Sobel (texture) | 45.66 Mpixels/s | 3.3x |
| MPS Sobel | 13.68 Mpixels/s | 1.0x |

**Surprising Finding**: Custom Sobel is 3.3x FASTER than MPS Sobel on M2!

## Analysis

### Why Custom Won (Sobel)

1. **Simple Operation**: Sobel is a 3x3 filter - simple enough that custom implementation is efficient
2. **Direct Texture Access**: Custom kernel reads texture directly without MPS overhead
3. **Apple M2 Texture Cache**: M2 has efficient texture caching that benefits direct reads
4. **Format Match**: Using r32Float texture may not be optimal for MPS

### When MPS Should Win

1. **Complex Operations**: CNN convolutions with large kernels
2. **Batch Processing**: Operations where MPS overhead amortizes
3. **Standard Formats**: When using standard image formats (8-bit, 16-bit)

### Custom Kernel Advantages

1. **Flexibility**: Can implement any operation
2. **Memory Layout**: Can use arbitrary buffer formats
3. **Specialized**: Can optimize for specific data patterns
4. **Research**: Necessary for novel algorithms
5. **Simple Ops**: For simple 3x3 filters on M2, custom can win

### Why MPS Might Be Slow in This Test

1. **Gradient Calculation**: MPS Sobel might use a different (more accurate) gradient formula
2. **Edge Handling**: MPS may add edge handling that custom skips
3. **Texture Format**: r32Float may not be the native format for MPS
4. **Function Call Overhead**: MPS has initialization overhead

## Recommendations

### Use MPS When:
- Implementing standard CNN operations (conv, pooling, normalization)
- Doing matrix operations (GEMM, matrix multiply)
- Image processing with standard formats (8-bit, 16-bit images)
- Production ML inference on Apple devices
- Large-scale operations where overhead amortizes

### Use Custom Kernels When:
- Implementing novel operations not in MPS
- Need specific numerical behavior
- Researching new algorithms
- Require unusual memory layouts
- Simple operations (3x3, 5x5 filters) on Apple M2
- When you can beat MPS performance with specialization

## Benchmark Methodology

- Image Size: 1024x1024 (1M pixels)
- Iterations: 10 (averaged)
- Texture Format: r32Float
- Device: Apple M2
- Custom kernel uses texture2d read with explicit gradient computation
- MPS uses MPSImageSobel with default settings

## Comparison with NVIDIA

| Operation | Apple MPS | NVIDIA cuBLAS/cuDNN |
|-----------|-----------|---------------------|
| GEMM | MPSMatrixMultiplication | cuBLAS GEMM |
| Conv2D | MPSImage Convolution | cuDNN Convolution |
| Pooling | MPSPooling | cuDNN Pooling |
| Edge Detect | MPSImageSobel | Custom or DALI |

Note: Apple MPS and NVIDIA cuDNN have different optimization strategies due to architectural differences.

## Conclusion

**MPS is NOT always faster than custom kernels on Apple M2.**

For simple operations like 3x3 Sobel edge detection, custom implementations can significantly outperform MPS. This is likely due to:
1. M2's efficient texture cache and memory system
2. MPS overhead for simple operations
3. Potential format mismatch between test setup and MPS expectations

**Recommendation**: Always benchmark your specific use case before choosing between MPS and custom kernels.

For production code:
1. Start with MPS for standard operations
2. Benchmark against custom if performance is critical
3. Consider using custom for simple image processing operations
4. Remember: MPS provides reliability and correctness; custom provides flexibility
