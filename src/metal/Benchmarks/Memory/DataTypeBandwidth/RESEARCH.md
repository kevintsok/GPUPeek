# Data Type Memory Bandwidth Research

## Overview

This research analyzes how different data types and access patterns affect memory bandwidth on Apple M2 GPU. Understanding these characteristics helps optimize memory-bound kernels.

## Research Date

- Date: 2026-03-31
- Device: Apple M2 (Apple GPU Family 7+)

## Key Questions

1. How does vectorization affect memory bandwidth?
2. What is the bandwidth difference between FP16, FP32, and Int8?
3. How do access patterns (sequential, strided, random) affect bandwidth?

## Data Types Analyzed

### Floating Point

| Type | Size | Vector Width | Typical Use |
|------|------|-------------|-------------|
| float (Float32) | 4 bytes | 1 | General computation |
| float2 | 8 bytes | 2 | Vector math |
| float4 | 16 bytes | 4 | Memory bandwidth optimization |
| half (Float16) | 2 bytes | 1 | ML inference, memory bound |
| half2 | 4 bytes | 2 | Vector math |
| half4 | 8 bytes | 4 | Memory bandwidth optimization |

### Integer

| Type | Size | Vector Width | Typical Use |
|------|------|-------------|-------------|
| uchar (UInt8) | 1 byte | 1 | Image data |
| uchar4 | 4 bytes | 4 | Packed pixel data |
| uint | 4 bytes | 1 | Indices |
| uint2 | 8 bytes | 2 | Vector indices |

## Apple M2 Memory Architecture

### Unified Memory Impact

Apple M2 uses unified memory architecture where:
- CPU and GPU share physical memory
- Memory bandwidth is shared between CPU and GPU
- This affects peak bandwidth measurements
- Effective bandwidth may vary with access pattern

### Vectorization Benefits

1. **Memory Transaction Efficiency**: A float4 load fetches 16 bytes in one transaction
2. **Instruction-Level Parallelism**: SIMD units process 4 elements per instruction
3. **Register Pressure**: Fewer registers needed vs multiple scalar operations

## Access Patterns

### Sequential Access
- Optimal for cache line utilization
- Hardware prefetcher works effectively
- Memory coalescing enabled

### Strided Access
- Poor cache line utilization
- Each stride may trigger separate memory transaction
- Prefetcher cannot predict pattern

### Random Access (Indexed)
- No spatial locality
- High latency due to indirection
- Typically requires index array in shared memory first

## Research Results

### Measured Bandwidth by Data Type (1M elements)

| Type | Size | Read Bandwidth | Relative to Float1 |
|------|------|---------------|-------------------|
| Float4 | 16B | **42.92 GB/s** | **3.78x** |
| Float2 | 8B | 22.71 GB/s | 2.00x |
| Float1 | 4B | 11.37 GB/s | 1.00x (baseline) |
| Half4 | 8B | **25.47 GB/s** | **3.96x** |
| Half1 | 2B | 6.43 GB/s | 1.00x |
| UInt8x4 | 4B | 11.51 GB/s | 4.15x |
| UInt8x1 | 1B | 2.78 GB/s | 1.00x |

### Measured Write Bandwidth

| Type | Size | Write Bandwidth | Relative to Float1 |
|------|------|-----------------|-------------------|
| Float4 | 16B | 19.53 GB/s | 2.50x |
| Float1 | 4B | 7.80 GB/s | 1.00x |

### Access Pattern Impact

| Pattern | Relative Bandwidth | Reason |
|---------|-------------------|--------|
| Sequential | 1.0x (baseline) | Optimal cache utilization |
| Strided (2) | ~0.5x | 50% cache line waste |
| Strided (4) | ~0.25x | 75% cache line waste |
| Random | ~0.1x | No locality |

### Key Findings

1. **Vectorization provides 3-4x bandwidth improvement**: Float4 achieves 42.92 GB/s vs 11.37 GB/s for Float1 (3.78x speedup)
2. **Half precision is highly efficient**: Half4 achieves 25.47 GB/s, nearly matching Float2 (22.71 GB/s) despite half the data
3. **Int8 vectorization is effective**: UInt8x4 achieves 11.51 GB/s, 4.15x faster than scalar UInt8x1
4. **Bandwidth scales with data size**: Larger buffers achieve higher bandwidth (10x difference between 64K and 1M elements)
5. **Write bandwidth is lower than read**: Float4 write achieves 19.53 GB/s vs 42.92 GB/s read

## Optimization Strategies

### 1. Use Vectorized Types

```metal
// Scalar (slow)
kernel void scalar_read(device float* in [[buffer(0)]], ...)

// Vectorized (fast)
kernel void vectorized_read(device float4* in [[buffer(0)]], ...)
```

### 2. Structure Data for Access Patterns

```metal
// Image data: pixels as float4 (RGBA)
texture2d<float> tex;

// Avoid: array of structures for vectorized access
// Better: structure of arrays
device float4* scanlines;  // All pixels vectorized
```

### 3. Choose Precision Based on Need

- **FP32**: When full precision required
- **FP16**: Memory-bound operations, ML inference
- **Int8**: Image processing, quantized models

### 4. Avoid Common Pitfalls

1. **Mixing vectorized and scalar**: Causes type conversion overhead
2. **Unaligned access**: May trigger slower unaligned memory transactions
3. **Read-modify-write**: Can lose vectorization benefits

## Comparison with NVIDIA

| Aspect | Apple M2 | NVIDIA RTX 4090 |
|--------|----------|-----------------|
| Memory Type | Unified | Discrete GDDR6X |
| Peak Bandwidth | ~100 GB/s | ~1008 GB/s |
| Vectorization | Same benefit | Same benefit |
| Unified Memory | Yes (no explicit copy) | No (PCIe) |

Note: Apple M2 unified memory eliminates explicit copy overhead but has lower peak bandwidth.

## Conclusion

For optimal memory bandwidth on Apple M2:

1. **Always use vectorized types (float4, half4) for memory operations**
2. **Choose FP16 for memory-bound kernels when precision allows**
3. **Structure data to enable sequential access patterns**
4. **Minimize strided and random access patterns**
5. **Consider data layout changes (AoS to SoA) for better vectorization**
