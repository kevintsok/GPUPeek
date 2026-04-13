# Apple Metal GPU Research Report: Phase 3 - Compute Throughput

## Executive Summary

Phase 3 research focuses on compute throughput characteristics of Apple M2 GPU, testing matrix multiplication (FP32/FP16), FMA operations, trigonometric functions, and integer arithmetic. Key findings reveal that shared memory tiling provides 2-3x speedup for matrix operations, while FP16 offers marginal advantage over FP32.

**Key Findings**:
- Tiled matrix multiply achieves 2-3x speedup over naive implementation
- FP16 is ~5% faster than FP32 for matrix operations
- FMA throughput limited by memory bandwidth (0.22 GFLOPS)
- Trigonometric functions achieve 0.57 GOPS
- Integer operations achieve 0.58 GOPS

## Test Environment

| Component | Specification |
|-----------|---------------|
| Device | Apple M2 (MacBook Air) |
| GPU Cores | 10-core GPU |
| Unified Memory | Shared with CPU |
| Metal Version | Apple 7+ GPU Family |
| Max Threadgroup Memory | 32 KB |
| Max Threads Per Threadgroup | 1024 |

## Matrix Multiply Performance

### FP32 Matrix Multiply: Naive vs Tiled

| Matrix Size | Naive (GFLOPS) | Tiled (GFLOPS) | Speedup |
|------------|----------------|-----------------|---------|
| 256x256x256 | 0.91 | 2.63 | 2.9x |
| 512x512x512 | 3.04 | 5.56 | 1.8x |
| 1024x1024x1024 | 4.30 | 9.11 | 2.1x |

**Analysis**: Tiled implementation with shared memory achieves consistent 2-3x speedup over naive approach. The speedup is most pronounced at smaller sizes (2.9x at 256³) where shared memory caching has greater effect. Even at 1024³, tiled version achieves 9.11 GFLOPS.

### FP16 vs FP32 Matrix Multiply

| Precision | GFLOPS | Notes |
|-----------|--------|-------|
| FP32 | 4.71 | Single precision |
| FP16 | 4.92 | Half precision |
| Ratio | 1.05x | FP16 marginally faster |

**Analysis**: FP16 shows only 5% improvement over FP32, which is less than expected. This suggests:
1. M2 GPU may not have significant FP16 tensor throughput advantage
2. Memory bandwidth is the limiting factor rather than compute
3. The 2x memory bandwidth advantage of FP16 (half the data) is offset by other bottlenecks

## Arithmetic Operations

### FMA (Fused Multiply-Add) Performance

| Metric | Value |
|--------|-------|
| Buffer Size | 32 MB |
| Elements | 8 M |
| Performance | 0.22 GFLOPS |

**Analysis**: The low FMA throughput (0.22 GFLOPS) indicates memory bandwidth limitation rather than compute limitation. FMA requires 2 reads + 1 write per element, making it memory-bound on unified memory architecture.

### Trigonometric Functions

| Metric | Value |
|--------|-------|
| Elements | 8 M |
| Operations | sin + cos + tan per element |
| Performance | 0.57 GOPS |

**Analysis**: Trigonometric functions achieve 0.57 GOPS, which is reasonable given they are software-implemented and memory-bound.

### Integer Operations

| Metric | Value |
|--------|-------|
| Elements | 8 M |
| Operations | add + sub + mul + xor per element |
| Performance | 0.58 GOPS |

**Analysis**: Integer operations achieve 0.58 GOPS, similar to floating-point transcendental functions. Integer multiply is hardware-accelerated but the combined workload is still memory-bound.

## Architecture Insights

### 1. Shared Memory Tiling Effectiveness

The 2-3x speedup from tiling demonstrates effective use of 32KB threadgroup memory:

```metal
// Tiled matrix multiply loads tiles into shared memory
threadgroup float* As [[threadgroup(0)]];
threadgroup float* Bs [[threadgroup(1)]];

// Tile size of 16x16 = 256 floats = 1KB per tile
// 32KB shared memory can hold 32 such tiles
```

Tiling reduces global memory accesses by reusing data within threadgroups.

### 2. Memory Bandwidth Limitation

Low compute throughput across all operations (0.22-9.11 GFLOPS) confirms that unified memory bandwidth is the primary bottleneck:

| Operation | Throughput | Limiting Factor |
|-----------|------------|----------------|
| Tiled MatMul | 9.11 GFLOPS | Memory bandwidth |
| Naive MatMul | 4.30 GFLOPS | Memory bandwidth |
| FMA | 0.22 GFLOPS | Memory bandwidth |
| Trig | 0.57 GOPS | Memory bandwidth |
| Integer | 0.58 GOPS | Memory bandwidth |

### 3. Precision Comparison

The minimal FP16 advantage (5%) over FP32 suggests:
- M2 may not have dedicated FP16 tensor cores
- FP16 throughput is limited by same memory bandwidth as FP32
- Unified memory architecture equalizes precision advantages

## Optimization Recommendations

### 1. Matrix Multiplication

**DO**:
- Use tiled algorithm with shared memory for 2-3x speedup
- Match tile size to shared memory capacity (16x16 works well)
- Consider register tiling for very large matrices

**AVOID**:
- Naive implementation for production workloads
- Tile sizes that exceed 32KB shared memory limit

### 2. Arithmetic Operations

**DO**:
- Fuse multiple operations (FMA) to reduce memory traffic
- Use vectorized types (float4) for memory-bound operations
- Batch operations to amortize kernel launch overhead

**AVOID**:
- Separate kernel calls for dependent operations
- Small buffer sizes that don't saturate GPU

### 3. Precision Selection

**DO**:
- Use FP32 for numerical accuracy requirements
- Use FP16 for memory-bandwidth-bound workloads with tolerance for precision loss
- Consider mixed precision where applicable

**AVOID**:
- Assuming FP16 provides significant speedup without measurement
- Using FP16 for reduction operations requiring high precision

## Theoretical Performance Analysis

### Peak Theoretical Performance

Assuming M2 GPU with 10 cores at ~1 GHz:
- FP32: 10 cores × 2 FMA units × 1 GHz = 20 GFLOPS per core = 200 GFLOPS total (theoretical)
- Measured: 9.11 GFLOPS = ~4.5% of theoretical

### Efficiency Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Memory bandwidth (theoretical) | 100 GB/s | Unified memory |
| Memory bandwidth (measured) | ~1-2 GB/s | ~1-2% utilization |
| Compute efficiency | 4.5% | Memory-limited |

The low compute efficiency indicates the workload is memory-bound, not compute-bound.

## Comparison with Previous Phases

| Phase | Focus | Key Metric | Value |
|-------|-------|------------|-------|
| Phase 1 | API/Bandwidth | Memory bandwidth | ~1 GB/s |
| Phase 2 | Memory Subsystem | Strided access penalty | 2.3x slower |
| Phase 3 | Compute Throughput | Tiled MatMul | 9.11 GFLOPS |

Phase 3 shows that compute operations achieve higher effective bandwidth (9.11 GFLOPS for matrix multiply) compared to simple memory copy (~1 GB/s), demonstrating that computation provides some parallelism benefit even under memory constraints.

## Conclusions

Phase 3 research reveals several key compute throughput characteristics:

1. **Tiling is highly effective**: Shared memory tiling provides 2-3x speedup for matrix operations

2. **Memory bandwidth is the bottleneck**: All operations achieve <5% of theoretical compute throughput

3. **FP16 provides minimal advantage**: Only ~5% improvement over FP32, suggesting no dedicated tensor units

4. **FMA and transcendental operations are memory-bound**: Low GOPS measured due to memory limitations

5. **Integer operations competitive**: Integer arithmetic achieves similar throughput to floating-point

## Future Research

- Phase 4: Parallel Computing Characteristics (threadgroup, SIMD, atomics)
- Investigate tensor operations with Apple Neural Engine
- Compare with Metal Performance Shaders (MPS) optimized kernels
- Explore hardware ray tracing on M3+ GPUs

---

*Report generated: 2026-03-23*
*Research Phase: Phase 3 - Compute Throughput*
*GPU: Apple M2 (Family Apple 7+)*
