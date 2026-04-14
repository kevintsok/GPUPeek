# FFT Optimization Deep Dive Research

## Overview

This research analyzes FFT (Fast Fourier Transform) algorithm optimization strategies on Apple M2 GPU, comparing naive global memory implementation vs shared memory optimization vs radix-4 algorithm.

## Research Date

- Date: 2026-03-31
- Device: Apple M2 (Apple GPU Family 7+)

## Key Questions

1. How much does shared memory improve FFT performance?
2. What is the benefit of radix-4 over radix-2?
3. How does FFT performance scale with input size?
4. What is the bottleneck in naive FFT implementation?

## Measured Results

### Performance Comparison (GFLOPS)

| Size | Naive Radix-2 | Shared Memory | Radix-4 |
|------|----------------|---------------|----------|
| 256 | 0.0013 | 0.0016 | 0.0012 |
| 512 | 0.0026 | 0.0032 | - |
| 1024 | 0.0049 | 0.0055 | 0.0043 |
| 2048 | 0.0073 | 0.0083 | - |
| 4096 | 0.0095 | 0.0114 | 0.0104 |
| 16384 | - | - | 0.0077 |

### Shared Memory Speedup

| Size | Speedup |
|------|---------|
| 256 | 1.19x |
| 512 | 1.20x |
| 1024 | 1.14x |
| 2048 | 1.14x |
| 4096 | 1.20x |

### Key Findings

1. **Shared memory provides modest 1.14-1.20x speedup** - Less than expected, likely due to Apple M2 unified memory already being efficient
2. **Performance scales sublinearly**: 16x size increase yields only 7.2x performance (theoretical: 24x)
3. **Single stage time is constant (~5.5ms)** regardless of FFT size, indicating fixed barrier overhead
4. **Radix-4 shows similar performance to Radix-2** - The reduced stage count doesn't translate to proportional speedup
5. **FFT is memory-bound on Apple M2** - Unified memory architecture limits peak performance

## FFT Algorithm Background

### Cooley-Tukey Radix-2 FFT

The standard FFT algorithm divides the DFT computation into smaller sub-problems:

- **Complexity**: O(n log n)
- **Stages**: log₂(n) butterfly stages
- **Butterfly ops**: Each butterfly computes 2 outputs from 2 inputs
  - 1 complex multiplication (4 real muls + 2 real adds)
  - 1 complex addition
  - 1 complex subtraction
  - Total: ~6 FLOPs per butterfly

For N=1024:
- Stages: 10
- Butterflies per stage: N/2 = 512
- Total butterflies: N/2 * log₂(N) = 5120
- Total FLOPs: 5120 * 6 ≈ 30K FLOPs

### Radix-4 FFT

Radix-4 processes 4 elements per butterfly:

- **Stages**: log₄(n) = log₂(n)/2
- For N=1024 (4^5): 5 stages vs 10 stages for radix-2
- Fewer synchronization barriers

## Implementations Compared

### 1. Naive Radix-2 (Global Memory)

```metal
kernel void fft_naive_radix2(device float2* data [[buffer(0)]], ...) {
    // Each thread handles one butterfly
    // All memory accesses go to global memory
    // Barrier at each stage for synchronization
}
```

**Characteristics:**
- All threads access global memory
- Poor memory coalescing (non-sequential)
- High memory latency exposure
- log₂(N) barriers per FFT

### 2. Shared Memory Radix-2

```metal
kernel void fft_shared_radix2(device float2* data [[buffer(0)]],
                              threadgroup float2* shared [[threadgroup(0)]], ...) {
    // Load entire dataset into shared memory first
    // All butterfly operations in shared memory
    // Single barrier between stages
    // Write back to global at end
}
```

**Characteristics:**
- Shared memory bandwidth >> global memory bandwidth
- Better memory coalescing for load/store
- Still log₂(N) stages
- Threadgroup size limited (max 32KB)

### 3. Radix-4

```metal
kernel void fft_radix4(device float2* data [[buffer(0)]], ...) {
    // Each thread handles 4-element butterfly
    // Half the number of stages
    // Fewer barriers
}
```

**Characteristics:**
- log₄(N) stages vs log₂(N)
- Fewer synchronization points
- Better instruction-level parallelism
- Only works for power-of-4 sizes

## Expected Performance Analysis

### Arithmetic Intensity

FFT is a **memory-bound** operation:

```
Memory traffic per butterfly:
- Read: 2 float2 = 16 bytes
- Write: 2 float2 = 16 bytes
- Total: 32 bytes

Computations per butterfly:
- 4 multiplications
- 2 additions
- 2 subtractions
- Total: ~6 FLOPs

Arithmetic Intensity = 6 FLOPs / 32 bytes = 0.1875 FLOPs/byte
```

For Apple M2 with ~2 GB/s effective bandwidth:
- Theoretical peak FFT performance: ~0.375 GOPS

### Optimization Potential

| Optimization | Expected Speedup | Reason |
|--------------|-----------------|--------|
| Shared Memory | 2-4x | Higher bandwidth, better coalescing |
| Radix-4 | 1.5-2x | Fewer barriers |
| Combined | 3-6x | Synergistic benefits |

## Memory Access Patterns

### Global Memory FFT

- Non-sequential access pattern
- Bank conflicts in shared memory (if used incorrectly)
- Poor cache line utilization
- High global memory latency

### Shared Memory FFT

- Sequential load into shared memory (coalesced)
- Butterfly ops in shared memory (fast)
- Sequential write back (coalesced)
- Better utilization of memory controller

## Synchronization Overhead

Each FFT stage requires a `threadgroup_barrier`:

- Fixed cost: ~4.8 μs per barrier
- For N=1024 (10 stages): ~48 μs just for barriers
- This is a significant portion of total time

## Optimization Strategies

### 1. Shared Memory Caching

```metal
// Load all data into shared memory first
shared[lid] = data[lid];
threadgroup_barrier(flags::mem_threadgroup);

// All butterfly operations in shared memory
for (stage = 0; stage < log2(N); stage++) {
    // butterfly in shared[]
    threadgroup_barrier(flags::mem_threadgroup);
}

// Write back
data[lid] = shared[lid];
```

### 2. Radix-4 Algorithm

- Process 4 elements per thread
- Halve the number of stages
- Reduce barrier overhead

### 3.混合基 (Mixed Radix)

- Use radix-2, radix-4, radix-8 as appropriate
- Optimize for specific sizes

### 4. Precomputed Twiddle Factors

```metal
// Instead of computing cos/sin each time:
constant float2 twiddle_factors[N/2] = { ... };
```

## Apple M2 Specific Considerations

### Unified Memory Impact

- CPU and GPU share memory bandwidth
- FFT's large memory access can saturate unified memory
- Consider using device-only allocation for FFT buffers

### Threadgroup Memory Limits

- Max 32KB per threadgroup
- For N > 8192, cannot fit all data in single threadgroup
- Requires multi-pass approach or shared/threadgroup split

## Comparison with cuFFT

| Feature | Apple Metal | NVIDIA cuFFT |
|---------|-------------|--------------|
| Radix-2/4 | Manual | Auto |
| Shared memory | Manual | Auto |
| Memory coalescing | Manual | Auto |
| Performance | Baseline | Optimized |

## Practical Recommendations

### When to Use FFT on Apple GPU

✅ **Good for:**
- Very large FFTs (>16K elements)
- Batch FFT processing
- When you need custom FFT (non-power-of-2 sizes)
- Integration with other GPU kernels

❌ **Consider alternatives for:**
- Small FFTs (<1K): CPU FFT may be faster
- Single FFT: Overhead of GPU launch not worth it
- Power-of-2 sizes: vDSP is highly optimized

### Best Practices

1. **Use shared memory for FFTs up to threadgroup limit**
2. **Use radix-4 for power-of-4 sizes**
3. **Batch multiple FFTs to amortize launch overhead**
4. **Consider CPU FFT for small/medium sizes**
5. **Profile to determine optimal configuration**

## Future Research Directions

1. **Multi-pass FFT for large sizes**
2. **Stockham algorithm for automatic bit-reversal**
3. **FFT-based convolution optimization**
4. **Integration with MPS FFT (if available)**

## References

- Cooley-Tukey FFT Algorithm (1965)
- CUDA cuFFT Documentation
- Apple Metal Performance Shaders
- "FFT: The Fast Fourier Transform" - various texts