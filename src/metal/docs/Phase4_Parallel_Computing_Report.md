# Apple Metal GPU Research Report: Phase 4 - Parallel Computing Characteristics

## Executive Summary

Phase 4 research investigates parallel computing characteristics of Apple M2 GPU, focusing on threadgroup scaling, SIMD operations, atomic operations with contention scaling, thread divergence behavior, and barrier overhead. Key findings reveal that threadgroup size has minimal impact on performance, atomic operations scale with counter count, and barrier overhead decreases significantly with iteration count.

**Key Findings**:
- Threadgroup size (64-1024) shows minimal performance difference (0.70-0.76 GB/s)
- SIMD float4 operations achieve 0.03-0.18 GFLOPS
- Atomic operations scale from 0.016 to 0.57 GOPS based on contention
- Thread divergence causes 10-15% performance variation
- Barrier overhead: ~4.8μs single barrier, drops to ~89ns with pipelining

## Test Environment

| Component | Specification |
|-----------|---------------|
| Device | Apple M2 (MacBook Air) |
| GPU Cores | 10-core GPU |
| Unified Memory | Shared with CPU |
| Metal Version | Apple 7+ GPU Family |
| Max Threadgroup Memory | 32 KB |
| Max Threads Per Threadgroup | 1024 |

## Threadgroup Performance

### Threadgroup Size Scaling

| Threadgroup Size | Bandwidth | Relative Performance |
|-----------------|-----------|---------------------|
| 64 | 0.70 GB/s | 0.92x |
| 256 | 0.76 GB/s | 1.00x (baseline) |
| 512 | 0.75 GB/s | 0.99x |
| 1024 | 0.72 GB/s | 0.95x |

**Analysis**: Threadgroup size scaling shows minimal impact on performance, with all sizes achieving 0.70-0.76 GB/s. This suggests:
1. Memory bandwidth is the limiting factor, not threadgroup configuration
2. The M2 GPU efficiently schedules threads regardless of threadgroup size
3. Shared memory access patterns dominate performance more than threadgroup size

## SIMD Vector Operations

### Float4 Vector Operations

| Operation | GFLOPS | Notes |
|-----------|--------|-------|
| Vector Add | 0.03 | Single add per float4 |
| Vector Mul | 0.04 | Single mul per float4 |
| Dot Product | 0.18 | 4 muls + 3 adds per float4 |
| SIMD Shuffle | 0.15 | 3 adds per float4 |

**Analysis**: SIMD operations achieve low GFLOPS due to memory bandwidth limitations. The dot product achieves higher performance because it performs more computation per memory access (4 multiplications + horizontal sum).

## Atomic Operations

### Contention Scaling

| Number of Counters | GOPS | Verification |
|-------------------|------|-------------|
| 1 | 0.016 | PASS |
| 16 | 0.118 | PASS |
| 64 | 0.361 | PASS |
| 256 | 0.565 | PASS |
| 1024 | 0.570 | PASS |

**Analysis**: Atomic operation throughput scales with the number of counters, demonstrating:
1. Reduced contention with more atomic locations
2. M2 GPU handles parallel atomic operations efficiently
3. Throughput plateaus at ~0.57 GOPS with 256+ counters

### Single Counter Bottleneck

| Configuration | GOPS |
|--------------|------|
| All threads to single counter | 0.461 |

**Analysis**: When all threads compete for a single atomic, throughput drops to 0.461 GOPS, which is still measurable but significantly lower than distributed atomic operations.

## Thread Divergence

### Branch Prediction Impact

| Threshold | Branch Distribution | GFLOPS |
|-----------|-------------------|--------|
| 25% | 25% take path A, 75% path B | 1.16 |
| 50% | 50% take path A, 50% path B | 1.31 |
| 75% | 75% take path A, 25% path B | 1.30 |

**Analysis**: Thread divergence causes ~10-15% performance variation:
1. Balanced branches (50/50) show best overall performance
2. Skewed branches (25/75) show lowest performance
3. The GPU's branch predictor may be optimizing for balanced paths

## Barrier Overhead

### Threadgroup Barrier Performance

| Barriers per Thread | Time per Barrier | Notes |
|---------------------|------------------|-------|
| 1 | 4,843 ns | High overhead per barrier |
| 10 | 305 ns | Reduced overhead |
| 100 | 108 ns | Near-optimal |
| 500 | 89 ns | Minimum overhead |

**Analysis**: Barrier overhead analysis reveals:
1. Single barrier costs ~4.8μs (extremely high)
2. Overhead amortized with pipelining: 305ns at 10 iterations
3. Near-optimal performance at 100+ barriers per thread
4. Suggests GPU batches barrier operations for efficiency

## Shared Memory Reduction

### Reduction Performance

| Array Size | GFLOPS | Reduction Elements |
|-----------|--------|-------------------|
| 65,536 | 0.01 | 256-element groups |
| 262,144 | 0.04 | 256-element groups |
| 1,048,576 | 0.11 | 256-element groups |

**Analysis**: Shared memory reduction scales with problem size:
1. Small reductions (65K) are inefficient (0.01 GFLOPS)
2. Larger reductions (1M) achieve better utilization (0.11 GFLOPS)
3. Logarithmic scaling: 4x size increase → ~4x performance

## Architecture Insights

### 1. Threadgroup Efficiency

Apple M2's threadgroup scheduler efficiently handles various threadgroup sizes:

```metal
// Threadgroup sizes tested: 64, 256, 512, 1024
// All achieve similar performance (0.70-0.76 GB/s)
// Suggests efficient hardware thread scheduling
```

### 2. SIMD Implementation

Float4 SIMD operations on M2:
- 128-bit vector registers
- Hardware-accelerated float4 operations
- Memory bandwidth limits effective throughput

### 3. Atomic Operation Scaling

M2 GPU atomic operations:
- Hardware atomic units scale with contention
- Peak throughput ~0.57 GOPS with low contention
- Memory order relaxed mode for best performance

### 4. Barrier Synchronization

Threadgroup barriers on M2:
- High fixed overhead (~4.8μs) for first barrier
- Amortized cost decreases with pipelining
- Hardware barrier implementation efficient for large batches

## Optimization Recommendations

### 1. Threadgroup Configuration

**DO**:
- Use threadgroup size of 256 as baseline (optimal balance)
- Adjust based on shared memory requirements
- Test different sizes to find workload-specific optimum

**AVOID**:
- Extremely small threadgroups (< 64) unless necessary
- Assuming larger threadgroups always better

### 2. SIMD Operations

**DO**:
- Use float4 (SIMD4<Float>) for memory-bound vector ops
- Pack data to utilize full vector width
- Consider half4 (float16) for even higher throughput

**AVOID**:
- Scalar operations when SIMD can be used
- Unaligned data that breaks vectorization

### 3. Atomic Operations

**DO**:
- Distribute atomics across multiple counters to reduce contention
- Use memory_order_relaxed when ordering not required
- Batch atomic updates when possible

**AVOID**:
- Many threads updating single atomic location
- Frequent atomics in tight loops
- Unnecessary atomic operations

### 4. Thread Divergence

**DO**:
- Design kernels with balanced branch probabilities
- Consider branchless alternatives for critical paths
- Test different branch thresholds

**AVOID**:
- Highly skewed branch probabilities (> 80/20)
- Nested branches that compound divergence
- Warp divergence in tight loops

### 5. Barrier Usage

**DO**:
- Pipeline barriers to amortize overhead
- Minimize unnecessary threadgroup_barrier calls
- Combine barriers when possible

**AVOID**:
- Single barriers in performance-critical paths
- Redundant barriers before/after data reuse
- Barriers inside divergent branches

## Comparison with Previous Phases

| Phase | Focus | Key Metric | Value |
|-------|-------|------------|-------|
| Phase 1 | API/Bandwidth | Memory bandwidth | ~1 GB/s |
| Phase 2 | Memory Subsystem | Strided access penalty | 2.3x slower |
| Phase 3 | Compute Throughput | Tiled MatMul | 9.11 GFLOPS |
| Phase 4 | Parallel Computing | Threadgroup efficiency | 0.70-0.76 GB/s |

Phase 4 reveals that parallel computing primitives (threadgroups, SIMD, atomics) all operate within the unified memory bandwidth constraints, achieving similar effective bandwidths.

## Conclusions

Phase 4 research reveals several key parallel computing characteristics:

1. **Threadgroup size is not performance-critical**: All sizes (64-1024) achieve similar bandwidth

2. **SIMD provides moderate benefit**: Float4 operations achieve 0.03-0.18 GFLOPS

3. **Atomics scale with contention**: Throughput ranges from 0.016 to 0.57 GOPS based on counter count

4. **Thread divergence has measurable impact**: 10-15% performance variation with branch skew

5. **Barrier overhead is high but amortizable**: ~4.8μs single, drops to ~89ns pipelined

6. **Memory bandwidth unifies all operations**: All primitives constrained by unified memory bandwidth

## Future Research

- Phase 5: Architecture Comparison (NVIDIA/AMD vs Apple)
- Investigate TBDR impact on compute workloads
- Explore hardware ray tracing capabilities on M3+
- Compare with Metal Performance Shaders

---

*Report generated: 2026-03-23*
*Research Phase: Phase 4 - Parallel Computing Characteristics*
*GPU: Apple M2 (Family Apple 7+)*
