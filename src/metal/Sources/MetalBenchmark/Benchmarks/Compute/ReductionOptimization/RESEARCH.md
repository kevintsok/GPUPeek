# Reduction Operations Optimization Research

## Overview

This research analyzes GPU reduction operations (sum, max) on Apple M2 Metal, comparing multiple optimization strategies: naive sequential, tree-based parallel, shared memory, warp-level (SIMD), and multi-warp approaches.

## Research Date

- Date: 2026-03-31
- Device: Apple M2 (Apple GPU Family 7+)

## Key Findings

### Performance Summary

| Strategy | Best Throughput | Notes |
|----------|----------------|-------|
| Naive Sequential | ~0.2 M/s | O(n) complexity, high overhead |
| Tree-based Parallel | ~18 M/s | O(log n) but global memory bottleneck |
| Shared Memory | ~91 M/s | Threadgroup caching, 3-5x faster |
| Warp-level (SIMD) | ~97 M/s | Hardware shuffle, fastest for sum |
| Multi-warp | ~127 M/s | Best overall, combines warp + threadgroup |

### Why Multi-warp is Fastest

1. **Warp-level reduction**: Uses `simd_shuffle_down` for O(log 32) = 5 steps
2. **Threadgroup accumulation**: Reduces global memory writes
3. **Hybrid approach**: Combines best of both worlds

### Shared Memory vs Parallel

For 1M elements:
- Parallel: 56.87 ms
- Shared: 11.47 ms → **5x faster**

This demonstrates that threadgroup memory caching significantly reduces global memory traffic.

## Implementation Details

### Tree-based Parallel Reduction

```metal
float sum = input[id];
for (uint stride = 1; stride < size; stride *= 2) {
    uint mask = stride * 2;
    uint index = (id / mask) * mask + stride;
    if ((id % mask) == stride && index < size) {
        sum += input[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

Complexity: O(log n) steps, but each step requires global memory access.

### Shared Memory Reduction

```metal
threadgroup float shared[256];
float sum = input[id];
shared[lid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = 256/2; stride > 0; stride /= 2) {
    if (lid < stride && id + stride < size) {
        shared[lid] += shared[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

Key insight: All reduction happens in threadgroup memory, then single write to global.

### Warp-level Reduction (SIMD)

```metal
float sum = input[id];
sum += simd_shuffle_down(sum, 16);
sum += simd_shuffle_down(sum, 8);
sum += simd_shuffle_down(sum, 4);
sum += simd_shuffle_down(sum, 2);
sum += simd_shuffle_down(sum, 1);
```

Key insight: No barrier needed - SIMD group is implicitly synchronized.

### Multi-warp Reduction

```metal
// Each warp does its own reduction
sum += simd_shuffle_down(sum, 16);
// ... more shuffle steps ...
shared[lid / 32] = sum;  // Store warp result

// Final warp reduction
sum = shared[lid];
sum += simd_shuffle_down(sum, 16);
// ...
```

## Max Reduction Performance

| Size | Parallel (ms) | Shared (ms) | Speedup |
|------|---------------|-------------|---------|
| 1024 | 5.27 | 5.32 | 0.99x |
| 4096 | 6.27 | 5.63 | 1.11x |
| 16384 | 6.99 | 5.95 | 1.17x |
| 65536 | 8.59 | 6.57 | 1.31x |
| 262144 | 23.52 | 9.05 | 2.60x |
| 1048576 | 57.15 | 10.82 | **5.28x** |

Key insight: Max reduction benefits even more from shared memory because fmax is more expensive than addition.

## Apple M2 Unified Memory Impact

On Apple M2 with unified memory:
- Threadgroup memory provides true on-chip storage
- Global memory accesses go through unified memory subsystem
- Shared memory reduction reduces unified memory traffic significantly
- Peak throughput: 127 M elements/sec for sum, ~10 M/sec for max

## Comparison with CUDA/NVIDIA

| Feature | Apple Metal | NVIDIA CUDA |
|---------|-------------|-------------|
| SIMD Width | 32 threads | 32 threads (warp) |
| Warp Reduction | simd_shuffle_down | __shfl_xor |
| Shared Memory | 32 KB | 48 KB (V100) |
| Threadgroup Barrier | threadgroup_barrier | __syncthreads |

## Optimization Recommendations

1. **Small reductions (< 1K elements)**: Use warp-level reduction
2. **Medium reductions (1K - 1M)**: Use multi-warp reduction
3. **Large reductions (> 1M)**: Use shared memory with multi-warp accumulation
4. **Max/Min reductions**: Use shared memory (more benefit from caching)

## Roofline Analysis

For reduction operations:
- Operational intensity: 1 FLOP per 4 bytes (load only)
- Memory-bound on Apple M2 unified architecture
- Shared memory helps by reducing global memory traffic
- Warp-level is compute-efficient but needs proper data layout

## Conclusions

1. Multi-warp reduction achieves best performance (127 M elem/s)
2. Shared memory is essential for large reductions (5x speedup vs parallel)
3. Warp-level primitives (simd_shuffle_down) are highly efficient
4. For max reduction, shared memory provides even greater benefit (5x vs parallel)
5. Apple M2 unified memory architecture benefits from threadgroup-local operations

## References

- WWDC2020: "Metal for GPU Debugging and Optimization"
- Metal Shading Language Specification
- "Optimizing Parallel Reduction in CUDA" - Mark Harris