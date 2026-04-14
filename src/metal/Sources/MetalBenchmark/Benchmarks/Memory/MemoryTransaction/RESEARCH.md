# Memory Transaction Efficiency Research

## Overview

This research analyzes memory transaction efficiency on Apple M2 Metal GPU, focusing on read/write asymmetry, access patterns, and atomic operation overhead in the unified memory architecture.

## Research Date

- Date: 2026-03-31
- Device: Apple M2 (Apple GPU Family 7+)

## Key Findings

### 1. Read vs Write Asymmetry

**Critical Discovery**: Read is 2.86x faster than write on Apple M2 unified memory.

| Size | Write (GB/s) | Read (GB/s) | Ratio |
|------|--------------|-------------|-------|
| 64K | 2.49 | 3.15 | 1.26x |
| 256K | 10.12 | 12.93 | 1.28x |
| 1024K | 32.49 | 51.90 | 1.60x |
| 4096K | 72.26 | 206.57 | **2.86x** |

**Interpretation**: This asymmetry is due to unified memory write-through behavior. Write operations must immediately update main memory, while reads can be satisfied from a more efficient cache hierarchy.

### 2. Access Pattern Impact

Surprisingly, spatial locality (sequential vs strided) shows minimal impact on Apple M2 unified memory:

| Pattern | Bandwidth (GB/s) | Relative |
|---------|------------------|----------|
| Sequential | 16.33 | 1.00x |
| Strided x4 | 17.09 | 0.96x |
| Strided x16 | 17.08 | 0.96x |
| Temporal (16x read same) | 16.10 | 1.01x |

**Interpretation**: Unlike discrete GPUs where coalesced access is critical, Apple M2 unified memory architecture handles varied access patterns more efficiently. The shared CPU/GPU memory subsystem has different optimization characteristics.

### 3. Temporal Locality (Cache Behavior)

Reading the same data 16 times shows **NO benefit**:

```
Temporal Locality Test (256K elements):
- First read: 16.33 GB/s
- 16 repeated reads of same data: 16.10 GB/s
- Ratio: 1.01x (essentially no caching benefit)
```

**Interpretation**: This is the most significant finding. Apple M2 unified memory does NOT benefit from traditional L1/L2 caching in the way discrete GPUs do. The unified memory architecture shares physical RAM between CPU and GPU, and the "cache" behavior is fundamentally different.

### 4. Read-Write Pattern Costs

| Pattern | Bandwidth | Notes |
|---------|-----------|-------|
| Write-Read | 1.67 GB/s | Barrier to ensure write completes before read |
| Read-Modify-Write | 1.55 GB/s | Classic RMW = slowest pattern |
| Bidirectional | 24.12 GB/s | Parallel read+write = most efficient |

**Key Insight**: Write-read dependency (where you must write then read the same location) is extremely expensive due to `threadgroup_barrier(mem_flags::mem_device)`.

### 5. Atomic Operations Overhead

Atomic increment: **0.168 GB/s** effective bandwidth

This is **1200x slower** than pure read operations!

**Why?**: Atomic operations require:
1. Cache line exclusivity (cache coherency)
2. Memory ordering guarantees
3. Hardware atomic primitives (lock-free)

On unified memory, these guarantees require coordination between CPU and GPU memory controllers.

## Implications for Algorithm Design

### DO:
- **Design for read-heavy workloads** - Reads are 3x faster
- **Use bidirectional operations** - Parallel read+write is 15x faster than RMW
- **Avoid read-modify-write patterns** - Use separate passes if possible
- **Batch atomic operations** - Reduce synchronization frequency

### DON'T:
- **Don't assume traditional GPU optimization rules apply** - Unified memory is different
- **Don't rely on temporal locality** - Same-data reuse doesn't help
- **Don't use atomic operations in hot paths** - Use parallel reduction instead
- **Don't use write-read dependencies** - Restructure to avoid barriers

## Comparison with Discrete GPUs

| Feature | Apple M2 Unified | NVIDIA Discrete |
|---------|------------------|----------------|
| Read/Write Ratio | ~3x (read faster) | ~1:1 |
| Temporal Locality | Minimal benefit | Strong benefit |
| Spatial Coalescing | Less critical | Critical |
| Atomic Overhead | Very high | Moderate |
| Memory Model | Unified (shared) | Separate GPU memory |

## Optimization Strategies for Apple M2

### 1. Read-First Design
```metal
// Bad: Write-heavy pattern
data[id] = compute();
threadgroup_barrier(mem_flags::mem_device);
use(data[id]);

// Good: Read-heavy pattern
let val1 = data[id];
let val2 = data[id + 1];
let result = compute(val1, val2);
data[id] = result;  // Single write at end
```

### 2. Avoid Synchronization Barriers
```metal
// Bad: Barrier for write-read dependency
data[id] = val;
threadgroup_barrier(mem_flags::mem_device);
float readback = data[id];

// Good: Separate buffers
writeBuffer[id] = val;
readBuffer[id] = val;  // Copy on same kernel, no barrier
```

### 3. Use Parallel Reduction for Atomics
```metal
// Bad: Atomic in hot path
atomic_fetch_add(&counter, 1);

// Good: Local reduction + single atomic at end
localCounter[lid]++;
// ... at end of threadgroup ...
atomic_fetch_add(&globalCounter, localCounter[0]);
```

## Roofline Analysis

For Apple M2 unified memory:

```
                    Memory Bound Region
                    (Most algorithms)
Peak Compute: 12 GFLOPS  |    Compute Bound
                               Region
                               (N-body, FFT)
                              /
                             /
Peak Memory: 100 GB/s ------/
                           /
                          /
                         /
            Operational Intensity (FLOPs/Byte)
```

Most algorithms on Apple M2 operate in the memory-bound region due to unified memory sharing.

## Conclusions

1. **Read is 3x faster than write** - Design algorithms to read more than write
2. **Temporal locality provides no benefit** - Traditional GPU cache optimization doesn't apply
3. **Spatial locality less critical** - Unified memory handles varied access better
4. **Read-modify-write is extremely slow** - Restructure algorithms to avoid
5. **Atomic operations are very expensive** - Use parallel reduction instead
6. **Bidirectional operations are efficient** - Parallel read+write is best pattern

## References

- WWDC2020: "Metal for GPU Debugging and Optimization"
- Apple M2 Technical Overview
- Metal Shading Language Specification