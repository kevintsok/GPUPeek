# Phase 3: Compute Throughput

## Overview

Third phase research focusing on compute throughput of Apple M2 GPU, including matrix multiplication and arithmetic operations.

## Key Findings

| Operation | Performance | Notes |
|-----------|------------|-------|
| FP32 MatMul (naive) | 4.30 GFLOPS | Memory-bound |
| FP32 MatMul (tiled) | 9.11 GFLOPS | **2.1x speedup** |
| FP16 MatMul | 4.92 GFLOPS | Only 5% faster than FP32 |
| FMA | 0.22 GFLOPS | Memory-bound |
| Trig Functions | 0.57 GOPS | sin+cos+tan |

**Conclusion**: Shared memory tiling provides 2-3x speedup for matrix operations.

## Files

- `Phase3_Compute_Throughput_Report.md` - Full English report
- `Phase3_Compute_Throughput_Report_CN.md` - Full Chinese report

## Research Notes

- Tiled matrix multiply achieves 9.11 GFLOPS (2.1x speedup over naive)
- FP16 shows minimal advantage (5% faster than FP32)
- Memory bandwidth remains the bottleneck for all operations
