# Phase 5: Architecture Deep Dive

## Overview

Fifth phase research on Apple M2 architecture including memory compression, access patterns, and latency characteristics.

## Key Findings

| Test | Result | Impact |
|------|--------|--------|
| Random vs Sequential | 0.03 vs 0.88 GB/s | **27x slower** |
| Write vs Read | 1.80 vs 0.62 GB/s | **2.9x faster** |
| Memory Latency (single) | 61 ns | High overhead |
| Memory Latency (pipelined) | 0.45 ns | 135x improvement |
| Cache Line Stride | 6-15x slower |跨越惩罚 |

**Conclusion**: Sequential access is critical. Write combining is highly effective. Memory latency amortizable.

## Files

- `Phase5_Architecture_Deep_Dive_Report.md` - Full English report
- `Phase5_Architecture_Deep_Dive_Report_CN.md` - Full Chinese report

## Research Notes

- Random access is 27x slower than sequential
- Write bandwidth exceeds read by 2.9x (write combining)
- Memory compression has minimal measurable effect
- Single-element memory access: 61ns, pipelined: 0.45ns
