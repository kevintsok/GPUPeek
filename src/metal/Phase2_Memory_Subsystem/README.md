# Phase 2: Memory Subsystem

## Overview

Second phase research focusing on Apple M2 memory subsystem characteristics, including access patterns, threadgroup memory, and atomic operations.

## Key Findings

| Memory Pattern | Performance | Impact |
|---------------|------------|--------|
| Sequential Access | 0.81 GB/s | Baseline |
| Strided Access (stride=4) | 0.35 GB/s | **2.3x slower** |
| Write-only | 1.57 GB/s | **Faster** |
| Read-Modify-Write | 0.93 GB/s | + overhead |

**Conclusion**: Apple implements aggressive write combining. Strided access has significant penalty.

## Files

- `Phase2_Memory_Subsystem_Report.md` - Full English report
- `Phase2_Memory_Subsystem_Report_CN.md` - Full Chinese report

## Research Notes

- Write combining makes writes 2x faster than reads
- Atomic operations: 0.093 GOPS (high contention)
- Threadgroup memory bandwidth: 0.02-0.07 GB/s (limited by transfer overhead)
