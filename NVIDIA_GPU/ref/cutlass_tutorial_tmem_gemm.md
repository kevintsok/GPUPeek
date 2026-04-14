# CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA® Blackwell GPUs

Source: https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/

## Summary

This technical article from Colfax Research provides a tutorial on writing GEMM (General Matrix Multiply) kernels using Tensor Memory for NVIDIA Blackwell GPUs. The article is Part 1 of a three-part series exploring new Blackwell architecture features.

## Key Topics Covered

### Blackwell MMA Overview

- Blackwell replaces Hopper's WGMMA instruction with the new `tcgen05.mma` (UMMA) instruction
- UMMA introduces several major differences including support for low-precision data types (FP4, FP6), built-in block scaling, and Tensor Memory integration
- Two adjacent CTAs within an SM cluster ("CTA pair") can collaborate on UMMA operations

### Tensor Memory (TMEM)

- Dedicated 256KB per-SM on-chip memory for Tensor Cores
- Organized as 512 columns × 128 rows of 32-bit cells
- Replaces registers for MMA operations, reducing register pressure
- Requires explicit allocation via `tcgen05.alloc` and deallocation via `tcgen05.dealloc`

### Instruction Details

The article covers `tcgen05.mma` syntax and operand requirements:

- Operand A: can be in TMEM or SMEM
- Operand B: must be in SMEM
- Accumulator: must be in TMEM

It also explains `tcgen05.ld` for copying data from TMEM to registers for post-processing.

### CUTLASS Interface

The tutorial introduces the CUTLASS/CuTe interface for UMMA, including MMA_Atom wrappers and MMA_Traits for CuTe layouts.

## Context

This is Part 1 of a series; Parts 2 and 3 cover cluster operations (including TMA multicast and CTA-pair concepts) and MMA with lower precision datatypes with block-scaling support respectively.
