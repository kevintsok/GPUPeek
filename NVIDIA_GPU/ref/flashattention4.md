# FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling

Source: https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/

## Overview

FlashAttention-4 is an algorithm and kernel co-design for Blackwell GPUs (B200) that addresses asymmetric hardware scaling, where tensor core throughput outpaces SFU and memory bandwidth resources.

## Key Hardware Features (Blackwell B200)

- **Tensor Memory (TMEM)**: 256 KB per SM for warp-synchronous intermediate storage
- **Async 5th Gen Tensor Cores**: MMA with 128×256×16 single CTA tile; can source operand A from TMEM
- **2-CTA MMA**: Spans TMEM across paired CTAs, reducing shared memory traffic and atomic reductions

## Feed/Speed Analysis

Per SM at M=N=D=128:
- Tensor Cores (BF16): 8192 ops/cycle
- Exponential unit: 16 ops/cycle
- Shared Memory: 128 bytes/cycle

Forward: Bottlenecked by compute and exp. Backward: Bottlenecked by SMEM bandwidth.

## Forward Pass Innovations

- Ping-pong schedule with 2× Q tiles and 2× O tiles per CTA
- 2× softmax warpgroups with explicit sync to reduce MUFU contention
- **Exp2 emulation via FMA**: Cody-Waite range reduction + Horner polynomial (p₀=1.0, p₁≈0.6951, p₂≈0.2276, p₃≈0.0771)
- Dedicated correction warpgroup for conditional rescaling
- Online softmax rescaling triggered only when max jump exceeds threshold τ

## Backward Pass Innovations

- Recompute S and P as transposed tiles; store P^T and dS^T in TMEM
- 2-CTA mode: M=256, N=K=128; halves operand B traffic
- DSMEM exchange resolves dQ reduction axis conflict
- Deterministic mode via CTA swizzling and SPT ordering (85-90% of nondet throughput)

## Scheduling

- **LPT (Longest Processing Time First)** for causal masking via batch-head section traversal
- Batch sorting kernel for variable sequence length with cached metadata

## Performance Results (B200 BF16)

- **Up to 1605 TFLOPs/s** (71% utilization)
- 1.1-1.3× faster than cuDNN 9.13
- 2.1-2.7× faster than Triton
- Consistent backward pass advantage at large sequence lengths

## Implementation

- Written in **CuTe-DSL** (CUTLASS Python kernel DSL)
- ~20-30× faster compile times vs C++ templates
- Available at: github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute
