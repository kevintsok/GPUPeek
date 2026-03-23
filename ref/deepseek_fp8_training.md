# DeepSeek-R1 and FP8 Mixed-Precision Training

Source: https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/

## Article Overview

DeepSeek-R1 is a reasoning model that generates "chain of thought" before responding, achieving parity with OpenAI's o1 on benchmarks including math, coding, and language understanding while being open-source and cost-effective.

## FP8 Mixed-Precision Strategy (DeepSeek-V3)

- Model weights: stored in FP8
- Matrix multiplications: performed in FP8 with FP32 accumulation
- Activations and gradients: stored in BF16
- Internal computations: use FP32

## Performance Data

- NVIDIA H100 SXM GPU FP8 Tensor Core throughput: approximately 2 petaFLOPS
- Scaling granularity: 128×128 submatrices (blockwise) and 1×128 subvectors (tilewise)

## Technical Challenges Addressed

1. **Outlier sensitivity**: Blockwise/tilewise scaling prevents large weights from forcing others to zero
2. **Accumulation precision**: Fixed-point accumulation limited to ~14 bits instead of true FP32
3. **Solution**: Alternate WGMMA operations (Tensor Cores) with FP32 accumulation (CUDA Cores)

## CUTLASS Implementation

- FP32 accumulation support since version 3.2
- Blockwise scaling added in version 3.7
- Available kernel schedule: `KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum`
- Reference example: 67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu
- Alternative (faster but lower precision): `KernelTmaWarpSpecializedFP8FastAccum`
