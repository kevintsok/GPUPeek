# CUTLASS Tutorial: Sub-byte GEMM on NVIDIA Blackwell GPUs

Source: https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/

## Key Technical Concepts

This article covers low-precision computation using 6-bit and 4-bit floating-point formats on NVIDIA Blackwell GPUs. Blackwell supports sub-byte precision with five floating-point types: E5M2, E4M3 (8-bit), E3M2, E2M3 (6-bit), and E2M1 (4-bit).

## Low Precision Rationale

Low precision formats reduce model size and computational load. Historical progression includes FP16 (Volta 2017), BF16 (Google/AMPERE), TF32 (AMPERE), and FP8 (Hopper).

## Mixed-Precision UMMA

The `f8f6f4` mixed-input UMMA allows any combination of supported 8-bit, 6-bit, and 4-bit operands with FP32 or FP16 accumulation. The K extent for MMA tiles is always 32 for dense GEMM.

## SMEM and GMEM Layouts

Sub-byte operands in SMEM must be stored in a 16-byte aligned format where 16 consecutive 4-bit or 6-bit elements are packed contiguously. Fully compressed contiguous data in SMEM is not supported with `.kind::f8f6f4`.

## TMA Loading

TMA handles unpacking from packed GMEM to the required padded SMEM format using specialized tensor map data types: `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` for 4-bit and `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` for 6-bit data.

## Alignment Requirements

TMA with sub-byte types requires:
- 32-byte base address alignment
- Leading dimension must be a multiple of 128 elements
- Only 128-byte swizzling patterns or no swizzling supported

## TMEM Support

UMMA can source operand A from TMEM (but not operand B), with sub-byte data padded to 1-byte containers.
