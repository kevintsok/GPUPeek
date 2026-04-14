# CUTLASS Repository for Blackwell (SM 12.0) GPU Research

## Repository Location
`D:\Projects\dissecting-sm110\ref\cutlass\`

## Cloned: 2026-03-23
- URL: https://github.com/NVIDIA/cutlass.git
- Branch: main (shallow clone, depth 1)

## Key Blackwell Examples (SM 12.0 / SM 120)

| Example | Description | Key Features |
|---------|-------------|--------------|
| 70_blackwell_gemm | Basic dense GEMM | Simple FP16/BF16 GEMM |
| 72_blackwell_narrow_precision_gemm | Block-scaled dense GEMM | Blockwise scaling |
| 78_blackwell_emulated_bf16x9_gemm | FastFP32 using BF16 | Emulated SGEMM |
| **79_blackwell_geforce_gemm** | **SM120 MMA for RTX 50 series** | **NVFP4/BF16, Block Scaling** |
| 79a | NVFP4 + BF16 GEMM | FP4 (e2m1) with block scaling |
| 79b | NVFP4 + NVFP4 GEMM | FP4 x FP4 with block scaling |
| 79c | Mixed MXFP8/MXFP6 + BF16 | 8-bit and 6-bit mixed precision |
| 79d | Grouped GEMM | Variable batch sizes |
| 80_blackwell_geforce_sparse_gemm | Sparse MMA | 2:4 structured sparsity |
| 81_blackwell_gemm_blockwise | Blockwise scaling | Per-block scale factors |
| 86_blackwell_mixed_dtype_gemm | Mixed precision | Multiple data types |
| 87_blackwell_geforce_gemm_blockwise | Blockwise for GeForce | Alternative block scaling |

## Key CUTLASS Include Files for UMMA/TCGen05

| File | Purpose |
|------|---------|
| `include/cute/arch/mma_sm100_umma.hpp` | UMMA MMA instruction wrappers |
| `include/cute/arch/mma_sm100.hpp` | SM100 MMA instructions |
| `include/cute/arch/mma_sm100_desc.hpp` | MMA descriptor definitions |
| `include/cute/atom/mma_traits_sm100.hpp` | MMA atom traits |
| `include/cutlass/arch/mma_sm100.h` | CUTLASS arch definitions |
| `include/cutlass/gemm/collective/builders/sm100_umma_builder.inl` | UMMA collective builder |
| `include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl` | Block scaled UMMA |

## TCGen05/UMMA Key Concepts

### TMEM (Tensor Memory)
- 256KB per SM on-chip memory for Tensor Cores
- Organized as 512 columns × 128 rows of 32-bit cells
- Source A can come from TMEM or SMEM; source B from SMEM; accumulator in TMEM

### Block Scaling
- Hardware-supported dequantization: D = (A * scale_A) @ (B * scale_B) + C
- Scale factors divide rows/columns into 16 or 32 element chunks
- Supported formats: MXF8 (E5M2/E4M3), MXF6 (E3M2/E2M3), MXF4 (E2M1)

### TCGen05 Instruction Syntax (PTX)
```asm
tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
    [d-tmem], a-desc, b-desc, idesc,
    [scale-A-tmem], [scale-B-tmem], enable-input-d;
```

### Key Differences: WMMA vs TCGen05

| Feature | WMMA (Legacy) | TCGen05/UMMA (Blackwell) |
|---------|---------------|---------------------------|
| Memory | Registers | TMEM (256KB/SM) |
| Shape | m16n16k16 | Various (m64nNk16, etc.) |
| Precision | FP16, BF16, TF32, FP64, INT8 | FP4, FP6, FP8, FP16, BF16, TF32 |
| Block Scaling | Not supported | Hardware supported |
| Clusters | Not supported | 2-CTA MMA |

## Building CUTLASS Examples

```bash
cd cutlass
mkdir -p build/cmake
cd build/cmake
cmake ../.. -DCUTLASS_NVCC_ARCHS="120" -DCMAKE_BUILD_TYPE=Release
make -j16
```

## Running Blackwell GEMM Examples

```bash
# NVFP4 + BF16 GEMM
./build/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm --m=2048 --n=2048 --k=2048

# Block scaled GEMM
./build/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise --m=1024 --n=1024 --k=1024
```

## GPUPeek Integration Notes

- WMMA API (`nvcuda::wmma`) is legacy warp-level MMA - tested in gpupeek
- TCGen05/UMMA API requires CUTLASS or raw PTX inline assembly
- For GPUPeek TCGen05 research: study examples/79 and include/cute/arch/mma_sm100_umma.hpp

## Reference Documentation

- [PTX ISA Section 9.7.16](https://docs.nvidia.com/cuda/parallel-thread-execution) - TCGen05 instructions
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [Colfax CUTLASS Tutorials](../ref/) - Additional research articles
