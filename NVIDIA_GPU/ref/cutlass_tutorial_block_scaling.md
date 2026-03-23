# CUTLASS Tutorial: Hardware-supported Block-scaling with NVIDIA Blackwell GPUs

Source: https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/

## Overview

Block-scaling is a dequantization technique where operand data is multiplied by scale factors before multiply-add operations. The formula is:

```
D = (A * scale_A) @ (B * scale_B) + C
```

Blackwell Tensor Cores provide hardware support for this, dividing rows/columns into 16 or 32 element chunks in K-mode, with each chunk multiplied by its own scale factor.

## Supported Data Types for Blockscaled GEMM

The article documents five different combinations of operand data type, vector length, and scale factor data type:

| Operand Type | Vector Length | Scale Factor Type |
|--------------|---------------|------------------|
| mxf8 (E5M2, E4M3) | 32 | UE8M0 |
| mxf6 (E3M2, E2M3) | 32 | UE8M0 |
| mxf4 (E2M1) | 32 | UE8M0 |
| nvf4 (E2M1) | 16 | UE4M3 |

Scale factors are unsigned 8-bit floating point numbers. UE4M3 offers more accuracy but reduced range (max 448), while UE8M0 can represent values as 2^x where -127 ≤ x ≤ 127.

## PTX Syntax for Blockscaled UMMA

The instruction syntax is:

```asm
tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
    [d-tmem], a-desc, b-desc, idesc,
    [scale-A-tmem], [scale-B-tmem], enable-input-d;
```

Key qualifiers:
- `.kind`: mxf8f6f4, mxf4, or mxf4nvf4
- `.scale_vectorsize`: .scale_vec::1X, 2X, 4X, .block16, or .block32

The `.scale_vectorsize` qualifier specifies how many scale factors per UMMA atom row:
- 1X: Shape M×1 (scale_A) and N×1 (scale_B)
- 2X: Shape M×2 and N×2
- 4X: Shape M×4 and N×4

## Scale Factor TMEM Layouts

Three examples are provided for dense MMA with bM = 128:

1. **block32/1X** (atom_K=32 for mxf8f6f4): Scale factors stored in 1-byte aligned sub-columns of a 32-lane × 4-column tile

2. **block32/2X** (atom_K=64 for mxf4/mxf4nvf4): Scale factors stored in two adjacent 2-byte-aligned sub-columns

3. **block16/4X** (atom_K=64 for mxf4nvf4): Scale factor vector M×4 per MMA block, requiring 16 columns for SFA and up to 32 for SFB

TMEM requirements vary by format—block32/1X needs at most 12 columns total, block32/2X needs up to 24 columns, and block16/4X needs up to 48 columns.

## CUTLASS Implementation

The CuTeDSL example `dense_blockscaled_gemm_persistent.py` demonstrates implementation. Key components:

### Tiled MMA Helper

```python
tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
    self.a_dtype,
    self.a_major_mode,
    self.b_major_mode,
    self.sf_dtype,
    self.sf_vec_size,
    self.cta_group,
    self.mma_inst_shape_mn,
)
```

### MXF8 Operation

```python
if ab_dtype in {Float8E4M3FN, Float8E5M2}:
    mma_op = MmaMXF8Op(
        ab_dtype,
        (*mma_tiler_mn, 32),  # atom_K must be 32 bytes
        cta_group,
        a_source,
        a_leading_mode,
        b_leading_mode,
    )
```

### MXF4 Operation

```python
elif ab_dtype == Float4E2M1FN:
    if sf_vec_size == 32:
        mma_op = MmaMXF4Op(
            (*mma_tiler_mn, 64),
            cta_group,
            a_source,)
    elif sf_vec_size == 16:
        mma_op = MmaMXF4NVF4Op(
            sf_dtype,  # can be either E8M0 or E4M3
            (*mma_tiler_mn, 64),
            cta_group,
            a_source,)
```

## Scale Factor Interleaving

Scale factors require an interleaved GMEM layout for efficient TMA loading. A helper function permutes K-major scale tensors:

```python
def interleave_sf_tensor(sf: torch.Tensor) -> torch.Tensor:
    M, SF_K = sf.shape
    REST_M = M // 128
    REST_K = SF_K // 4
    out = sf.reshape(REST_M, 4, 32, REST_K, 4)
    out = out.permute(0, 3, 2, 1, 4).contiguous()
    out = out.permute(2, 3, 0, 4, 1)
    return out
```

The broadcasted layout for block16 is:
```
(((32, 4), REST_M), ((16, 4), REST_K)) : (((16, 4), 512 * REST_K), ((0, 1), 512))
```

## Key Constraints

- bM = 128 for 1-CTA MMA, and 128 or 256 for 2-CTA MMA
- atom_K = 32 for MXF8, atom_K = 64 for MXF4/MXF4NVF4
- bK = 4 × atom_K for optimal performance
- 4-bit data types can be packed 2 elements per byte using `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`
- The persistent tile scheduler separates worktile logic from data-dependent operations
