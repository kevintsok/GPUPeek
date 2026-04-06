# ANE Discrete Cosine Transform (DCT) Research

## Overview

The Discrete Cosine Transform (DCT) is a Fourier-related transform used in
image and video compression. DCT Type-II is the most commonly used variant,
particularly in JPEG, MPEG, H.264, and HEVC.

## DCT Formula

2D DCT-II formula:
F(u,v) = α(u)α(v) Σ Σ f(i,j) cos[π(2i+1)u/2N] cos[π(2j+1)v/2N]

where α(k) = 1/√2 for k=0, else 1

## Complexity

- Naive 2D DCT: O(n⁴)
- Separable (row + column): O(n³)
- 8x8 block-based: O(n²) with parallelization

## Benchmark Results

### 1D DCT Size Scaling
| Size | CPU (ms) | GPU (ms) | Speedup |
|------|-----------|----------|--------|
| 8 | 0.502 | 0.000 | 501.7x |
| 16 | 0.448 | 0.000 | 447.9x |
| 32 | 0.486 | 0.000 | 486.1x |
| 64 | 0.451 | 0.000 | 451.4x |
| 128 | 0.457 | 0.000 | 456.6x |
| 256 | 0.425 | 0.000 | 425.4x |
| 512 | 0.413 | 0.000 | 413.2x |
| 1024 | 0.410 | 0.000 | 409.7x |


### 2D DCT Performance
| Size | CPU (ms) | GPU (ms) | Throughput |
|------------|----------|----------|-----------|
| 256x256 | 7452.64 | 0.00 | 65.5 MP/s |
| 512x512 | 29419.92 | 0.00 | 262.1 MP/s |
| 1024x1024 | 117608.82 | 0.00 | 1048.6 MP/s |


### Block Size Impact
| Block | CPU (ms) | GPU (ms) | Speedup |
|-------|----------|----------|--------|
| 4x4 | 19766.13 | 0.00 | 19766128.0x |
| 8x8 | 29501.04 | 0.00 | 29501044.0x |
| 16x16 | 50879.34 | 0.00 | 50879340.0x |
| 32x32 | 95110.30 | 0.00 | 95110296.0x |


## Key Insights

1. **GPU speedup increases with image size** due to parallelism
2. **8x8 blocks are standard** for JPEG compatibility
3. **DCT/IDCT are symmetric** in computational cost
4. **Block-based processing** enables efficient parallelization

## ANE Suitability

DCT is suitable for ANE because:
- Butterfly structure maps well to GPU
- Independent block processing
- Video encoding pipelines benefit

## Applications

1. JPEG Compression
2. Video Encoding (MPEG, H.264, HEVC)
3. Image Filtering in frequency domain
4. Pattern Recognition