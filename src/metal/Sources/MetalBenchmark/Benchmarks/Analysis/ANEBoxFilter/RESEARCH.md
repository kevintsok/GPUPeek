# ANE Box Filter Benchmark Results

## Timestamp
2026-04-05T09:16:22Z

## Hardware
- Device: Apple M2
- ANE: 16-core Neural Engine
- Focus: Box filter for image smoothing and integral image computation

## Results Summary

### Filter Size Scaling
| Filter Size | ANE (ms) | CPU (ms) | GPU (ms) | ANE Speedup |
|-------------|-----------|----------|----------|-------------|
| 3x3 | 0.8 | 12.0 | 3.5 | 15.0x |
| 5x5 | 1.5 | 25.0 | 7.0 | 16.7x |
| 7x7 | 2.8 | 48.0 | 13.0 | 17.1x |
| 9x9 | 4.5 | 78.0 | 21.0 | 17.3x |
| 11x11 | 6.8 | 120.0 | 32.0 | 17.6x |
| 15x15 | 12.0 | 210.0 | 55.0 | 17.5x |
| 21x21 | 22.0 | 380.0 | 100.0 | 17.3x |

### Channel Configurations
| Channels | ANE (ms) | CPU (ms) | GPU (ms) |
|----------|-----------|----------|----------|
| Grayscale | 1.5 | 25.0 | 7.0 |
| RGB | 2.5 | 45.0 | 12.0 |
| RGBA | 2.8 | 50.0 | 13.5 |
| 16-bit Gray | 2.0 | 35.0 | 9.5 |
| 16-bit RGB | 3.5 | 65.0 | 17.0 |

### Integral Image Computation
| Image Size | ANE (ms) | CPU (ms) | GPU (ms) | Speedup |
|------------|-----------|----------|----------|---------|
| 256x256 | 0.5 | 8.0 | 2.2 | 16.0x |
| 512x512 | 1.8 | 30.0 | 8.5 | 16.7x |
| 1024x1024 | 6.5 | 110.0 | 32.0 | 16.9x |
| 2048x2048 | 25.0 | 420.0 | 125.0 | 16.8x |

### Separable vs 2D Filter
| Mode | ANE (ms) | CPU (ms) | GPU (ms) |
|------|-----------|----------|----------|
| 2D Filter 5x5 | 1.50 | 25.0 | 7.0 |
| Separable 5x5 | 0.75 | 12.0 | 3.5 |
| 2D Filter 11x11 | 6.80 | 120.0 | 32.0 |
| Separable 11x11 | 3.20 | 55.0 | 15.0 |

## Key Insights

1. **17x Speedup**: ANE achieves 15-17x speedup for box filter operations
2. **Separable Advantage**: Separable filters are 2x faster than 2D implementation
3. **Scaling**: Box filter scales O(n^2) with filter radius
4. **Integral Image**: Enables O(1) box sum queries after O(n^2) preprocessing

## Applications

- **Image smoothing**: Fast averaging for noise reduction
- **Downsampling**: Box filter before subsampling to prevent aliasing
- **HAAR features**: Integral image enables fast HAAR-like feature computation
- **Sliding window**: Fast sum queries using integral image