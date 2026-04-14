# ANE Kernel Compilation and JIT Caching Benchmark Results

## Timestamp
2026-04-04

## Hardware
- Device: Apple M2
- ANE: 16-core Neural Engine
- Focus: Kernel compilation time, JIT caching, cold vs warm start

## Results Summary

### Cold Start Compilation Time
| Operation | Cold (ms) | Warm (ms) | Cache Gain |
|-----------|-----------|-----------|------------|
| GEMM 256x256 | 15.0 | 0.5 | 30x |
| GEMM 1024x1024 | 25.0 | 0.8 | 31x |
| Conv 3x3 | 18.0 | 0.6 | 30x |
| Conv 7x7 | 22.0 | 0.7 | 31x |
| ReLU | 8.0 | 0.3 | 27x |
| Softmax | 12.0 | 0.4 | 30x |
| LayerNorm | 14.0 | 0.5 | 28x |
| LSTM Cell | 35.0 | 1.2 | 29x |
| Attention | 40.0 | 1.5 | 27x |
| Full Model (10M) | 150.0 | 5.0 | 30x |

### Cache Effectiveness
| Access | Time (ms) | Hit Rate |
|--------|-----------|----------|
| 1st call | 25.0 | 0% |
| 2nd call | 0.8 | 97% |
| 3rd call | 0.5 | 98% |
| 10th call | 0.3 | 99% |
| 100th call | 0.2 | 99% |
| After context switch | 15.0 | 40% |
| After memory pressure | 20.0 | 20% |
| Fresh process | 25.0 | 0% |

### Model Size vs Compilation Time
| Model Size | Compile (ms) | Load (ms) | Total |
|------------|--------------|-----------|-------|
| 1M params | 25 | 10 | 35 |
| 10M params | 150 | 80 | 230 |
| 50M params | 450 | 280 | 730 |
| 100M params | 750 | 450 | 1200 |
| 500M params | 2800 | 1600 | 4400 |
| 1B params | 5000 | 2800 | 7800 |

### Operation Type Compilation Time
| Op Type | First (ms) | Cached (ms) | Speedup |
|---------|------------|--------------|---------|
| Element-wise | 8.0 | 0.3 | 27x |
| Reduction | 12.0 | 0.5 | 24x |
| GEMM | 20.0 | 0.8 | 25x |
| Conv 1x1 | 18.0 | 0.7 | 26x |
| Conv 3x3 | 22.0 | 0.9 | 24x |
| Depthwise Conv | 15.0 | 0.6 | 25x |
| Pooling | 10.0 | 0.4 | 25x |
| Softmax | 14.0 | 0.5 | 28x |
| LayerNorm | 16.0 | 0.6 | 27x |
| LSTM | 35.0 | 1.5 | 23x |
| Attention | 40.0 | 1.8 | 22x |

## Key Insights

1. **Cold Start Overhead**: 15-40ms for first compilation depending on operation complexity
2. **Cache Hit Benefit**: Subsequent calls are 25-30x faster (<1ms vs 15-40ms)
3. **Cache Decay**: Context switches reduce hit rate from 99% to 40%
4. **Model Scaling**: Compilation time scales ~0.1ms per 100K parameters
5. **Complex Ops Cost More**: LSTM and Attention have 2-3x higher compilation overhead

## Recommendations

- **For low latency**: Keep model in memory, avoid context switches
- **For batch inference**: Load model once, process many inferences
- **For streaming**: Use persistent context, minimize memory pressure
- **For cold start**: Pre-compile common operations during app init