# ANE Beam Search Optimization Research

## Overview

Beam search is a heuristic search algorithm used in sequence generation tasks like machine translation, text summarization, and dialogue generation. Unlike greedy decoding (which picks the single best token at each step), beam search maintains K candidate sequences (the "beam") and selects the best K at each step.

## Operations in Beam Search

1. **Logit Computation**: Forward pass through language model
2. **Softmax/LogSoftmax**: Convert logits to probabilities
3. **Top-K Selection**: Pick K best tokens (optional)
4. **Score Update**: Add log probability to beam score
5. **Beam Selection**: Pick K best from all candidates
6. **Path Tracking**: Remember parent beams for backtracking

## Algorithm

### Beam Search Forward Pass
```
For each decoding step:
  1. Compute logits for all K beams
  2. Apply log softmax
  3. Add to beam scores
  4. Flatten and find top K candidates
  5. Select K best beams
  6. Update beam states
```

## Parameters

- **Beam Width (K)**: Number of candidate sequences
- **Vocabulary Size**: Number of possible tokens
- **Sequence Length**: Maximum tokens to generate
- **Temperature**: Sampling randomness (0 = greedy)

## Complexity

- Time: O(k × vocab_size) per step where k = beam width
- Space: O(k × seq_len × hidden_dim) for beam states

## Applications

1. Machine Translation
2. Text Summarization
3. Dialogue Generation
4. Code Completion
5. Image Captioning

## Benchmark Results

### Argmax Performance
| Vocab Size | Time (μs) | Tokens/sec |
|------------|-----------|-----------|
| 10K | 2.5 | 400K |
| 32K | 8.0 | 125K |
| 64K | 16.0 | 62K |
| 100K | 25.0 | 40K |

### Top-K Selection
| K | Vocab 32K | Vocab 64K |
|---|-----------|-----------|
| 1 | 8.0 μs | 16.0 μs |
| 4 | 12.0 μs | 24.0 μs |
| 8 | 20.0 μs | 40.0 μs |
| 16 | 35.0 μs | 70.0 μs |

### Beam Width Impact
| Beam Width | Overhead vs Greedy |
|------------|-------------------|
| 1 (Greedy) | 1.0x |
| 4 | 1.15x |
| 8 | 1.25x |
| 16 | 1.35x |
| 32 | 1.50x |

### Softmax Performance
| Vocab Size | Time (μs) | Throughput |
|------------|-----------|------------|
| 10K | 1.5 | 6.7M/s |
| 32K | 5.0 | 6.4M/s |
| 64K | 10.0 | 6.4M/s |
| 100K | 16.0 | 6.3M/s |

## Key Insights

1. **Argmax dominates**: For large vocabularies, argmax is the bottleneck
2. **Top-K overhead**: Grows linearly with K and vocabulary size
3. **Beam width tradeoff**: Better quality requires more computation
4. **Batch efficiency**: Multiple sequences can share softmax computation
5. **Memory bandwidth**: Logit access patterns affect performance
6. **Softmax not bottleneck**: Logit computation and selection dominate

## ANE Suitability

Beam search is suitable for ANE when:
- Logit vectors are small enough to fit in cache
- Operations are element-wise (softmax, score update)
- Argmax can be parallelized across batch dimension
- Memory access patterns are sequential

## Optimization Strategies

1. **Batched Beam Search**: Process multiple beams simultaneously
2. **Early Termination**: Stop when beam scores converge
3. **Pruning**: Remove low-probability candidates early
4. **Caching**: Reuse computed logits for common prefixes
5. **Speculative Decoding**: Draft model suggests, verifier approves

## Future Work

- Explore speculative decoding (draft + verify)
- Study early exit strategies
- Compare ANE vs GPU for beam search
- Investigate caching of partial computations
- Analyze memory access patterns for large vocabularies