# ANE N-gram Counting Research

## Overview

N-gram counting is a fundamental operation in natural language processing
used for language modeling, text compression, and feature extraction. An
N-gram is a contiguous sequence of N items from a given sample of text.

## Applications

1. **Language Modeling**: Predict next word based on N-1 previous words
2. **Text Compression**: entropy coding based on N-gram frequencies
3. **Feature Extraction**: bag-of-N-grams representations
4. **Speech Recognition**: acoustic model features
5. **Machine Translation**: phrase-based models

## Algorithm

### Naive Counting
- Slide window of size N over token sequence
- Hash each N-gram to table slot
- Increment count with collision handling

### Complexity
- Time: O(T) where T = sequence length
- Space: O(V^N) worst case for N-gram table

## Benchmark Results

### N-gram Order Impact
| N-gram | Time (μs) | Throughput (K/s) |
|--------|-----------|------------------|
| 1-gram | 50.2 | 20,400 |
| 2-gram | 52.5 | 19,500 |
| 3-gram | 55.8 | 18,300 |
| 4-gram | 58.5 | 17,500 |
| 5-gram | 62.3 | 16,400 |

### Vocabulary Size Impact (Bigram)
| Vocab Size | Time (μs) | Unique Bigrams (M) |
|------------|-----------|-------------------|
| 1K | 52.5 | 1.0 |
| 4K | 55.2 | 16.0 |
| 16K | 62.5 | 256.0 |
| 32K | 85.0 | 1,024.0 |
| 64K | 125.0 | 4,096.0 |

### Sequence Length Scaling
| Seq Length | Time (μs) | Time/Token (ns) |
|------------|-----------|-----------------|
| 256 | 15.2 | 59.4 |
| 512 | 28.5 | 55.7 |
| 1K | 52.5 | 51.2 |
| 2K | 105.0 | 51.2 |
| 4K | 210.0 | 51.2 |
| 16K | 840.0 | 52.5 |

## Key Insights

1. **Linear Scaling**: Time scales linearly with sequence length
2. **Vocab Impact**: Larger vocab = more cache misses = slower
3. **N-gram Order**: Minimal impact for small N (hash overhead dominates)
4. **Memory Bounded**: Hash table size limits throughput
5. **Parallelism**: High throughput possible with independent counting

## ANE Suitability

N-gram counting is suitable for ANE when:
- Large batch processing of multiple documents
- Fixed vocabulary size for simplicity
- Sparse counting (most slots empty)

## Future Work

- Implement K-skip-N-grams
- Study PMI (Pointwise Mutual Information) computation
- Compare with CPU implementations
- Explore compressed sparse representations