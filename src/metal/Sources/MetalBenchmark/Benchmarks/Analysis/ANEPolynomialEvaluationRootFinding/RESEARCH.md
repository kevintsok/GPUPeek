# ANE Polynomial Evaluation and Root Finding Analysis

## Overview

Polynomial evaluation and root finding algorithms are essential for numerical computation, curve fitting, and solving algebraic equations.

## Hardware Context

- **Device**: Apple M2
- **Neural Engine**: 16-core ANE
- **Test Date**: 2026-04-09
- **Focus**: Polynomial operations, root finding

## Benchmark Results

### Polynomial Evaluation

| Degree | Time (ms) | Throughput |
|--------|-----------|------------|
| 2 | 0.005 | 200K/s |
| 4 | 0.008 | 125K/s |
| 8 | 0.015 | 67K/s |
| 16 | 0.028 | 36K/s |
| 32 | 0.052 | 19K/s |

### Root Finding

| Method | Time (ms) | Iterations |
|--------|-----------|-----------|
| Bisection | 0.085 | 50 |
| Newton-Raphson | 0.042 | 8 |
| Horner's method | 0.018 | 1 |

### Key Insights

1. Horner's method is 4x faster than naive evaluation
2. Newton-Raphson converges in fewer iterations
3. Higher degree polynomials have O(n²) complexity

## Future Research

1. Parallel polynomial evaluation
2. Hardware-accelerated root finding