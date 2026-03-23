#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Unified Memory Research Kernels
// =============================================================================
//
// Unified Memory API Functions:
// - cudaMallocManaged - Allocate unified memory
// - cudaMemPrefetchAsync - Prefetch to device/host
// - cudaMemAdvise - Set memory advice
// - cudaMemsetAccessAdvise - Set access advice
// - cudaPointerGetAttributes - Query pointer attributes
//
// Concepts:
// - Managed memory: Single allocation for host/device
// - Page fault: On-demand migration
// - Prefetching: Explicit data migration
// - Access counters: Track device access patterns
//
// Use cases:
// - Simplify memory management
// - GPU memory expansion
// - Heterogeneous computing
// - Out-of-core processing
// =============================================================================

// =============================================================================
// Simple Kernels for Unified Memory Testing
// =============================================================================

template <typename T>
__global__ void vectorAddKernel(const T* __restrict__ a,
                                 const T* __restrict__ b,
                                 T* __restrict__ c,
                                 size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
__global__ void matrixMultiplyKernel(const T* __restrict__ A,
                                      const T* __restrict__ B,
                                      T* __restrict__ C,
                                      size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (size_t k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template <typename T>
__global__ void vectorScaleKernel(T* __restrict__ data,
                                   T scalar,
                                   size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        data[idx] = data[idx] * scalar;
    }
}

template <typename T>
__global__ void vectorReduceKernel(const T* __restrict__ data,
                                    T* __restrict__ result,
                                    size_t N) {
    __shared__ T shared[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Load data
    T val = (idx < N) ? data[idx] : 0;
    shared[tid] = val;
    __syncthreads();

    // Reduce
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}

// =============================================================================
// Access Pattern Kernels
// =============================================================================

template <typename T>
__global__ void sequentialAccessKernel(T* __restrict__ data,
                                       size_t N, int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        for (int i = 0; i < iterations; i++) {
            data[idx] = data[idx] * 1.001f + 0.001f;
        }
    }
}

template <typename T>
__global__ void randomAccessKernel(T* __restrict__ data,
                                     size_t N, size_t* indices,
                                     int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        for (int i = 0; i < iterations; i++) {
            size_t access_idx = indices[(idx + i) % N];
            data[access_idx] = data[access_idx] * 1.001f + 0.001f;
        }
    }
}

template <typename T>
__global__ void stridedAccessKernel(T* __restrict__ data,
                                      size_t N, size_t stride,
                                      int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N / stride) {
        for (int i = 0; i < iterations; i++) {
            size_t access_idx = (idx * stride) % N;
            data[access_idx] = data[access_idx] * 1.001f + 0.001f;
        }
    }
}

// =============================================================================
// Page Fault Detection Kernels
// =============================================================================

template <typename T>
__global__ void touchAllPagesKernel(T* __restrict__ data,
                                    size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Touch each cache line (typically 128 bytes)
    size_t line_size = 128 / sizeof(T);
    size_t num_lines = (N + line_size - 1) / line_size;

    if (idx < num_lines) {
        size_t line_idx = idx * line_size;
        // Write to first element of each line to fault pages
        data[line_idx] = (T)1.0f;
    }
}

// =============================================================================
// Write Combiner Kernels
// =============================================================================

template <typename T>
__global__ void writeCombiningKernel(T* __restrict__ data,
                                      size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Sequential write - benefits from write combining
        data[idx] = (T)(idx & 0xFF);
    }
}

template <typename T>
__global__ void writeScatterKernel(T* __restrict__ data,
                                      size_t N, size_t* indices) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Scattered write - less benefit from write combining
        data[indices[idx % N]] = (T)(idx & 0xFF);
    }
}

// =============================================================================
// GPU-CPU Synchronization Kernels
// =============================================================================

template <typename T>
__global__ void spinWaitKernel(T* __restrict__ data,
                                 volatile int* __restrict__ flag,
                                 size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Spin until flag is set
    while (*flag == 0) {
        // Busy wait
    }

    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

template <typename T>
__global__ void conditionalComputeKernel(T* __restrict__ data,
                                            volatile int* __restrict__ ready,
                                            size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute if ready
    if (*ready != 0 && idx < N) {
        data[idx] = data[idx] * 3.0f;
    }
}
