#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Multi-Stream Research Kernels
// =============================================================================
//
// Multi-Stream Concepts:
// - cudaStreamCreate / cudaStreamCreateWithPriority
// - cudaStreamSynchronize / cudaStreamQuery
// - cudaStreamWaitEvent / cudaStreamRecordEvent
// - Concurrent kernel execution
// - Memory transfer overlap
//
// Use cases:
// - Pipelining data processing
// - Concurrent kernel execution
// - Memory transfer + compute overlap
// - Priority scheduling
// =============================================================================

// =============================================================================
// Basic Kernels for Stream Testing
// =============================================================================

template <typename T>
__global__ void streamVectorAddKernel(const T* __restrict__ a,
                                      const T* __restrict__ b,
                                      T* __restrict__ c,
                                      size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
__global__ void streamVectorScaleKernel(T* __restrict__ data,
                                        T scalar,
                                        size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * scalar;
    }
}

template <typename T>
__global__ void streamMatrixMulKernel(const T* __restrict__ A,
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

// =============================================================================
// Reduction Kernel for Verification
// =============================================================================

template <typename T>
__global__ void streamReduceKernel(const T* __restrict__ data,
                                   T* __restrict__ result,
                                   size_t N) {
    __shared__ T shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    T val = (idx < N) ? data[idx] : 0;
    shared[tid] = val;
    __syncthreads();

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
// Memory Intensive Kernel
// =============================================================================

template <typename T>
__global__ void streamMemoryIntensiveKernel(T* __restrict__ data,
                                             size_t N, int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        T val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.001f + 0.001f;
            val = sin(val) + cos(val);
        }
        data[idx] = val;
    }
}

// =============================================================================
// Compute Intensive Kernel
// =============================================================================

template <typename T>
__global__ void streamComputeIntensiveKernel(T* __restrict__ data,
                                              size_t N, int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        T val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * val;
            val = sqrtf(val + 1.0f);
        }
        data[idx] = val;
    }
}

// =============================================================================
// Pipeline Stage Kernels
// =============================================================================

template <typename T>
__global__ void pipelineLoadKernel(const T* __restrict__ input,
                                   T* __restrict__ output,
                                   size_t N, size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

template <typename T>
__global__ void pipelineProcessKernel(const T* __restrict__ input,
                                      T* __restrict__ output,
                                      size_t N, size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < N) {
        T val = input[idx];
        for (int i = 0; i < 10; i++) {
            val = val * 1.01f + 0.01f;
        }
        output[idx] = val;
    }
}

template <typename T>
__global__ void pipelineStoreKernel(const T* __restrict__ input,
                                    T* __restrict__ output,
                                    size_t N, size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < N) {
        output[idx] = input[idx] + 1.0f;
    }
}

// =============================================================================
// Wait Kernel (for Event Synchronization)
// =============================================================================

template <typename T>
__global__ void waitKernel(T* __restrict__ data,
                           volatile int* __restrict__ flag,
                           size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (*flag == 0) {
        // Spin wait
    }

    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}
