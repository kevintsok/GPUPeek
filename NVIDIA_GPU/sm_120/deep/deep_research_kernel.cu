#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// L2 Cache Deep Analysis - Working Set vs Bandwidth
// =============================================================================

// L2 working set test - access pattern that tests L2 cache effectiveness
template <typename T>
__global__ void l2WorkingSetKernel(const T* __restrict__ src,
                                    T* __restrict__ dst,
                                    size_t N, size_t blockSize) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalThreads = gridDim.x * blockDim.x;

    // Each thread works on a block of 'blockSize' elements
    for (size_t base = idx * blockSize; base < N; base += totalThreads * blockSize) {
        T sum = 0;
        for (size_t i = 0; i < blockSize && (base + i) < N; i++) {
            sum += src[base + i];
        }
        if (idx < N / blockSize) {
            dst[idx] = sum;
        }
    }
}

// L2 cache line access - test sequential access within cache line size
template <typename T>
__global__ void l2CacheLineAccessKernel(const T* __restrict__ src,
                                        T* __restrict__ dst,
                                        size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // RTX 5080 L2 cache line is 64 bytes (16 floats or 8 doubles)
    // Access 16 floats per thread to test cache line efficiency
    for (size_t i = idx * 16; i < N; i += stride * 16) {
        T sum = 0;
        // Sequential access within likely same cache line
        for (size_t j = 0; j < 16 && (i + j) < N; j++) {
            sum += src[i + j];
        }
        dst[idx] = sum;
    }
}

// L2 thrashing test - access pattern that evicts cache lines quickly
template <typename T>
__global__ void l2ThrashKernel(const T* __restrict__ src,
                                 T* __restrict__ dst,
                                 size_t N, size_t stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalThreads = gridDim.x * blockDim.x;

    // Strided access with large stride to thrash L2
    for (size_t i = idx; i < N; i += totalThreads) {
        size_t pos = (i * stride) % N;
        dst[idx] = src[pos];
    }
}

// =============================================================================
// Tensor Core Matrix Multiply Test
// =============================================================================

// Naive matrix multiply (for comparison)
template <typename T>
__global__ void naiveMatrixMultiplyKernel(const T* __restrict__ A,
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
// Warp-Level Operation Detailed Analysis
// =============================================================================

// Warp reduce using shuffle
template <typename T>
__global__ void warpShuffleReduceKernel(const T* __restrict__ input,
                                        T* __restrict__ output,
                                        size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = input[idx];

    // Full warp shuffle reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if ((threadIdx.x & 31) == 0) {
        output[blockIdx.x] = val;
    }
}

// Warp butterfly reduce
template <typename T>
__global__ void warpButterflyReduceKernel(const T* __restrict__ input,
                                          T* __restrict__ output,
                                          size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = input[idx];

    // Butterfly pattern reduce
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 16);

    if ((threadIdx.x & 31) == 0) {
        output[blockIdx.x] = val;
    }
}

// Warp vote ballot test
template <typename T>
__global__ void warpBallotKernel(const int* __restrict__ pred,
                                  int* __restrict__ result,
                                  size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int ballot = __ballot_sync(0xffffffff, pred[idx]);

    if ((threadIdx.x & 31) == 0) {
        result[blockIdx.x] = (int)ballot;
    }
}

// =============================================================================
// Instruction Throughput Test
// =============================================================================

// FMA throughput test
template <typename T>
__global__ void fmaThroughputKernel(const T* __restrict__ a,
                                     const T* __restrict__ b,
                                     const T* __restrict__ c,
                                     T* __restrict__ dst,
                                     size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = a[i] * b[i] + c[i];
    }
}

// Integer throughput test
__global__ void intThroughputKernel(const int* __restrict__ a,
                                     const int* __restrict__ b,
                                     int* __restrict__ dst,
                                     size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = (a[i] + b[i]) * (a[i] - b[i]) + a[i];
    }
}

// Memory fence impact test
template <typename T>
__global__ void memoryFenceImpactKernel(const T* __restrict__ src,
                                         T* __restrict__ dst,
                                         size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T val = src[i];
        __threadfence();  // Memory fence
        dst[i] = val * 2.0f;
    }
}

// No-fence baseline
template <typename T>
__global__ void noFenceKernel(const T* __restrict__ src,
                               T* __restrict__ dst,
                               size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T val = src[i];
        dst[i] = val * 2.0f;
    }
}
