#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Topic 1: Global Memory Bandwidth vs Data Size Test Kernels
// =============================================================================

// Sequential Read - test sequential read bandwidth
template <typename T>
__global__ void sequentialReadGlobalKernel(const T* __restrict__ src,
                                           T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    T sum = 0;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    if (idx < gridDim.x * blockDim.x) {
        dst[idx] = sum;
    }
}

// Sequential Write - test sequential write bandwidth
template <typename T>
__global__ void sequentialWriteGlobalKernel(const T* __restrict__ src,
                                            T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

// Read-Modify-Write - test read-write mixed bandwidth
template <typename T>
__global__ void readModifyWriteGlobalKernel(const T* __restrict__ src,
                                           T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i] * 2.0f + src[i];
    }
}

// =============================================================================
// Topic 2: Global -> L1 -> L2 Memory Hierarchy Bandwidth Test Kernels
// =============================================================================

// Shared Memory (L1-like) bandwidth test
template <typename T>
__global__ void sharedMemoryBandwidthKernel(const T* __restrict__ global_src,
                                            T* __restrict__ global_dst,
                                            size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Phase 1: Global -> Shared (L1)
    if (idx < N) {
        shared_buf[tid] = global_src[idx];
    }
    __syncthreads();

    // Phase 2: Shared -> Global
    if (idx < N) {
        global_dst[idx] = shared_buf[tid] * 2.0f;
    }
}

// L2 Streaming test - strided access to trigger L2
template <typename T>
__global__ void l2StreamingAccessKernel(const T* __restrict__ src,
                                        T* __restrict__ dst,
                                        size_t N, size_t stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalThreads = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += totalThreads) {
        T sum = 0;
        size_t k = i;
        size_t count = 0;
        while (count < stride && k < N) {
            sum += src[k];
            k += totalThreads;
            count++;
        }
        if (idx < N) {
            dst[idx] = sum;
        }
    }
}

// L2 Cache Bypass test - using __ldg
template <typename T>
__global__ void l2BypassAccessKernel(const T* __restrict__ src,
                                      T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T val = __ldg(&src[i]);
        dst[i] = val * 1.0f;
    }
}

// L1 Preference test - use more registers to avoid spills
template <typename T>
__global__ void l1PreferenceKernel(const T* __restrict__ src,
                                   T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    T r0, r1, r2, r3, r4, r5, r6, r7;
    T sum = 0;

    for (size_t i = idx; i < N; i += stride) {
        r0 = src[i];
        r1 = r0 * 2.0f;
        r2 = r1 * 3.0f;
        r3 = r2 * 4.0f;
        r4 = r3 * 5.0f;
        r5 = r4 * 6.0f;
        r6 = r5 * 7.0f;
        r7 = r6 * 8.0f;
        sum = (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7) * 0.125f;
        dst[i] = sum;
    }
}

// =============================================================================
// Topic 3: TMA (Tensor Memory Accelerator) Copy Test
// =============================================================================

// TMA 1D copy test
template <typename T>
__global__ void tmaCopy1DKernel(const T* __restrict__ src,
                                 T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

// TMA 2D copy simulation - block copy
template <typename T>
__global__ void tmaCopy2DKernel(const T* __restrict__ src,
                                 T* __restrict__ dst,
                                 size_t rows, size_t cols,
                                 size_t src_stride, size_t dst_stride) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        size_t src_idx = row * src_stride + col;
        size_t dst_idx = row * dst_stride + col;
        dst[dst_idx] = src[src_idx];
    }
}

// =============================================================================
// Topic 4: Memory Access Pattern Impact on Performance
// =============================================================================

// Sequential Access
template <typename T>
__global__ void sequentialAccessKernel(const T* __restrict__ src,
                                       T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i] * 2.0f;
    }
}

// Strided Access - test different strides
template <typename T>
__global__ void stridedAccessKernel(const T* __restrict__ src,
                                    T* __restrict__ dst,
                                    size_t N, int stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalThreads = gridDim.x * blockDim.x;

    for (size_t base = idx; base < N; base += totalThreads * (size_t)stride) {
        size_t i = base;
        T sum = 0;
        int s = 0;
        while (s < stride && i < N) {
            sum += src[i];
            i++;
            s++;
        }
        if (idx < N) {
            dst[idx] = sum;
        }
    }
}

// Pointer Chasing - test dependent read latency
template <typename T>
__global__ void pointerChaseKernel(const T* __restrict__ src,
                                    T* __restrict__ dst,
                                    size_t* __restrict__ indices,
                                    size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        size_t pos = indices[i] % N;
        dst[i] = src[pos] * 1.0f;
    }
}

// Write-Combining test
template <typename T>
__global__ void writeCombiningKernel(const T* __restrict__ src,
                                      T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

// Different data types bandwidth test
template <typename T>
__global__ void typeBandwidthKernel(const T* __restrict__ src,
                                    T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

// Broadcast write test
template <typename T>
__global__ void broadcastWriteKernel(T* __restrict__ dst, size_t N, T val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = val;
    }
}

// Reduction pattern (read many, write one)
template <typename T>
__global__ void reductionPatternKernel(const T* __restrict__ src,
                                       T* __restrict__ result, size_t N) {
    __shared__ T shared_sum[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    T sum = 0;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = shared_sum[0];
    }
}
