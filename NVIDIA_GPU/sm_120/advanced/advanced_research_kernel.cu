#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Occupancy Analysis Kernels
// =============================================================================

// Kernel with high register usage
template <typename T>
__global__ void highRegisterKernel(const T* __restrict__ src,
                                   T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Many local variables to force high register usage
    T r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;
    T r10, r11, r12, r13, r14, r15, r16, r17, r18, r19;

    for (size_t i = idx; i < N; i += stride) {
        r0 = src[i];
        r1 = r0 * 2.0f; r2 = r1 * 3.0f; r3 = r2 * 4.0f;
        r4 = r3 * 5.0f; r5 = r4 * 6.0f; r6 = r5 * 7.0f;
        r7 = r6 * 8.0f; r8 = r7 * 9.0f; r9 = r8 * 10.0f;
        r10 = r9 * 11.0f; r11 = r10 * 12.0f; r12 = r11 * 13.0f;
        r13 = r12 * 14.0f; r14 = r13 * 15.0f; r15 = r14 * 16.0f;
        r16 = r15 * 17.0f; r17 = r16 * 18.0f; r18 = r17 * 19.0f;
        r19 = r18 * 20.0f;
        T sum = (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 +
                 r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19) * 0.05f;
        dst[i] = sum;
    }
}

// Kernel with low register usage
template <typename T>
__global__ void lowRegisterKernel(const T* __restrict__ src,
                                   T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i] * 2.0f;
    }
}

// Kernel with shared memory usage
template <typename T>
__global__ void sharedMemoryKernel(const T* __restrict__ src,
                                   T* __restrict__ dst, size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Safe access: only use tid % 256 to stay within bounds
    if (idx < N && tid < 256) {
        shared_buf[tid] = src[idx];
    }
    __syncthreads();

    if (idx < N && tid < 256) {
        dst[idx] = shared_buf[tid] * 2.0f;
    }
}

// =============================================================================
// Bank Conflict Analysis Kernels
// =============================================================================

// Sequential shared memory access (no bank conflict)
template <typename T>
__global__ void sequentialSharedAccessKernel(T* __restrict__ dst, size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Sequential write - bank conflict free (only first 256 threads)
    if (tid < 256) {
        shared_buf[tid] = tid;
    }
    __syncthreads();

    // Sequential read (only first 32 threads do this)
    T sum = 0;
    if (tid < 32) {
        for (int i = 0; i < 32; i++) {
            sum += shared_buf[i];
        }
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = sum;
    }
}

// Strided shared memory access (with bank conflict)
template <typename T>
__global__ void stridedSharedAccessKernel(T* __restrict__ dst, size_t N, int stride) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory to avoid garbage values
    if (tid < 256) {
        shared_buf[tid] = 0;
    }
    __syncthreads();

    // Strided write - causes bank conflicts (only first 256 threads)
    if (tid < 256) {
        for (int i = tid; i < 256; i += stride) {
            shared_buf[i] = tid;
        }
    }
    __syncthreads();

    // Strided read - all threads participate
    T sum = 0;
    if (tid < 256) {
        for (int i = 0; i < 256; i += stride) {
            sum += shared_buf[i];
        }
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = sum;
    }
}

// =============================================================================
// Branch Divergence Analysis Kernels
// =============================================================================

// No divergence kernel
template <typename T>
__global__ void noDivergenceKernel(const int* __restrict__ pred,
                                    T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        if (pred[i % 256] == 1) {
            dst[i] = pred[i] * 2.0f;
        } else {
            dst[i] = pred[i] * 3.0f;
        }
    }
}

// High divergence kernel
template <typename T>
__global__ void highDivergenceKernel(const int* __restrict__ pred,
                                      T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        if (pred[idx % 2] == 1) {
            dst[i] = pred[i] * 2.0f;
        } else {
            dst[i] = pred[i] * 3.0f;
        }
    }
}

// =============================================================================
// Atomic Operations Performance Kernels
// =============================================================================

__global__ void atomicAddKernel(const float* __restrict__ src,
                                 float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        atomicAdd(dst, src[i]);
    }
}

__global__ void atomicAddBlockKernel(const float* __restrict__ src,
                                      float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = threadIdx.x;

    __shared__ float shared_sum[256];

    // Each thread computes partial sum
    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory - all threads participate
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Only one thread per block does the atomic add
    if (tid == 0) {
        atomicAdd(dst, shared_sum[0]);
    }
}

__global__ void atomicCASKernel(unsigned int* __restrict__ data,
                                 unsigned int* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        unsigned int old = atomicCAS(&data[0], 0u, 1u);
        if (old == 0u) {
            result[i] = 1;
        }
    }
}

// =============================================================================
// Constant Memory Bandwidth Test
// =============================================================================

__constant__ float const_data[4096];

template <typename T>
__global__ void constantMemoryKernel(T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        // All threads read same location (broadcast)
        dst[i] = const_data[0];
    }
}

template <typename T>
__global__ void constantMemoryStrideKernel(T* __restrict__ dst, size_t N, int stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalThreads = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += totalThreads) {
        T sum = 0;
        for (int s = 0; s < stride; s++) {
            sum += const_data[s];
        }
        dst[i] = sum;
    }
}

// =============================================================================
// Instruction Latency Analysis Kernels
// =============================================================================

// Single instruction latency test
__global__ void singleFMAKernel(float* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float a = data[idx];
    float b = data[idx + 1];
    float c = data[idx + 2];

    // Chain of dependent FMAs - tests latency
    for (int i = 0; i < 32; i++) {
        c = a * b + c;
    }

    data[idx] = c;
}

// Independent FMA throughput test
__global__ void independentFMAKernel(float* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        data[i] = data[i] * data[i] + data[i];
    }
}

// Memory bound vs Compute bound test
__global__ void memoryBoundKernel(const float* __restrict__ src,
                                  float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float val = src[i];
        val = val * 2.0f;
        dst[i] = val;
    }
}

__global__ void computeBoundKernel(const float* __restrict__ src,
                                    float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float val = src[i];
        // Heavy computation to hide memory latency
        for (int j = 0; j < 16; j++) {
            val = val * 2.0f + 1.0f;
            val = val * 0.5f - 0.5f;
        }
        dst[i] = val;
    }
}

// =============================================================================
// PCIe Bandwidth Test Kernels
// =============================================================================

template <typename T>
__global__ void hostToDeviceCopyKernel(const T* __restrict__ src,
                                       T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}
