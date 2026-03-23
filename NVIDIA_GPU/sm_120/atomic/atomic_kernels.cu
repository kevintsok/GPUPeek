#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// B.1 Warp-Level Atomic Operations
// =============================================================================

// Warp-level reduction then single atomic per warp
__global__ void atomicWarpLevelAdd(const float* __restrict__ src,
                                    float* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    // Each thread accumulates its portion
    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread of each warp does the atomic add
    if (tid % 32 == 0) {
        atomicAdd(result, sum);
    }
}

// Warp-level reduction using warp vote/ballot
__global__ void atomicWarpLevelMin(const int* __restrict__ src,
                                    int* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    // Each thread finds min in its portion
    int local_min = src[idx % N];
    for (size_t i = idx; i < N; i += stride) {
        local_min = min(local_min, src[i % N]);
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xffffffff, local_min, offset);
        local_min = min(local_min, other);
    }

    // First thread of each warp does the atomic min
    if (tid % 32 == 0) {
        atomicMin(result, local_min);
    }
}

// =============================================================================
// B.2 Block-Level Atomic Operations
// =============================================================================

// Block-level reduction then single atomic per block
__global__ void atomicBlockLevelAdd(const float* __restrict__ src,
                                     float* __restrict__ result, size_t N) {
    __shared__ float shared_sum[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    // Each thread accumulates its portion
    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < 32) {
        float warp_sum = (tid < blockDim.x) ? shared_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (tid == 0) {
            atomicAdd(result, warp_sum);
        }
    }
}

// Block-level reduction with multiple atomics (for comparison)
__global__ void atomicBlockLevelMultiAdd(const float* __restrict__ src,
                                          float* __restrict__ result, size_t N) {
    __shared__ float shared_sum[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    // Each thread accumulates its portion
    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Every thread does an atomic (high contention)
    if (tid == 0) {
        atomicAdd(&result[blockIdx.x], shared_sum[0]);
    }
}

// =============================================================================
// B.3 Grid-Level Atomic Operations (High Contention)
// =============================================================================

// All threads directly atomic add to same location (maximum contention)
__global__ void atomicGridDirectAdd(const float* __restrict__ src,
                                     float* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        atomicAdd(result, src[i]);
    }
}

// Grid-level atomic with thread-level reduction
__global__ void atomicGridReduction(const float* __restrict__ src,
                                    float* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    __shared__ float shared[256];

    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    // Accumulate in register
    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    shared[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Single atomic per block
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

// =============================================================================
// B.4 Atomic Operation Comparison (Add vs CAS vs Min/Max)
// =============================================================================

// Atomic Add
__global__ void atomicOperationAdd(const float* __restrict__ src,
                                    float* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        atomicAdd(result, src[i]);
    }
}

// Atomic Compare-And-Swap (CAS) based addition
__global__ void atomicOperationCAS(const float* __restrict__ src,
                                    float* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        float old_val = result[0];
        float assumed;
        do {
            assumed = old_val;
            old_val = atomicCAS((unsigned int*)&result[0],
                                __float_as_uint(assumed),
                                __float_as_uint(assumed + src[i]));
        } while (old_val != assumed);
    }
}

// Atomic Min
__global__ void atomicOperationMin(const int* __restrict__ src,
                                    int* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        atomicMin(result, src[i]);
    }
}

// Atomic Max
__global__ void atomicOperationMax(const int* __restrict__ src,
                                    int* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        atomicMax(result, src[i]);
    }
}

// =============================================================================
// B.5 Atomic with Different Data Types
// =============================================================================

// 64-bit atomic add (for double or long long)
__global__ void atomic64Add(const double* __restrict__ src,
                              double* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        atomicAdd(result, src[i]);
    }
}

// 32-bit unsigned int atomic add
__global__ void atomic32Add(const unsigned int* __restrict__ src,
                              unsigned int* __restrict__ result, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        atomicAdd(result, src[i]);
    }
}

// =============================================================================
// Helper: Block-level reduction without atomics (for baseline)
// =============================================================================

__global__ void noAtomicBaseline(const float* __restrict__ src,
                                  float* __restrict__ result, size_t N) {
    __shared__ float shared[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    // Accumulate
    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    shared[tid] = sum;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Store per-block result
    if (tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}
