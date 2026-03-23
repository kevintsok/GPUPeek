#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// D.1 Warp Specialization Basic (2-Warp Producer/Consumer)
// =============================================================================

// Warp specialization: First half of warp = producer, second half = consumer
template <typename T>
__global__ void warpSpecializationBasicKernel(const T* __restrict__ src,
                                              T* __restrict__ dst,
                                              size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t wid = tid / 32;  // Warp ID
    size_t lane = tid % 32;  // Lane within warp

    size_t idx = blockIdx.x * blockDim.x + tid;

    // Warp specialization: Different roles for different warps
    if (wid % 2 == 0) {
        // Even warp IDs: Producer - load data to shared memory
        if (lane < 16) {  // Only first half of each warp
            size_t shared_idx = wid * 16 + lane;
            if (shared_idx < 256 && idx < N) {
                shared_buf[shared_idx] = src[idx];
            }
        }
        // Sync within warp (implicit)
        // Sync across warps
        __syncthreads();

        // Consumer phase: all threads read
        if (idx < N) {
            T val = shared_buf[tid % 256];
            dst[idx] = val * 2.0f;
        }
    } else {
        // Odd warp IDs: Consumer - wait and process
        __syncthreads();

        if (idx < N) {
            T val = shared_buf[tid % 256];
            dst[idx] = val * 3.0f;
        }
    }
}

// =============================================================================
// D.2 TMA (Tensor Memory Accelerator) + Barrier Synchronization
// =============================================================================

#if __CUDA_ARCH__ >= 900

// TMA 1D copy with barrier
template <typename T>
__global__ void tmaBarrierCopyKernel(const T* __restrict__ src,
                                      T* __restrict__ dst,
                                      size_t N,
                                      int* __restrict__ barrier) {
    __shared__ T shared_buf[128];

    size_t tid = threadIdx.x;
    size_t block_size = blockDim.x;
    size_t block_start = blockIdx.x * block_size;

    // Phase 1: TMA-style async copy (using cp.async)
    if (block_start + tid < N) {
        // Using cp.async to load globally
        // cp.async is a warp-level operation
        if (lane < 8) {  // Cooperative load
            size_t src_idx = block_start + tid + lane * block_size;
            if (src_idx < N) {
                shared_buf[tid] = src[src_idx];
            }
        }
    }

    // Barrier sync
    __syncthreads();

    // Phase 2: Process from shared memory
    if (block_start + tid < N) {
        dst[block_start + tid] = shared_buf[tid] * 2.0f;
    }
}

#endif

// Alternative: Standard copy with explicit barriers
template <typename T>
__global__ void barrierCopyKernel(const T* __restrict__ src,
                                   T* __restrict__ dst,
                                   size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Phase 1: Load to shared
    if (idx < N) {
        shared_buf[tid] = src[idx];
    }
    __syncthreads();

    // Phase 2: Process from shared
    if (idx < N) {
        dst[idx] = shared_buf[tid] * 2.0f;
    }
}

// =============================================================================
// D.3 Multi-Stage Pipeline (Load/Compute/Store)
// =============================================================================

// 3-stage pipeline: Load -> Compute -> Store
template <typename T>
__global__ void pipelineKernel(const T* __restrict__ src,
                               T* __restrict__ dst,
                               size_t N) {
    __shared__ T load_buf[256];
    __shared__ T compute_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Stage 1: Load from global to shared (Load)
    if (idx < N) {
        load_buf[tid] = src[idx];
    }
    __syncthreads();

    // Stage 2: Compute (Compute)
    T result = 0;
    if (idx < N) {
        result = load_buf[tid] * 2.0f + 1.0f;
        // Simulate more compute
        for (int i = 0; i < 4; i++) {
            result = result * 0.5f + 1.0f;
        }
        compute_buf[tid] = result;
    }
    __syncthreads();

    // Stage 3: Store from shared to global (Store)
    if (idx < N) {
        dst[idx] = compute_buf[tid];
    }
}

// 3-stage pipeline with producer-consumer overlap
template <typename T>
__global__ void overlappedPipelineKernel(const T* __restrict__ src,
                                          T* __restrict__ dst,
                                          size_t N) {
    __shared__ T stage1[256];
    __shared__ T stage2[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t num_iterations = (N + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);

    for (size_t iter = 0; iter < num_iterations; iter++) {
        size_t base_idx = iter * gridDim.x * blockDim.x + idx;

        // Stage 1: Load
        if (base_idx < N) {
            stage1[tid] = src[base_idx];
        }
        __syncthreads();

        // Stage 2: Compute (overlaps with next load)
        if (base_idx < N) {
            stage2[tid] = stage1[tid] * 2.0f + 1.0f;
        }
        __syncthreads();

        // Stage 3: Store (overlaps with next compute)
        if (base_idx < N) {
            dst[base_idx] = stage2[tid];
        }
        __syncthreads();
    }
}

// =============================================================================
// D.4 Block Specialization (Half Block = Producer, Half = Consumer)
// =============================================================================

// Block specialization: First half of block threads = producer, second half = consumer
template <typename T>
__global__ void blockSpecializationKernel(const T* __restrict__ src,
                                           T* __restrict__ dst,
                                           size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t block_half = blockDim.x / 2;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Producer phase: First half of threads load data
    if (tid < block_half) {
        if (idx < N) {
            shared_buf[tid] = src[idx];
        }
    }
    __syncthreads();

    // Consumer phase: Second half of threads process data
    if (tid >= block_half) {
        if (idx < N) {
            T val = shared_buf[tid - block_half];
            dst[idx] = val * 2.0f;
        }
    }
}

// Block specialization with warp-level producer/consumer
template <typename T>
__global__ void warpBlockSpecializationKernel(const T* __restrict__ src,
                                               T* __restrict__ dst,
                                               size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t warp_in_block = tid / 32;
    size_t lane = tid % 32;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Warp-level specialization within block
    if (warp_in_block % 2 == 0) {
        // Producer warp: load data
        if (lane < 16) {
            size_t shared_idx = warp_in_block * 16 + lane;
            if (shared_idx < 256 && idx < N) {
                shared_buf[shared_idx] = src[idx];
            }
        }
    } else {
        // Consumer warp: wait for data
        __syncthreads();
        if (idx < N) {
            T val = shared_buf[(warp_in_block - 1) * 16 + lane];
            dst[idx] = val * 2.0f;
        }
    }
}

// =============================================================================
// D.5 Warp-Level Synchronization Primitives
// =============================================================================

// Warp-level mutex using atomic operations
__global__ void warpMutexKernel(int* __restrict__ mutex,
                                 int* __restrict__ result,
                                 size_t N) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Each warp tries to acquire the mutex
    if (idx < N) {
        // Try to acquire mutex
        int acquired = 0;
        while (atomicCAS(mutex, 0, 1) != 0) {
            // Spin - mutex is held
        }

        // Critical section
        result[idx] = idx;

        // Release mutex
        atomicExch(mutex, 0);
    }
}

// Warp-level barrier using shuffle
__global__ void warpBarrierShuffleKernel(float* __restrict__ data,
                                           size_t N) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    float val = data[idx];

    // Simulate warp barrier using shuffle
    // All threads in warp must reach this point
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        val = __shfl_sync(0xffffffff, val, tid % 32);
    }

    data[idx] = val;
}

// Warp-level reduction with barrier
__global__ void warpReduceWithBarrierKernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              size_t N) {
    __shared__ float warp_results[8];  // One per warp in block

    size_t tid = threadIdx.x;
    size_t wid = tid / 32;
    size_t lane = tid % 32;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Per-thread reduction in registers
    float val = 0.0f;
    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        val += src[i];
    }

    // Warp-level reduction (no explicit barrier needed)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Store warp result
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    // Block-level reduction
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            block_sum += warp_results[i];
        }
        atomicAdd(dst, block_sum);
    }
}

// Warp-level scan (prefix sum)
__global__ void warpScanKernel(const int* __restrict__ src,
                                int* __restrict__ dst,
                                size_t N) {
    size_t tid = threadIdx.x;
    size_t lane = tid % 32;

    int val = src[tid];

    // Warp-level scan using shuffle
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n;
        }
    }

    dst[tid] = val;
}

// =============================================================================
// D.6 TMA + Warp Specialization Combined
// =============================================================================

#if __CUDA_ARCH__ >= 900

template <typename T>
__global__ void tmaWarpSpecializationKernel(const T* __restrict__ src,
                                             T* __restrict__ dst,
                                             size_t N) {
    extern __shared__ T shared[];
    __shared__ int barrier;

    size_t tid = threadIdx.x;
    size_t wid = tid / 32;
    size_t lane = tid % 32;

    // Producer warps: Load data using TMA-style access
    if (wid % 2 == 0) {
        // Even warps are producers
        if (lane < 16) {
            size_t src_idx = blockIdx.x * blockDim.x + wid * 16 + lane;
            if (src_idx < N) {
                shared[wid * 16 + lane] = src[src_idx];
            }
        }
    } else {
        // Odd warps are consumers - wait for producers
        __syncthreads();

        if (blockIdx.x * blockDim.x + tid < N) {
            size_t prod_idx = (wid - 1) * 16 + lane;
            T val = shared[prod_idx];
            dst[blockIdx.x * blockDim.x + tid] = val * 2.0f;
        }
    }
}

#endif

// =============================================================================
// Helper: Simple producer-consumer template
// =============================================================================

template <typename T>
__global__ void simpleProducerConsumerKernel(const T* __restrict__ src,
                                             T* __restrict__ dst,
                                             size_t N) {
    __shared__ T shared[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Producer: Thread 0 in each block loads data
    if (tid == 0) {
        for (int i = 0; i < 256 && idx + i < N; i++) {
            shared[i] = src[idx + i];
        }
    }
    __syncthreads();

    // Consumer: All threads process
    if (idx < N) {
        dst[idx] = shared[tid % 256] * 2.0f;
    }
}
