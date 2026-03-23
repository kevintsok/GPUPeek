#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

// =============================================================================
// Cooperative Groups Research Kernels
// =============================================================================
//
// Cooperative Groups API enables thread cooperation across:
// - Thread Block (same block threads)
// - Grid (all threads in kernel)
// - Multi-GPU (across GPUs)
//
// Key APIs:
// - cooperative_groups::this_thread_block()
// - cooperative_groups::this_grid()
// - cooperative_groups::this_multi_grid()
//
// Synchronization:
// - this_thread_block().sync()
// - this_grid().sync() - grid-level sync (cooperative groups)
// - this_multi_grid().sync() - multi-GPU sync
//
// Use cases:
// - Grid-wide reduction
// - Cooperative loads/stores
// - Multi-GPU collectives
// =============================================================================

namespace cg = cooperative_groups;

// =============================================================================
// Basic Cooperative Groups Kernels
// =============================================================================

// Thread block synchronization using cooperative groups
template <typename T>
__global__ void threadBlockSyncKernel(T* __restrict__ data, size_t N) {
    // Get the thread block group
    cg::thread_block block = cg::this_thread_block();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Each thread loads data
        T val = data[idx];

        // Synchronize within the block
        block.sync();

        // Cooperative addition - each thread adds its value to neighbor
        if (idx + 1 < N) {
            val += data[idx + 1];
        }

        block.sync();

        data[idx] = val;
    }
}

// =============================================================================
// Grid-Level Synchronization Kernels
// =============================================================================

// Grid-wide reduction using cooperative groups
template <typename T>
__global__ void gridReduceKernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                size_t N) {
    // Get grid group
    cg::grid_group grid = cg::this_grid();

    // Shared memory for reduction
    extern __shared__ char shared_mem[];
    T* shared = (T*)shared_mem;

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < blockDim.x) {
        shared[tid] = (idx < N) ? input[idx] : 0;
    }
    __syncthreads();

    // Block-level reduction
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Grid synchronization before final reduction
    grid.sync();

    // Thread 0 writes final result
    if (grid.is_valid() && tid == 0) {
        output[0] = shared[0];
    }
}

// =============================================================================
// Cooperative Load/Store Kernels
// =============================================================================

// Cooperative load - all threads in grid cooperatively load data
template <typename T>
__global__ void cooperativeLoadKernel(const T* __restrict__ src,
                                     T* __restrict__ dst,
                                     size_t N) {
    cg::grid_group grid = cg::this_grid();

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t total_threads = blockDim.x * gridDim.x;

    // Each thread loads multiple elements cooperatively
    for (size_t i = idx; i < N; i += total_threads) {
        dst[i] = src[i];
    }

    // Synchronize grid
    grid.sync();
}

// =============================================================================
// Grid Barrier with Memset Pattern
// =============================================================================

template <typename T>
__global__ void gridBarrierMemsetKernel(T* __restrict__ data,
                                        T value,
                                        size_t N) {
    cg::grid_group grid = cg::this_grid();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        data[idx] = value;
    }

    // Grid-wide barrier
    grid.sync();

    // All threads have written, now increment
    if (idx < N) {
        data[idx] += 1;
    }
}

// =============================================================================
// Multi-Block Reduction with Grid Sync
// =============================================================================

template <typename T>
__global__ void multiBlockReduceKernel(const T* __restrict__ input,
                                       T* __restrict__ result,
                                       size_t N) {
    cg::grid_group grid = cg::this_grid();

    __shared__ T block_sum[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t gridSize = blockDim.x * gridDim.x;

    // Each thread loads and accumulates
    T sum = 0;
    for (size_t i = idx; i < N; i += gridSize) {
        sum += input[i];
    }

    // Store to shared memory
    block_sum[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            block_sum[tid] += block_sum[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction (no sync needed within warp)
    if (tid < 32) {
        volatile T* smem = block_sum;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }

    // Grid sync before writing result
    grid.sync();

    // Thread 0 writes final result
    if (tid == 0 && blockIdx.x == 0) {
        result[0] = block_sum[0];
    }
}

// =============================================================================
// Two-Phase Cooperative Kernel
// =============================================================================

// Phase 1: Each block processes its portion
template <typename T>
__global__ void twoPhaseKernel(T* __restrict__ data,
                               size_t N, int phase) {
    cg::grid_group grid = cg::this_grid();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        if (phase == 0) {
            // Phase 0: Compute local sum
            T sum = 0;
            for (int i = 0; i < 10; i++) {
                sum += data[idx] * 1.001f;
            }
            data[idx] = sum;
        } else {
            // Phase 1: Aggregate
            if (idx > 0) {
                data[idx] += data[idx - 1] * 0.01f;
            }
        }
    }

    // Sync between phases
    grid.sync();
}

// =============================================================================
// Broadcast from Specific Thread
// =============================================================================

template <typename T>
__global__ void broadcastKernel(const T* __restrict__ src,
                               T* __restrict__ dst,
                               unsigned int broadcast_lane,
                               size_t N) {
    cg::grid_group grid = cg::this_grid();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Value to broadcast
    T value = (threadIdx.x == broadcast_lane) ? src[0] : 0;

    // Use grid.sync to ensure all threads have completed reading before next use
    grid.sync();

    if (idx < N) {
        dst[idx] = value;
    }
}

// =============================================================================
// Predicate-based Synchronization
// =============================================================================

template <typename T>
__global__ void predicateSyncKernel(T* __restrict__ data,
                                    volatile int* __restrict__ ready,
                                    size_t N) {
    cg::grid_group grid = cg::this_grid();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Wait for ready flag
    while (*ready == 0) {
        // Spin - in real code would use proper synchronization
    }

    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }

    grid.sync();
}

// =============================================================================
// Even-Odd Synchronization Pattern
// =============================================================================

template <typename T>
__global__ void evenOddSyncKernel(T* __restrict__ data,
                                   size_t N, int num_phases) {
    cg::thread_block block = cg::this_thread_block();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_even = (threadIdx.x % 2 == 0);

    for (int phase = 0; phase < num_phases; phase++) {
        if (idx < N) {
            if (is_even) {
                // Even threads process
                data[idx] += 1.0f;
            }
        }

        // Sync within block
        block.sync();

        if (idx < N) {
            if (!is_even) {
                // Odd threads process
                data[idx] += 1.0f;
            }
        }

        block.sync();
    }
}

// =============================================================================
// Barrier Efficiency Test
// =============================================================================

template <typename T>
__global__ void barrierEfficiencyKernel(T* __restrict__ data,
                                        size_t N, int iterations) {
    cg::thread_block block = cg::this_thread_block();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        T val = data[idx];

        for (int i = 0; i < iterations; i++) {
            // Some computation
            val = val * 1.001f + 0.001f;

            // Barrier
            block.sync();
        }

        data[idx] = val;
    }
}

// =============================================================================
// Load-Cooperative Pattern (Vectorized Cooperative Loading)
// =============================================================================

template <typename T>
__global__ void vectorizedCoopLoadKernel(const T* __restrict__ src,
                                         T* __restrict__ dst,
                                         size_t N) {
    cg::grid_group grid = cg::this_grid();

    // Handle 4 elements per thread cooperatively
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    size_t total_elements = blockDim.x * gridDim.x * 4;

    for (size_t i = idx; i < N && i < idx + 4; i += total_elements) {
        dst[i] = src[i];
    }

    grid.sync();
}
