#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

// =============================================================================
// C.1 __syncthreads() Overhead Measurement
// =============================================================================

// Minimal kernel with no sync (baseline)
__global__ void noSyncKernel(float* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

// Kernel with single __syncthreads() per iteration
__global__ void singleSyncKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared_buf[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared_buf[tid] = data[idx];
    }
    __syncthreads();

    if (idx < N) {
        data[idx] = shared_buf[tid] * 2.0f + 1.0f;
    }
}

// Kernel with multiple __syncthreads() per iteration
__global__ void multiSyncKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared_buf[256];
    __shared__ float shared_buf2[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared_buf[tid] = data[idx];
    }
    __syncthreads();

    if (idx < N) {
        shared_buf2[tid] = shared_buf[tid] * 2.0f;
    }
    __syncthreads();

    if (idx < N) {
        data[idx] = shared_buf2[tid] + 1.0f;
    }
}

// =============================================================================
// C.2 Barrier Stall Analysis Kernels
// =============================================================================

// Kernel where all warps reach barrier together (no stall)
__global__ void barrierNoStallKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        float val = data[idx];
        // All threads do same work
        for (int i = 0; i < 8; i++) {
            val = val * 2.0f + 1.0f;
        }
        shared[tid] = val;
    }
    __syncthreads();

    if (idx < N) {
        data[idx] = shared[tid];
    }
}

// Kernel with divergent sync (some threads reach barrier first)
__global__ void barrierDivergentKernel(float* __restrict__ data, size_t N,
                                        const int* __restrict__ pred) {
    __shared__ float shared[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        float val = data[idx];
        if (pred[tid % 32] == 1) {
            // Some threads do more work before sync
            for (int i = 0; i < 16; i++) {
                val = val * 2.0f + 1.0f;
            }
        }
        shared[tid] = val;
    }
    __syncthreads();

    if (idx < N) {
        data[idx] = shared[tid];
    }
}

// Kernel with dependent loads before sync (causes stall)
__global__ void barrierDependentLoadKernel(float* __restrict__ src,
                                            float* __restrict__ dst, size_t N) {
    __shared__ float shared[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        // Dependent load pattern - next thread depends on previous
        float val = src[idx];
        if (tid > 0) {
            val = val + shared[tid - 1];
        }
        shared[tid] = val;
    }
    __syncthreads();

    if (idx < N) {
        dst[idx] = shared[tid];
    }
}

// =============================================================================
// C.3 Block Size vs Barrier Efficiency
// =============================================================================

// Test kernel with __syncthreads() at different block sizes
template <int BlockSize>
__global__ void blockSizeBarrierKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[BlockSize];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * BlockSize + tid;

    if (idx < N) {
        float val = data[idx];
        shared[tid] = val * 2.0f;
    }
    __syncthreads();

    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < BlockSize; i++) {
            sum += shared[i];
        }
        data[idx] = sum / (float)BlockSize;
    }
}

// =============================================================================
// C.4 Multi-Block Synchronization (Flag-Based)
// =============================================================================

// Grid-level flag synchronization (INEFFICIENT - spin wait)
__global__ void gridFlagSyncKernel(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int* __restrict__ flag, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Phase 1: Compute
    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }

    // Phase 2: Spin-wait for other blocks (INEFFICIENT)
    if (blockIdx.x == 0) {
        // Block 0 waits for all other blocks to finish
        __threadfence();
        atomicAdd(&flag[0], 1);

        // Spin wait
        while (atomicAdd(&flag[0], 0) < gridDim.x) {
            // Spin
        }

        // All blocks done
        if (threadIdx.x == 0) {
            dst[0] = sum;
        }
    } else {
        // Other blocks signal completion and exit
        __threadfence();
        atomicAdd(&flag[0], 1);

        // Exit without writing (inefficient pattern)
        if (idx < N) {
            dst[idx] = 0;
        }
    }
}

// Proper pattern: No inter-block sync needed (just use grid sync)
__global__ void noGridSyncKernel(const float* __restrict__ src,
                                   float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }

    // Per-block result (no inter-block sync needed)
    if (idx < N) {
        dst[idx] = sum;
    }
}

// =============================================================================
// C.5 Warp-Level Synchronization (No Barrier Needed)
// =============================================================================

// Warp-level reduction using shuffle (no __syncthreads needed)
__global__ void warpShuffleReductionKernel(const float* __restrict__ src,
                                            float* __restrict__ dst, size_t N) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        val += src[i];
    }

    // Warp-level reduction using shuffle (no barrier needed within warp)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Store warp result
    if (tid % 32 == 0) {
        dst[blockIdx.x * (blockDim.x / 32) + tid / 32] = val;
    }
}

// Warp-level vote and ballot operations
__global__ void warpVoteBallotKernel(const int* __restrict__ pred,
                                      int* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        int val = pred[idx];

        // Warp-level any() - any thread in warp has non-zero?
        int any_result = __any_sync(0xffffffff, val);

        // Warp-level all() - all threads in warp have non-zero?
        int all_result = __all_sync(0xffffffff, val);

        // Warp-level ballot() - which threads have non-zero?
        unsigned int ballot_result = __ballot_sync(0xffffffff, val);

        // Store results
        if (tid % 32 == 0) {
            result[idx / 32 * 3 + 0] = any_result;
            result[idx / 32 * 3 + 1] = all_result;
            result[idx / 32 * 3 + 2] = (int)ballot_result;
        }
    }
}

// =============================================================================
// C.6 CTA (Cooperative Thread Array) Synchronization
// =============================================================================

// Cooperative grid sync using cuda::thread_block::sync()
__global__ void cooperativeGridSyncKernel(float* __restrict__ data, size_t N) {
    // Using CTA-level sync (grid sync via cooperative groups)
    namespace cg = cooperative_groups;
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;

    // Phase 1
    if (blockIdx.x < N / blockDim.x) {
        size_t idx = blockIdx.x * blockDim.x + tid;
        if (idx < N) {
            ((float*)shared)[tid] = data[idx] * 2.0f;
        }
    }

    // Sync across grid (cooperative groups)
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // Phase 2
    if (blockIdx.x < N / blockDim.x) {
        size_t idx = blockIdx.x * blockDim.x + tid;
        if (idx < N) {
            data[idx] = ((float*)shared)[tid] + 1.0f;
        }
    }
}

// =============================================================================
// B.7 bar.red Reduction Barrier
// =============================================================================

// bar.red.popc: Count threads where predicate is true
// Using warp vote to simulate barrier reduction behavior
__global__ void barRedPopcKernel(const int* __restrict__ pred,
                                  unsigned int* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    unsigned int count = 0;

    for (size_t i = tid; i < N; i += blockDim.x) {
        count += (pred[i] != 0) ? 1 : 0;
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }

    // First thread of each warp reports
    if (tid % 32 == 0) {
        result[blockIdx.x * (blockDim.x / 32) + tid / 32] = count;
    }
}

// bar.red.and: All threads must have predicate true
// Using warp all() to check if all threads have predicate true
__global__ void barRedAndKernel(const int* __restrict__ pred,
                                 int* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    int val = 1;  // Assume true

    for (size_t i = tid; i < N; i += blockDim.x) {
        val = val && (pred[i] != 0);
    }

    // Warp-level AND reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = val && __shfl_down_sync(0xffffffff, val, offset);
    }

    // First thread of each warp reports
    if (tid % 32 == 0) {
        result[blockIdx.x * (blockDim.x / 32) + tid / 32] = val;
    }
}

// bar.red.or: At least one thread has predicate true
// Using warp any() to check if any thread has predicate true
__global__ void barRedOrKernel(const int* __restrict__ pred,
                                int* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    int val = 0;  // Assume false

    for (size_t i = tid; i < N; i += blockDim.x) {
        val = val || (pred[i] != 0);
    }

    // Warp-level OR reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = val || __shfl_down_sync(0xffffffff, val, offset);
    }

    // First thread of each warp reports
    if (tid % 32 == 0) {
        result[blockIdx.x * (blockDim.x / 32) + tid / 32] = val;
    }
}

// =============================================================================
// B.8 bar.arrive vs bar.sync vs bar.wait
// Note: These use __syncthreads() as baseline; true bar.arrive/wait requires
// special hardware support and are shown for documentation purposes
// =============================================================================

// Pattern 1: Traditional __syncthreads() (blocking)
__global__ void barSyncBlockingKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx];
    }
    // Blocking sync - all threads wait
    __syncthreads();

    if (idx < N) {
        data[idx] = shared[tid] * 2.0f + 1.0f;
    }
}

// Pattern 2: Two-phase barrier (simulate arrive+wait)
__global__ void barArriveWaitKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    __shared__ int arrive_count;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        arrive_count = 0;
    }
    __syncthreads();

    if (idx < N) {
        shared[tid] = data[idx];
    }

    // Phase 1: Arrive (signal arrival)
    __syncthreads();

    // Do other work (simulated)
    for (int i = 0; i < 4; i++) {
        if (idx < N) {
            shared[tid] = shared[tid] * 1.01f;
        }
    }

    // Phase 2: Wait for all arrivals
    __syncthreads();

    if (idx < N) {
        data[idx] = shared[tid] * 2.0f + 1.0f;
    }
}

// Pattern 3: Two-barrier producer-consumer simulation
__global__ void producerConsumerKernel(float* __restrict__ data,
                                       size_t N, int phase) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (phase == 0) {
        // Producer phase
        if (idx < N) {
            shared[tid] = data[idx] * 2.0f;
        }
        __syncthreads();
    } else {
        // Consumer phase
        __syncthreads();

        if (idx < N) {
            data[idx] = shared[tid] + 1.0f;
        }
        __syncthreads();
    }
}

// =============================================================================
// B.9 Named Barriers (SM90+)
// Note: Named barriers are documented but require special hardware support
// Using __syncthreads() as functional equivalent for baseline
// =============================================================================

__global__ void namedBarrierKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx];
    }

    // __syncthreads() equivalent to named barrier
    __syncthreads();

    if (idx < N) {
        data[idx] = shared[tid] * 2.0f;
    }
}

// =============================================================================
// B.10 mbarrier (Memory Barrier) Operations
// Note: mbarrier requires sm_80+ hardware, shown for documentation
// Using __threadfence as baseline
// =============================================================================

// Memory fence + sync (baseline for mbarrier operations)
__global__ void mbarrierInitWaitKernel(uint64_t* __restrict__ mbarrier,
                                        float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx] * 2.0f;
    }
    __syncthreads();

    if (idx < N) {
        data[idx] = shared[tid] + 1.0f;
    }
}

// __threadfence as memory barrier baseline
__global__ void mbarrierArriveKernel(uint64_t* __restrict__ mbarrier,
                                     float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx] * 2.0f;
    }
    __threadfence();

    if (idx < N) {
        data[idx] = shared[tid] + 1.0f;
    }
}

// mbarrier.expect_tx simulation
__global__ void mbarrierExpectTxKernel(uint64_t* __restrict__ mbarrier,
                                        float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx] * 2.0f;
    }
    __threadfence();

    if (idx < N) {
        data[idx] = shared[tid] + 1.0f;
    }
}

// =============================================================================
// B.11 cp.async + mbarrier Pipeline
// Note: cp.async requires sm_80+; using regular loads as baseline
// =============================================================================

// Producer with regular loads (baseline for cp.async)
__global__ void cpAsyncProducerKernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       uint64_t* __restrict__ mbarrier,
                                       size_t N, size_t block_size) {
    __shared__ float smem[256];
    size_t tid = threadIdx.x;
    size_t block_start = blockIdx.x * block_size;
    size_t idx = block_start + tid;

    // Regular load (baseline for cp.async)
    if (idx < N && idx < block_start + block_size) {
        smem[tid] = src[idx];
    }
    __syncthreads();

    // Store to global
    if (idx < N && idx < block_start + block_size) {
        dst[idx] = smem[tid];
    }
}

// Consumer waits for producer
__global__ void cpAsyncConsumerKernel(const float* __restrict__ src,
                                        float* __restrict__ dst,
                                        uint64_t* __restrict__ mbarrier,
                                        size_t N) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    __syncthreads();

    if (idx < N) {
        dst[idx] = src[idx] * 2.0f + 1.0f;
    }
}

// =============================================================================
// B.12 __threadfence vs __syncthreads
// =============================================================================

// Memory fence only (no sync)
__global__ void threadFenceOnlyKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    if (idx < N) {
        shared[tid] = data[idx] * 2.0f;
    }
    // Memory fence - ensures ordering but NOT synchronization
    __threadfence();
    // Note: This is UNSAFE for cross-thread communication without sync!
}

// __syncthreads() (both fence and sync)
__global__ void syncthreadsKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx] * 2.0f;
    }
    // Both memory fence AND synchronization
    __syncthreads();

    if (idx < N && tid > 0) {
        data[idx] = shared[tid] + shared[tid - 1];
    }
}

// __threadfence_block (CUDA 9+)
__global__ void threadFenceBlockKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx] * 2.0f;
    }
    // Memory fence for block only (cheaper than __syncthreads)
    __threadfence_block();

    if (idx < N && tid > 0) {
        data[idx] = shared[tid] + shared[tid - 1];
    }
}

// =============================================================================
// Helper: Simple barrier test kernel
// =============================================================================

__global__ void simpleBarrierTest(float* __restrict__ data, size_t N) {
    __shared__ float shared[1024];  // Max block size is 1024

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx];
    }
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx < N && idx + s < N) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && idx < N) {
        data[blockIdx.x] = shared[0];
    }
}
