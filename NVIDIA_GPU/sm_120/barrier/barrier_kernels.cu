#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    cuda::thread_block block = this_thread_block();
    cuda::grid_sync(grid_block, cuda::grid_scope);

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
__global__ void barRedPopcKernel(const int* __restrict__ pred,
                                  unsigned int* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    unsigned int count = 0;

    for (size_t i = tid; i < N; i += blockDim.x) {
        count += (pred[i] != 0) ? 1 : 0;
    }

    // Use bar.red.popc to reduce across CTA
    unsigned int total;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %1, 0;\n\t"
        "bar.red.popc.u32 %0, %2, p;\n\t"
        "}"
        : "=r"(total)
        : "r"(count), "r"(blockDim.x)
        : "p");

    if (tid == 0) {
        result[blockIdx.x] = total;
    }
}

// bar.red.and: All threads must have predicate true
__global__ void barRedAndKernel(const int* __restrict__ pred,
                                 int* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    int val = 1;  // Assume true

    for (size_t i = tid; i < N; i += blockDim.x) {
        val = val && (pred[i] != 0);
    }

    // Use bar.red.and to check all threads agree
    int all_true;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %1, 0;\n\t"
        "bar.red.and.pred %0, %2, p;\n\t"
        "}"
        : "=r"(all_true)
        : "r"(val), "r"(blockDim.x)
        : "p");

    if (tid == 0) {
        result[blockIdx.x] = all_true;
    }
}

// bar.red.or: At least one thread has predicate true
__global__ void barRedOrKernel(const int* __restrict__ pred,
                                int* __restrict__ result, size_t N) {
    size_t tid = threadIdx.x;
    int val = 0;  // Assume false

    for (size_t i = tid; i < N; i += blockDim.x) {
        val = val || (pred[i] != 0);
    }

    // Use bar.red.or to check any thread is true
    int any_true;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %1, 0;\n\t"
        "bar.red.or.pred %0, %2, p;\n\t"
        "}"
        : "=r"(any_true)
        : "r"(val), "r"(blockDim.x)
        : "p");

    if (tid == 0) {
        result[blockIdx.x] = any_true;
    }
}

// =============================================================================
// B.8 bar.arrive vs bar.sync vs bar.wait
// =============================================================================

// Pattern 1: Traditional bar.sync (blocking)
__global__ void barSyncBlockingKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx];
    }
    // Blocking sync - all threads wait
    asm volatile("bar.sync 0, %0;" : : "r"(blockDim.x));

    if (idx < N) {
        data[idx] = shared[tid] * 2.0f + 1.0f;
    }
}

// Pattern 2: bar.arrive (non-blocking) + bar.wait
__global__ void barArriveWaitKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    __shared__ int barrier_id;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        barrier_id = 0;
    }
    __syncthreads();

    if (idx < N) {
        shared[tid] = data[idx];
    }
    // Non-blocking arrive - threads don't wait
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(blockDim.x));

    // ... do other work here (simulated)
    for (int i = 0; i < 4; i++) {
        if (idx < N) {
            shared[tid] = shared[tid] * 1.01f;
        }
    }

    // Then wait for all to arrive
    asm volatile("bar.wait %0, %1;" : : "r"(barrier_id), "r"(blockDim.x));

    if (idx < N) {
        data[idx] = shared[tid] * 2.0f + 1.0f;
    }
}

// Pattern 3: Two-barrier producer-consumer
__global__ void producerConsumerKernel(float* __restrict__ data,
                                       size_t N, int phase) {
    __shared__ float shared[256];
    __shared__ int produce_barrier;
    __shared__ int consume_barrier;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        produce_barrier = 0;
        consume_barrier = 1;
    }
    __syncthreads();

    if (phase == 0) {
        // Producer phase
        if (idx < N) {
            shared[tid] = data[idx] * 2.0f;
        }
        asm volatile("bar.arrive %0, %1;" : : "r"(produce_barrier), "r"(blockDim.x));
    } else {
        // Consumer phase
        asm volatile("bar.wait %0, %1;" : : "r"(consume_barrier), "r"(blockDim.x));

        if (idx < N) {
            data[idx] = shared[tid] + 1.0f;
        }
        asm volatile("bar.arrive %0, %1;" : : "r"(consume_barrier), "r"(blockDim.x));
    }
}

// =============================================================================
// B.9 Named Barriers (SM90+)
// =============================================================================

// Note: Named barriers require SM90+ hardware
// Using inline PTX to access named barrier functionality
__global__ void namedBarrierKernel(float* __restrict__ data, size_t N) {
    __shared__ float shared[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx];
    }

    // Named barrier ID 8 (first available after reserved 0-7)
    // All threads in CTA must participate
    // bar.sync with named barrier
    asm volatile("bar.sync %0, %1;" : : "r"(8), "r"(blockDim.x));

    if (idx < N) {
        data[idx] = shared[tid] * 2.0f;
    }
}

// =============================================================================
// B.10 mbarrier (Memory Barrier) Operations
// =============================================================================

// mbarrier.init + mbarrier.test_wait
__global__ void mbarrierInitWaitKernel(uint64_t* __restrict__ mbarrier,
                                        float* __restrict__ data, size_t N) {
    __shared__ uint64_t barrier;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        // Init mbarrier with arrival count = blockDim.x
        uint32_t addr = (uint32_t)(uintptr_t)&barrier;
        asm volatile(
            "{\n\t"
            "mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
            "}"
            :
            : "r"(addr), "r"(blockDim.x)
            : "memory");
    }
    __syncthreads();

    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }

    __syncthreads();

    if (tid == 0) {
        uint32_t addr = (uint32_t)(uintptr_t)&barrier;
        // Wait for all threads to arrive
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT:\n\t"
            "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], 0;\n\t"
            "@P1 bra DONE;\n\t"
            "bra LAB_WAIT;\n\t"
            "DONE:\n\t"
            "}"
            :
            : "r"(addr)
            : "p", "memory");
    }
}

// mbarrier.arrive (non-blocking)
__global__ void mbarrierArriveKernel(uint64_t* __restrict__ mbarrier,
                                     float* __restrict__ data, size_t N) {
    __shared__ uint64_t barrier;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        uint32_t addr = (uint32_t)(uintptr_t)&barrier;
        asm volatile(
            "{\n\t"
            "mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
            "}"
            :
            : "r"(addr), "r"(blockDim.x)
            : "memory");
    }
    __syncthreads();

    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }

    // Arrive (non-blocking) - decrement arrive count
    if (tid == 0) {
        uint32_t addr = (uint32_t)(uintptr_t)&barrier;
        asm volatile(
            "{\n\t"
            "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
            "}"
            :
            : "r"(addr)
            : "memory");
    }

    // All threads wait
    __syncthreads();
}

// mbarrier.expect_tx (transaction count)
__global__ void mbarrierExpectTxKernel(uint64_t* __restrict__ mbarrier,
                                        float* __restrict__ data, size_t N) {
    __shared__ uint64_t barrier;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        uint32_t addr = (uint32_t)(uintptr_t)&barrier;
        // Init with transaction count expectation
        asm volatile(
            "{\n\t"
            "mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
            "mbarrier.expect_tx.shared::cta.b64 [%0], %2;\n\t"
            "}"
            :
            : "r"(addr), "r"(blockDim.x), "r"(256)  // Expect 256 bytes
            : "memory");
    }
    __syncthreads();

    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }

    __syncthreads();
}

// =============================================================================
// B.11 cp.async + mbarrier Pipeline
// =============================================================================

// Producer: async copy with mbarrier arrive
__global__ void cpAsyncProducerKernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       uint64_t* __restrict__ mbarrier,
                                       size_t N, size_t block_size) {
    __shared__ float smem[256];
    __shared__ uint64_t barrier;
    size_t tid = threadIdx.x;
    size_t block_start = blockIdx.x * block_size;

    if (tid == 0) {
        uint32_t addr = (uint32_t)(uintptr_t)&barrier;
        asm volatile(
            "{\n\t"
            "mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
            "}"
            :
            : "r"(addr), "r"(blockDim.x)
            : "memory");
    }
    __syncthreads();

    // cp.async commit group
    size_t idx = block_start + tid;

    // Async copy 16 bytes (4 floats)
    if (idx + 3 < N && idx < block_start + block_size) {
        asm volatile(
            "{\n\t"
            "cp.async.ca.shared::cta.b32 [%0], [%1], 16;\n\t"
            "}"
            :
            : "r"((uint32_t)(uintptr_t)(smem + tid)),
              "l"(src + idx)
            : "memory");
    }

    // Arrive on mbarrier
    if (tid == 0) {
        uint32_t addr = (uint32_t)(uintptr_t)&barrier;
        asm volatile(
            "{\n\t"
            "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
            "}"
            :
            : "r"(addr)
            : "memory");

        // Commit async copies
        asm volatile("cp.async.commit_group;");
    }

    __syncthreads();

    // Store to global
    if (idx + 3 < N && idx < block_start + block_size) {
        dst[idx] = smem[tid];
    }
}

// Consumer: wait for mbarrier before using data
__global__ void cpAsyncConsumerKernel(const float* __restrict__ src,
                                        float* __restrict__ dst,
                                        uint64_t* __restrict__ mbarrier,
                                        size_t N) {
    extern __shared__ uint64_t shared_barrier[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (tid < blockDim.x) {
        shared_barrier[tid] = mbarrier[blockIdx.x];
    }
    __syncthreads();

    // Wait for producer's async copies
    if (tid == 0) {
        uint32_t addr = (uint32_t)(uintptr_t)shared_barrier;
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT:\n\t"
            "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], 0;\n\t"
            "@P1 bra DONE;\n\t"
            "bra LAB_WAIT;\n\t"
            "DONE:\n\t"
            "}"
            :
            : "r"(addr)
            : "p", "memory");
    }

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
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Write to shared memory
        __shared__ float shared[256];
        shared[threadIdx.x] = data[idx] * 2.0f;
        // Memory fence - ensures ordering but NOT synchronization
        __threadfence();
        // This is NOT safe - other threads may not see the write!
    }
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
    __shared__ float shared[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        shared[tid] = data[idx];
    }
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && idx < N) {
        data[blockIdx.x] = shared[0];
    }
}
