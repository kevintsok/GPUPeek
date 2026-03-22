#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Redux.sync Research Kernels
// =============================================================================
//
// PTX ISA: Section 9.7.12 - Redux Operations
//
// redux.sync performs warp-level reduction operations.
// Supported operations: ADD, MIN, MAX, AND, OR, XOR
//
// Syntax: redux.sync [dest], [src], op
// Where op: .add, .min, .max, .and, .or, .xor
//
// SASS equivalent: RRED (reduction) instruction
//
// Key advantage over shuffle-based reduction:
// - Single instruction instead of multiple shuffles
// - Hardware-accelerated
// - Lower latency
// =============================================================================

// =============================================================================
// Redux.sync Basic Operations
// =============================================================================

// Redux.sync ADD - warp-level addition reduction
template <typename T>
__global__ void reduxAddKernel(const T* __restrict__ input,
                              T* __restrict__ output,
                              size_t N) {
    int tid = threadIdx.x;
    int wid = threadIdx.x / 32;

    // Each warp processes 32 elements
    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    // Initialize with first element in warp
    T val = input[warp_start];

    // Sequential within warp (for demonstration - redux does this in hw)
    for (int i = warp_start + 1; i < warp_end; i++) {
        val = val + input[i];
    }

    // Store partial result
    if (tid % 32 == 0) {
        output[wid] = val;
    }
}

// Redux.sync MIN - warp-level minimum reduction
template <typename T>
__global__ void reduxMinKernel(const T* __restrict__ input,
                              T* __restrict__ output,
                              size_t N) {
    int tid = threadIdx.x;
    int wid = threadIdx.x / 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = input[warp_start];

    for (int i = warp_start + 1; i < warp_end; i++) {
        val = min(val, input[i]);
    }

    if (tid % 32 == 0) {
        output[wid] = val;
    }
}

// Redux.sync MAX - warp-level maximum reduction
template <typename T>
__global__ void reduxMaxKernel(const T* __restrict__ input,
                              T* __restrict__ output,
                              size_t N) {
    int tid = threadIdx.x;
    int wid = threadIdx.x / 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = input[warp_start];

    for (int i = warp_start + 1; i < warp_end; i++) {
        val = max(val, input[i]);
    }

    if (tid % 32 == 0) {
        output[wid] = val;
    }
}

// Redux.sync AND - warp-level AND reduction
template <typename T>
__global__ void reduxAndKernel(const unsigned int* __restrict__ input,
                              unsigned int* __restrict__ output,
                              size_t N) {
    int tid = threadIdx.x;
    int wid = threadIdx.x / 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    unsigned int val = input[warp_start];

    for (int i = warp_start + 1; i < warp_end; i++) {
        val = val & input[i];
    }

    if (tid % 32 == 0) {
        output[wid] = val;
    }
}

// Redux.sync OR - warp-level OR reduction
template <typename T>
__global__ void reduxOrKernel(const unsigned int* __restrict__ input,
                             unsigned int* __restrict__ output,
                             size_t N) {
    int tid = threadIdx.x;
    int wid = threadIdx.x / 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    unsigned int val = input[warp_start];

    for (int i = warp_start + 1; i < warp_end; i++) {
        val = val | input[i];
    }

    if (tid % 32 == 0) {
        output[wid] = val;
    }
}

// Redux.sync XOR - warp-level XOR reduction
template <typename T>
__global__ void reduxXorKernel(const unsigned int* __restrict__ input,
                               unsigned int* __restrict__ output,
                               size_t N) {
    int tid = threadIdx.x;
    int wid = threadIdx.x / 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    unsigned int val = input[warp_start];

    for (int i = warp_start + 1; i < warp_end; i++) {
        val = val ^ input[i];
    }

    if (tid % 32 == 0) {
        output[wid] = val;
    }
}

// =============================================================================
// Redux.sync with Inline Assembly (Conceptual)
// =============================================================================

// This demonstrates the concept - actual redux.sync requires inline PTX
template <typename T>
__global__ void reduxConceptualKernel(const T* __restrict__ input,
                                      T* __restrict__ output,
                                      size_t N) {
    int tid = threadIdx.x;
    int wid = threadIdx.x / 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    // Load value
    T val = input[warp_start + (tid % 32)];

    // redux.sync would do:
    // asm volatile("redux.sync.add.s32 %0, %1;" : "=r"(val) : "r"(val));

    // For now, simulate with shuffle-based reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = val + other;
    }

    if (tid % 32 == 0) {
        output[wid] = val;
    }
}

// =============================================================================
// Redux.sync for Different Data Types
// =============================================================================

// Float redux.add
template <typename T>
__global__ void reduxFloatAddKernel(const T* __restrict__ input,
                                    T* __restrict__ output,
                                    size_t N) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = (lane < warp_end - warp_start) ? input[warp_start + lane] : 0;

    // Shuffle-based reduction (hardware redux would be single instruction)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val += other;
    }

    if (lane == 0) {
        output[wid] = val;
    }
}

// Float2 redux.add (two parallel reductions)
template <typename T>
__global__ void reduxFloat2AddKernel(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       size_t N) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    // Load two floats per thread
    T val0 = (lane < warp_end - warp_start) ? input[(warp_start + lane) * 2] : 0;
    T val1 = (lane < warp_end - warp_start) ? input[(warp_start + lane) * 2 + 1] : 0;

    // Reduction for val0
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other0 = __shfl_down_sync(0xffffffff, val0, offset);
        val0 += other0;
    }

    // Reduction for val1
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other1 = __shfl_down_sync(0xffffffff, val1, offset);
        val1 += other1;
    }

    if (lane == 0) {
        output[wid * 2] = val0;
        output[wid * 2 + 1] = val1;
    }
}

// =============================================================================
// Redux.sync Performance Comparison
// =============================================================================

// Shuffle-based reduction (baseline - what redux.sync replaces)
template <typename T>
__global__ void shuffleReductionKernel(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       size_t N) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = (lane < warp_end - warp_start) ? input[warp_start + lane] : 0;

    // Classic warp reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val += other;
    }

    if (lane == 0) {
        output[wid] = val;
    }
}

// Warp reduction using butterfly pattern
template <typename T>
__global__ void butterflyReductionKernel(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         size_t N) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = (lane < warp_end - warp_start) ? input[warp_start + lane] : 0;

    // Butterfly reduction pattern
    #pragma unroll
    for (int offset = 1; offset <= 16; offset <<= 1) {
        T other = __shfl_xor_sync(0xffffffff, val, offset);
        val += other;
    }

    if (lane == 0) {
        output[wid] = val;
    }
}

// =============================================================================
// Redux.sync for Atomic Operations
// =============================================================================

// Warp-level reduction then atomic (simulates redux + atomic)
template <typename T>
__global__ void reduxAtomicKernel(const T* __restrict__ input,
                                  T* __restrict__ global_sum,
                                  size_t N) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = (lane < warp_end - warp_start) ? input[warp_start + lane] : 0;

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val += other;
    }

    // Thread 0 in warp does atomic add
    if (lane == 0) {
        atomicAdd(global_sum, val);
    }
}

// =============================================================================
// Redux.sync for Block-Level Reduction
// =============================================================================

// Full block reduction using redux concept
template <typename T>
__global__ void blockReduceReduxKernel(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       size_t N) {
    __shared__ T shared[256];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    size_t idx = bid * blockDim.x + tid;

    // Load
    T val = (idx < N) ? input[idx] : 0;
    shared[tid] = val;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Warp reduction (would use redux.sync here)
    if (tid < 32) {
        val = shared[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            T other = __shfl_down_sync(0xffffffff, val, offset);
            val += other;
        }
        if (tid == 0) {
            output[bid] = val;
        }
    }
}

// =============================================================================
// Vote Operations (for comparison with redux)
// =============================================================================

// Warp vote - any (returns non-zero if any thread has condition)
template <typename T>
__global__ void warpVoteAnyKernel(const T* __restrict__ input,
                                  int* __restrict__ output,
                                  size_t N) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = input[warp_start + lane];
    int any_nonzero = (val != 0) ? 1 : 0;

    // Warp vote any
    int result = __any_sync(0xffffffff, any_nonzero);

    if (lane == 0) {
        output[wid] = result;
    }
}

// Warp vote - all (returns non-zero if all threads have condition)
template <typename T>
__global__ void warpVoteAllKernel(const T* __restrict__ input,
                                  int* __restrict__ output,
                                  size_t N) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = input[warp_start + lane];
    int nonzero = (val > 0) ? 1 : 0;

    // Warp vote all
    int result = __all_sync(0xffffffff, nonzero);

    if (lane == 0) {
        output[wid] = result;
    }
}

// =============================================================================
// Match Operations (PTX match.sync)
// =============================================================================

// Match sync - threads with matching values register
template <typename T>
__global__ void matchSyncKernel(const T* __restrict__ input,
                                unsigned int* __restrict__ matched,
                                unsigned int* __restrict__ count,
                                size_t N) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int warp_start = wid * 32;
    int warp_end = min(warp_start + 32, (int)N);

    if (warp_start >= (int)N) return;

    T val = input[warp_start + lane];

    // Count matching values in warp
    unsigned int match_mask = 0;
    for (int i = 0; i < 32; i++) {
        if (i < warp_end - warp_start) {
            T other = __shfl_sync(0xffffffff, val, i);
            if (val == other) {
                match_mask |= (1 << i);
            }
        }
    }

    if (lane == 0) {
        matched[wid] = match_mask;
        count[wid] = __popc(match_mask);
    }
}
