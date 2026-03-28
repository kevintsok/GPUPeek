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

// =============================================================================
// Topic 5: Cache Line Size Effect Research Kernels
// =============================================================================
// Research question: How does access granularity affect bandwidth?
// CUDA cache line is 128B (32B x 4 segments), but L1 is 32B

// Aligned 32B access - single L1 cache line
template <typename T>
__global__ void cacheLine32BKernel(const T* __restrict__ src,
                                    T* __restrict__ dst, size_t N) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * (32 / sizeof(T));
    size_t stride = gridDim.x * blockDim.x * (32 / sizeof(T));

    for (size_t i = idx; i < N; i += stride) {
        // Read one full 32B cache line
        T val = src[i];
        dst[i] = val;
    }
}

// Aligned 64B access - two L1 cache lines
template <typename T>
__global__ void cacheLine64BKernel(const T* __restrict__ src,
                                    T* __restrict__ dst, size_t N) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * (64 / sizeof(T));
    size_t stride = gridDim.x * blockDim.x * (64 / sizeof(T));

    for (size_t i = idx; i < N; i += stride) {
        T val0 = src[i];
        T val1 = src[i + 1];
        dst[i] = val0;
        dst[i + 1] = val1;
    }
}

// Aligned 128B access - full cache segment
template <typename T>
__global__ void cacheLine128BKernel(const T* __restrict__ src,
                                     T* __restrict__ dst, size_t N) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * (128 / sizeof(T));
    size_t stride = gridDim.x * blockDim.x * (128 / sizeof(T));

    for (size_t i = idx; i < N; i += stride) {
        T v0 = src[i];
        T v1 = src[i + 1];
        T v2 = src[i + 2];
        T v3 = src[i + 3];
        dst[i] = v0;
        dst[i + 1] = v1;
        dst[i + 2] = v2;
        dst[i + 3] = v3;
    }
}

// Misaligned access - spanning cache lines
template <typename T>
__global__ void misalignedAccessKernel(const T* __restrict__ src,
                                        T* __restrict__ dst, size_t N,
                                        int offset_bytes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    const T* src_ptr = (const T*)((const char*)src + offset_bytes);

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src_ptr[i];
    }
}

// =============================================================================
// Topic 6: Read vs Write Asymmetry Research
// =============================================================================
// Research question: Is read bandwidth truly different from write bandwidth?

// Pure read bandwidth
template <typename T>
__global__ void pureReadKernel(const T* __restrict__ src,
                                T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    T sum = 0;
    for (size_t i = idx; i < N; i += stride) {
        sum += src[i];
    }
    if (idx < 32) {
        dst[0] = sum;  // Prevent optimization
    }
}

// Pure write bandwidth (without read)
template <typename T>
__global__ void pureWriteKernel(const T* __restrict__ src,
                                 T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = (T)1.0f;
    }
}

// Read-write dependency test (RAW hazard)
template <typename T>
__global__ void readAfterWriteKernel(T* __restrict__ data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        data[i] = data[i] * 2.0f;  // Read, modify, write (dependent)
    }
}

// Write-after-read (WAR hazard) - less common in GPU
template <typename T>
__global__ void writeAfterReadKernel(const T* __restrict__ src,
                                     T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T val = src[i];
        dst[i] = val + 1.0f;  // Read old, write new
    }
}

// =============================================================================
// Topic 7: Non-Temporal vs Cached Access Research
// =============================================================================
// Research question: When does non-temporal (write-combining) beat cached?

// Cached read (default)
template <typename T>
__global__ void cachedReadKernel(const T* __restrict__ src,
                                  T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T val = src[i];
        dst[i] = val * 1.0f;
    }
}

// Non-temporal read hint (compiler may honor this)
template <typename T>
__global__ void nonTemporalReadKernel(const T* __restrict__ src,
                                       T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        T val;
        asm volatile("ld.global.gc.L1::128 {%0, %1}, [%2];"
                     : "=f"(val), "=f"(val)
                     : "l"(&src[i]));
        dst[i] = val;
    }
}

// Write with write-combining hint
template <typename T>
__global__ void writeCombiningWriteKernel(const T* __restrict__ src,
                                          T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

// =============================================================================
// Topic 8: Memory Coalescing Effectiveness Research
// =============================================================================
// Research question: How does thread arrangement affect coalescing?

// Best case: Sequential threads access sequential addresses
template <typename T>
__global__ void coalescedAccessKernel(const T* __restrict__ src,
                                       T* __restrict__ dst, size_t N) {
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t block_size = blockDim.x;

    size_t idx = bid * block_size + tid;
    size_t stride = gridDim.x * block_size;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i] * 1.0f;
    }
}

// Worst case: Sequential threads access strided addresses
template <typename T>
__global__ void uncoalescedAccessKernel(const T* __restrict__ src,
                                         T* __restrict__ dst,
                                         size_t N, size_t stride) {
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t block_size = blockDim.x;

    size_t idx = bid * block_size + tid;

    for (size_t i = idx; i < N; i += gridDim.x * block_size) {
        size_t access_idx = i * stride;
        if (access_idx < N) {
            dst[access_idx] = src[access_idx] * 1.0f;
        }
    }
}

// Half-warp divergence test (odd/even threads access different patterns)
template <typename T>
__global__ void halfWarpDivergenceKernel(const T* __restrict__ src,
                                         T* __restrict__ dst, size_t N) {
    size_t tid = threadIdx.x & 0xF;  // Half-warp (16 threads)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        // Even half-warp threads: sequential, odd half-warp: strided
        size_t addr = (tid < 8) ? i : i + (i % 32) * 16;
        if (addr < N) {
            dst[addr] = src[addr] * 1.0f;
        }
    }
}

// =============================================================================
// Topic 9: Software Prefetch Effectiveness Research
// =============================================================================
// Research question: Does software prefetch help hide memory latency?

// Prefetch ahead kernel
template <typename T>
__global__ void prefetchReadKernel(const T* __restrict__ src,
                                    T* __restrict__ dst, size_t N,
                                    int prefetch_dist) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        size_t prefetch_idx = i + prefetch_dist;
        if (prefetch_idx < N) {
            // Software prefetch hint
            asm volatile("prefetch.global.L1 [%0];" : : "l"(&src[prefetch_idx]));
        }
        dst[i] = src[i] * 1.0f;
    }
}

// Double-buffer pipeline (producer-consumer pattern)
template <typename T>
__global__ void doubleBufferKernel(const T* __restrict__ src,
                                    T* __restrict__ dst, size_t N) {
    __shared__ T shared_buf[256];

    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;

    size_t elements_per_block = (N + gridDim.x - 1) / gridDim.x;
    size_t block_start = bid * elements_per_block;
    size_t block_end = min(block_start + elements_per_block, N);

    // Even blocks: load to shared, Odd blocks: store from shared
    if (bid % 2 == 0) {
        // Producer: load to shared
        for (size_t i = block_start + tid; i < block_end; i += blockDim.x) {
            shared_buf[tid] = src[i];
        }
        __syncthreads();
        // Consumer (same block): store from shared
        for (size_t i = block_start + tid; i < block_end; i += blockDim.x) {
            dst[i] = shared_buf[tid] * 2.0f;
        }
    } else {
        // Wait for previous producer
        __syncthreads();
        for (size_t i = block_start + tid; i < block_end; i += blockDim.x) {
            dst[i] = shared_buf[tid] * 2.0f;
        }
    }
}
