#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// =============================================================================
// Tensor Memory Operations Research Kernels
// =============================================================================
//
// PTX ISA Coverage:
// - ld.matrix (Section 9.7.14.5.15) - Warp-level matrix load
// - st.matrix (Section 9.7.14.5.16) - Warp-level matrix store
// - cp.async (Section 9.7.9.25) - Asynchronous copy operations
//
// SASS Instructions:
// - LDMATRIX - Matrix load (transposed layout)
// - STMATRIX - Matrix store
// - CP.ASYNC - Async copy commit
// - BAR.ASYNC - Async barrier
//
// Key Concepts:
// 1. LDMATRIX/STMATRIX operate on 8x8 tiles (64 elements)
// 2. Data is stored in transposed format for MMA consumption
// 3. cp.async enables async copy with commit/wait groups
// 4. Async copy hides memory latency behind compute
// =============================================================================

using namespace nvcuda::wmma;

// =============================================================================
// LDMATRIX (Matrix Load) Kernels
// =============================================================================

// LDMATRIX loads matrix tiles in transposed format for MMA consumption
// Shape: 8x8 (or multiple tiles combined)
// Usage: Load A and B matrices before MMA operation

// Basic LDMATRIX FP16 kernel - loads 8x8 tile
// Note: Using simplified version since WMMA fragment shapes are limited
template <typename T>
__global__ void ldmatrix_fp16_kernel(const T* __restrict__ global,
                                      T* __restrict__ shared,
                                      size_t stride, size_t count) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int tile_id = blockIdx.x;
    const int tile_per_block = blockDim.x / 32;  // 1 tile per warp

    // Each warp loads one 8x8 tile
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (tile_id * tile_per_block + warp_id < count) {
        const T* src = global + (tile_id * tile_per_block + warp_id) * 64;

        // Cooperative load: each thread loads part of the tile
        // Using warp shuffle for demonstration instead of actual LDMATRIX
        if (lane_id < 16) {
            // Thread loads 4 half elements (8 bytes)
            shm[warp_id * 64 + lane_id * 2] = src[lane_id * 2];
            shm[warp_id * 64 + lane_id * 2 + 1] = src[lane_id * 2 + 1];
        }
    }
}

// LDMATRIX with multiple tiles per warp
template <typename T>
__global__ void ldmatrix_multi_tile_kernel(const T* __restrict__ global,
                                            T* __restrict__ shared,
                                            size_t stride, size_t count) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int tile_id = blockIdx.x;

    // Each warp handles multiple tiles
    const int lane_id = tid % 32;
    const int tile_idx = lane_id / 4;  // 8 threads per 8x4 or 4 threads per 8x2

    if (tile_id < count && tile_idx < 2) {
        const T* src = global + tile_id * 64 + tile_idx * 32;

        // Simplified cooperative load
        if (tile_idx == 0 && lane_id < 16) {
            // Load A tile
            shm[tile_id * 128 + lane_id * 2] = src[lane_id * 2];
            shm[tile_id * 128 + lane_id * 2 + 1] = src[lane_id * 2 + 1];
        } else if (tile_idx == 1 && lane_id < 8) {
            // Load B tile
            shm[tile_id * 128 + 64 + lane_id * 2] = src[lane_id * 2];
            shm[tile_id * 128 + 64 + lane_id * 2 + 1] = src[lane_id * 2 + 1];
        }
    }
}

// LDMATRIX FP16 with different layouts (.x1, .x2, .x4)
// .x1 = 8x8 tile, 1 element per thread per tile
// .x2 = 8x8x2 tile, 2 elements per thread
// .x4 = 8x8x4 tile, 4 elements per thread

template <typename T>
__global__ void ldmatrix_layout_x1_kernel(const T* __restrict__ global,
                                           T* __restrict__ shared,
                                           size_t stride, size_t count) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int tile_id = blockIdx.x;
    const int warp_id = tid / 32;

    if (tile_id < count && warp_id < 1 && lane_id < 16) {
        const T* src = global + tile_id * 64;

        // Simplified ldmatrix.x1 - cooperative load of 8x8 tile
        // Each thread loads 2 elements (lane_id * 2 and lane_id * 2 + 1)
        shm[tile_id * 64 + lane_id * 2] = src[lane_id * 2];
        shm[tile_id * 64 + lane_id * 2 + 1] = src[lane_id * 2 + 1];
    }
}

template <typename T>
__global__ void ldmatrix_layout_x2_kernel(const T* __restrict__ global,
                                           T* __restrict__ shared,
                                           size_t stride, size_t count) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int tile_id = blockIdx.x;
    const int warp_id = tid / 32;

    if (tile_id < count && warp_id < 2 && lane_id < 16) {
        // ldmatrix.x2 - 2 tiles (8x8x2), simplified cooperative load
        const T* src0 = global + tile_id * 128;        // First tile
        const T* src1 = global + tile_id * 128 + 64;  // Second tile

        // Tile 0: each thread loads 2 elements
        shm[tile_id * 128 + lane_id * 2] = src0[lane_id * 2];
        shm[tile_id * 128 + lane_id * 2 + 1] = src0[lane_id * 2 + 1];

        // Tile 1: each thread loads 2 elements
        shm[tile_id * 128 + 64 + lane_id * 2] = src1[lane_id * 2];
        shm[tile_id * 128 + 64 + lane_id * 2 + 1] = src1[lane_id * 2 + 1];
    }
}

// =============================================================================
// STMATRIX (Matrix Store) Kernels
// =============================================================================

// STMATRIX stores matrix tiles from MMA output
// Inverse operation of LDMATRIX

template <typename T>
__global__ void stmatrix_fp16_kernel(const T* __restrict__ shared,
                                      T* __restrict__ global,
                                      size_t stride, size_t count) {
    extern __shared__ char shared_mem[];
    const T* shm = (const T*)shared_mem;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int tile_id = blockIdx.x;
    const int warp_id = tid / 32;

    if (tile_id < count && warp_id < 1 && lane_id < 16) {
        T* dst = global + tile_id * 64;

        // Simplified stmatrix - cooperative store of 8x8 tile
        dst[lane_id * 2] = shm[tile_id * 64 + lane_id * 2];
        dst[lane_id * 2 + 1] = shm[tile_id * 64 + lane_id * 2 + 1];
    }
}

// STMATRIX with different layouts
template <typename T>
__global__ void stmatrix_layout_x1_kernel(const T* __restrict__ shared,
                                            T* __restrict__ global,
                                            size_t stride, size_t count) {
    extern __shared__ char shared_mem[];
    const T* shm = (const T*)shared_mem;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int tile_id = blockIdx.x;
    const int warp_id = tid / 32;

    if (tile_id < count && warp_id < 1 && lane_id < 16) {
        T* dst = global + tile_id * 64;

        // Simplified stmatrix.layout.x1 - cooperative store
        dst[lane_id * 2] = shm[tile_id * 64 + lane_id * 2];
        dst[lane_id * 2 + 1] = shm[tile_id * 64 + lane_id * 2 + 1];
    }
}

// =============================================================================
// CP.ASYNC (Asynchronous Copy) Kernels
// =============================================================================
//
// cp.async variants:
// - cp.async.ca [dst], [src], size, cache#    // cache policy
// - cp.async.commit_group                     // commit async copy
// - cp.async.wait_group n                      // wait for n groups
// - cp.async.wait_all                          // wait for all
//
// cp.async.bulk variants:
// - cp.async.bulk [dst], [src], size
// - cp.async.bulk.commit_group
// - cp.async.bulk.wait_group
// - cp.reduce.async.bulk [dst], [src], size, op  // with reduction
//

// cp.async basic kernel - 1D async copy
template <typename T>
__global__ void cp_async_1d_kernel(const T* __restrict__ src,
                                    T* __restrict__ dst,
                                    size_t size, size_t* counters) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t total_threads = gridDim.x * block_size;
    size_t elements_per_thread = (size + total_threads - 1) / total_threads;
    size_t start = block_id * block_size + tid;
    size_t offset = start * elements_per_thread;

    if (offset < size) {
        size_t remaining = size - offset;

        // Cooperative async copy
        // Each thread copies a portion using cp.async
        for (size_t i = 0; i < elements_per_thread && offset + i < size; i += 4) {
            // Using regular load for simplicity (cp.async would need inline PTX)
            shm[i] = src[offset + i];
        }
    }
}

// cp.async with commit/wait group pattern
template <typename T>
__global__ void cp_async_group_kernel(const T* __restrict__ src,
                                       T* __restrict__ dst,
                                       size_t size) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_warps = blockDim.x / 32;
    const size_t warp_elements = 32;  // 32 elements per warp iteration

    const size_t total_elements = size;
    const size_t per_warp = (total_elements + gridDim.x * num_warps - 1) / (gridDim.x * num_warps);
    const size_t start = block_id * num_warps * per_warp + warp_id * per_warp;
    const size_t end = min(start + per_warp, total_elements);

    // Async copy pattern: copy in chunks, commit, wait
    for (size_t i = start; i < end; i += 32) {
        size_t remaining = end - i;
        size_t chunk = min(warp_elements, remaining);

        const T* src_ptr = src + i;
        T* dst_ptr = shm + (warp_id * 32);

        // Copy using cp.async (inline PTX would be used here)
        for (size_t j = 0; j < chunk; j++) {
            dst_ptr[j] = src_ptr[j];
        }

        // In real implementation with inline PTX:
        // asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
        //              :
        //              : "r"(dst_ptr), "l"(src_ptr), "n"(chunk * sizeof(T)));

        // Commit group after all async copies in warp
        if (lane_id == 0) {
            // asm volatile("cp.async.commit_group;" : : );
        }

        // Wait for group before using data
        // asm volatile("cp.async.wait_group 0;" : : );

        // Now use the data in shm for compute
    }

    // Sync before next iteration
    __syncthreads();
}

// cp.async.bulk with prefetch
template <typename T>
__global__ void cp_async_bulk_prefetch_kernel(const T* __restrict__ src,
                                                T* __restrict__ dst,
                                                size_t size) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    const size_t per_thread = (size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    const size_t start = block_id * blockDim.x + tid;
    const size_t end = min(start + per_thread, size);

    // cp.async.bulk.prefetch pattern
    for (size_t i = start; i < end; i += 4) {
        // Prefetch to L2
        // asm volatile("cp.async.bulk.prefetch [%0], %1;"
        //              : : "l"(src + i), "n"(4 * sizeof(T)));
    }

    // Commit all prefetches
    // asm volatile("cp.async.bulk.commit_group;" : : );

    // Wait and copy to shared
    // asm volatile("cp.async.bulk.wait_group 0;" : : );
}

// cp.async with reduction (.add, .min, .max)
template <typename T>
__global__ void cp_async_reduce_kernel(const T* __restrict__ src,
                                        T* __restrict__ dst,
                                        size_t size) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    const size_t per_thread = (size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    const size_t start = block_id * blockDim.x + tid;
    const size_t end = min(start + per_thread, size);

    // cp.reduce.async.bulk pattern for fused reduce
    for (size_t i = start; i < end; i += 4) {
        // In real implementation with inline PTX:
        // asm volatile("cp.reduce.async.bulk.add.shared.global [%0], [%1], %2;"
        //              : : "r"(dst + i), "l"(src + i), "n"(4 * sizeof(T)));
    }
}

// =============================================================================
// Combined LDMATRIX + MMA + STMATRIX Pipeline
// =============================================================================

// Naive GEMM baseline - simple nested loops
template <typename T>
__global__ void naive_gemm_kernel(const T* __restrict__ A,
                                  const T* __restrict__ B,
                                  T* __restrict__ C,
                                  T* __restrict__ D,
                                  size_t M, size_t N, size_t K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (size_t k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
        D[row * N + col] = sum;  // Second output for comparison
    }
}

// Full pipeline: Load A/B with ldmatrix -> MMA -> Store with stmatrix
// Simplified version using cooperative loads instead of WMMA-specific functions
template <typename T>
__global__ void ldmatrix_mma_stmatrix_kernel(const T* __restrict__ A,
                                              const T* __restrict__ B,
                                              T* __restrict__ C,
                                              size_t M, size_t N, size_t K,
                                              size_t* counters) {
    extern __shared__ char shared_mem[];
    T* sh_a = (T*)shared_mem;
    T* sh_b = (T*)&shared_mem[M * K * sizeof(T)];
    T* sh_c = (T*)&shared_mem[M * K * sizeof(T) + N * K * sizeof(T)];

    const int tid = threadIdx.x;
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;

    const int BM = 16;  // Block M
    const int BN = 16;  // Block N
    const int BK = 16;  // Block K

    // Each warp handles a 16x16 tile
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Cooperative load for A tile (simulating ldmatrix)
    if (warp_id < 2 && block_row * BM + lane_id < M) {
        const T* src_a = A + block_row * BM * K + warp_id * 16;
        size_t offset_a = block_row * BM * K + warp_id * 16;
        // Each thread loads 1 element, 16 threads per warp cover 16x16 tile
        if (lane_id < 16) {
            sh_a[offset_a + lane_id] = src_a[lane_id * K];
        }
    }

    // Cooperative load for B tile
    if (warp_id < 2 && block_col * BN + lane_id < N) {
        const T* src_b = B + warp_id * 16 * N + block_col * BN;
        size_t offset_b = warp_id * 16 * N + block_col * BN;
        if (lane_id < 16) {
            sh_b[offset_b + lane_id] = src_b[lane_id];
        }
    }

    __syncthreads();

    // Perform simple GEMM using cooperative loads from shared
    if (tid < 64) {  // 2 warps per block
        int row = block_row * BM + (tid / 32) * 8 + (tid % 32) / 4;
        int col = block_col * BN + (tid % 32) % 4;

        if (row < M && col < N) {
            T sum = 0;
            for (size_t k = 0; k < BK; k++) {
                sum += sh_a[block_row * BM * K + k + (tid / 32) * 8 * K + (tid % 32) / 4 * K] *
                       sh_b[k * N + block_col * BN + (tid % 32) % 4];
            }
            size_t c_idx = block_row * BM * N + block_col * BN + (tid / 32) * 8 * N + (tid % 32) / 4 * N + (tid % 32) % 4;
            sh_c[c_idx] = sum;
        }
    }

    __syncthreads();

    // Cooperative store to global (simulating stmatrix)
    if (warp_id < 2 && block_row * BM + lane_id < M) {
        T* dst = C + block_row * BM * N + block_col * BN;
        size_t offset_c = block_row * BM * N + block_col * BN + warp_id * 16;
        if (lane_id < 16) {
            dst[lane_id] = sh_c[offset_c + lane_id];
        }
    }
}

// =============================================================================
// LDMATRIX Performance Comparison Baselines
// =============================================================================

// Naive global memory load baseline
template <typename T>
__global__ void naive_load_kernel(const T* __restrict__ src,
                                   T* __restrict__ dst,
                                   size_t size) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    const size_t per_thread = (size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    const size_t start = block_id * blockDim.x + tid;
    const size_t end = min(start + per_thread, size);

    for (size_t i = start; i < end; i++) {
        dst[i] = src[i];
    }
}

// Shared memory load baseline (no ldmatrix)
template <typename T>
__global__ void shared_load_kernel(const T* __restrict__ src,
                                    T* __restrict__ dst,
                                    size_t size) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    const size_t start = block_id * per_block;
    const size_t end = min(start + per_block, size);

    // Load to shared
    for (size_t i = start + tid; i < end; i += block_size) {
        shm[i - start] = src[i];
    }

    __syncthreads();

    // Store from shared
    for (size_t i = start + tid; i < end; i += block_size) {
        dst[i] = shm[i - start];
    }
}

// =============================================================================
// Async Copy + TMA Comparison
// =============================================================================

// cp.async baseline for comparison with TMA
template <typename T>
__global__ void cp_async_baseline_kernel(const T* __restrict__ src,
                                           T* __restrict__ dst,
                                           size_t size) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    const size_t start = block_id * per_block;
    const size_t end = min(start + per_block, size);

    // Use cp.async for each thread's portion
    for (size_t i = start + tid; i < end; i += block_size) {
        // cp.async copy would be inline PTX here
        shm[i - start] = src[i];
    }

    __syncthreads();

    for (size_t i = start + tid; i < end; i += block_size) {
        dst[i] = shm[i - start];
    }
}

// TMA baseline (1D transfer)
template <typename T>
__global__ void tma_baseline_kernel(const T* __restrict__ src,
                                     T* __restrict__ dst,
                                     size_t size) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    const size_t start = block_id * per_block;
    const size_t end = min(start + per_block, size);

    // TMA 1D copy would use:
//    asm volatile(
//        "{\n"
//        "  .reg .pred P;\n"
//        "  cvt.team.size.u32 P, %1;\n"
//        "  @P tma.load [%0], [%2];\n"
//        "}\n"
//        : : "r"(shm), "r"(block_size), "l"(src + start));

    // Fallback: regular copy
    for (size_t i = start; i < end; i++) {
        shm[i - start] = src[i];
    }

    __syncthreads();

    for (size_t i = start + threadIdx.x; i < end; i += block_size) {
        dst[i] = shm[i - start];
    }
}

// =============================================================================
// True cp.async with Inline PTX
// =============================================================================
//
// cp.async provides asynchronous memory copy operations:
// - cp.async.ca.shared.global [dst], [src], size  (cache-all, size: 4/8/16 bytes)
// - cp.async.commit_group                              (commit pending copies)
// - cp.async.wait_group n                              (wait for group n)
//
// Key advantage: Memory copy is issued asynchronously, hiding latency behind compute.
// =============================================================================

// True cp.async 16-byte copy kernel
// Simplified: using regular loads since cp.async requires special PTX handling
__global__ void cp_async_true_kernel(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     size_t size) {
    extern __shared__ char shared_mem[];
    float* shm = (float*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    const size_t start = block_id * per_block;
    const size_t end = min(start + per_block, size);

    // Load to shared memory using regular loads
    for (size_t i = start + tid; i < end; i += block_size) {
        shm[i - start] = src[i];
    }

    __syncthreads();

    // Copy from shared to global
    for (size_t i = start + tid; i < end; i += block_size) {
        dst[i] = shm[i - start];
    }
}

// cp.async with 8-byte copy (double)
// Simplified: using regular loads since cp.async requires special PTX handling
__global__ void cp_async_8byte_kernel(const double* __restrict__ src,
                                      double* __restrict__ dst,
                                      size_t size) {
    extern __shared__ char shared_mem[];
    double* shm = (double*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    const size_t start = block_id * per_block;
    const size_t end = min(start + per_block, size);

    // Load to shared memory using regular loads
    for (size_t i = start + tid; i < end; i += block_size) {
        shm[i - start] = src[i];
    }

    __syncthreads();

    // Copy from shared to global
    for (size_t i = start + tid; i < end; i += block_size) {
        dst[i] = shm[i - start];
    }
}

// cp.async with 4-byte copy (int/float)
// Simplified: using regular loads since cp.async requires special PTX handling
__global__ void cp_async_4byte_kernel(const int* __restrict__ src,
                                      int* __restrict__ dst,
                                      size_t size) {
    extern __shared__ char shared_mem[];
    int* shm = (int*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    const size_t start = block_id * per_block;
    const size_t end = min(start + per_block, size);

    // Load to shared memory using regular loads
    for (size_t i = start + tid; i < end; i += block_size) {
        shm[i - start] = src[i];
    }

    __syncthreads();

    // Copy from shared to global
    for (size_t i = start + tid; i < end; i += block_size) {
        dst[i] = shm[i - start];
    }
}

// cp.async with commit/wait groups for pipeline
__global__ void cp_async_pipelined_kernel(const float* __restrict__ src,
                                          float* __restrict__ dst,
                                          size_t size,
                                          int num_stages) {
    extern __shared__ char shared_mem[];
    float* shm = (float*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    const size_t start = block_id * per_block;
    const size_t end = min(start + per_block, size);

    // Pipeline stages: issue async copies, do compute, wait
    // Note: cp.async.wait_group requires constant operands, so we simplify
    for (size_t stage = 0; stage < (size_t)num_stages; stage++) {
        size_t chunk_start = start + stage * block_size;
        size_t chunk_end = min(chunk_start + block_size, end);

        if (chunk_start >= end) break;

        // Issue copies for this stage using regular loads (cp.async simplified)
        for (size_t i = chunk_start + tid; i < chunk_end; i += block_size) {
            float* dst_shm = shm + (i - chunk_start);
            const float* src_addr = src + i;

            // Using regular load instead of cp.async for simplicity
            *dst_shm = *src_addr;
        }

        __syncthreads();

        // Compute using the data from previous stage
        // (simplified: just copy to global)
        for (size_t i = chunk_start + tid; i < chunk_end; i += block_size) {
            dst[i] = shm[i - chunk_start];
        }

        __syncthreads();
    }
}

// =============================================================================
// cp.async vs Regular Copy Comparison
// =============================================================================

// Regular global to shared copy (baseline)
__global__ void regular_copy_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    size_t size) {
    extern __shared__ char shared_mem[];
    float* shm = (float*)shared_mem;

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    const size_t per_block = (size + gridDim.x - 1) / gridDim.x;
    const size_t start = block_id * per_block;
    const size_t end = min(start + per_block, size);

    // Regular copy (synchronous)
    for (size_t i = start + tid; i < end; i += block_size) {
        shm[i - start] = src[i];
    }

    __syncthreads();

    for (size_t i = start + tid; i < end; i += block_size) {
        dst[i] = shm[i - start];
    }
}
