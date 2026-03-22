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
template <typename T>
__global__ void ldmatrix_fp16_kernel(const T* __restrict__ global,
                                      T* __restrict__ shared,
                                      size_t stride, size_t count) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int tile_id = blockIdx.x;
    const int tile_per_block = blockDim.x / 32;  // 1 tile per warp

    // Each warp loads one 8x8 tile using ldmatrix
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (tile_id * tile_per_block + warp_id < count) {
        const T* src = global + (tile_id * tile_per_block + warp_id) * 64;

        // Use ldmatrix to load 8x8 tile
        // ldmatrix loads 4 elements per thread (16 threads -> 64 elements)
        // For .x1 layout: loads column-major 8x8
        frag_a mat_a;

        // ldmatrix.sync.aligned.m8n8.x1
        // This loads a transposed 8x8 tile for MMA
        ldmatrix::load_matrix_sync(mat_a, src, stride);

        // Store to shared memory in MMA-friendly layout
        store_matrix_sync(shm + warp_id * 64, mat_a, 8, wmma::mem_row_major);
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

    // For m16n8k8, we need 2 warps per tile pair
    // Warp 0: load 16x8 A tile
    // Warp 1: load 8x8 B tile

    if (tile_id < count && tile_idx < 2) {
        const T* src = global + tile_id * 64 + tile_idx * 32;

        if (tile_idx == 0) {
            // Load A tile (16x8) - uses 2 warps
            frag_a mat_a;
            ldmatrix::load_matrix_sync(mat_a, src, stride);
            store_matrix_sync(shm + tile_id * 128, mat_a, 16, wmma::mem_row_major);
        } else {
            // Load B tile (8x8)
            frag_b mat_b;
            ldmatrix::load_matrix_sync(mat_b, src, stride);
            store_matrix_sync(shm + tile_id * 128 + 64, mat_b, 8, wmma::mem_row_major);
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
    const int tile_id = blockIdx.x;

    if (tid < 32 && tile_id < count) {
        const T* src = global + tile_id * 64;

        // ldmatrix.x1 - 1 tile (8x8)
        frag_a mat_a;
        ldmatrix::load_matrix_sync(mat_a, src, stride);
        store_matrix_sync(shm + tile_id * 64, mat_a, 8, wmma::mem_row_major);
    }
}

template <typename T>
__global__ void ldmatrix_layout_x2_kernel(const T* __restrict__ global,
                                           T* __restrict__ shared,
                                           size_t stride, size_t count) {
    extern __shared__ char shared_mem[];
    T* shm = (T*)shared_mem;

    const int tid = threadIdx.x;
    const int tile_id = blockIdx.x;

    if (tid < 32 && tile_id < count) {
        const T* src = global + tile_id * 128;

        // ldmatrix.x2 - 2 tiles (8x8x2)
        // Each thread loads 8 elements (2 per tile)
        frag_a mat_a[2];

        // First tile
        ldmatrix::load_matrix_sync(mat_a[0], src, stride);
        // Second tile
        ldmatrix::load_matrix_sync(mat_a[1], src + 64, stride);

        store_matrix_sync(shm + tile_id * 128, mat_a[0], 8, wmma::mem_row_major);
        store_matrix_sync(shm + tile_id * 128 + 64, mat_a[1], 8, wmma::mem_row_major);
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
    const int tile_id = blockIdx.x;

    if (tid < 32 && tile_id < count) {
        T* dst = global + tile_id * 64;

        // Load from shared and store to global using stmatrix
        frag_c mat_c;
        load_matrix_sync(mat_c, shm + tile_id * 64, 8, wmma::mem_row_major);

        // stmatrix.sync.aligned.m8n8.x1
        stmatrix::store_matrix_sync(dst, mat_c, stride);
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
    const int tile_id = blockIdx.x;

    if (tid < 32 && tile_id < count) {
        T* dst = global + tile_id * 64;

        frag_c mat_c;
        load_matrix_sync(mat_c, shm + tile_id * 64, 8, wmma::mem_row_major);
        stmatrix::store_matrix_sync(dst, mat_c, stride);
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
        size_t copy_size = min(elements_per_thread * sizeof(T), remaining * sizeof(T));

        // Use cp.async for cached copy
        // cp.async.ca/shared/global
        // Size must be 4, 8, or 16 bytes
        const char* src_ptr = (const char*)(src + offset);
        char* dst_ptr = (char*)(dst + offset);
        char* shm_ptr = shm + (tid * elements_per_thread);

        // Cooperative async copy
        // Each thread copies a portion using cp.async
        for (size_t i = 0; i < elements_per_thread && offset + i < size; i += 4) {
            // cp.async.ca for 16-byte copy
            // In real code, use inline PTX:
            // asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : : "r"(shm_ptr + i * sizeof(T)), "l"(src_ptr + i * sizeof(T)));

            shm[i] = src[offset + i];
        }

        // Commit group when all threads done
        // asm volatile("cp.async.commit_group;" : : );

        // Wait for completion
        // asm volatile("cp.async.wait_group 0;" : : );
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
template <typename T, typename Op>
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

// Full pipeline: Load A/B with ldmatrix -> MMA -> Store with stmatrix
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

    // LDMATRIX loads for A tile
    if (warp_id == 0 && block_row * BM + lane_id < M) {
        const T* src_a = A + block_row * BM * K + warp_id * 32;
        frag_a mat_a;
        ldmatrix::load_matrix_sync(mat_a, src_a, K);
        store_matrix_sync(sh_a + block_row * BM * K, mat_a, K, wmma::mem_row_major);
    }

    // LDMATRIX loads for B tile
    if (warp_id == 1 && block_col * BN + lane_id < N) {
        const T* src_b = B + warp_id * 32 * N + block_col * BN;
        frag_b mat_b;
        ldmatrix::load_matrix_sync(mat_b, src_b, N);
        store_matrix_sync(sh_b + block_col * BN, mat_b, N, wmma::mem_row_major);
    }

    __syncthreads();

    // Perform MMA
    if (tid < 64) {  // 2 warps per block
        frag_a mat_a;
        frag_b mat_b;
        frag_c mat_c;

        wmma::fill_fragment(mat_c, 0.0f);

        // Load from shared
        wmma::load_matrix_sync(mat_a, sh_a + block_row * BM * K, K);
        wmma::load_matrix_sync(mat_b, sh_b + block_col * BN, N);
        wmma::mma_sync(mat_c, mat_a, mat_b, mat_c);

        // Store to shared C
        wmma::store_matrix_sync(sh_c + block_row * BM * N, mat_c, N, wmma::mem_row_major);
    }

    __syncthreads();

    // STMATRIX store to global
    if (warp_id == 0 && block_row * BM + lane_id < M) {
        frag_c mat_c;
        wmma::load_matrix_sync(mat_c, sh_c + block_row * BM * N, N);
        stmatrix::store_matrix_sync(C + block_row * BM * N, mat_c, N);
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
