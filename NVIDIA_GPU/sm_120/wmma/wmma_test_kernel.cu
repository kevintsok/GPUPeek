#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// =============================================================================
// WMMA Research Kernels - Based on PTX ISA WMMA Implementation
// =============================================================================
//
// Reference: PTX ISA Section 9.7.14 (Warp-Level Matrix Instructions)
//
// WMMA m16n16k16 operations require:
// - 8x .b32 registers for result/input storage
// - wmma.load.c.sync to load accumulator matrix C
// - wmma.mma.sync to execute matrix multiply-accumulate
// - wmma.store.d.sync to store result matrix D
//
// C++ API uses nvcuda::wmma namespace
//
// Supported data types and Shapes:
// - FP16: m16n16k16 (primary test)
// - BF16: m16n16k16 (requires cuda_bf16.h)
// - TF32: m16n8k4
// - FP32: m16n16k16
// - INT8: m16n16k16
// =============================================================================

using namespace nvcuda::wmma;

// =============================================================================
// WMMA FP16 Kernel (m16n16k16)
// =============================================================================

__global__ void wmma_fp16_test_kernel(const __half* a, const __half* b, float* d,
                                       int M, int N, int K) {
    // WMMA requires blockDim.x == 32 (warp size)
    if (threadIdx.x >= 32) return;

    // Each warp handles one 16x16 output block
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    // Calculate starting position for this warp's output block
    int row_start = block_row * 16;
    int col_start = block_col * 16;

    // Define fragment types - m16n16k16 shape
    fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_d;

    // Initialize accumulator to 0
    fill_fragment(frag_d, 0.0f);

    // Iterate through K dimension
    for (int k = 0; k < K; k += 16) {
        // Load A matrix fragment
        load_matrix_sync(frag_a, a + row_start * K + k, K);

        // Load B matrix fragment
        load_matrix_sync(frag_b, b + k * N + col_start, N);

        // Execute matrix multiply-accumulate
        mma_sync(frag_d, frag_a, frag_b, frag_d);
    }

    // Store result
    store_matrix_sync(d + row_start * N + col_start, frag_d, N, mem_row_major);
}

// =============================================================================
// WMMA FP16 Large Matrix Kernel
// =============================================================================

__global__ void wmma_fp16_large_kernel(const __half* a, const __half* b, float* d,
                                        int M, int N, int K) {
    if (threadIdx.x >= 32) return;

    // Each warp handles 16x16 output
    int warp_row = blockIdx.x * 2 + (threadIdx.y / 16);
    int warp_col = blockIdx.y * 2 + (threadIdx.y % 16);

    int row_start = warp_row * 16;
    int col_start = warp_col * 16;

    if (row_start >= M || col_start >= N) return;

    fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_d;

    fill_fragment(frag_d, 0.0f);

    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(frag_a, a + row_start * K + k, K);
        load_matrix_sync(frag_b, b + k * N + col_start, N);
        mma_sync(frag_d, frag_a, frag_b, frag_d);
    }

    store_matrix_sync(d + row_start * N + col_start, frag_d, N, mem_row_major);
}
