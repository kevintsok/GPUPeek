#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// =============================================================================
// MMA Research Kernels - WMMA (Warp-level Matrix Multiply-Accumulate)
// =============================================================================
//
// This file implements working WMMA kernels using the nvcuda::wmma API
// Only FP16 and BF16 kernels are currently working
// =============================================================================

using namespace nvcuda::wmma;

// =============================================================================
// WMMA FP16 Kernel (m16n16k16)
// =============================================================================

template <typename T>
__global__ void wmma_fp16_kernel(const T* __restrict__ a,
                                  const T* __restrict__ b,
                                  float* __restrict__ c,
                                  float* __restrict__ d,
                                  size_t M, size_t N, size_t K) {
    // WMMA fragment types for m16n16k16 with __half input and float accumulator
    using frag_a_t = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    using frag_b_t = fragment<matrix_b, 16, 16, 16, __half, col_major>;
    using frag_c_t = fragment<accumulator, 16, 16, 16, float>;

    const int BM = 16;
    const int BN = 16;
    const int BK = 16;

    extern __shared__ char shared_mem[];
    __half* sh_a = (__half*)shared_mem;
    __half* sh_b = (__half*)&shared_mem[BM * BK * sizeof(__half)];

    size_t block_row = blockIdx.x;
    size_t block_col = blockIdx.y;

    frag_a_t mat_a;
    frag_b_t mat_b;
    frag_c_t mat_c;

    fill_fragment(mat_c, 0.0f);

    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i++) {
            for (int j = 0; j < BK; j++) {
                size_t gaddr_a = (block_row * BM + i) * K + (k + j);
                size_t saddr_a = i * BK + j;
                sh_a[saddr_a] = __float2half(a[gaddr_a]);
            }
        }
        for (int i = 0; i < BN; i++) {
            for (int j = 0; j < BK; j++) {
                size_t gaddr_b = (k + i) * N + (block_col * BN + j);
                size_t saddr_b = i * BK + j;
                sh_b[saddr_b] = __float2half(b[gaddr_b]);
            }
        }
        __syncthreads();

        load_matrix_sync(mat_a, sh_a, BK);
        load_matrix_sync(mat_b, sh_b, BK);
        mma_sync(mat_c, mat_a, mat_b, mat_c);

        __syncthreads();
    }

    store_matrix_sync(d, mat_c, N, mem_row_major);
}

// =============================================================================
// WMMA BF16 Kernel (m16n16k16)
// =============================================================================

template <typename T>
__global__ void wmma_bf16_kernel(const T* __restrict__ a,
                                   const T* __restrict__ b,
                                   float* __restrict__ c,
                                   float* __restrict__ d,
                                   size_t M, size_t N, size_t K) {
    // WMMA fragment types for BF16
    using frag_a_t = fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>;
    using frag_b_t = fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major>;
    using frag_c_t = fragment<accumulator, 16, 16, 16, float>;

    const int BM = 16;
    const int BN = 16;
    const int BK = 16;

    extern __shared__ char shared_mem[];
    __nv_bfloat16* sh_a = (__nv_bfloat16*)shared_mem;
    __nv_bfloat16* sh_b = (__nv_bfloat16*)&shared_mem[BM * BK * sizeof(__nv_bfloat16)];

    size_t block_row = blockIdx.x;
    size_t block_col = blockIdx.y;

    frag_a_t mat_a;
    frag_b_t mat_b;
    frag_c_t mat_c;

    fill_fragment(mat_c, 0.0f);

    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i++) {
            for (int j = 0; j < BK; j++) {
                size_t gaddr_a = (block_row * BM + i) * K + (k + j);
                size_t saddr_a = i * BK + j;
                sh_a[saddr_a] = __float2bfloat16(a[gaddr_a]);
            }
        }
        for (int i = 0; i < BN; i++) {
            for (int j = 0; j < BK; j++) {
                size_t gaddr_b = (k + i) * N + (block_col * BN + j);
                size_t saddr_b = i * BK + j;
                sh_b[saddr_b] = __float2bfloat16(b[gaddr_b]);
            }
        }
        __syncthreads();

        load_matrix_sync(mat_a, sh_a, BK);
        load_matrix_sync(mat_b, sh_b, BK);
        mma_sync(mat_c, mat_a, mat_b, mat_c);

        __syncthreads();
    }

    store_matrix_sync(d, mat_c, N, mem_row_major);
}

// =============================================================================
// Simple GEMM Kernel (Non-WMMA baseline for comparison)
// =============================================================================

template <typename T>
__global__ void simpleGemmKernel(const T* __restrict__ a,
                                   const T* __restrict__ b,
                                   T* __restrict__ c,
                                   size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum += (float)a[row * K + k] * (float)b[k * N + col];
        }
        c[row * N + col] = (T)sum;
    }
}
