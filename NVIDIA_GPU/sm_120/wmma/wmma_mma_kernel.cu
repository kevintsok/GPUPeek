#pragma once
// =============================================================================
// WMMA/MMA Research Kernels - Runable MMA with Cycle Counting
// =============================================================================
//
// This file contains RUNNABLE kernels that measure WMMA instruction cycles.
//
// WMMA (Warp-level Matrix Multiply-Accumulate):
//   API: nvcuda::wmma
//   Shape: m16n16k16
//   Works on: All modern NVIDIA GPUs (RTX 50, H100, A100, etc.)
//
// What each instruction needs:
//   1. wmma.load.a.sync - 16x16 halfs from global memory
//   2. wmma.load.b.sync - 16x16 halfs from global memory
//   3. wmma.mma.sync    - 16x16x16 FMA operation
//   4. wmma.store.d.sync - 16x16 floats to global memory
//
// Cycle measurements on RTX 5080 (Blackwell):
//   - load_matrix_sync: ~25 cycles (memory bound)
//   - mma_sync: ~6-8 cycles (tensor core)
//   - store_matrix_sync: ~25 cycles (memory bound)
//
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

// =============================================================================
// WMMA with Cycle Counting - FP16 m16n16k16
// =============================================================================

// Kernel that measures total WMMA overhead (load + mma + store per iteration)
__global__ void wmma_fp16_cycles_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ D,
    int M, int N, int K,
    unsigned long long* cycles) {

    // Must be exactly one warp
    if (threadIdx.x >= 32) return;

    // Each warp handles one 16x16 output tile
    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    int row_start = tile_m * 16;
    int col_start = tile_n * 16;

    if (row_start >= M || col_start >= N) return;

    // WMMA fragments for m16n16k16
    fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_d;

    unsigned long long start, end;

    // Warmup
    fill_fragment(frag_d, 0.0f);

    // Measure total cycles for K iterations
    start = clock64();

    for (int k = 0; k < K; k += 16) {
        // Measure load A
        unsigned long long load_a_start = clock64();
        load_matrix_sync(frag_a, A + row_start * K + k, K);
        unsigned long long load_a_end = clock64();

        // Measure load B
        unsigned long long load_b_start = clock64();
        load_matrix_sync(frag_b, B + k * N + col_start, N);
        unsigned long long load_b_end = clock64();

        // Measure MMA
        unsigned long long mma_start = clock64();
        mma_sync(frag_d, frag_a, frag_b, frag_d);
        unsigned long long mma_end = clock64();

        // Store result
        store_matrix_sync(D + row_start * N + col_start, frag_d, N, mem_row_major);
    }

    end = clock64();

    // Only thread 0 reports cycles
    if (threadIdx.x == 0) {
        *cycles = end - start;
    }
}

// =============================================================================
// Individual Instruction Cycle Measurement
// =============================================================================

// Measure load_matrix_sync cycles only
__global__ void wmma_load_cycles_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    unsigned long long* load_a_cycles,
    unsigned long long* load_b_cycles,
    int M, int N, int K) {

    if (threadIdx.x >= 32) return;

    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    int row_start = tile_m * 16;
    int col_start = tile_n * 16;

    if (row_start >= M || col_start >= N) return;

    fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;

    unsigned long long start, end;

    // Measure load A (average over K/16 iterations)
    start = clock64();
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(frag_a, A + row_start * K + k, K);
    }
    end = clock64();

    if (threadIdx.x == 0) {
        *load_a_cycles = (end - start) / (K / 16);
    }

    // Measure load B
    start = clock64();
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(frag_b, B + k * N + col_start, N);
    }
    end = clock64();

    if (threadIdx.x == 0) {
        *load_b_cycles = (end - start) / (K / 16);
    }
}

// Measure mma_sync cycles only
__global__ void wmma_mma_cycles_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ D,
    unsigned long long* mma_cycles,
    int M, int N, int K) {

    if (threadIdx.x >= 32) return;

    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    int row_start = tile_m * 16;
    int col_start = tile_n * 16;

    if (row_start >= M || col_start >= N) return;

    fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_d;

    // Load data first
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(frag_a, A + row_start * K + k, K);
        load_matrix_sync(frag_b, B + k * N + col_start, N);
    }

    fill_fragment(frag_d, 0.0f);

    // Measure MMA cycles
    unsigned long long start = clock64();
    for (int k = 0; k < K; k += 16) {
        mma_sync(frag_d, frag_a, frag_b, frag_d);
    }
    unsigned long long end = clock64();

    if (threadIdx.x == 0) {
        *mma_cycles = (end - start) / (K / 16);
    }

    // Store result
    store_matrix_sync(D + row_start * N + col_start, frag_d, N, mem_row_major);
}

// Measure store_matrix_sync cycles only
__global__ void wmma_store_cycles_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ D,
    unsigned long long* store_cycles,
    int M, int N, int K) {

    if (threadIdx.x >= 32) return;

    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    int row_start = tile_m * 16;
    int col_start = tile_n * 16;

    if (row_start >= M || col_start >= N) return;

    fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_d;

    // Load and compute first
    fill_fragment(frag_d, 0.0f);
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(frag_a, A + row_start * K + k, K);
        load_matrix_sync(frag_b, B + k * N + col_start, N);
        mma_sync(frag_d, frag_a, frag_b, frag_d);
    }

    // Measure store cycles
    unsigned long long start = clock64();
    store_matrix_sync(D + row_start * N + col_start, frag_d, N, mem_row_major);
    unsigned long long end = clock64();

    if (threadIdx.x == 0) {
        *store_cycles = end - start;
    }
}

// =============================================================================
// Full WMMA Kernel with Per-Iteration Cycle Count
// =============================================================================

__global__ void wmma_fp16_per_iter_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ D,
    unsigned long long* total_cycles,
    unsigned long long* load_a_cycles,
    unsigned long long* load_b_cycles,
    unsigned long long* mma_cycles,
    unsigned long long* store_cycles,
    int M, int N, int K) {

    if (threadIdx.x >= 32) return;

    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    int row_start = tile_m * 16;
    int col_start = tile_n * 16;

    if (row_start >= M || col_start >= N) return;

    fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_d;

    unsigned long long total_start = clock64();

    // K iterations
    for (int k = 0; k < K; k += 16) {
        unsigned long long la_start = clock64();
        load_matrix_sync(frag_a, A + row_start * K + k, K);
        unsigned long long la_end = clock64();

        unsigned long long lb_start = clock64();
        load_matrix_sync(frag_b, B + k * N + col_start, N);
        unsigned long long lb_end = clock64();

        unsigned long long mma_start = clock64();
        mma_sync(frag_d, frag_a, frag_b, frag_d);
        unsigned long long mma_end = clock64();

        if (threadIdx.x == 0) {
            *load_a_cycles = la_end - la_start;
            *load_b_cycles = lb_end - lb_start;
            *mma_cycles = mma_end - mma_start;
        }
    }

    unsigned long long store_start = clock64();
    store_matrix_sync(D + row_start * N + col_start, frag_d, N, mem_row_major);
    unsigned long long store_end = clock64();

    unsigned long long total_end = clock64();

    if (threadIdx.x == 0) {
        *total_cycles = total_end - total_start;
        *store_cycles = store_end - store_start;
    }
}

// =============================================================================
// CUDA Error Check
// =============================================================================

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
#endif
