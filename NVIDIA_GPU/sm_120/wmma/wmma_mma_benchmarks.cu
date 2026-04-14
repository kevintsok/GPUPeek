#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "wmma_mma_kernel.cu"

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

// =============================================================================
// WMMA/MMA Research Benchmarks - Cycle Counting
// =============================================================================
//
// This benchmark measures cycle counts for WMMA instructions on Blackwell.
//
// What each WMMA instruction needs:
//   1. load_matrix_sync(frag_a, ...):
//      - Input: 16x16 halfs from global memory (512 bytes)
//      - Layout: row_major (A) or col_major (B)
//      - Register usage: 8 x uint32 (packed halfs)
//
//   2. load_matrix_sync(frag_b, ...):
//      - Same as frag_a
//
//   3. mma_sync(frag_d, frag_a, frag_b, frag_d):
//      - Input: frag_a (16x16), frag_b (16x16)
//      - Output: frag_d (16x16) accumulated
//      - 256 FMA operations per iteration
//      - Latency: ~6-8 cycles on Blackwell
//
//   4. store_matrix_sync(d, frag_d, ...):
//      - Output: 16x16 floats to global memory (1024 bytes)
//
// Theoretical throughput:
//   - FP16 tensor core: 512 FLOPS per cycle per warp
//   - At 1.9 GHz: ~973 GFLOPS per tensor core
//   - RTX 5080 has 60 tensor cores
//
// =============================================================================

// =============================================================================
// Test 1: Total WMMA Cycles
// =============================================================================

static void runTotalCyclesTest() {
    printf("\n--- Test 1: Total WMMA Cycles (m16n16k16) ---\n");

    const int M = 256;
    const int N = 256;
    const int K = 256;
    const int iterations = 10;

    printf("Matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Output tiles: %d x %d = %d warps\n", M/16, N/16, (M/16)*(N/16));

    __half *h_a, *h_b;
    float *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; i++) h_d[i] = 0.0f;

    __half *d_a, *d_b;
    float *d_d;
    unsigned long long *d_cycles;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);

    // Warmup
    wmma_fp16_cycles_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K, d_cycles);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure
    GPUTimer timer;
    timer.start();

    for (int i = 0; i < iterations; i++) {
        wmma_fp16_cycles_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K, d_cycles);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    unsigned long long cycles;
    CHECK_CUDA(cudaMemcpy(&cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double time_ms = timer.elapsed_ms() / iterations;
    double cycles_per_iter = (double)cycles / iterations;

    printf("Grid: %dx%d, Block: %d\n", gridDim.x, gridDim.y, blockDim.x);
    printf("Time: %.3f ms/iteration\n", time_ms);
    printf("Total cycles (all warps): %llu\n", cycles);
    printf("Cycles per warp: %.0f\n", cycles_per_iter);

    // Calculate per-MMA-iteration breakdown
    int k_iterations = K / 16;
    printf("K iterations: %d\n", k_iterations);
    printf("Cycles per K-iteration: %.0f\n", cycles_per_iter / k_iterations);

    double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
    printf("Performance: %.2f GFLOPS\n", gflops);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFree(d_cycles));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Test 2: Individual Instruction Cycles
// =============================================================================

static void runIndividualCyclesTest() {
    printf("\n--- Test 2: Individual WMMA Instruction Cycles ---\n");

    const int M = 256;
    const int N = 256;
    const int K = 256;

    printf("Matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);
    printf("K iterations: %d\n", K / 16);

    __half *h_a, *h_b;
    float *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);

    __half *d_a, *d_b;
    float *d_d;
    unsigned long long *d_load_a, *d_load_b, *d_mma, *d_store;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_load_a, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_load_b, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_mma, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_store, sizeof(unsigned long long)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);

    // Measure load A cycles
    unsigned long long load_a_cycles = 0, load_b_cycles = 0, mma_cycles = 0, store_cycles = 0;

    wmma_load_cycles_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_load_a, d_load_b, M, N, K);
    CHECK_CUDA(cudaMemcpy(&load_a_cycles, d_load_a, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&load_b_cycles, d_load_b, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Measure MMA cycles
    wmma_mma_cycles_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, d_mma, M, N, K);
    CHECK_CUDA(cudaMemcpy(&mma_cycles, d_mma, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Measure store cycles
    wmma_store_cycles_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, d_store, M, N, K);
    CHECK_CUDA(cudaMemcpy(&store_cycles, d_store, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    printf("\nPer-Instruction Cycle Count (per warp, per K-iteration of 16):\n");
    printf("  load_matrix_sync (frag_a): %llu cycles\n", load_a_cycles);
    printf("  load_matrix_sync (frag_b): %llu cycles\n", load_b_cycles);
    printf("  mma_sync:                 %llu cycles\n", mma_cycles);
    printf("  store_matrix_sync:        %llu cycles\n", store_cycles);

    unsigned long long total = load_a_cycles + load_b_cycles + mma_cycles;
    printf("\nTotal (load_a + load_b + mma): %llu cycles\n", total);
    printf("Memory bound: loads take %.1f%% of MMA time\n",
           100.0 * (load_a_cycles + load_b_cycles) / (double)total);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFree(d_load_a));
    CHECK_CUDA(cudaFree(d_load_b));
    CHECK_CUDA(cudaFree(d_mma));
    CHECK_CUDA(cudaFree(d_store));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Test 3: What Data Each Instruction Needs
// =============================================================================

static void runDataRequirementsTest() {
    printf("\n--- Test 3: WMMA Data Requirements ---\n");

    printf("\n=== WMMA m16n16k16 Data Layout ===\n\n");

    printf("Shape: m16n16k16 (16 rows x 16 cols x 16 K-dimension)\n\n");

    printf("Per warp (32 threads) memory access:\n");
    printf("  load_matrix_sync(frag_a):\n");
    printf("    - Reads: 16 x 16 = 256 halfs = 512 bytes\n");
    printf("    - Layout: row_major (A[row*K + col])\n");
    printf("    - Each thread: 8 halfs (16 bytes)\n\n");

    printf("  load_matrix_sync(frag_b):\n");
    printf("    - Reads: 16 x 16 = 256 halfs = 512 bytes\n");
    printf("    - Layout: col_major (B[row*N + col])\n");
    printf("    - Each thread: 8 halfs (16 bytes)\n\n");

    printf("  mma_sync:\n");
    printf("    - Input A: 256 halfs (register file)\n");
    printf("    - Input B: 256 halfs (register file)\n");
    printf("    - Output D: 256 floats (register file)\n");
    printf("    - FLOPs: 2 x 16 x 16 x 16 = 8192 FMA per iteration\n");
    printf("    - Throughput: ~512 FLOPS per cycle per warp\n\n");

    printf("  store_matrix_sync:\n");
    printf("    - Writes: 16 x 16 = 256 floats = 1024 bytes\n");
    printf("    - Each thread: 8 floats (32 bytes)\n\n");

    printf("Register usage per warp:\n");
    printf("  A fragment: 8 x uint32 (packed halfs)\n");
    printf("  B fragment: 8 x uint32 (packed halfs)\n");
    printf("  D fragment: 8 x float\n");
    printf("  Total: 24 registers minimum\n\n");

    printf("Memory bandwidth per tile:\n");
    printf("  Load A + B: 512 + 512 = 1024 bytes\n");
    printf("  Store D: 1024 bytes\n");
    printf("  Total: 2048 bytes per K-iteration\n\n");

    printf("For full GEMM M=%d N=%d K=%d:\n", 256, 256, 256);
    size_t tiles_m = 256 / 16;
    size_t tiles_n = 256 / 16;
    size_t k_iters = 256 / 16;
    size_t total_loads = tiles_m * tiles_n * k_iters * 2 * 512;
    size_t total_stores = tiles_m * tiles_n * 1024;
    printf("  Total loads: %zu tiles x 2 x 512 bytes = %.1f MB\n",
           tiles_m * tiles_n * k_iters, total_loads / 1024.0 / 1024.0);
    printf("  Total stores: %zu tiles x 1024 bytes = %.1f MB\n",
           tiles_m * tiles_n, total_stores / 1024.0 / 1024.0);
}

// =============================================================================
// Test 4: Throughput Analysis
// =============================================================================

static void runThroughputTest() {
    printf("\n--- Test 4: WMMA Throughput Analysis ---\n");

    const int M = 512;
    const int N = 512;
    const int K = 512;
    const int iterations = 10;

    printf("Matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);

    __half *h_a, *h_b;
    float *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);

    __half *d_a, *d_b;
    float *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);

    // Warmup
    wmma_fp16_test_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    GPUTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_fp16_test_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
    double utilized_tflops = gflops / 1000.0; // Convert to TFLOPS

    printf("Time: %.3f ms\n", time_ms);
    printf("Performance: %.2f GFLOPS (%.2f TFLOPS)\n", gflops, utilized_tflops);
    printf("Grid: %dx%d = %d warps\n", gridDim.x, gridDim.y, gridDim.x * gridDim.y);

    // Theoretical peak for FP16 on RTX 5080
    printf("\nTheoretical FP16 tensor peak: ~89 TFLOPS\n");
    printf("Achieved efficiency: %.1f%%\n", utilized_tflops / 89.0 * 100);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Main Runner
// =============================================================================

void runWMMA_MMA_ResearchBenchmarks(size_t N) {
    printf("\n");
    printf("================================================================================\n");
    printf("WMMA/MMA Research - Cycle Counting and Data Requirements\n");
    printf("================================================================================\n");
    printf("\n");
    printf("API: nvcuda::wmma (C++ wrapper for PTX wmma instructions)\n");
    printf("Shape: m16n16k16 (16x16 output, 16 K-dimension)\n");
    printf("Precision: FP16 input, FP32 accumulator\n");
    printf("\n");
    printf("PTX Instructions:\n");
    printf("  wmma.load.a.sync  - Load matrix A fragment\n");
    printf("  wmma.load.b.sync  - Load matrix B fragment\n");
    printf("  wmma.mma.sync     - Matrix multiply-accumulate\n");
    printf("  wmma.store.d.sync - Store result fragment\n");
    printf("\n");
    printf("================================================================================\n");

    runDataRequirementsTest();
    runIndividualCyclesTest();
    runTotalCyclesTest();
    runThroughputTest();

    printf("\n================================================================================\n");
    printf("NCU Profiling Commands:\n");
    printf("================================================================================\n");
    printf("  # Overall tensor utilization\n");
    printf("  ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe mma\n");
    printf("\n");
    printf("  # MMA instruction throughput\n");
    printf("  ncu --set full --metrics sm__inst_executed.mma.sum ./gpupeek.exe mma\n");
    printf("\n");
    printf("  # Memory throughput\n");
    printf("  ncu --set full --metrics dram__bytes.sum ./gpupeek.exe mma\n");
    printf("\n");
    printf("  # Detailed cycle analysis\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek.exe mma\n");
}
