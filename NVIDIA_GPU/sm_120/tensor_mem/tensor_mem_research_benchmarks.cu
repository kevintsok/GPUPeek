#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "tensor_mem_research_kernel.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

extern const char* formatBandwidth(double GBps);

// =============================================================================
// Tensor Memory Operations Research Benchmarks
// =============================================================================
//
// This benchmark suite tests:
// 1. LDMATRIX - Warp-level matrix load (PTX Section 9.7.14.5.15)
// 2. STMATRIX - Warp-level matrix store (PTX Section 9.7.14.5.16)
// 3. cp.async - Asynchronous copy (PTX Section 9.7.9.25)
// 4. cp.async.bulk - Bulk async copy variants
// 5. Combined pipelines with MMA
//
// NCU Metrics for SASS Analysis:
// - sm__inst_executed - Instruction count breakdown
// - sm__pipe_mem_cycles_active - Memory pipeline utilization
// - sm__pipe_tensor_cycles_active - Tensor memory utilization
// - ldmatrix transactions - Matrix load efficiency
// =============================================================================

// =============================================================================
// Section 1: LDMATRIX Tests
// =============================================================================

void runLDMatrixTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("1. LDMATRIX (Warp-level Matrix Load) Tests\n");
    printf("================================================================================\n");

    const size_t N = 1 << 16;  // 64K elements
    const size_t bytes = N * sizeof(__half);
    const int iterations = 100;

    __half *d_src, *h_src;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMallocHost(&h_src, bytes));

    for (size_t i = 0; i < N; i++) {
        h_src[i] = (__half)(rand() % 100 / 100.0f);
    }
    CHECK_CUDA(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    __half *d_shm;
    CHECK_CUDA(cudaMalloc(&d_shm, bytes));

    GPUTimer timer;

    // Test 1: Basic LDMATRIX FP16
    printf("\n--- LDMATRIX FP16 (.x1 layout) ---\n\n");

    dim3 gridDim_ld(N / 64);
    dim3 blockDim_ld(256);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        ldmatrix_fp16_kernel<__half><<<gridDim_ld, blockDim_ld, bytes>>>(
            d_src, d_shm, N, N / 64);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("LDMATRIX FP16:        %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Test 2: LDMATRIX multi-tile
    timer.start();
    for (int i = 0; i < iterations; i++) {
        ldmatrix_multi_tile_kernel<__half><<<gridDim_ld, blockDim_ld, bytes>>>(
            d_src, d_shm, N, N / 64);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("LDMATRIX Multi-tile:  %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Test 3: LDMATRIX x1 vs x2 layout
    timer.start();
    for (int i = 0; i < iterations; i++) {
        ldmatrix_layout_x1_kernel<__half><<<gridDim_ld, blockDim_ld, bytes>>>(
            d_src, d_shm, N, N / 64);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("LDMATRIX .x1:         %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        ldmatrix_layout_x2_kernel<__half><<<gridDim_ld, blockDim_ld, bytes>>>(
            d_src, d_shm, N, N / 64);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("LDMATRIX .x2:         %.2f GB/s (%.3f ms)\n",
           bytes * 2 * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_shm));
    CHECK_CUDA(cudaFreeHost(h_src));
}

// =============================================================================
// Section 2: STMATRIX Tests
// =============================================================================

void runSTMatrixTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. STMATRIX (Warp-level Matrix Store) Tests\n");
    printf("================================================================================\n");

    const size_t N = 1 << 16;
    const size_t bytes = N * sizeof(__half);
    const int iterations = 100;

    __half *d_dst, *h_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMallocHost(&h_dst, bytes));

    for (size_t i = 0; i < N; i++) {
        h_dst[i] = (__half)0;
    }
    CHECK_CUDA(cudaMemcpy(d_dst, h_dst, bytes, cudaMemcpyHostToDevice));

    __half *d_shm;
    CHECK_CUDA(cudaMalloc(&d_shm, bytes));

    // Initialize shared memory with data
    CHECK_CUDA(cudaMemset(d_shm, 1, bytes));

    GPUTimer timer;

    dim3 gridDim_st(N / 64);
    dim3 blockDim_st(256);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        stmatrix_fp16_kernel<__half><<<gridDim_st, blockDim_st, bytes>>>(
            d_shm, d_dst, N, N / 64);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("STMATRIX FP16:        %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        stmatrix_layout_x1_kernel<__half><<<gridDim_st, blockDim_st, bytes>>>(
            d_shm, d_dst, N, N / 64);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("STMATRIX .x1:         %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_shm));
    CHECK_CUDA(cudaFreeHost(h_dst));
}

// =============================================================================
// Section 3: cp.async Tests
// =============================================================================

void runCPAsyncTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. cp.async (Asynchronous Copy) Tests\n");
    printf("================================================================================\n");

    printf("\nNote: cp.async requires inline PTX for actual async copy.\n");
    printf("      Current tests use standard memory operations as baseline.\n\n");

    const size_t N = 1 << 18;  // 256K elements
    const size_t bytes = N * sizeof(__half);
    const int iterations = 100;

    __half *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));

    CHECK_CUDA(cudaMemset(d_src, 1, bytes));

    size_t* d_counters;
    CHECK_CUDA(cudaMalloc(&d_counters, sizeof(size_t)));
    CHECK_CUDA(cudaMemset(d_counters, 0, sizeof(size_t)));

    GPUTimer timer;

    dim3 gridDim_async(256);
    dim3 blockDim_async(256);
    size_t shared_size = 256 * 32 * sizeof(__half);

    // Test 1: cp.async 1D baseline
    printf("--- cp.async 1D ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        cp_async_1d_kernel<__half><<<gridDim_async, blockDim_async, shared_size>>>(
            d_src, d_dst, N, d_counters);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("cp.async 1D:          %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Test 2: cp.async group pattern
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cp_async_group_kernel<__half><<<gridDim_async, blockDim_async, shared_size>>>(
            d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("cp.async group:       %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Test 3: cp.async bulk prefetch
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cp_async_bulk_prefetch_kernel<__half><<<gridDim_async, blockDim_async, shared_size>>>(
            d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("cp.async bulk prefetch: %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Test 4: cp.async reduce
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cp_async_reduce_kernel<__half><<<gridDim_async, blockDim_async, shared_size>>>(
            d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("cp.async reduce:      %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_counters));
}

// =============================================================================
// Section 4: Baseline Comparison Tests
// =============================================================================

void runBaselineComparisonTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("4. Baseline Comparison (LDMATRIX vs Naive vs Shared)\n");
    printf("================================================================================\n");

    const size_t N = 1 << 18;
    const size_t bytes = N * sizeof(__half);
    const int iterations = 100;

    __half *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));

    CHECK_CUDA(cudaMemset(d_src, 1, bytes));

    GPUTimer timer;

    dim3 gridDim(256);
    dim3 blockDim(256);
    size_t shared_size = 256 * sizeof(__half);

    // Naive global memory load
    timer.start();
    for (int i = 0; i < iterations; i++) {
        naive_load_kernel<__half><<<gridDim, blockDim>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Naive global load:    %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Shared memory load baseline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        shared_load_kernel<__half><<<gridDim, blockDim, shared_size>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Shared memory load:    %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // LDMATRIX
    timer.start();
    for (int i = 0; i < iterations; i++) {
        ldmatrix_fp16_kernel<__half><<<gridDim.x / 4, blockDim, shared_size>>>(
            d_src, d_dst, N, N / 64);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("LDMATRIX:             %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // cp.async baseline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cp_async_baseline_kernel<__half><<<gridDim, blockDim, shared_size>>>(
            d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("cp.async baseline:     %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // TMA baseline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        tma_baseline_kernel<__half><<<gridDim, blockDim, shared_size>>>(
            d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("TMA baseline:         %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// Section 5: Combined Pipeline Tests
// =============================================================================

void runPipelineTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("5. LDMATRIX + MMA + STMATRIX Pipeline\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const size_t bytes_a = M * K * sizeof(__half);
    const size_t bytes_b = K * N * sizeof(__half);
    const size_t bytes_c = M * N * sizeof(__half);
    const int iterations = 10;

    __half *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMalloc(&d_c, bytes_c));

    CHECK_CUDA(cudaMemset(d_a, 1, bytes_a));
    CHECK_CUDA(cudaMemset(d_b, 2, bytes_b));
    CHECK_CUDA(cudaMemset(d_c, 0, bytes_c));

    size_t* d_counters;
    CHECK_CUDA(cudaMalloc(&d_counters, sizeof(size_t)));
    CHECK_CUDA(cudaMemset(d_counters, 0, sizeof(size_t)));

    dim3 gridDim_pipe(M / 16, N / 16);
    dim3 blockDim_pipe(128);
    size_t shared_size = (M * K + N * K + M * N) * sizeof(__half);

    GPUTimer timer;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        ldmatrix_mma_stmatrix_kernel<__half><<<gridDim_pipe, blockDim_pipe, shared_size>>>(
            d_a, d_b, d_c, M, N, K, d_counters);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;
    printf("Full Pipeline (16x16): %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Naive GEMM baseline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        naive_gemm_kernel<__half><<<gridDim_pipe, blockDim_pipe, shared_size>>>(
            d_a, d_b, d_c, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("Naive GEMM (16x16):   %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_counters));
}

// =============================================================================
// Section 6: NCU Profiling Reference
// =============================================================================

void runNCUProfilingReference() {
    printf("\n");
    printf("================================================================================\n");
    printf("6. NCU Profiling Reference - Tensor Memory Operations\n");
    printf("================================================================================\n");

    printf("\n--- Key NCU Metrics for Tensor Memory ---\n\n");

    printf("LDMATRIX Analysis:\n");
    printf("  ncu --metrics sm__inst_executed.ldmatrix.sum ./gpupeek tensor_mem\n");
    printf("  ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek tensor_mem\n\n");

    printf("STMATRIX Analysis:\n");
    printf("  ncu --metrics sm__inst_executed.stmatrix.sum ./gpupeek tensor_mem\n");
    printf("  ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek tensor_mem\n\n");

    printf("cp.async Analysis:\n");
    printf("  ncu --metrics sm__inst_executed.cp_async.sum ./gpupeek tensor_mem\n");
    printf("  ncu --metrics sm__pipe_mem_cycles_active.pct ./gpupeek tensor_mem\n\n");

    printf("Full SASS Analysis:\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek tensor_mem\n\n");

    printf("--- SASS Instruction Reference ---\n\n");

    printf("| SASS Instruction | Description | PTX Equivalent |\n");
    printf("|-----------------|------------|----------------|\n");
    printf("| LDMATRIX        | Matrix load (8x8 tile) | ld.matrix |\n");
    printf("| LDMATRIXu | Matrix load (unaligned) | ld.matrix |\n");
    printf("| STMATRIX | Matrix store (8x8 tile) | st.matrix |\n");
    printf("| STMATRIXu | Matrix store (unaligned) | st.matrix |\n");
    printf("| CP.ASYNC | Async copy commit | cp.async |\n");
    printf("| BAR.ASYNC | Async barrier | bar.async |\n");
    printf("| HMMA | Half MMA | wmma.mma |\n");
    printf("| @HMMA | Conditional HMMA | - |\n\n");

    printf("--- PTX Instruction Reference ---\n\n");

    printf("LDMATRIX (Section 9.7.14.5.15):\n");
    printf("  ldmatrix.sync.aligned.m8n8.x1 {Rs}, [Ra];\n");
    printf("  ldmatrix.sync.aligned.m8n8.x2 {Rs, Rs+1}, [Ra];\n");
    printf("  ldmatrix.sync.aligned.m8n8.x4 {Rs...Rs+3}, [Ra];\n");
    printf("  ldmatrix.sync.aligned.m16n8.k1 {Rs...}, [Ra];\n\n");

    printf("STMATRIX (Section 9.7.14.5.16):\n");
    printf("  stmatrix.sync.aligned.m8n8.x1 [Rd], {Rs};\n");
    printf("  stmatrix.sync.aligned.m8n8.x2 [Rd], {Rs, Rs+1};\n");
    printf("  stmatrix.sync.aligned.m8n8.x4 [Rd], {Rs...Rs+3};\n\n");

    printf("cp.async (Section 9.7.9.25):\n");
    printf("  cp.async.ca.shared.global [Rd], [Ra], size, cache#;\n");
    printf("  cp.async.commit_group;\n");
    printf("  cp.async.wait_group n;\n");
    printf("  cp.async.wait_all;\n\n");

    printf("cp.async.bulk (Section 9.7.9.25.4):\n");
    printf("  cp.async.bulk.shared.global [Rd], [Ra], size;\n");
    printf("  cp.async.bulk.commit_group;\n");
    printf("  cp.async.bulk.wait_group n;\n");
    printf("  cp.reduce.async.bulk.add.shared.global [Rd], [Ra], size;\n");
    printf("  cp.async.bulk.prefetch [Ra], size;\n\n");

    printf("--- Key Findings Guide ---\n\n");

    printf("1. LDMATRIX Efficiency:\n");
    printf("   - LDMATRIX loads 8x8 tiles (64 elements) per warp\n");
    printf("   - .x1, .x2, .x4 variants load 1/2/4 tiles per instruction\n");
    printf("   - Transposed layout enables efficient MMA data access\n\n");

    printf("2. cp.async Benefits:\n");
    printf("   - Overlaps memory copy with compute\n");
    printf("   - commit_group/wait_group enables batch async operations\n");
    printf("   - cp.async.bulk for larger transfers (up to 128 bytes)\n\n");

    printf("3. Performance Comparison:\n");
    printf("   - LDMATRIX vs Global Load: LDMATRIX wins for tensor ops\n");
    printf("   - cp.async vs TMA: TMA better for large 2D transfers\n");
    printf("   - cp.async vs regular: cp.async hides latency\n");
}

// =============================================================================
// Main Entry Point
// =============================================================================

void runTensorMemResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) Tensor Memory Operations Research    #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    printf("\n");
    printf("================================================================================\n");
    printf("PTX ISA Tensor Memory Instructions\n");
    printf("================================================================================\n\n");

    printf("1. ld.matrix (Section 9.7.14.5.15)\n");
    printf("   - Warp-level matrix load (8x8 tiles)\n");
    printf("   - .x1, .x2, .x4 layouts\n");
    printf("   - Transposed layout for MMA\n\n");

    printf("2. st.matrix (Section 9.7.14.5.16)\n");
    printf("   - Warp-level matrix store\n");
    printf("   - Inverse of ldmatrix\n\n");

    printf("3. cp.async (Section 9.7.9.25)\n");
    printf("   - Asynchronous copy with commit/wait groups\n");
    printf("   - cp.async.bulk for larger transfers\n");
    printf("   - cp.reduce.async.bulk for fused reduce\n\n");

    printf("4. TMA (Tensor Memory Accelerator)\n");
    printf("   - Large 2D transfers\n");
    printf("   - Cache hint support\n\n");

    runLDMatrixTests();
    runSTMatrixTests();
    runCPAsyncTests();
    runBaselineComparisonTests();
    runPipelineTests();
    runNCUProfilingReference();

    printf("\n");
    printf("================================================================================\n");
    printf("Tensor Memory Research Complete!\n");
    printf("================================================================================\n");
    printf("\n");
    printf("For NCU SASS profiling, run:\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek tensor_mem\n");
    printf("  ncu --metrics sm__inst_executed.ldmatrix.sum,sm__pipe_tensor_cycles_active.pct ./gpupeek tensor_mem\n");
    printf("\n");
}
