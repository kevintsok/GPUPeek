#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/timer.h"
#include "wmma_test_kernel.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

extern const char* formatBandwidth(double GBps);
extern const char* formatGFLOPS(double gflops);

double computeGemmgfLOPS(size_t M, size_t N, size_t K, double time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

// =============================================================================
// WMMA Test Runner
// =============================================================================

void runWMMATest() {
    printf("\n");
    printf("================================================================================\n");
    printf("WMMA (Warp-level Matrix Multiply-Accumulate) Test\n");
    printf("================================================================================\n");

    const int M = 256;
    const int N = 256;
    const int K = 256;
    const int iterations = 10;

    printf("\nMatrix sizes: M=%d, N=%d, K=%d\n", M, N, K);
    printf("WMMA shape: m16n16k16\n\n");

    // Allocate host memory
    __half *h_a, *h_b;
    float *h_d;

    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    // Initialize with random data
    for (int i = 0; i < M * K; i++) {
        h_a[i] = __float2half((rand() % 100) / 100.0f);
    }
    for (int i = 0; i < K * N; i++) {
        h_b[i] = __float2half((rand() % 100) / 100.0f);
    }
    for (int i = 0; i < M * N; i++) {
        h_d[i] = 0.0f;
    }

    // Allocate device memory
    __half *d_a, *d_b;
    float *d_d;

    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // Launch kernel - grid dimensions for 16x16 output tiles
    dim3 gridDim(N / 16, M / 16);
    dim3 blockDim(32);  // WMMA uses 32 threads (warp)

    printf("Grid: %dx%d, Block: %d\n", gridDim.x, gridDim.y, blockDim.x);
    printf("Launching wmma_fp16_test_kernel...\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_fp16_test_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);

    printf("Results:\n");
    printf("  Time: %.3f ms per iteration\n", time_ms);
    printf("  GFLOPS: %.2f GFLOPS\n", gflops);

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_d, d_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result (just check it's not all zeros)
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) {
        sum += h_d[i];
    }
    printf("  Result sum: %.2f (should be non-zero)\n", sum);

    // Cleanup
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));

    printf("\n================================================================================\n");
}

void runWMMAResearchBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("WMMA Research Benchmarks\n");
    printf("========================================\n");
    printf("Testing WMMA (Warp-level MMA) API on SM 12.0\n");
    printf("Shape: m16n16k16\n");
    printf("========================================\n");

    runWMMATest();

    printf("\nNCU Profiling Hints:\n");
    printf("  ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe wmma\n");
}
