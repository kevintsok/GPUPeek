#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "wmma_test_kernel.cu"

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

extern const char* formatBandwidth(double GBps);
extern const char* formatGFLOPS(double gflops);

double computeGemmgfLOPS(size_t M, size_t N, size_t K, double time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

// =============================================================================
// WMMA FP16 Test (m16n16k16)
// =============================================================================

static void runWMMAFP16Test() {
    printf("\n--- Test 1: WMMA FP16 (m16n16k16) ---\n");

    const int M = 256;
    const int N = 256;
    const int K = 256;
    const int iterations = 10;

    printf("Matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);

    __half *h_a, *h_b;
    float *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half((rand() % 100) / 100.0f);
    for (int i = 0; i < M * N; i++) h_d[i] = 0.0f;

    __half *d_a, *d_b;
    float *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);

    GPUTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_fp16_test_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);

    printf("Grid: %dx%d, Block: %d\n", gridDim.x, gridDim.y, blockDim.x);
    printf("Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);

    CHECK_CUDA(cudaMemcpy(h_d, d_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) sum += h_d[i];
    printf("Result sum: %.2f (non-zero=correct)\n", sum);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// WMMA FP16 Large Matrix Test
// =============================================================================

static void runWMMAFP16LargeTest() {
    printf("\n--- Test 2: WMMA FP16 Large Matrix ---\n");

    const int M = 512;
    const int N = 512;
    const int K = 512;
    const int iterations = 10;

    printf("Matrix sizes: M=%d, N=%d, K=%d (4x larger)\n", M, N, K);

    __half *h_a, *h_b;
    float *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half((rand() % 100) / 100.0f);
    for (int i = 0; i < M * N; i++) h_d[i] = 0.0f;

    __half *d_a, *d_b;
    float *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);

    GPUTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_fp16_test_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);

    printf("Grid: %dx%d, Block: %d\n", gridDim.x, gridDim.y, blockDim.x);
    printf("Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);

    CHECK_CUDA(cudaMemcpy(h_d, d_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) sum += h_d[i];
    printf("Result sum: %.2f (non-zero=correct)\n", sum);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Main WMMA Test Runner
// =============================================================================

void runWMMATest() {
    printf("\n");
    printf("================================================================================\n");
    printf("WMMA (Warp-level Matrix Multiply-Accumulate) Tests\n");
    printf("================================================================================\n");
    printf("PTX ISA: Section 9.7.14 (Warp-Level Matrix Instructions)\n");
    printf("C++ API: nvcuda::wmma namespace\n");
    printf("Shape: m16n16k16\n");
    printf("================================================================================\n");

    runWMMAFP16Test();
    runWMMAFP16LargeTest();

    printf("\n================================================================================\n");
    printf("WMMA Tests Complete\n");
    printf("================================================================================\n");
}

void runWMMAResearchBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("WMMA Research Benchmarks\n");
    printf("========================================\n");
    printf("Testing WMMA (Warp-level MMA) API on SM 12.0\n");
    printf("========================================\n");

    runWMMATest();

    printf("\nNCU Profiling Hints:\n");
    printf("  ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe wmma\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek.exe wmma\n");
}
