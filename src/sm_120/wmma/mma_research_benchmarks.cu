#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "mma_research_kernel.cu"
#include "wmma_mma_kernel.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

extern const char* formatBandwidth(double GBps);
extern const char* formatGFLOPS(double gflops);

// =============================================================================
// MMA Research Benchmarks - Working WMMA Kernels
// =============================================================================
//
// This benchmark suite tests WMMA (Warp-level Matrix Multiply-Accumulate):
// - FP16 (Half precision)
// - BF16 (BFloat16)
// - TF32 (TensorFloat-32)
// - INT8 (Integer)
//
// NCU Metrics:
// - sm__pipe_tensor_cycles_active - Tensor core utilization
// - sm__inst_executed - Instruction count
// =============================================================================

// Helper to compute GFLOPS for matrix multiply
double computeGemmgfLOPS(size_t M, size_t N, size_t K, double time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

// =============================================================================
// WMMA FP16 Tests
// =============================================================================

void runWMMATests() {
    printf("\n");
    printf("================================================================================\n");
    printf("1. WMMA (Warp-level MMA) Tests - m16n16k16 Shape\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    __half *h_a, *h_b;
    float *h_c, *h_d;

    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_a[i] = (__half)(rand() % 100 / 100.0f);
    for (size_t i = 0; i < K * N; i++) h_b[i] = (__half)(rand() % 100 / 100.0f);
    for (size_t i = 0; i < M * N; i++) h_c[i] = 0.0f;

    __half *d_a, *d_b;
    float *d_c, *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);  // WMMA uses 32 threads (warp) per block

    GPUTimer timer;

    // WMMA FP16 Test
    printf("\n--- WMMA FP16 (m16n16k16) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_fp16_kernel<__half><<<gridDim, blockDim, 0>>>(d_a, d_b, d_c, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("WMMA FP16:           %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    // Cleanup
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// WMMA BF16 Tests
// =============================================================================

void runMMABF16Tests() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. WMMA BF16 (BFloat16) Tests\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    float *h_a, *h_b;
    float *h_c, *h_d;

    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_a[i] = rand() % 100 / 100.0f;
    for (size_t i = 0; i < K * N; i++) h_b[i] = rand() % 100 / 100.0f;
    for (size_t i = 0; i < M * N; i++) h_c[i] = 0.0f;

    float *d_a, *d_b;
    float *d_c, *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);  // WMMA uses 32 threads (warp) per block

    GPUTimer timer;

    printf("\n--- WMMA BF16 ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_bf16_kernel<float><<<gridDim, blockDim, 0>>>(d_a, d_b, d_c, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("WMMA BF16:           %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Simple GEMM Baseline
// =============================================================================

void runBaselineComparisonTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. Simple GEMM Baseline (Non-WMMA)\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    float *h_a, *h_b;
    float *h_c;

    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_c, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_a[i] = rand() % 100 / 100.0f;
    for (size_t i = 0; i < K * N; i++) h_b[i] = rand() % 100 / 100.0f;

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    GPUTimer timer;

    printf("\n--- Simple GEMM (FP32) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        simpleGemmKernel<float><<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("Simple GEMM FP32:    %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
}

// =============================================================================
// Main MMA Research Benchmark Runner
// =============================================================================

void runMMAResearchBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("MMA (Tensor Core) Research Benchmarks\n");
    printf("========================================\n");
    printf("Blackwell 5th-gen Tensor Core supports:\n");
    printf("  - WMMA: m16n16k16 shape\n");
    printf("  - Types: FP16, BF16, FP32, FP64, INT8\n");
    printf("  - TF32 support on Ampere+\n");
    printf("========================================\n");

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("========================================\n");

    // Run test suites
    runWMMATests();
    runMMABF16Tests();
    runBaselineComparisonTests();
    runWMMAcycleCountingTests();

    printf("\n=== MMA Research Complete ===\n");
    printf("NCU Profiling Hints:\n");
    printf("  ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe mma\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek.exe mma\n");
}

// =============================================================================
// WMMA Cycle Counting Tests
// =============================================================================

void runWMMAcycleCountingTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("WMMA Cycle Counting & Data Requirements\n");
    printf("================================================================================\n");

    runWMMA_MMA_ResearchBenchmarks(0);
}
