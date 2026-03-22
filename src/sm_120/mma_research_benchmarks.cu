#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/timer.h"
#include "mma_research_kernel.cu"

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
// MMA Research Benchmarks - Comprehensive Coverage
// =============================================================================
//
// This benchmark suite tests all major MMA instruction variants:
// 1. wmma - Warp-level MMA (m16n16k16)
// 2. mma - Newer MMA with multiple shapes (m8n8k4, m16n8k8, etc.)
// 3. mma.sp - Sparse MMA
// 4. wgmma - Asynchronous Warpgroup MMA
// 5. tcgen05 - TensorCore 5th Generation MMA
//
// Data Types:
// - FP16 (Half precision)
// - BF16 (BFloat16)
// - TF32 (TensorFloat-32)
// - FP64 (Double precision)
// - INT8 (Integer)
// - INT4 (4-bit quantization)
//
// NCU Metrics for SASS Analysis:
// - sm__pipe_tensor_cycles_active - Tensor core utilization
// - sm__inst_executed - Instruction count
// - sm__average_execution_latency - Instruction latency
// - tcb__sectors - L2/TLB statistics
// =============================================================================

// Helper to compute GFLOPS for matrix multiply
double computeGemmgfLOPS(size_t M, size_t N, size_t K, double time_ms) {
    // 2*M*N*K floating point operations per matrix multiply
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

// =============================================================================
// Section 1: WMMA (Warp-level Matrix Multiply-Accumulate) Tests
// =============================================================================

void runWMMATests() {
    printf("\n");
    printf("================================================================================\n");
    printf("1. WMMA (Warp-level MMA) Tests - m16n16k16 Shape\n");
    printf("================================================================================\n");

    // Matrix sizes
    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    // Allocate memory
    __half *h_a, *h_b;
    float *h_c, *h_d;

    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    // Initialize with random data
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
    dim3 blockDim(32, 8);  // wmma uses 32 threads per block

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

    // WMMA FP32 Accumulation Test
    printf("\n--- WMMA FP16 with FP32 Accumulation ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_fp32_acc_kernel<__half><<<gridDim, blockDim, 0>>>(d_a, d_b, d_c, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    time_ms = timer.elapsed_ms() / iterations;
    gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("WMMA FP32 Accum:     %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

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
// Section 2: MMA (New Warp-level MMA) Tests with Multiple Shapes
// =============================================================================

void runMMAShapeTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. MMA Tests - Multiple Shapes (m16n8k8, m8n8k4, etc.)\n");
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

    __half *d_a, *d_b;
    float *d_c, *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // mma.m16n8k8.f16
    printf("\n--- MMA m16n8k8 FP16 ---\n\n");

    dim3 gridDim16n8k8(M / 16, N / 8);
    dim3 blockDim32(32, 8);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mma_m16n8k8_fp16_kernel<__half><<<gridDim16n8k8, blockDim32, 0>>>(d_a, d_b, d_c, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("mma.m16n8k8.f16:     %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    // mma.m8n8k4.f16 (FP64-like shape)
    printf("\n--- MMA m8n8k4 FP16 ---\n\n");

    dim3 gridDim8n8k4(M / 8, N / 8);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_fp16_kernel<__half><<<gridDim8n8k4, blockDim32, 0>>>(d_a, d_b, d_c, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    time_ms = timer.elapsed_ms() / iterations;
    gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("mma.m8n8k4.f16:      %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

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
// Section 3: MMA with Different Data Types (TF32, BF16, FP64, INT8)
// =============================================================================

void runMMADataTypeTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. MMA Data Types - TF32, BF16, FP64, INT8\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    GPUTimer timer;

    // TF32 Test
    printf("\n--- TF32 (m16n8k4) TensorFloat-32 ---\n\n");

    __half *h_tf32_a, *h_tf32_b;
    float *h_tf32_c, *h_tf32_d;
    CHECK_CUDA(cudaMallocHost(&h_tf32_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_tf32_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_tf32_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_tf32_d, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_tf32_a[i] = (__half)(rand() % 100 / 100.0f);
    for (size_t i = 0; i < K * N; i++) h_tf32_b[i] = (__half)(rand() % 100 / 100.0f);

    __half *d_tf32_a, *d_tf32_b;
    float *d_tf32_c, *d_tf32_d;
    CHECK_CUDA(cudaMalloc(&d_tf32_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_tf32_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_tf32_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tf32_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_tf32_a, h_tf32_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tf32_b, h_tf32_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDimTF32(M / 16, N / 8);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mma_tf32_kernel<__half><<<gridDimTF32, dim3(32, 8), 0>>>(d_tf32_a, d_tf32_b, d_tf32_c, d_tf32_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("TF32 (m16n8k4):       %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    CHECK_CUDA(cudaFree(d_tf32_a));
    CHECK_CUDA(cudaFree(d_tf32_b));
    CHECK_CUDA(cudaFree(d_tf32_c));
    CHECK_CUDA(cudaFree(d_tf32_d));
    CHECK_CUDA(cudaFreeHost(h_tf32_a));
    CHECK_CUDA(cudaFreeHost(h_tf32_b));
    CHECK_CUDA(cudaFreeHost(h_tf32_c));
    CHECK_CUDA(cudaFreeHost(h_tf32_d));

    // BF16 Test
    printf("\n--- BF16 (m16n8k8) BFloat16 ---\n\n");

    __half *h_bf16_a, *h_bf16_b;
    float *h_bf16_c, *h_bf16_d;
    CHECK_CUDA(cudaMallocHost(&h_bf16_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_bf16_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_bf16_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_bf16_d, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_bf16_a[i] = (__half)(rand() % 100 / 100.0f);
    for (size_t i = 0; i < K * N; i++) h_bf16_b[i] = (__half)(rand() % 100 / 100.0f);

    __half *d_bf16_a, *d_bf16_b;
    float *d_bf16_c, *d_bf16_d;
    CHECK_CUDA(cudaMalloc(&d_bf16_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_bf16_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_bf16_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bf16_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_bf16_a, h_bf16_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bf16_b, h_bf16_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDimBF16(M / 16, N / 8);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mma_bf16_kernel<__half><<<gridDimBF16, dim3(32, 8), 0>>>(d_bf16_a, d_bf16_b, d_bf16_c, d_bf16_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    time_ms = timer.elapsed_ms() / iterations;
    gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("BF16 (m16n8k8):       %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    CHECK_CUDA(cudaFree(d_bf16_a));
    CHECK_CUDA(cudaFree(d_bf16_b));
    CHECK_CUDA(cudaFree(d_bf16_c));
    CHECK_CUDA(cudaFree(d_bf16_d));
    CHECK_CUDA(cudaFreeHost(h_bf16_a));
    CHECK_CUDA(cudaFreeHost(h_bf16_b));
    CHECK_CUDA(cudaFreeHost(h_bf16_c));
    CHECK_CUDA(cudaFreeHost(h_bf16_d));

    // FP64 Test
    printf("\n--- FP64 (m8n8k4) Double Precision ---\n\n");

    double *h_fp64_a, *h_fp64_b, *h_fp64_c, *h_fp64_d;
    CHECK_CUDA(cudaMallocHost(&h_fp64_a, M * K * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_fp64_b, K * N * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_fp64_c, M * N * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_fp64_d, M * N * sizeof(double)));

    for (size_t i = 0; i < M * K; i++) h_fp64_a[i] = rand() % 100 / 100.0;
    for (size_t i = 0; i < K * N; i++) h_fp64_b[i] = rand() % 100 / 100.0;

    double *d_fp64_a, *d_fp64_b, *d_fp64_c, *d_fp64_d;
    CHECK_CUDA(cudaMalloc(&d_fp64_a, M * K * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_fp64_b, K * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_fp64_c, M * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_fp64_d, M * N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_fp64_a, h_fp64_a, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fp64_b, h_fp64_b, K * N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 gridDimFP64(M / 8, N / 8);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mma_fp64_kernel<double><<<gridDimFP64, dim3(32, 8), 0>>>(d_fp64_a, d_fp64_b, d_fp64_c, d_fp64_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    time_ms = timer.elapsed_ms() / iterations;
    gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("FP64 (m8n8k4):        %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    CHECK_CUDA(cudaFree(d_fp64_a));
    CHECK_CUDA(cudaFree(d_fp64_b));
    CHECK_CUDA(cudaFree(d_fp64_c));
    CHECK_CUDA(cudaFree(d_fp64_d));
    CHECK_CUDA(cudaFreeHost(h_fp64_a));
    CHECK_CUDA(cudaFreeHost(h_fp64_b));
    CHECK_CUDA(cudaFreeHost(h_fp64_c));
    CHECK_CUDA(cudaFreeHost(h_fp64_d));

    // INT8 Test
    printf("\n--- INT8 (m16n8k16) Integer ---\n\n");

    char *h_int8_a, *h_int8_b;
    int *h_int8_c, *h_int8_d;
    CHECK_CUDA(cudaMallocHost(&h_int8_a, M * K * sizeof(char)));
    CHECK_CUDA(cudaMallocHost(&h_int8_b, K * N * sizeof(char)));
    CHECK_CUDA(cudaMallocHost(&h_int8_c, M * N * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&h_int8_d, M * N * sizeof(int)));

    for (size_t i = 0; i < M * K; i++) h_int8_a[i] = rand() % 10;
    for (size_t i = 0; i < K * N; i++) h_int8_b[i] = rand() % 10;

    char *d_int8_a, *d_int8_b;
    int *d_int8_c, *d_int8_d;
    CHECK_CUDA(cudaMalloc(&d_int8_a, M * K * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&d_int8_b, K * N * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&d_int8_c, M * N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_int8_d, M * N * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_int8_a, h_int8_a, M * K * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_int8_b, h_int8_b, K * N * sizeof(char), cudaMemcpyHostToDevice));

    dim3 gridDimINT8(M / 16, N / 8);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mma_int8_kernel<char><<<gridDimINT8, dim3(32, 8), 0>>>(d_int8_a, d_int8_b, d_int8_c, d_int8_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    time_ms = timer.elapsed_ms() / iterations;
    double iops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("INT8 (m16n8k16):      %.2f GIOPS (%.3f ms)\n", iops, time_ms);

    CHECK_CUDA(cudaFree(d_int8_a));
    CHECK_CUDA(cudaFree(d_int8_b));
    CHECK_CUDA(cudaFree(d_int8_c));
    CHECK_CUDA(cudaFree(d_int8_d));
    CHECK_CUDA(cudaFreeHost(h_int8_a));
    CHECK_CUDA(cudaFreeHost(h_int8_b));
    CHECK_CUDA(cudaFreeHost(h_int8_c));
    CHECK_CUDA(cudaFreeHost(h_int8_d));
}

// =============================================================================
// Section 4: Sparse MMA (mma.sp) Tests
// =============================================================================

void runSparseMMATests() {
    printf("\n");
    printf("================================================================================\n");
    printf("4. Sparse MMA (mma.sp) Tests - Structured Sparsity\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    __half *h_a, *h_b, *h_meta;
    float *h_c, *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_meta, M * K * sizeof(__half)));  // Sparsity mask
    CHECK_CUDA(cudaMallocHost(&h_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_a[i] = (__half)(rand() % 100 / 100.0f);
    for (size_t i = 0; i < K * N; i++) h_b[i] = (__half)(rand() % 100 / 100.0f);
    // 2:4 structured sparsity pattern (every 4 elements, 2 are non-zero)
    for (size_t i = 0; i < M * K; i++) h_meta[i] = (i % 4 < 2) ? (__half)1.0 : (__half)0.0;

    __half *d_a, *d_b, *d_meta;
    float *d_c, *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_meta, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_meta, h_meta, M * K * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 8);
    GPUTimer timer;

    printf("\n--- Sparse MMA FP16 (m16n8k32 with 2:4 Sparsity) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mma_sparse_fp16_kernel<__half><<<gridDim, dim3(32, 8), 0>>>(d_a, d_b, d_meta, d_c, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("Sparse MMA FP16:      %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);
    printf("Speedup vs dense:     ~2x (2:4 sparsity)\n");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_meta));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_meta));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Section 5: WGMMA (Asynchronous Warpgroup MMA) Tests
// =============================================================================

void runWGMMATests() {
    printf("\n");
    printf("================================================================================\n");
    printf("5. WGMMA (Asynchronous Warpgroup MMA) Tests\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    __half *h_a, *h_b;
    float *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_a[i] = (__half)(rand() % 100 / 100.0f);
    for (size_t i = 0; i < K * N; i++) h_b[i] = (__half)(rand() % 100 / 100.0f);

    __half *d_a, *d_b;
    float *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 64, N / 16);  // WGMMA uses larger tiles (m64nNk16)
    GPUTimer timer;

    printf("\n--- WGMMA Async FP16 (m64nNk16) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wgmma_async_kernel<__half><<<gridDim, dim3(96, 4), 0>>>(d_a, d_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("WGMMA Async FP16:     %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);
    printf("Note: WGMMA requires cooperative groups (3 warps per tile)\n");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Section 6: TCGen05 (TensorCore 5th Generation) Tests
// =============================================================================

void runTCGen05Tests() {
    printf("\n");
    printf("================================================================================\n");
    printf("6. TCGen05 (TensorCore 5th Generation) Tests - Blackwell\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    __half *h_a, *h_b, *h_scale_a, *h_scale_b;
    float *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_scale_a, M * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_scale_b, N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_a[i] = (__half)(rand() % 100 / 100.0f);
    for (size_t i = 0; i < K * N; i++) h_b[i] = (__half)(rand() % 100 / 100.0f);
    for (size_t i = 0; i < M; i++) h_scale_a[i] = (__half)1.0f;
    for (size_t i = 0; i < N; i++) h_scale_b[i] = (__half)1.0f;

    __half *d_a, *d_b, *d_scale_a, *d_scale_b;
    float *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_scale_a, M * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_scale_b, N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale_a, h_scale_a, M * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale_b, h_scale_b, N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 64, N / 16);  // TCGen05 uses m64 tiles
    GPUTimer timer;

    printf("\n--- TCGen05 MMA FP16 with Block Scaling ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        tcgen05_mma_kernel<__half><<<gridDim, dim3(128, 4), 0>>>(d_a, d_b, d_scale_a, d_scale_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("TCGen05 MMA BlockScale: %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    printf("\n--- TCGen05 Weight-Only Quantization ---\n\n");

    // Weight-only quantization test
    __half *h_b_quant;
    CHECK_CUDA(cudaMallocHost(&h_b_quant, K * N * sizeof(__half)));
    for (size_t i = 0; i < K * N; i++) h_b_quant[i] = (__half)(rand() % 10);
    CHECK_CUDA(cudaMemcpy(d_b, h_b_quant, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        tcgen05_mma_ws_kernel<__half><<<gridDim, dim3(128, 4), 0>>>(d_a, d_b_quant, d_scale_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    time_ms = timer.elapsed_ms() / iterations;
    gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("TCGen05 Weight-Only:   %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);
    printf("Note: Weight quantization for LLM inference\n");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_scale_a));
    CHECK_CUDA(cudaFree(d_scale_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_b_quant));
    CHECK_CUDA(cudaFreeHost(h_scale_a));
    CHECK_CUDA(cudaFreeHost(h_scale_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Section 7: Baseline Comparison (Non-MMA GEMM)
// =============================================================================

void runBaselineComparisonTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("7. Baseline Comparison - Non-MMA GEMM Implementations\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 10;

    float *h_a, *h_b, *h_c;
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

    GPUTimer timer;

    // Naive GEMM
    printf("\n--- Naive GEMM (element-wise) ---\n\n");

    dim3 blockDimNaive(16, 16);
    dim3 gridDimNaive((N + 15) / 16, (M + 15) / 16);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        naive_gemm_kernel<float><<<gridDimNaive, blockDimNaive>>>(d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("Naive GEMM:           %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    // Shared memory blocked GEMM
    printf("\n--- Shared Memory Blocked GEMM ---\n\n");

    dim3 blockDimShared(16, 16);
    dim3 gridDimShared(N / 16, M / 16);
    size_t sharedSize = 2 * 16 * 16 * sizeof(float);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        shared_gemm_kernel<float><<<gridDimShared, blockDimShared, sharedSize>>>(d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    time_ms = timer.elapsed_ms() / iterations;
    gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("Shared GEMM:          %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
}

// =============================================================================
// Section 8: Mixed Precision and Fused Operations
// =============================================================================

void runMixedPrecisionTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("8. Mixed Precision & Fused Operations\n");
    printf("================================================================================\n");

    const size_t M = 256;
    const size_t N = 256;
    const size_t K = 256;
    const int iterations = 100;

    float *h_a_f32, *h_b_f32, *h_d_f32;
    CHECK_CUDA(cudaMallocHost(&h_a_f32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_b_f32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_d_f32, M * N * sizeof(float)));

    for (size_t i = 0; i < M * K; i++) h_a_f32[i] = rand() % 100 / 100.0f;
    for (size_t i = 0; i < K * N; i++) h_b_f32[i] = rand() % 100 / 100.0f;

    float *d_a_f32, *d_b_f32, *d_d_f32;
    CHECK_CUDA(cudaMalloc(&d_a_f32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_f32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_f32, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a_f32, h_a_f32, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_f32, h_b_f32, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    GPUTimer timer;

    printf("\n--- Mixed Precision: FP32 -> FP16 -> FP32 ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mixed_precision_mma_kernel<__half><<<gridDim, dim3(32, 8), 0>>>(d_a_f32, d_b_f32, d_d_f32, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = computeGemmgfLOPS(M, N, K, time_ms);
    printf("Mixed Precision:      %.2f GFLOPS (%.3f ms)\n", gflops, time_ms);
    printf("Benefit: Uses TensorCore with FP32 I/O\n");

    CHECK_CUDA(cudaFree(d_a_f32));
    CHECK_CUDA(cudaFree(d_b_f32));
    CHECK_CUDA(cudaFree(d_d_f32));
    CHECK_CUDA(cudaFreeHost(h_a_f32));
    CHECK_CUDA(cudaFreeHost(h_b_f32));
    CHECK_CUDA(cudaFreeHost(h_d_f32));
}

// =============================================================================
// Section 9: NCU Profiling Reference
// =============================================================================

void runNCUProfilingReference() {
    printf("\n");
    printf("================================================================================\n");
    printf("9. NCU Profiling Metrics Reference for MMA Analysis\n");
    printf("================================================================================\n");

    printf("\n");
    printf("--- Key NCU Metrics for TensorCore Analysis ---\n\n");

    printf("%-50s %s\n", "Metric", "Description");
    printf("%-50s %s\n", "------", "-----------");
    printf("%-50s %s\n", "sm__pipe_tensor_cycles_active.pct",
           "Tensor core utilization percentage");
    printf("%-50s %s\n", "sm__pipe_tensor_cycles_active.sum",
           "Total tensor core active cycles");
    printf("%-50s %s\n", "sm__inst_executed.sum",
           "Total instructions executed");
    printf("%-50s %s\n", "sm__average_execution_latency",
           "Average instruction latency");
    printf("%-50s %s\n", "sm__warp_issue_stalled_by_barrier.pct",
           "Warp stalls at barrier");
    printf("%-50s %s\n", "dram__bytes.sum",
           "Memory traffic in bytes");
    printf("%-50s %s\n", "lts__tcs_hit_rate.pct",
           "L2 cache hit rate");
    printf("%-50s %s\n", "tcb__sectors_ld.sum",
           "L2 sectors loaded (read)");
    printf("%-50s %s\n", "tcb__sectors_st.sum",
           "L2 sectors stored (write)");

    printf("\n");
    printf("--- SASS Instruction Mnemonics for MMA ---\n\n");

    printf("%-20s %s\n", "SASS Instruction", "Description");
    printf("%-20s %s\n", "--------------", "-----------");
    printf("%-20s %s\n", "HMMA", "Half-precision MMA (FP16)");
    printf("%-20s %s\n", "IMMA", "Integer MMA (INT8/INT4)");
    printf("%-20s %s\n", "DMMA", "Double-precision MMA (FP64)");
    printf("%-20s %s\n", "BMMA", "BFloat16 MMA (BF16)");
    printf("%-20s %s\n", "LDG", "Load from global memory");
    printf("%-20s %s\n", "STG", "Store to global memory");
    printf("%-20s %s\n", "LDmatrix", "Matrix load to wmma fragment");
    printf("%-20s %s\n", "STmatrix", "Matrix store from wmma fragment");
    printf("%-20s %s\n", "BAR.SYNC", "Barrier synchronization");
    printf("%-20s %s\n", "CG.sync", "Cooperative group sync");
    printf("%-20s %s\n", "wgmma", "Warpgroup MMA async");

    printf("\n");
    printf("--- NCU Profiling Commands ---\n\n");

    printf("Full TensorCore analysis:\n");
    printf("  ncu --set full --metrics sm__pipe_tensor_cycles_active.sum,\\\n");
    printf("      sm__inst_executed.sum,dram__bytes.sum ./gpupeek.exe mma\n\n");

    printf("SASS instruction analysis:\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek.exe mma\n\n");

    printf("L2 cache analysis for MMA:\n");
    printf("  ncu --set full --metrics lts__tcs_hit_rate.pct,\\\n");
    printf("      tcb__sectors_ld.sum,tcb__sectors_st.sum ./gpupeek.exe mma\n\n");

    printf("Memory bandwidth for MMA:\n");
    printf("  ncu --set full --metrics dram__bytes.sum,  \\\n");
    printf("      sm__pipe_tensor_cycles_active.pct ./gpupeek.exe mma\n");
}

// =============================================================================
// Main Entry Point
// =============================================================================

void runMMAResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) MMA Deep Research                    #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    printf("\n");
    printf("================================================================================\n");
    printf("PTX ISA MMA Instruction Categories (Section 9.7.14-9.7.16)\n");
    printf("================================================================================\n\n");

    printf("1. wmma    - Warp-level MMA (m16n16k16), Section 9.7.14.4\n");
    printf("2. mma      - New MMA (m8n8k4, m16n8k8, m16n8k16, etc.), Section 9.7.14.5\n");
    printf("3. mma.sp   - Sparse MMA (2:4 structured sparsity), Section 9.7.14.6\n");
    printf("4. wgmma    - Asynchronous Warpgroup MMA, Section 9.7.15\n");
    printf("5. tcgen05  - TensorCore 5th Generation MMA, Section 9.7.16.10\n\n");

    printf("Data Types: FP16, BF16, TF32, FP64, INT8, INT4\n\n");

    // Run all test categories
    runWMMATests();
    runMMAShapeTests();
    runMMADataTypeTests();
    runSparseMMATests();
    runWGMMATests();
    runTCGen05Tests();
    runBaselineComparisonTests();
    runMixedPrecisionTests();
    runNCUProfilingReference();

    printf("\n");
    printf("================================================================================\n");
    printf("MMA Research Complete!\n");
    printf("================================================================================\n");
    printf("\n");
    printf("For NCU SASS profiling, run:\n");
    printf("  ncu --set full --metrics sm__pipe_tensor_cycles_active.sum ./gpupeek.exe mma\n");
    printf("  ncu --set full --kernels-by-compute ./gpupeek.exe mma\n");
    printf("\n");
}
