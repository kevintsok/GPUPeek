#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "../common/timer.h"
#include "fp4_fp6_research_kernel.cu"

// =============================================================================
// FP4/FP6 Low-Precision MMA Research Benchmarks
// =============================================================================
//
// Blackwell (SM 12.0) 5th-gen Tensor Core supports FP4 and FP6 formats.
//
// FP4 Format: e2m1 (2-bit exponent, 1-bit mantissa)
// FP6 Format: e2m3 or e3m2 (configurable exponent/mantissa)
//
// PTX ISA (CUDA 12.9+):
//   mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32   (FP4)
//   mma.sync.aligned.m16n8k32.row.col.f32.e2m3.e2m3.f32   (FP6 e2m3)
//   mma.sync.aligned.m16n8k32.row.col.f32.e3m2.e3m2.f32   (FP6 e3m2)
//
// Shape: m16n8k32 (different from FP8's m16n8k16)
//
// Use cases:
// - LLM quantization (4-bit weights)
// - Inference acceleration
// - Extreme quantization
// =============================================================================

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)
#endif

// =============================================================================
// FP4/FP6 Conversion Tests
// =============================================================================

static void runConversionTests(size_t N) {
    printf("\n--- FP4/FP6 Conversion Tests ---\n");

    size_t bytes = N * sizeof(float);
    size_t quant_bytes = N * sizeof(unsigned char);

    float *d_input = nullptr;
    unsigned char *d_fp4 = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_fp4, quant_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // Initialize input with test values
    float* h_input = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    for (size_t i = 0; i < N; i++) {
        h_input[i] = static_cast<float>((i % 100) / 10.0f);
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_input));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 1: FP32 to FP4 conversion
    printf("\n[Test 1] FP32 to FP4 Conversion:\n");
    timer.start();
    floatToFp4Kernel<float><<<numBlocks, blockSize>>>(d_input, d_fp4, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  FP4 conversion: %.3f ms for %zu elements\n", timer.elapsed_ms(), N);

    // Test 2: FP4 to FP32 conversion
    printf("\n[Test 2] FP4 to FP32 Conversion:\n");
    timer.start();
    fp4ToFloatKernel<float><<<numBlocks, blockSize>>>(d_fp4, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  FP4 deconversion: %.3f ms for %zu elements\n", timer.elapsed_ms(), N);

    // Verify some values
    float* h_output = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_output, bytes));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("  Sample values: %.3f, %.3f, %.3f\n",
           h_output[0], h_output[10], h_output[50]);
    CHECK_CUDA(cudaFreeHost(h_output));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_fp4));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// FP4/FP6 GEMM Simulation Tests
// =============================================================================

static void runFpStyleMmaTests(size_t M, size_t N, size_t K) {
    printf("\n--- FP4/FP6 Style MMA Tests ---\n");

    size_t size_a = M * K * sizeof(__half);
    size_t size_b = K * N * sizeof(__half);
    size_t size_c = M * N * sizeof(float);

    __half *d_A = nullptr, *d_B = nullptr;
    float *d_C_fp4 = nullptr, *d_C_fp6 = nullptr, *d_C_fp16 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, size_a));
    CHECK_CUDA(cudaMalloc(&d_B, size_b));
    CHECK_CUDA(cudaMalloc(&d_C_fp4, size_c));
    CHECK_CUDA(cudaMalloc(&d_C_fp6, size_c));
    CHECK_CUDA(cudaMalloc(&d_C_fp16, size_c));

    // Initialize with test data
    __half* h_A = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_A, size_a));
    __half* h_B = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_B, size_b));

    for (size_t i = 0; i < M * K; i++) {
        h_A[i] = __float2half(static_cast<float>((i % 7) + 1) / 7.0f);
    }
    for (size_t i = 0; i < K * N; i++) {
        h_B[i] = __float2half(static_cast<float>((i % 5) + 1) / 5.0f);
    }

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    GPUTimer timer;
    const int iterations = 10;

    // Test 3: FP16 baseline GEMM
    printf("\n[Test 3] FP16 GEMM Baseline:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp16GemmKernel<__half><<<gridDim, blockDim>>>(d_A, d_B, (__half*)d_C_fp16, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    float gflops_fp16 = (2.0f * M * N * K * iterations) / (timer.elapsed_ms() * 1e6);
    printf("  FP16 GEMM: %.2f GFLOPS (%.3f ms)\n", gflops_fp16, timer.elapsed_ms() / iterations);

    // Test 4: FP4-style GEMM (simulated)
    printf("\n[Test 4] FP4-Style GEMM (Simulated):\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp4StyleMmaKernel<__half><<<gridDim, blockDim>>>(d_A, d_B, d_C_fp4, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    float gflops_fp4 = (2.0f * M * N * K * iterations) / (timer.elapsed_ms() * 1e6);
    printf("  FP4-style GEMM: %.2f GFLOPS (%.3f ms)\n", gflops_fp4, timer.elapsed_ms() / iterations);

    // Test 5: FP6-style GEMM (simulated)
    printf("\n[Test 5] FP6-Style GEMM (Simulated):\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp6StyleMmaKernel<__half><<<gridDim, blockDim>>>(d_A, d_B, d_C_fp6, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    float gflops_fp6 = (2.0f * M * N * K * iterations) / (timer.elapsed_ms() * 1e6);
    printf("  FP6-style GEMM: %.2f GFLOPS (%.3f ms)\n", gflops_fp6, timer.elapsed_ms() / iterations);

    // Speedup vs FP16
    printf("\n  Speedup vs FP16:\n");
    printf("    FP4-style: %.2fx\n", gflops_fp4 / gflops_fp16);
    printf("    FP6-style: %.2fx\n", gflops_fp6 / gflops_fp16);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_fp4));
    CHECK_CUDA(cudaFree(d_C_fp6));
    CHECK_CUDA(cudaFree(d_C_fp16));
}

// =============================================================================
// Block Scaling Tests
// =============================================================================

static void runBlockScalingTests(size_t N) {
    printf("\n--- Block Scaling Tests ---\n");

    size_t bytes = N * sizeof(__half);
    size_t block_size = 32;
    size_t num_blocks = (N + block_size - 1) / block_size;
    size_t scale_bytes = num_blocks * sizeof(float);

    __half *d_input = nullptr, *d_output = nullptr;
    float *d_scales = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_scales, scale_bytes));

    // Initialize input
    __half* h_input = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    for (size_t i = 0; i < N; i++) {
        h_input[i] = __float2half(static_cast<float>(i % 10) / 10.0f);
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_input));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 6: Block scaling
    printf("\n[Test 6] Block Scaling:\n");
    timer.start();
    blockScalingKernel<__half><<<numBlocks, blockSize>>>(d_input, d_output, d_scales, N, block_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Block scaling: %.3f ms for %zu elements\n", timer.elapsed_ms(), N);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_scales));
}

// =============================================================================
// Weight-Only Quantization Tests
// =============================================================================

static void runWeightOnlyQuantTests(size_t N) {
    printf("\n--- Weight-Only Quantization Tests ---\n");

    size_t bytes = N * sizeof(float);
    size_t block_size = 64;
    size_t num_blocks = (N + block_size - 1) / block_size;
    size_t quant_bytes = N * sizeof(unsigned char);
    size_t scale_bytes = num_blocks * sizeof(float);

    float *d_weights = nullptr;
    unsigned char *d_quantized = nullptr;
    float *d_scales = nullptr;
    float *d_dequantized = nullptr;

    CHECK_CUDA(cudaMalloc(&d_weights, bytes));
    CHECK_CUDA(cudaMalloc(&d_quantized, quant_bytes));
    CHECK_CUDA(cudaMalloc(&d_scales, scale_bytes));
    CHECK_CUDA(cudaMalloc(&d_dequantized, bytes));

    // Initialize with random weights
    float* h_weights = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_weights, bytes));
    for (size_t i = 0; i < N; i++) {
        h_weights[i] = static_cast<float>((i % 100) - 50) / 10.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_weights));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 7: Weight-only quantization
    printf("\n[Test 7] Weight-Only Quantization:\n");
    timer.start();
    weightOnlyQuantKernel<float><<<numBlocks, blockSize>>>(d_weights, d_quantized, d_scales, N, block_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Quantization: %.3f ms for %zu elements\n", timer.elapsed_ms(), N);

    // Test 8: Dequantization
    printf("\n[Test 8] Dequantization:\n");
    timer.start();
    dequantizeKernel<float><<<numBlocks, blockSize>>>(d_quantized, d_scales, d_dequantized, N, block_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Dequantization: %.3f ms for %zu elements\n", timer.elapsed_ms(), N);

    // Verify some values
    float* h_original = nullptr;
    float* h_dequantized = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_original, bytes));
    CHECK_CUDA(cudaMallocHost(&h_dequantized, bytes));
    CHECK_CUDA(cudaMemcpy(h_original, d_weights, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dequantized, d_dequantized, bytes, cudaMemcpyDeviceToHost));

    printf("  Sample comparison (original -> dequantized):\n");
    for (int i = 0; i < 5; i++) {
        printf("    [%d]: %.3f -> %.3f\n", i*10, h_original[i*10], h_dequantized[i*10]);
    }

    CHECK_CUDA(cudaFreeHost(h_original));
    CHECK_CUDA(cudaFreeHost(h_dequantized));

    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_quantized));
    CHECK_CUDA(cudaFree(d_scales));
    CHECK_CUDA(cudaFree(d_dequantized));
}

// =============================================================================
// LLM Inference Pattern Tests
// =============================================================================

static void runLLMInferenceTests(size_t seq_len, size_t head_dim) {
    printf("\n--- LLM Inference Pattern Tests ---\n");

    size_t q_size = seq_len * head_dim * sizeof(float);
    size_t k_size = seq_len * head_dim * sizeof(unsigned char);
    size_t k_scale_size = seq_len * sizeof(float);
    size_t att_size = seq_len * seq_len * sizeof(float);

    float *d_Q = nullptr;
    unsigned char *d_K_quant = nullptr;
    float *d_K_scales = nullptr;
    unsigned char *d_V_quant = nullptr;
    float *d_V_scales = nullptr;
    float *d_attention = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q, q_size));
    CHECK_CUDA(cudaMalloc(&d_K_quant, k_size));
    CHECK_CUDA(cudaMalloc(&d_K_scales, k_scale_size));
    CHECK_CUDA(cudaMalloc(&d_V_quant, k_size));
    CHECK_CUDA(cudaMalloc(&d_V_scales, k_scale_size));
    CHECK_CUDA(cudaMalloc(&d_attention, att_size));

    // Initialize Q with test data
    float* h_Q = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_Q, q_size));
    for (size_t i = 0; i < seq_len * head_dim; i++) {
        h_Q[i] = static_cast<float>(i % 10) / 10.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_Q));

    const int blockSize = 256;
    int numBlocks = (seq_len * seq_len + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 9: Quantized attention computation
    printf("\n[Test 9] Quantized Attention (FP4 K/V):\n");
    timer.start();
    quantizedAttentionKernel<float><<<numBlocks, blockSize>>>(
        d_Q, d_K_quant, d_K_scales, d_V_quant, d_V_scales, d_attention, seq_len, head_dim);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Quantized attention: %.3f ms for seq_len=%zu, head_dim=%zu\n",
           timer.elapsed_ms(), seq_len, head_dim);

    // Test 10: Softmax
    printf("\n[Test 10] Softmax:\n");
    timer.start();
    softmaxKernel<float><<<numBlocks, blockSize>>>(d_attention, seq_len);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Softmax: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K_quant));
    CHECK_CUDA(cudaFree(d_K_scales));
    CHECK_CUDA(cudaFree(d_V_quant));
    CHECK_CUDA(cudaFree(d_V_scales));
    CHECK_CUDA(cudaFree(d_attention));
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

void runFP4FP6ResearchBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("FP4/FP6 Low-Precision MMA Research Benchmarks\n");
    printf("========================================\n");
    printf("Blackwell 5th-gen Tensor Core supports:\n");
    printf("  - FP4 (e2m1): 2-bit exp, 1-bit mantissa\n");
    printf("  - FP6 (e2m3, e3m2): 6-bit formats\n");
    printf("  - Shape: m16n8k32\n");
    printf("========================================\n");

    // Get device info
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("Note: Actual FP4/FP6 MMA requires CUDA 12.9+ with .kind::f8f6f4\n");
    printf("========================================\n");

    // Run all test categories
    runConversionTests(N);
    runFpStyleMmaTests(256, 256, 128);
    runBlockScalingTests(N);
    runWeightOnlyQuantTests(N);
    runLLMInferenceTests(128, 64);

    printf("\n--- FP4/FP6 Research Complete ---\n");
    printf("NCU Profiling Hints:\n");
    printf("  ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe fp4\n");
    printf("  ncu --set full --metrics sm__inst_executed.fma.sum ./gpupeek.exe fp4\n");
}
