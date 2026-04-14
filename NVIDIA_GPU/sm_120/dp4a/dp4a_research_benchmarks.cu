#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "dp4a_research_kernel.cu"

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
// DP4A Research Benchmarks
// =============================================================================
//
// PTX ISA Section 9.7.1.23 - Integer Arithmetic: dp4a
//
// DP4A variants:
// - dp4a.s32.s8.s8   - signed INT8, signed 32-bit result
// - dp4a.u32.u8.u8   - unsigned INT8, unsigned 32-bit result
// - dp4a.s32.rmi.*   - with rounding mode
// - dp4a.s32.satfinite.* - with saturation
//
// SASS: DP4A instruction
//
// NCU Metrics:
// - sm__inst_executed.dp4a.sum - DP4A instruction count
// - sm__pipe_tensor_cycles_active.pct - Tensor/INT8 utilization
// =============================================================================

// =============================================================================
// Section 1: Basic DP4A Tests
// =============================================================================

void runDP4ABasicTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("1. DP4A Basic Tests (INT8 Dot Product of 4 Bytes)\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;  // 1M vectors
    const size_t bytes = N * 4 * sizeof(int8_t);
    const int iterations = 100;

    int8_t *d_a, *d_b;
    int32_t *d_result;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(int32_t)));

    // Initialize with random data
    int8_t* h_a = (int8_t*)malloc(bytes);
    int8_t* h_b = (int8_t*)malloc(bytes);
    for (size_t i = 0; i < N * 4; i++) {
        h_a[i] = (int8_t)(rand() % 256 - 128);  // -128 to 127
        h_b[i] = (int8_t)(rand() % 256 - 128);
    }
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    // Test 1: DP4A signed INT8 -> INT32
    printf("\n--- DP4A Signed INT8 -> INT32 ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_s32_kernel<int8_t><<<gridDim, blockDim>>>(d_a, d_b, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double gops = N * iterations / (timer.elapsed_ms() * 1e6);
    printf("DP4A S32:             %.2f GOPS (%.3f ms)\n", gops, timer.elapsed_ms() / iterations);

    // Test 2: DP4A unsigned INT8 -> UINT32
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_u32_kernel<uint8_t><<<gridDim, blockDim>>>(d_a, d_b, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    gops = N * iterations / (timer.elapsed_ms() * 1e6);
    printf("DP4A U32:             %.2f GOPS (%.3f ms)\n", gops, timer.elapsed_ms() / iterations);

    // Test 3: DP4A with saturation
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_satfinite_kernel<int8_t><<<gridDim, blockDim>>>(d_a, d_b, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    gops = N * iterations / (timer.elapsed_ms() * 1e6);
    printf("DP4A SatFinite:       %.2f GOPS (%.3f ms)\n", gops, timer.elapsed_ms() / iterations);

    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_result));
}

// =============================================================================
// Section 2: DP4A with Accumulation
// =============================================================================

void runDP4AAccumulationTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. DP4A Accumulation Tests\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;
    const size_t bytes = N * 4 * sizeof(int8_t);
    const int iterations = 100;

    int8_t *d_a, *d_b;
    int32_t *d_result;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(int32_t)));

    CHECK_CUDA(cudaMemset(d_result, 0, N * sizeof(int32_t)));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_accumulate_kernel<int8_t><<<gridDim, blockDim>>>(d_a, d_b, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double gops = N * iterations / (timer.elapsed_ms() * 1e6);
    printf("DP4A Accumulate:      %.2f GOPS (%.3f ms)\n", gops, timer.elapsed_ms() / iterations);

    // Batch processing
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_batch_kernel<int8_t><<<gridDim, blockDim>>>(d_a, d_b, d_result, N, 4);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    gops = N * 4 * iterations / (timer.elapsed_ms() * 1e6);
    printf("DP4A Batch (4):       %.2f GOPS (%.3f ms)\n", gops, timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_result));
}

// =============================================================================
// Section 3: Baseline Comparisons
// =============================================================================

void runBaselineComparisonTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. Baseline Comparisons (DP4A vs Other Approaches)\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;
    const size_t bytes_int8 = N * 4 * sizeof(int8_t);
    const size_t bytes_fp32 = N * 4 * sizeof(float);
    const int iterations = 100;

    int8_t *d_a_int8, *d_b_int8;
    float *d_a_fp32, *d_b_fp32;
    int32_t *d_result_i32;
    float *d_result_fp32;

    CHECK_CUDA(cudaMalloc(&d_a_int8, bytes_int8));
    CHECK_CUDA(cudaMalloc(&d_b_int8, bytes_int8));
    CHECK_CUDA(cudaMalloc(&d_a_fp32, bytes_fp32));
    CHECK_CUDA(cudaMalloc(&d_b_fp32, bytes_fp32));
    CHECK_CUDA(cudaMalloc(&d_result_i32, N * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_result_fp32, N * sizeof(float)));

    // Initialize
    int8_t* h_a_int8 = (int8_t*)malloc(bytes_int8);
    int8_t* h_b_int8 = (int8_t*)malloc(bytes_int8);
    for (size_t i = 0; i < N * 4; i++) {
        h_a_int8[i] = (int8_t)(rand() % 256 - 128);
        h_b_int8[i] = (int8_t)(rand() % 256 - 128);
    }
    CHECK_CUDA(cudaMemcpy(d_a_int8, h_a_int8, bytes_int8, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_int8, h_b_int8, bytes_int8, cudaMemcpyHostToDevice));

    // Convert to FP32
    float* h_a_fp32 = (float*)malloc(bytes_fp32);
    float* h_b_fp32 = (float*)malloc(bytes_fp32);
    for (size_t i = 0; i < N * 4; i++) {
        h_a_fp32[i] = (float)h_a_int8[i];
        h_b_fp32[i] = (float)h_b_int8[i];
    }
    CHECK_CUDA(cudaMemcpy(d_a_fp32, h_a_fp32, bytes_fp32, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_fp32, h_b_fp32, bytes_fp32, cudaMemcpyHostToDevice));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    // DP4A baseline
    printf("\n--- Performance Comparison ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_s32_kernel<int8_t><<<gridDim, blockDim>>>(d_a_int8, d_b_int8, d_result_i32, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("DP4A (INT8):          %.2f GOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    // Naive INT8 dot product
    timer.start();
    for (int i = 0; i < iterations; i++) {
        naive_dot4_kernel<int8_t><<<gridDim, blockDim>>>(d_a_int8, d_b_int8, d_result_i32, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Naive INT8 Dot4:      %.2f GOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    // FP32 MAD4
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp32_mad4_kernel<float><<<gridDim, blockDim>>>(d_a_fp32, d_b_fp32, d_result_fp32, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("FP32 MAD4:           %.2f GFLOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    // FP16 baseline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp16_dot4_kernel<__half><<<gridDim, blockDim>>>(
            (__half*)d_a_fp32, (__half*)d_b_fp32, d_result_fp32, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("FP16 Dot4:           %.2f GFLOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    // Packed INT8
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_packed_kernel<uint32_t><<<gridDim, blockDim>>>(
            (uint32_t*)d_a_int8, (uint32_t*)d_b_int8, d_result_i32, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("DP4A Packed (u32):   %.2f GOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    free(h_a_int8);
    free(h_b_int8);
    free(h_a_fp32);
    free(h_b_fp32);
    CHECK_CUDA(cudaFree(d_a_int8));
    CHECK_CUDA(cudaFree(d_b_int8));
    CHECK_CUDA(cudaFree(d_a_fp32));
    CHECK_CUDA(cudaFree(d_b_fp32));
    CHECK_CUDA(cudaFree(d_result_i32));
    CHECK_CUDA(cudaFree(d_result_fp32));
}

// =============================================================================
// Section 4: Quantized Inference Patterns
// =============================================================================

void runQuantizedInferenceTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("4. Quantized Inference Patterns\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;
    const size_t bytes = N * 4 * sizeof(int8_t);
    const int iterations = 100;

    int8_t *d_a, *d_b;
    float *d_scale_a, *d_scale_b, *d_result;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_scale_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scale_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(float)));

    // Initialize
    int8_t* h_a = (int8_t*)malloc(bytes);
    int8_t* h_b = (int8_t*)malloc(bytes);
    float* h_scale_a = (float*)malloc(N * sizeof(float));
    float* h_scale_b = (float*)malloc(N * sizeof(float));

    for (size_t i = 0; i < N * 4; i++) {
        h_a[i] = (int8_t)(rand() % 256 - 128);
        h_b[i] = (int8_t)(rand() % 256 - 128);
    }
    for (size_t i = 0; i < N; i++) {
        h_scale_a[i] = 0.1f + (rand() % 100) / 1000.0f;
        h_scale_b[i] = 0.1f + (rand() % 100) / 1000.0f;
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale_a, h_scale_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale_b, h_scale_b, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    // Quantized inference: INT8 -> DP4A -> Dequantize -> FP32
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_quantized_kernel<int8_t><<<gridDim, blockDim>>>(
            d_a, d_b, d_scale_a, d_scale_b, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("INT8 Quantized (DP4A): %.2f GOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    // Block scaling
    size_t block_dim = 32;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_block_scale_kernel<int8_t><<<gridDim, blockDim>>>(
            d_a, d_b, d_scale_a, d_result, N, block_dim);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("INT8 Block Scaling:   %.2f GOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    free(h_a);
    free(h_b);
    free(h_scale_a);
    free(h_scale_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_scale_a));
    CHECK_CUDA(cudaFree(d_scale_b));
    CHECK_CUDA(cudaFree(d_result));
}

// =============================================================================
// Section 5: Shared Memory and Warp Reduction
// =============================================================================

void runReductionTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("5. Shared Memory and Warp Reduction\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;
    const size_t bytes = N * 4 * sizeof(int8_t);
    const int iterations = 100;

    int8_t *d_a, *d_b;
    int32_t *d_result;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(int32_t)));

    int8_t* h_a = (int8_t*)malloc(bytes);
    int8_t* h_b = (int8_t*)malloc(bytes);
    for (size_t i = 0; i < N * 4; i++) {
        h_a[i] = (int8_t)(rand() % 256 - 128);
        h_b[i] = (int8_t)(rand() % 256 - 128);
    }
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    size_t block_size = 256;
    dim3 gridDim((N + block_size - 1) / block_size);
    dim3 blockDim(block_size);
    size_t shared_size = block_size * sizeof(int32_t);

    GPUTimer timer;

    // Shared memory reduction
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_shared_kernel<int8_t><<<gridDim, blockDim, shared_size>>>(
            d_a, d_b, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("DP4A Shared Reduce:   %.2f GOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    // Warp reduction
    dim3 gridDimWarp(N / 32);
    dim3 blockDimWarp(256);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        dp4a_warp_reduce_kernel<int8_t><<<gridDimWarp, blockDimWarp>>>(
            d_a, d_b, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("DP4A Warp Reduce:     %.2f GOPS (%.3f ms)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_result));
}

// =============================================================================
// Section 6: NCU Profiling Reference
// =============================================================================

void runNCUProfilingReference() {
    printf("\n");
    printf("================================================================================\n");
    printf("6. NCU Profiling Reference - DP4A\n");
    printf("================================================================================\n");

    printf("\n--- Key NCU Metrics for DP4A ---\n\n");

    printf("DP4A Instruction Count:\n");
    printf("  ncu --metrics sm__inst_executed.dp4a.sum ./gpupeek dp4a\n\n");

    printf("INT8 Pipeline Utilization:\n");
    printf("  ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek dp4a\n");
    printf("  (DP4A uses INT8/tensor pipeline)\n\n");

    printf("Memory Analysis:\n");
    printf("  ncu --metrics dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek dp4a\n\n");

    printf("--- SASS Instruction Reference ---\n\n");

    printf("| SASS | Description | PTX |\n");
    printf("|------|------------|-----|\n");
    printf("| DP4A | Dot Product 4 INT8 | dp4a.s32.s8.s8 |\n");
    printf("| DP4A | Dot Product 4 UINT8 | dp4a.u32.u8.u8 |\n");
    printf("| LEA | Address calculation | add |\n");
    printf("| IMAD | Integer multiply-add | mul.add |\n\n");

    printf("--- PTX Instruction Reference ---\n\n");

    printf("DP4A (Section 9.7.1.23):\n");
    printf("  dp4a.s32.s8.s8   Rd, Ra, Rb, Rc;  // Rd = Ra dot Rb + Rc\n");
    printf("  dp4a.u32.u8.u8   Rd, Ra, Rb, Rc;\n");
    printf("  dp4a.s32.rmi     Rd, Ra, Rb, Rc;  // with rounding\n");
    printf("  dp4a.s32.satfinite.s8.s8 Rd, Ra, Rb, Rc;\n\n");

    printf("--- Key Findings Guide ---\n\n");

    printf("1. DP4A Performance:\n");
    printf("   - Theoretical peak: 4 INT8 ops per instruction\n");
    printf("   - Throughput: ~4x higher than FP32 MAD\n");
    printf("   - Lower precision, higher throughput for inference\n\n");

    printf("2. Use Cases:\n");
    printf("   - INT8 quantization inference\n");
    printf("   - Neural network fully-connected layers\n");
    printf("   - Recommendation systems\n");
    printf("   - Image processing (pixel operations)\n\n");

    printf("3. Comparison with Tensor Core MMA:\n");
    printf("   - MMA is for matrix operations (16x16 tiles)\n");
    printf("   - DP4A is for vector dot products (4 elements)\n");
    printf("   - MMA requires more setup, DP4A is lightweight\n\n");

    printf("4. Optimization Tips:\n");
    printf("   - Pack 4 INT8 into 32-bit for vectorized loads\n");
    printf("   - Use warp shuffle for fast reductions\n");
    printf("   - Combine with quantization scales for inference\n");
}

// =============================================================================
// Main Entry Point
// =============================================================================

void runDP4AResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) DP4A Research                       #\n");
    printf("#           Dot Product of 4 Bytes (INT8)                                     #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    printf("\n");
    printf("================================================================================\n");
    printf("PTX ISA DP4A Instruction (Section 9.7.1.23)\n");
    printf("================================================================================\n\n");

    printf("DP4A performs: result = sum(a[i]*b[i]) for i=0..3\n\n");

    printf("Variants:\n");
    printf("  dp4a.s32.s8.s8   - Signed INT8, signed INT32 result\n");
    printf("  dp4a.u32.u8.u8   - Unsigned UINT8, unsigned UINT32 result\n");
    printf("  dp4a.s32.rmi     - With rounding mode\n");
    printf("  dp4a.s32.satfinite - With saturation\n\n");

    printf("SASS: DP4A instruction\n\n");

    runDP4ABasicTests();
    runDP4AAccumulationTests();
    runBaselineComparisonTests();
    runQuantizedInferenceTests();
    runReductionTests();
    runNCUProfilingReference();

    printf("\n");
    printf("================================================================================\n");
    printf("DP4A Research Complete!\n");
    printf("================================================================================\n");
    printf("\n");
    printf("For NCU profiling, run:\n");
    printf("  ncu --set full --metrics sm__inst_executed.dp4a.sum ./gpupeek.exe dp4a\n");
    printf("  ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe dp4a\n");
    printf("\n");
}
