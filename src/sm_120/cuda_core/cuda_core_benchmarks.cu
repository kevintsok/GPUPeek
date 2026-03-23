#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/timer.h"
#include "cuda_core_kernels.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

extern const char* formatBandwidth(double GBps);

// Helper function to format GFLOPS
const char* formatGFLOPS(double gflops) {
    static char buf[32];
    if (gflops >= 1000.0) {
        snprintf(buf, sizeof(buf), "%.2f TFOPS", gflops / 1000.0);
    } else {
        snprintf(buf, sizeof(buf), "%.2f GFOPS", gflops);
    }
    return buf;
}

// =============================================================================
// A.1 Data Type Throughput Test
// =============================================================================

void runDataTypeThroughputTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("A.1 Data Type Throughput\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("\n--- Arithmetic Throughput by Data Type ---\n\n");

    printf("%-12s %-12s %-15s %-15s\n", "Data Type", "Size", "Bandwidth", "Performance");
    printf("%-12s %-12s %-15s %-15s\n", "------------", "------------", "---------------", "---------------");

    // FP64 (Double)
    {
        double *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(double)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(double)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(double)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            fp64ArithmeticKernel<double><<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(double) * iterations / (timer.elapsed_ms() * 1e6);
        double flops = N * 2 * iterations / (timer.elapsed_ms() * 1e6);  // 2 ops per iteration
        printf("%-12s %-12zu %-15s %-15s\n", "FP64 (D)", sizeof(double), formatBandwidth(bw), formatGFLOPS(flops));

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // FP32 (Float)
    {
        float *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(float)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            fp32ArithmeticKernel<float><<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        double flops = N * 2 * iterations / (timer.elapsed_ms() * 1e6);
        printf("%-12s %-12zu %-15s %-15s\n", "FP32 (S)", sizeof(float), formatBandwidth(bw), formatGFLOPS(flops));

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // FP16 (Half)
    {
        __half *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(__half)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            fp16ArithmeticKernel<__half><<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(__half) * iterations / (timer.elapsed_ms() * 1e6);
        double flops = N * 2 * iterations / (timer.elapsed_ms() * 1e6);
        printf("%-12s %-12zu %-15s %-15s\n", "FP16 (H)", sizeof(__half), formatBandwidth(bw), formatGFLOPS(flops));

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // BF16 (BFloat16)
    {
        __nv_bfloat16 *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(__nv_bfloat16)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            bf16ArithmeticKernel<__nv_bfloat16><<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(__nv_bfloat16) * iterations / (timer.elapsed_ms() * 1e6);
        double flops = N * 2 * iterations / (timer.elapsed_ms() * 1e6);
        printf("%-12s %-12zu %-15s %-15s\n", "BF16 (B)", sizeof(__nv_bfloat16), formatBandwidth(bw), formatGFLOPS(flops));

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // INT8
    {
        char *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(char)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(char)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(char)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(char)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(char)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            int8ArithmeticKernel<char><<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(char) * iterations / (timer.elapsed_ms() * 1e6);
        double iops = N * 2 * iterations / (timer.elapsed_ms() * 1e6);
        printf("%-12s %-12zu %-15s %-15s\n", "INT8 (I1)", sizeof(char), formatBandwidth(bw), formatGFLOPS(iops));

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // INT32
    {
        int *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(int)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            int32ArithmeticKernel<int><<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(int) * iterations / (timer.elapsed_ms() * 1e6);
        double iops = N * 2 * iterations / (timer.elapsed_ms() * 1e6);
        printf("%-12s %-12zu %-15s %-15s\n", "INT32 (I4)", sizeof(int), formatBandwidth(bw), formatGFLOPS(iops));

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    printf("\nNote: RTX 5080 has dedicated FP64, FP32, FP16, and INT8 units.\n");
    printf("FP64 (double) typically runs at 1/64 or 1/32 of FP32 speed.\n");
}

// =============================================================================
// A.2 Instruction Latency vs Throughput Test
// =============================================================================

void runLatencyVsThroughputTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("A.2 Instruction Latency vs Throughput\n");
    printf("================================================================================\n");

    const size_t N = 1 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, (N + 2) * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, (N + 2) * sizeof(float)));

    GPUTimer timer;

    // Dependent FMA chain (tests latency)
    printf("\n--- FMA Latency vs Throughput ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        dependentFMAChainKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double latencyBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Dependent FMA (latency limited):  %s (%.3f ms/kernel)\n",
           formatBandwidth(latencyBw), timer.elapsed_ms() / iterations);

    // Independent FMA (tests throughput)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        independentFMAThroughputKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double throughputBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Independent FMA (throughput):      %s (%.3f ms/kernel)\n",
           formatBandwidth(throughputBw), timer.elapsed_ms() / iterations);

    double ratio = throughputBw / latencyBw;
    printf("Throughput/Latency Ratio:          %.1fx\n", ratio);

    CHECK_CUDA(cudaFree(d_data));

    printf("\nNote: Dependent operations are limited by instruction latency (~10 cycles for FMA).\n");
    printf("Independent operations can be overlapped by the GPU's many parallel units.\n");
}

// =============================================================================
// A.3 Vector Instruction Test
// =============================================================================

void runVectorInstructionTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("A.3 Vector Instruction Performance\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("\n--- Vector Operation Performance ---\n\n");

    // Float4 (128-bit) operations
    {
        float4 *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float4)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float4)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float4)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(float4)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(float4)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            float4Kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float4) * iterations / (timer.elapsed_ms() * 1e6);
        double flops = N * 4 * 2 * iterations / (timer.elapsed_ms() * 1e6);  // 4 elements, 2 ops each
        printf("float4 (128-bit):  %s, %.2f GFLOP/s\n", formatBandwidth(bw), flops);

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // Float2 (64-bit) operations
    {
        float2 *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float2)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(float2)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(float2)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            float2Kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float2) * iterations / (timer.elapsed_ms() * 1e6);
        double flops = N * 2 * 2 * iterations / (timer.elapsed_ms() * 1e6);  // 2 elements, 2 ops each
        printf("float2 (64-bit):   %s, %.2f GFLOP/s\n", formatBandwidth(bw), flops);

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // Scalar FP32 baseline
    {
        float *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(float)));

        GPUTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            fp32ArithmeticKernel<float><<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        double flops = N * 2 * iterations / (timer.elapsed_ms() * 1e6);
        printf("float (32-bit):    %s, %.2f GFLOP/s\n", formatBandwidth(bw), flops);

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    printf("\nNote: Vector instructions (float2, float4) can improve instruction throughput.\n");
}

// =============================================================================
// A.4 Transcendental Functions Test
// =============================================================================

void runTranscendentalFunctionsTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("A.4 Transcendental Functions (sin/cos/exp/log)\n");
    printf("================================================================================\n");

    const size_t N = 1 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

    GPUTimer timer;

    // sin/cos test
    timer.start();
    for (int i = 0; i < iterations; i++) {
        sinCosKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("sin/cos (8x per element):  %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    // exp/log test
    timer.start();
    for (int i = 0; i < iterations; i++) {
        expLogKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("exp/log (8x per element): %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    // Baseline FMA for comparison
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        fmaKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, d_a, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Baseline FMA:              %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    printf("\nNote: Transcendental functions (sin, cos, exp, log) have higher latency\n");
    printf("than simple arithmetic. Use __sinf, __cosf for faster approximations.\n");
}

// =============================================================================
// A.5 Mixed Precision Test
// =============================================================================

void runMixedPrecisionTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("A.5 Mixed Precision (FP32 -> FP16 -> FP32)\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_fma, *d_fmb, *d_fp16, *d_fp32;
    CHECK_CUDA(cudaMalloc(&d_fma, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fmb, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp16, N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_fp32, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_fma, 1, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_fmb, 2, N * sizeof(float)));

    GPUTimer timer;

    // Mixed precision (FP32 -> FP16 -> FP32)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        mixedPrecisionKernel<<<numBlocks, blockSize>>>(d_fma, d_fmb, d_fp16, d_fp32, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw = N * sizeof(float) * 2 * iterations / (timer.elapsed_ms() * 1e6);  // 2 reads + 1 write
    printf("Mixed Precision (FP32->FP16->FP32):  %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    // FP32 baseline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fmaKernel<<<numBlocks, blockSize>>>(d_fma, d_fmb, d_fma, d_fma, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("FP32 Baseline:                    %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_fma));
    CHECK_CUDA(cudaFree(d_fmb));
    CHECK_CUDA(cudaFree(d_fp16));
    CHECK_CUDA(cudaFree(d_fp32));

    printf("\nNote: Mixed precision uses Tensor Cores for FP16 math but FP32 accumulation.\n");
    printf("This can provide significant speedup for DL/ML workloads.\n");
}

// =============================================================================
// Main Entry
// =============================================================================

void runCudaCoreBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) CUDA Core Arithmetic Research         #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    runDataTypeThroughputTest();
    runLatencyVsThroughputTest();
    runVectorInstructionTest();
    runTranscendentalFunctionsTest();
    runMixedPrecisionTest();

    printf("\n");
    printf("================================================================================\n");
    printf("CUDA Core Arithmetic Research Complete!\n");
    printf("================================================================================\n");
}
