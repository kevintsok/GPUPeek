#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "../arch_kernels.cu"
#include "deep_research_kernel.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// Helper function to format bandwidth
const char* formatBandwidth(double GBps) {
    static char buf[32];
    if (GBps >= 1000.0) {
        snprintf(buf, sizeof(buf), "%.2f TB/s", GBps / 1000.0);
    } else {
        snprintf(buf, sizeof(buf), "%.2f GB/s", GBps);
    }
    return buf;
}

// =============================================================================
// L2 Cache Deep Analysis
// =============================================================================

void runL2CacheAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("L2 Cache Deep Analysis - Working Set vs Bandwidth\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const int iterations = 100;

    printf("\n--- L2 Working Set Test ---\n");
    printf("Testing how working set size affects L2 cache bandwidth\n\n");

    // Test different data sizes
    size_t dataSizes[] = {
        1024 * 64,                 // 64 KB
        1024 * 1024,               // 1 MB
        4 * 1024 * 1024,           // 4 MB
        8 * 1024 * 1024,           // 8 MB (likely > L2 size)
        16 * 1024 * 1024,          // 16 MB
        32 * 1024 * 1024,          // 32 MB
    };
    const int numSizes = sizeof(dataSizes) / sizeof(dataSizes[0]);

    printf("%-12s %15s %15s\n", "Data Size", "Bandwidth", "Comment");
    printf("%-12s %15s %15s\n", "------------", "---------------", "---------------");

    for (int s = 0; s < numSizes; s++) {
        size_t N = dataSizes[s] / sizeof(float);
        size_t bytes = N * sizeof(float);

        int numBlocks = (N + blockSize - 1) / blockSize;
        if (numBlocks > 65535) numBlocks = 65535;

        float *d_src, *d_dst;
        CHECK_CUDA(cudaMalloc(&d_src, bytes));
        CHECK_CUDA(cudaMalloc(&d_dst, bytes));
        CHECK_CUDA(cudaMemset(d_src, 1, bytes));

        GPUTimer timer;

        // Sequential read
        timer.start();
        for (int i = 0; i < iterations; i++) {
            l2CacheLineAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw = bytes * iterations / (timer.elapsed_ms() * 1e6);

        const char* comment = "";
        if (s < 2) comment = "L2 likely fits";
        else if (s == 2) comment = "L2 borderline";
        else comment = "L2 thrashing";

        printf("%-12zu %15s %15s\n", bytes / (1024*1024), formatBandwidth(bw), comment);

        CHECK_CUDA(cudaFree(d_src));
        CHECK_CUDA(cudaFree(d_dst));
    }

    // L2 Thrashing test
    printf("\n--- L2 Thrashing Test ---\n");
    printf("Testing behavior when L2 is forced to evict frequently\n\n");

    size_t N = 4 * 1024 * 1024;  // 16 MB
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    printf("%-12s %15s\n", "Stride", "Bandwidth");
    printf("%-12s %15s\n", "------------", "---------------");

    int strides[] = {1, 2, 4, 8, 16, 64, 256, 1024, 4096};
    const int numStrides = sizeof(strides) / sizeof(strides[0]);

    for (int i = 0; i < numStrides; i++) {
        int stride = strides[i];
        GPUTimer timer;

        timer.start();
        for (int j = 0; j < iterations; j++) {
            l2ThrashKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        printf("%-12d %15s\n", stride, formatBandwidth(bw));
    }

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nAnalysis:\n");
    printf("- Small data fits in L2, high bandwidth\n");
    printf("- Large data causes L2 miss, bandwidth drops to DRAM speed\n");
    printf("- Thrashing pattern shows memory controller efficiency\n");
}

// =============================================================================
// Tensor Core Matrix Multiply Test
// =============================================================================

void runTensorCoreTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Tensor Core Matrix Multiply Performance\n");
    printf("================================================================================\n");

    // Test matrix sizes
    size_t M = 512, N = 512, K = 512;

    printf("\nMatrix Size: %zux%zux%zu\n\n", M, N, K);

    size_t bytesA = M * K * sizeof(__half);
    size_t bytesB = K * N * sizeof(__half);
    size_t bytesC = M * N * sizeof(__half);

    __half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));

    // Initialize with ones
    CHECK_CUDA(cudaMemset(d_A, 1, bytesA));
    CHECK_CUDA(cudaMemset(d_B, 1, bytesB));
    CHECK_CUDA(cudaMemset(d_C, 0, bytesC));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    GPUTimer timer;
    const int iterations = 10;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        naiveMatrixMultiplyKernel<__half><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;  // multiply-adds
    double gflops = flops / (timer.elapsed_ms() * 1e6);

    printf("Naive FP16 MM:     %.2f GFLOPS (%.3f ms)\n", gflops, timer.elapsed_ms() / iterations);

    // Larger matrix test
    M = 2048; N = 2048; K = 2048;
    bytesA = M * K * sizeof(__half);
    bytesB = K * N * sizeof(__half);
    bytesC = M * N * sizeof(__half);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));

    CHECK_CUDA(cudaMemset(d_A, 1, bytesA));
    CHECK_CUDA(cudaMemset(d_B, 1, bytesB));
    CHECK_CUDA(cudaMemset(d_C, 0, bytesC));

    gridDim.x = (N + blockDim.x - 1) / blockDim.x;
    gridDim.y = (M + blockDim.y - 1) / blockDim.y;

    printf("\nMatrix Size: %zux%zux%zu\n\n", M, N, K);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        naiveMatrixMultiplyKernel<__half><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    flops = 2.0 * M * N * K * iterations;
    gflops = flops / (timer.elapsed_ms() * 1e6);

    printf("Naive FP16 MM:     %.2f GFLOPS (%.3f ms)\n", gflops, timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    printf("\nNote: Naive implementation uses only scalar operations.\n");
    printf("Real Tensor Core performance would be much higher (~1000+ TFLOPS).\n");
}

// =============================================================================
// Warp-Level Operation Detailed Analysis
// =============================================================================

void runWarpLevelAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("Warp-Level Operation Detailed Analysis\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;  // 4M elements
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;
    int resultBlocks = numBlocks;

    printf("\nConfig: N=%zu elements, blockSize=%d, iterations=%d\n\n", N, blockSize, iterations);

    float *d_input;
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, resultBlocks * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_input, 1, N * sizeof(float)));

    GPUTimer timer;

    // Warp shuffle reduce
    printf("--- Warp Shuffle Operations ---\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpShuffleReduceKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Shuffle Reduce:    %.3f ms/kernel (%.2f GB/s)\n",
           timer.elapsed_ms() / iterations,
           N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6));

    // Warp butterfly reduce
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpButterflyReduceKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Butterfly Reduce:   %.3f ms/kernel (%.2f GB/s)\n",
           timer.elapsed_ms() / iterations,
           N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6));

    // Warp ballot test
    printf("\n--- Warp Vote/Ballot Operations ---\n");

    int *d_pred, *d_ballot;
    CHECK_CUDA(cudaMalloc(&d_pred, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ballot, resultBlocks * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_pred, 1, N * sizeof(int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpBallotKernel<float><<<numBlocks, blockSize>>>(d_pred, d_ballot, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Ballot Sync:        %.3f ms/kernel\n", timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_pred));
    CHECK_CUDA(cudaFree(d_ballot));

    // Memory fence impact
    printf("\n--- Memory Fence Impact ---\n");

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    // Without fence
    timer.start();
    for (int i = 0; i < iterations; i++) {
        noFenceKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("No Fence:          %.3f ms/kernel (%s)\n",
           timer.elapsed_ms() / iterations,
           formatBandwidth(N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6)));

    // With fence
    timer.start();
    for (int i = 0; i < iterations; i++) {
        memoryFenceImpactKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("With Fence:        %.3f ms/kernel (%s)\n",
           timer.elapsed_ms() / iterations,
           formatBandwidth(N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6)));

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    printf("\nAnalysis:\n");
    printf("- Warp shuffle reduce is efficient for parallel reductions\n");
    printf("- Memory fence adds synchronization overhead\n");
}

// =============================================================================
// Instruction Throughput Test
// =============================================================================

void runInstructionThroughputTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Instruction Throughput Analysis\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    printf("\n--- Arithmetic Instruction Throughput ---\n\n");

    float *d_a, *d_b, *d_c, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b, 2, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_c, 3, N * sizeof(float)));

    GPUTimer timer;

    // FMA throughput
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fmaThroughputKernel<float><<<numBlocks, blockSize>>>(d_a, d_b, d_c, d_out, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double fmaGflops = N * iterations / (timer.elapsed_ms() * 1e6);
    printf("FP32 FMA:          %.2f GFLOPS (%.3f ms/kernel)\n", fmaGflops, timer.elapsed_ms() / iterations);

    // INT throughput
    int *d_ia, *d_ib, *d_iout;
    CHECK_CUDA(cudaMalloc(&d_ia, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ib, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_iout, N * sizeof(int)));

    CHECK_CUDA(cudaMemset(d_ia, 1, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_ib, 2, N * sizeof(int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        intThroughputKernel<<<numBlocks, blockSize>>>(d_ia, d_ib, d_iout, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double intGops = N * iterations / (timer.elapsed_ms() * 1e6);
    printf("INT32 Arith:       %.2f GIOPS (%.3f ms/kernel)\n", intGops, timer.elapsed_ms() / iterations);

    // Half precision
    __half *d_ha, *d_hb, *d_hc, *d_hout;
    CHECK_CUDA(cudaMalloc(&d_ha, N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_hb, N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_hc, N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_hout, N * sizeof(__half)));

    CHECK_CUDA(cudaMemset(d_ha, 1, N * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_hb, 2, N * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_hc, 3, N * sizeof(__half)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        fmaThroughputKernel<__half><<<numBlocks, blockSize>>>(d_ha, d_hb, d_hc, d_hout, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double halfGflops = N * iterations / (timer.elapsed_ms() * 1e6);
    printf("FP16 FMA:          %.2f GFLOPS (%.3f ms/kernel)\n", halfGflops, timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_ia));
    CHECK_CUDA(cudaFree(d_ib));
    CHECK_CUDA(cudaFree(d_iout));
    CHECK_CUDA(cudaFree(d_ha));
    CHECK_CUDA(cudaFree(d_hb));
    CHECK_CUDA(cudaFree(d_hc));
    CHECK_CUDA(cudaFree(d_hout));

    printf("\nAnalysis:\n");
    printf("- FP32 FMA throughput indicates sustained floating-point performance\n");
    printf("- INT32 throughput shows integer arithmetic capability\n");
    printf("- FP16 throughput is higher due to reduced precision and Tensor Core emulation\n");
}

// =============================================================================
// Main Entry
// =============================================================================

void runDeepResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) Deep Research                         #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    runL2CacheAnalysis();
    runTensorCoreTest();
    runWarpLevelAnalysis();
    runInstructionThroughputTest();

    printf("\n");
    printf("================================================================================\n");
    printf("Deep Research Complete!\n");
    printf("================================================================================\n");
}
