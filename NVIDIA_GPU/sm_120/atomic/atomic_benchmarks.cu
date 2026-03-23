#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "atomic_kernels.cu"

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
// B.1 Warp-Level Atomic Operations Test
// =============================================================================

void runWarpLevelAtomicTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.1 Warp-Level Atomic Operations\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 10;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_result;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // Warp-level atomic add
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicWarpLevelAdd<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("\n--- Warp-Level Atomic Add ---\n\n");
    printf("Warp-Level Reduced + Atomic:  %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    // Warp-level atomic min
    int *d_int_result;
    CHECK_CUDA(cudaMalloc(&d_int_result, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_int_result, 0x7fffffff, sizeof(int)));

    int *d_int_src;
    CHECK_CUDA(cudaMalloc(&d_int_src, N * sizeof(int)));
    for (size_t i = 0; i < N; i++) {
        ((int*)h_src)[i] = i % 1000;
    }
    CHECK_CUDA(cudaMemcpy(d_int_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicWarpLevelMin<<<numBlocks, blockSize>>>(d_int_src, d_int_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    bw = N * sizeof(int) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Warp-Level Atomic Min:        %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFree(d_int_src));
    CHECK_CUDA(cudaFree(d_int_result));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: Warp-level reduction reduces atomic contention by combining\n");
    printf("32 values into a single atomic operation per warp.\n");
}

// =============================================================================
// B.2 Block-Level Atomic Operations Test
// =============================================================================

void runBlockLevelAtomicTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.2 Block-Level Atomic Operations\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 10;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_result;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    GPUTimer timer;

    // Block-level reduction + single atomic
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicBlockLevelAdd<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("\n--- Block-Level Atomic ---\n\n");
    printf("Block-Reduced + Single Atomic:  %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    // Multiple atomics per block (for comparison)
    CHECK_CUDA(cudaMemset(d_result, 0, numBlocks * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicBlockLevelMultiAdd<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Block-Reduced + Multi-Atomic:   %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: Block-level reduction significantly reduces atomic contention.\n");
}

// =============================================================================
// B.3 Grid-Level Atomic Operations Test
// =============================================================================

void runGridLevelAtomicTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.3 Grid-Level Atomic Operations (High Contention)\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 10;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_result;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    GPUTimer timer;

    // Direct atomic (maximum contention)
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicGridDirectAdd<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double directBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("\n--- Grid-Level Atomic ---\n\n");
    printf("Direct Atomic (high contention):  %s (%.3f ms/kernel)\n",
           formatBandwidth(directBw), timer.elapsed_ms() / iterations);

    // Grid reduction with block-level atomics
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicGridReduction<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double reducedBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Block-Reduced + Grid Atomic:     %s (%.3f ms/kernel)\n",
           formatBandwidth(reducedBw), timer.elapsed_ms() / iterations);

    printf("Speedup from Reduction:           %.1fx\n", reducedBw / directBw);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nWarning: Direct atomic operations to the same location cause severe\n");
    printf("contention. Always reduce first, then do a single atomic when possible.\n");
}

// =============================================================================
// B.4 Atomic Operation Comparison Test
// =============================================================================

void runAtomicOperationComparisonTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.4 Atomic Operation Comparison (Add vs CAS vs Min/Max)\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 10;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_result;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    GPUTimer timer;

    // atomicAdd
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicOperationAdd<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double addBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("\n--- Atomic Operation Performance ---\n\n");
    printf("atomicAdd:    %s (%.3f ms/kernel)\n",
           formatBandwidth(addBw), timer.elapsed_ms() / iterations);

    // atomicCAS-based add
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicOperationCAS<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double casBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("atomicCAS:    %s (%.3f ms/kernel)\n",
           formatBandwidth(casBw), timer.elapsed_ms() / iterations);

    printf("CAS/Add Ratio: %.1fx (CAS is much slower)\n", casBw / addBw);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_result));

    // atomicMin and atomicMax
    int *d_int_src, *d_int_result;
    CHECK_CUDA(cudaMalloc(&d_int_src, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_int_result, sizeof(int)));
    for (size_t i = 0; i < N; i++) ((int*)h_src)[i] = rand() % 1000;

    CHECK_CUDA(cudaMemcpy(d_int_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_int_result, 0x7fffffff, sizeof(int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicOperationMin<<<numBlocks, blockSize>>>(d_int_src, d_int_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double minBw = N * sizeof(int) * iterations / (timer.elapsed_ms() * 1e6);
    printf("atomicMin:    %s (%.3f ms/kernel)\n",
           formatBandwidth(minBw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaMemset(d_int_result, 0, sizeof(int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicOperationMax<<<numBlocks, blockSize>>>(d_int_src, d_int_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double maxBw = N * sizeof(int) * iterations / (timer.elapsed_ms() * 1e6);
    printf("atomicMax:    %s (%.3f ms/kernel)\n",
           formatBandwidth(maxBw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_int_src));
    CHECK_CUDA(cudaFree(d_int_result));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: atomicCAS is significantly slower than atomicAdd as it\n");
    printf("requires retry loops on contention. Use atomicAdd when possible.\n");
}

// =============================================================================
// B.5 Atomic Data Type Comparison Test
// =============================================================================

void runAtomicDataTypeTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.5 Atomic Operations by Data Type\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 10;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_result;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    GPUTimer timer;

    // 32-bit atomic add
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicOperationAdd<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw32 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("\n--- Atomic by Data Type ---\n\n");
    printf("32-bit (float):    %s (%.3f ms/kernel)\n",
           formatBandwidth(bw32), timer.elapsed_ms() / iterations);

    // 64-bit atomic add
    double *d_double_src, *d_double_result;
    CHECK_CUDA(cudaMalloc(&d_double_src, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_double_result, sizeof(double)));
    for (size_t i = 0; i < N; i++) ((double*)h_src)[i] = 1.0;

    CHECK_CUDA(cudaMemcpy(d_double_src, h_src, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_double_result, 0, sizeof(double)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomic64Add<<<numBlocks, blockSize>>>(d_double_src, d_double_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw64 = N * sizeof(double) * iterations / (timer.elapsed_ms() * 1e6);
    printf("64-bit (double):   %s (%.3f ms/kernel)\n",
           formatBandwidth(bw64), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFree(d_double_src));
    CHECK_CUDA(cudaFree(d_double_result));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: 64-bit atomics may be slower than 32-bit on some architectures.\n");
}

// =============================================================================
// Main Entry
// =============================================================================

void runAtomicBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) Atomic Operations Research            #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    runWarpLevelAtomicTest();
    runBlockLevelAtomicTest();
    runGridLevelAtomicTest();
    runAtomicOperationComparisonTest();
    runAtomicDataTypeTest();

    printf("\n");
    printf("================================================================================\n");
    printf("Atomic Operations Research Complete!\n");
    printf("================================================================================\n");
}
