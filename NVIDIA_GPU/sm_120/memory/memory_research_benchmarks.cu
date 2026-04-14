#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "memory_research_kernel.cu"
#include "../arch_kernels.cu"

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

// Helper function to format bytes
const char* formatBytes(size_t bytes) {
    static char buf[32];
    if (bytes >= 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
    } else if (bytes >= 1024) {
        snprintf(buf, sizeof(buf), "%.2f KB", bytes / 1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return buf;
}

// =============================================================================
// Topic 1: Global Memory Bandwidth vs Data Size Test
// =============================================================================

void runGlobalMemorySizeBandwidthTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 1: Global Memory Bandwidth vs Data Size Test\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const int iterations = 100;

    // Test different data sizes
    size_t testSizes[] = {
        1024,                      // 1 KB
        1024 * 64,                 // 64 KB
        1024 * 256,                // 256 KB
        1024 * 1024,               // 1 MB
        4 * 1024 * 1024,           // 4 MB
        16 * 1024 * 1024,          // 16 MB
        64 * 1024 * 1024,          // 64 MB
        128 * 1024 * 1024,         // 128 MB
        256 * 1024 * 1024,         // 256 MB
    };
    const int numSizes = sizeof(testSizes) / sizeof(testSizes[0]);

    printf("\nConfig: blockSize=%d, iterations=%d\n\n", blockSize, iterations);

    printf("%-12s %15s %15s %15s\n", "Data Size", "Seq Read", "Seq Write", "Read-Modify-Write");
    printf("%-12s %15s %15s %15s\n", "------------", "---------------", "---------------", "---------------");

    for (int s = 0; s < numSizes; s++) {
        size_t N = testSizes[s] / sizeof(float);
        size_t bytes = N * sizeof(float);

        int numBlocks = (N + blockSize - 1) / blockSize;
        if (numBlocks > 65535) numBlocks = 65535;

        float *d_src, *d_dst;
        CHECK_CUDA(cudaMalloc(&d_src, bytes));
        CHECK_CUDA(cudaMalloc(&d_dst, bytes));
        CHECK_CUDA(cudaMemset(d_src, 1, bytes));

        GPUTimer timer;

        // Sequential Read
        timer.start();
        for (int i = 0; i < iterations; i++) {
            sequentialReadGlobalKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double readBW = bytes * iterations / (timer.elapsed_ms() * 1e6);

        // Sequential Write
        timer.start();
        for (int i = 0; i < iterations; i++) {
            sequentialWriteGlobalKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double writeBW = bytes * iterations / (timer.elapsed_ms() * 1e6);

        // Read-Modify-Write
        timer.start();
        for (int i = 0; i < iterations; i++) {
            readModifyWriteGlobalKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double rmwBW = bytes * iterations / (timer.elapsed_ms() * 1e6);

        printf("%-12s %15s %15s %15s\n",
               formatBytes(bytes),
               formatBandwidth(readBW),
               formatBandwidth(writeBW),
               formatBandwidth(rmwBW));

        CHECK_CUDA(cudaFree(d_src));
        CHECK_CUDA(cudaFree(d_dst));
    }

    printf("\nNote: As data size increases, memory bandwidth typically reaches saturation point.\n");
    printf("Bandwidth should approach peak around 4MB and above.\n");
}

// =============================================================================
// Topic 2: Global -> L1 -> L2 Memory Hierarchy Bandwidth Test
// =============================================================================

void runMemoryHierarchyBandwidthTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 2: Global -> L1 -> L2 Memory Hierarchy Bandwidth Test\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;  // 4M elements (16 MB)
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    printf("\nConfig: N=%zu elements (%.2f MB), blockSize=%d, iterations=%d\n\n",
           N, N * sizeof(float) / (1024 * 1024), blockSize, iterations);

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    GPUTimer timer;

    // 1. Global Memory Direct Access (baseline)
    printf("--- Global Memory Access Patterns ---\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        sequentialReadGlobalKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Global Direct Read:  %s (%.3f ms/kernel)\n",
           formatBandwidth(N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6)),
           timer.elapsed_ms() / iterations);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        sequentialWriteGlobalKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Global Direct Write: %s (%.3f ms/kernel)\n",
           formatBandwidth(N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6)),
           timer.elapsed_ms() / iterations);

    // 2. Shared Memory (L1-like) Bandwidth Test
    printf("\n--- Shared Memory (L1) Access Pattern ---\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        sharedMemoryBandwidthKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Shared Memory Round-Trip: %s (%.3f ms/kernel)\n",
           formatBandwidth(N * sizeof(float) * iterations * 2 / (timer.elapsed_ms() * 1e6)),
           timer.elapsed_ms() / iterations);

    // 3. L2 Streaming Access - different strides
    printf("\n--- L2 Cache Streaming Access (Strided) ---\n");
    printf("%-12s %15s %15s\n", "Stride", "Bandwidth", "Time/kernel");
    printf("%-12s %15s %15s\n", "------------", "---------------", "---------------");

    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    const int numStrides = sizeof(strides) / sizeof(strides[0]);

    for (int i = 0; i < numStrides; i++) {
        int stride = strides[i];
        timer.start();
        for (int j = 0; j < iterations; j++) {
            l2StreamingAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double effectiveBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        printf("%-12d %15s %15.3f ms\n", stride, formatBandwidth(effectiveBW), timer.elapsed_ms() / iterations);
    }

    // 4. L2 Bypass using __ldg
    printf("\n--- L2 Cache Bypass (__ldg) ---\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        l2BypassAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("__ldg Bypass:        %s (%.3f ms/kernel)\n",
           formatBandwidth(N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6)),
           timer.elapsed_ms() / iterations);

    // 5. L1 Preference (register optimization)
    printf("\n--- L1 Preference (Register Optimization) ---\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        l1PreferenceKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("L1 Preference:        %s (%.3f ms/kernel)\n",
           formatBandwidth(N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6)),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nAnalysis:\n");
    printf("- Shared Memory has highest bandwidth as it's on-chip memory\n");
    printf("- L2 Streaming strided access reduces effective bandwidth\n");
    printf("- __ldg bypasses cache, beneficial for one-time access data\n");
}

// =============================================================================
// Topic 3: TMA Copy Test
// =============================================================================

void runTMACopyTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 3: TMA (Tensor Memory Accelerator) Copy Test\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const int iterations = 100;

    printf("\nTesting TMA-style copy performance (using async copy pattern)\n\n");

    // Test different data sizes
    size_t testSizes[] = {
        1024 * 64,                 // 64 KB
        1024 * 256,                // 256 KB
        1024 * 1024,               // 1 MB
        4 * 1024 * 1024,           // 4 MB
        16 * 1024 * 1024,          // 16 MB
        64 * 1024 * 1024,          // 64 MB
        128 * 1024 * 1024,         // 128 MB
    };
    const int numSizes = sizeof(testSizes) / sizeof(testSizes[0]);

    printf("%-12s %15s %15s %15s\n", "Data Size", "1D Copy", "cudaMemcpy", "Speedup");
    printf("%-12s %15s %15s %15s\n", "------------", "---------------", "---------------", "---------------");

    for (int s = 0; s < numSizes; s++) {
        size_t N = testSizes[s] / sizeof(float);
        size_t bytes = N * sizeof(float);

        int numBlocks = (N + blockSize - 1) / blockSize;
        if (numBlocks > 65535) numBlocks = 65535;

        float *d_src, *d_dst;
        CHECK_CUDA(cudaMalloc(&d_src, bytes));
        CHECK_CUDA(cudaMalloc(&d_dst, bytes));
        CHECK_CUDA(cudaMemset(d_src, 1, bytes));

        GPUTimer timer;
        double tmaBW, memcpyBW;

        // TMA-style 1D copy (using kernel)
        timer.start();
        for (int i = 0; i < iterations; i++) {
            tmaCopy1DKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        tmaBW = bytes * iterations / (timer.elapsed_ms() * 1e6);

        // cudaMemcpy (device to device)
        void *d_temp;
        CHECK_CUDA(cudaMalloc(&d_temp, bytes));

        timer.start();
        for (int i = 0; i < iterations; i++) {
            CHECK_CUDA(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        memcpyBW = bytes * iterations / (timer.elapsed_ms() * 1e6);

        double speedup = tmaBW / memcpyBW;

        printf("%-12s %15s %15s %15.2fx\n",
               formatBytes(bytes),
               formatBandwidth(tmaBW),
               formatBandwidth(memcpyBW),
               speedup);

        CHECK_CUDA(cudaFree(d_src));
        CHECK_CUDA(cudaFree(d_dst));
        CHECK_CUDA(cudaFree(d_temp));
    }

    // 2D TMA Copy Test
    printf("\n--- TMA 2D Block Copy Test ---\n");

    size_t rows = 1024;
    size_t cols = 1024;
    size_t src_stride = 2048;  // pitched source
    size_t dst_stride = 2048;  // pitched destination

    size_t bytes = rows * src_stride * sizeof(float);
    printf("\n2D Test: %zux%zu (pitch=%zu), %.2f MB\n\n", rows, cols, src_stride, bytes / (1024*1024));

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, rows * src_stride * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, rows * dst_stride * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, rows * src_stride * sizeof(float)));

    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((rows + blockDim2D.x - 1) / blockDim2D.x,
                   (cols + blockDim2D.y - 1) / blockDim2D.y);

    GPUTimer timer;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        tmaCopy2DKernel<float><<<gridDim2D, blockDim2D>>>(d_src, d_dst, rows, cols, src_stride, dst_stride);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double tma2dBW = bytes * iterations / (timer.elapsed_ms() * 1e6);

    printf("TMA 2D Copy Bandwidth: %s (%.3f ms/kernel)\n",
           formatBandwidth(tma2dBW), timer.elapsed_ms() / iterations);

    // Use cudaMemcpy2D for comparison
    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy2D(d_dst, dst_stride * sizeof(float),
                                d_src, src_stride * sizeof(float),
                                cols * sizeof(float), rows,
                                cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double memcpy2dBW = bytes * iterations / (timer.elapsed_ms() * 1e6);

    printf("cudaMemcpy2D Bandwidth: %s (%.3f ms/kernel)\n",
           formatBandwidth(memcpy2dBW), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nAnalysis:\n");
    printf("- TMA-style kernel copy vs cudaMemcpy comparison shows parallel copy advantages\n");
    printf("- 2D copy test verifies block memory access efficiency\n");
}

// =============================================================================
// Topic 4: Memory Access Pattern Impact on Performance
// =============================================================================

void runAccessPatternTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 4: Memory Access Pattern Impact on Performance\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;  // 4M elements (16 MB)
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    printf("\nConfig: N=%zu elements (%.2f MB)\n\n", N, N * sizeof(float) / (1024 * 1024));

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    GPUTimer timer;

    // 1. Sequential Access
    printf("--- Access Pattern Comparison ---\n");
    printf("%-20s %15s %15s\n", "Pattern", "Bandwidth", "Time/kernel");
    printf("%-20s %15s %15s\n", "--------------------", "---------------", "---------------");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        sequentialAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("%-20s %15s %15.3f ms\n", "Sequential", formatBandwidth(N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6)), timer.elapsed_ms() / iterations);

    // 2. Strided Access
    printf("\n--- Strided Access ---\n");
    printf("%-12s %15s %15s\n", "Stride", "Bandwidth", "Efficiency");
    printf("%-12s %15s %15s\n", "------------", "---------------", "---------------");

    int strides[] = {2, 4, 8, 16, 32, 64, 128, 256};
    const int numStrides = sizeof(strides) / sizeof(strides[0]);

    double seqBW = 0;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    seqBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    for (int i = 0; i < numStrides; i++) {
        int stride = strides[i];
        timer.start();
        for (int j = 0; j < iterations; j++) {
            stridedAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double strideBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        double ratio = strideBW / seqBW * 100;
        printf("%-12d %15s %15.1f%%\n", stride, formatBandwidth(strideBW), ratio);
    }

    // 3. Broadcast Write
    printf("\n--- Broadcast Write (All threads write same value) ---\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        broadcastWriteKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 1.0f);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Broadcast Write:     %s (%.3f ms/kernel)\n",
           formatBandwidth(N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6)),
           timer.elapsed_ms() / iterations);

    // 4. Reduction Pattern
    printf("\n--- Reduction Pattern (Read many, write few) ---\n");

    float *d_result;
    CHECK_CUDA(cudaMalloc(&d_result, numBlocks * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        reductionPatternKernel<float><<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double redReadBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    double redWriteBW = numBlocks * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Reduction (Read):     %s\n", formatBandwidth(redReadBW));
    printf("Reduction (Write):    %s\n", formatBandwidth(redWriteBW));

    CHECK_CUDA(cudaFree(d_result));

    // 5. Different Data Types
    printf("\n--- Different Data Type Bandwidth ---\n");
    printf("%-12s %15s %15s\n", "Type", "Size", "Bandwidth");
    printf("%-12s %15s %15s\n", "------------", "---------------", "---------------");

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    // Float
    size_t floatN = N;
    size_t floatBytes = floatN * sizeof(float);
    float *d_float_src, *d_float_dst;
    CHECK_CUDA(cudaMalloc(&d_float_src, floatBytes));
    CHECK_CUDA(cudaMalloc(&d_float_dst, floatBytes));
    CHECK_CUDA(cudaMemset(d_float_src, 1, floatBytes));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        typeBandwidthKernel<float><<<numBlocks, blockSize>>>(d_float_src, d_float_dst, floatN);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("%-12s %15zu B %15s\n", "float", sizeof(float), formatBandwidth(floatBytes * iterations / (timer.elapsed_ms() * 1e6)));

    CHECK_CUDA(cudaFree(d_float_src));
    CHECK_CUDA(cudaFree(d_float_dst));

    // Double
    size_t doubleN = N;
    size_t doubleBytes = doubleN * sizeof(double);
    double *d_double_src, *d_double_dst;
    CHECK_CUDA(cudaMalloc(&d_double_src, doubleBytes));
    CHECK_CUDA(cudaMalloc(&d_double_dst, doubleBytes));
    CHECK_CUDA(cudaMemset(d_double_src, 1, doubleBytes));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        typeBandwidthKernel<double><<<numBlocks, blockSize>>>(d_double_src, d_double_dst, doubleN);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("%-12s %15zu B %15s\n", "double", sizeof(double), formatBandwidth(doubleBytes * iterations / (timer.elapsed_ms() * 1e6)));

    CHECK_CUDA(cudaFree(d_double_src));
    CHECK_CUDA(cudaFree(d_double_dst));

    // Int
    size_t intN = N;
    size_t intBytes = intN * sizeof(int);
    int *d_int_src, *d_int_dst;
    CHECK_CUDA(cudaMalloc(&d_int_src, intBytes));
    CHECK_CUDA(cudaMalloc(&d_int_dst, intBytes));
    CHECK_CUDA(cudaMemset(d_int_src, 1, intBytes));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        typeBandwidthKernel<int><<<numBlocks, blockSize>>>(d_int_src, d_int_dst, intN);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("%-12s %15zu B %15s\n", "int", sizeof(int), formatBandwidth(intBytes * iterations / (timer.elapsed_ms() * 1e6)));

    CHECK_CUDA(cudaFree(d_int_src));
    CHECK_CUDA(cudaFree(d_int_dst));

    // Half (FP16)
    size_t halfN = N;
    size_t halfBytes = halfN * sizeof(__half);
    __half *d_half_src, *d_half_dst;
    CHECK_CUDA(cudaMalloc(&d_half_src, halfBytes));
    CHECK_CUDA(cudaMalloc(&d_half_dst, halfBytes));
    CHECK_CUDA(cudaMemset(d_half_src, 1, halfBytes));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        typeBandwidthKernel<__half><<<numBlocks, blockSize>>>(d_half_src, d_half_dst, halfN);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("%-12s %15zu B %15s\n", "half(FP16)", sizeof(__half), formatBandwidth(halfBytes * iterations / (timer.elapsed_ms() * 1e6)));

    CHECK_CUDA(cudaFree(d_half_src));
    CHECK_CUDA(cudaFree(d_half_dst));

    printf("\nAnalysis:\n");
    printf("- Sequential access achieves highest bandwidth\n");
    printf("- Strided access effective bandwidth drops sharply as stride increases\n");
    printf("- Different data types may have different bandwidth due to memory alignment\n");
}

// =============================================================================
// Main Entry Function
// =============================================================================

// =============================================================================
// Topic 5: Cache Line Size Effect Research
// =============================================================================

void runCacheLineEffectTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 5: Cache Line Size Effect Research\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;  // 16 MB
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    printf("\nResearch: How does access granularity affect effective bandwidth?\n");
    printf("CUDA L1 cache line: 32B, L2 cache line: 128B\n\n");

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    GPUTimer timer;

    printf("%-15s %15s %15s %15s\n", "Access Size", "Bandwidth", "Time/kernel", "Efficiency");
    printf("%-15s %15s %15s %15s\n", "---------------", "---------------", "---------------", "---------------");

    // 32B access
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cacheLine32BKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double bw32 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-15s %15s %15.3f ms %15s\n", "32B (L1 line)", formatBandwidth(bw32),
           timer.elapsed_ms() / iterations, "baseline");

    // 64B access
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cacheLine64BKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double bw64 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-15s %15s %15.3f ms %15.0f%%\n", "64B (2xL1)", formatBandwidth(bw64),
           timer.elapsed_ms() / iterations, bw64/bw32*100);

    // 128B access
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cacheLine128BKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double bw128 = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-15s %15s %15.3f ms %15.0f%%\n", "128B (L2 line)", formatBandwidth(bw128),
           timer.elapsed_ms() / iterations, bw128/bw32*100);

    // Misaligned access tests
    printf("\n--- Misaligned Access Impact ---\n");
    printf("%-15s %15s %15s %15s\n", "Offset", "Bandwidth", "Time/kernel", "vs Aligned");

    int offsets[] = {0, 4, 8, 16, 32, 64};
    for (int o = 0; o < 6; o++) {
        int offset = offsets[o];
        timer.start();
        for (int i = 0; i < iterations; i++) {
            misalignedAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, offset);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double bw_mis = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        const char* aligned = (offset == 0) ? "(aligned)" : "";
        printf("%-15d %15s %15.3f ms %15.1f%%%s\n", offset, formatBandwidth(bw_mis),
               timer.elapsed_ms() / iterations, bw_mis/bw32*100, aligned);
    }

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Finding:\n");
    printf("- Access granularity affects memory transaction efficiency\n");
    printf("- Misaligned access can reduce effective bandwidth\n");
}

// =============================================================================
// Topic 6: Read vs Write Asymmetry Research
// =============================================================================

void runReadWriteAsymmetryTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 6: Read vs Write Asymmetry Research\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;  // 16 MB
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    printf("\nResearch: Is read bandwidth truly different from write bandwidth?\n\n");

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

    GPUTimer timer;

    printf("%-25s %15s %15s\n", "Operation", "Bandwidth", "Time/kernel");
    printf("%-25s %15s %15s\n", "-------------------------", "---------------", "---------------");

    // Pure read
    timer.start();
    for (int i = 0; i < iterations; i++) {
        pureReadKernel<float><<<numBlocks, blockSize>>>(d_data, d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double readBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-25s %15s %15.3f ms\n", "Pure Read (accumulate)", formatBandwidth(readBW),
           timer.elapsed_ms() / iterations);

    // Pure write
    timer.start();
    for (int i = 0; i < iterations; i++) {
        pureWriteKernel<float><<<numBlocks, blockSize>>>(d_data, d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double writeBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-25s %15s %15.3f ms\n", "Pure Write (no read)", formatBandwidth(writeBW),
           timer.elapsed_ms() / iterations);

    printf("\n%25s Read/Write = %.2f%%\n", "Asymmetry Ratio:", readBW/writeBW*100);

    // RAW dependency
    printf("\n--- Read-After-Write Dependency ---\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        readAfterWriteKernel<float><<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double rawBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-25s %15s %15.3f ms\n", "RAW (in-place *2)", formatBandwidth(rawBW),
           timer.elapsed_ms() / iterations);

    // WAR (write after read)
    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        writeAfterReadKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double warBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-25s %15s %15.3f ms\n", "WAR (separate arrays)", formatBandwidth(warBW),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Finding:\n");
    printf("- Read vs Write asymmetry typically 3-5%% on modern GPUs\n");
    printf("- RAW dependencies prevent write-combining optimizations\n");
}

// =============================================================================
// Topic 7: Non-Temporal vs Cached Access Research
// =============================================================================

void runCachedVsNontemporalTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 7: Non-Temporal vs Cached Access Research\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;  // 16 MB
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    printf("\nResearch: When does non-temporal (write-combining) beat cached?\n\n");

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    GPUTimer timer;

    printf("%-25s %15s %15s\n", "Access Type", "Bandwidth", "Time/kernel");
    printf("%-25s %15s %15s\n", "-------------------------", "---------------", "---------------");

    // Cached read
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cachedReadKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double cachedBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-25s %15s %15.3f ms\n", "Cached Read (default)", formatBandwidth(cachedBW),
           timer.elapsed_ms() / iterations);

    // Write-combining write
    timer.start();
    for (int i = 0; i < iterations; i++) {
        writeCombiningWriteKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double wcBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-25s %15s %15.3f ms\n", "Write-Combining Write", formatBandwidth(wcBW),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Finding:\n");
    printf("- Write-combining benefits: one-time writes, large streaming data\n");
    printf("- Cached access benefits: data reuse, sequential reads\n");
}

// =============================================================================
// Topic 8: Memory Coalescing Effectiveness Research
// =============================================================================

void runCoalescingEffectivenessTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 8: Memory Coalescing Effectiveness Research\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;  // 16 MB
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    printf("\nResearch: How does thread arrangement affect memory coalescing?\n\n");

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    GPUTimer timer;

    printf("%-25s %15s %15s %15s\n", "Pattern", "Bandwidth", "Time/kernel", "Efficiency");
    printf("%-25s %15s %15s %15s\n", "-------------------------", "---------------", "---------------", "---------------");

    // Best case: coalesced
    timer.start();
    for (int i = 0; i < iterations; i++) {
        coalescedAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double coalBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-25s %15s %15.3f ms %15s\n", "Coalesced (best)", formatBandwidth(coalBW),
           timer.elapsed_ms() / iterations, "100%");

    // Uncoalesced with different strides
    printf("\n--- Uncoalesced Stride Impact ---\n");
    int strides[] = {2, 4, 8, 16, 32};
    for (int s = 0; s < 5; s++) {
        int stride = strides[s];
        timer.start();
        for (int i = 0; i < iterations; i++) {
            uncoalescedAccessKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double uncoalBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        printf("%-25d %15s %15.3f ms %15.0f%%\n", stride, formatBandwidth(uncoalBW),
               timer.elapsed_ms() / iterations, uncoalBW/coalBW*100);
    }

    // Half-warp divergence
    printf("\n--- Half-Warp Divergence Impact ---\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        halfWarpDivergenceKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double divBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-25s %15s %15.3f ms %15.0f%%\n", "Half-warp divergence", formatBandwidth(divBW),
           timer.elapsed_ms() / iterations, divBW/coalBW*100);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Finding:\n");
    printf("- Coalesced access: threads in warp access sequential addresses\n");
    printf("- Uncoalesced: strided access wastes memory transactions\n");
    printf("- Half-warp divergence splits warp, reducing efficiency\n");
}

// =============================================================================
// Topic 9: Software Prefetch Effectiveness Research
// =============================================================================

void runPrefetchEffectivenessTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("Topic 9: Software Prefetch Effectiveness Research\n");
    printf("================================================================================\n");

    const int blockSize = 256;
    const size_t N = 4 * 1024 * 1024;  // 16 MB
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    printf("\nResearch: Does software prefetch help hide memory latency?\n\n");

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    GPUTimer timer;

    printf("%-30s %15s %15s %15s\n", "Prefetch Distance", "Bandwidth", "Time/kernel", "Speedup");
    printf("%-30s %15s %15s %15s\n", "------------------------------", "---------------", "---------------", "---------------");

    // Baseline (no prefetch)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cachedReadKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double baseBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-30s %15s %15.3f ms %15s\n", "No Prefetch (baseline)", formatBandwidth(baseBW),
           timer.elapsed_ms() / iterations, "1.00x");

    // Prefetch distances
    int distances[] = {32, 64, 128, 256, 512};
    for (int d = 0; d < 5; d++) {
        int prefetch_dist = distances[d];
        timer.start();
        for (int i = 0; i < iterations; i++) {
            prefetchReadKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, prefetch_dist);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double prefetchBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        printf("%-30d %15s %15.3f ms %15.2fx\n", prefetch_dist, formatBandwidth(prefetchBW),
               timer.elapsed_ms() / iterations, prefetchBW/baseBW);
    }

    // Double-buffer pipeline
    printf("\n--- Double Buffer Pipeline ---\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        doubleBufferKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    double pipeBW = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-30s %15s %15.3f ms %15.2fx\n", "Double Buffer (2-stage)", formatBandwidth(pipeBW),
           timer.elapsed_ms() / iterations, pipeBW/baseBW);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("\nKey Finding:\n");
    printf("- Software prefetch can hide memory latency\n");
    printf("- Optimal prefetch distance depends on memory latency\n");
    printf("- Double-buffering enables producer-consumer overlap\n");
}

void runMemoryResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) Memory Deep Research                  #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    // Topic 1: Global Memory Bandwidth vs Data Size
    runGlobalMemorySizeBandwidthTest();

    // Topic 2: Global -> L1 -> L2 Hierarchy Bandwidth
    runMemoryHierarchyBandwidthTest();

    // Topic 3: TMA Copy Test
    runTMACopyTest();

    // Topic 4: Memory Access Pattern Impact
    runAccessPatternTest();

    // Topic 5: Cache Line Size Effect
    runCacheLineEffectTest();

    // Topic 6: Read vs Write Asymmetry
    runReadWriteAsymmetryTest();

    // Topic 7: Non-Temporal vs Cached Access
    runCachedVsNontemporalTest();

    // Topic 8: Memory Coalescing Effectiveness
    runCoalescingEffectivenessTest();

    // Topic 9: Software Prefetch Effectiveness
    runPrefetchEffectivenessTest();

    printf("\n");
    printf("================================================================================\n");
    printf("Memory Research Complete!\n");
    printf("================================================================================\n");
}
