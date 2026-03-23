#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "advanced_research_kernel.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// Note: formatBandwidth is defined in memory_research_benchmarks.cu

const char* formatTime(double ms) {
    static char buf[32];
    if (ms >= 1000.0) {
        snprintf(buf, sizeof(buf), "%.2f s", ms / 1000.0);
    } else if (ms >= 1.0) {
        snprintf(buf, sizeof(buf), "%.2f ms", ms);
    } else if (ms >= 0.001) {
        snprintf(buf, sizeof(buf), "%.3f us", ms * 1000.0);
    } else {
        snprintf(buf, sizeof(buf), "%.3f ns", ms * 1000.0 * 1000.0);
    }
    return buf;
}

// =============================================================================
// 1. Occupancy Analysis
// =============================================================================

void runOccupancyAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. Occupancy Analysis - Block Size vs Performance\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;

    printf("\n--- Register Usage vs Performance ---\n");
    printf("Testing how different register usage affects performance.\n\n");

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    const int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

    printf("%-12s %-15s %-15s %-15s\n", "BlockSize", "Registers", "Bandwidth", "Time/kernel");
    printf("%-12s %-15s %-15s %-15s\n", "------------", "---------------", "---------------", "---------------");

    for (int b = 0; b < numBlockSizes; b++) {
        int blockSize = blockSizes[b];
        int numBlocks = (N + blockSize - 1) / blockSize;
        if (numBlocks > 65535) numBlocks = 65535;

        float *d_src, *d_dst;
        CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

        GPUTimer timer;

        // Low register kernel
        timer.start();
        for (int i = 0; i < iterations; i++) {
            lowRegisterKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        printf("%-12d %-15s %-15s %-15s\n", blockSize, "~4", formatBandwidth(bw), formatTime(timer.elapsed_ms() / iterations));

        CHECK_CUDA(cudaFree(d_src));
        CHECK_CUDA(cudaFree(d_dst));
    }

    printf("\n--- Shared Memory Usage Impact ---\n\n");

    for (int b = 0; b < numBlockSizes; b++) {
        int blockSize = blockSizes[b];
        int numBlocks = (N + blockSize - 1) / blockSize;
        if (numBlocks > 65535) numBlocks = 65535;

        float *d_src, *d_dst;
        CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

        GPUTimer timer;

        timer.start();
        for (int i = 0; i < iterations; i++) {
            sharedMemoryKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

        printf("BlockSize=%4d: %s (%s/kernel)\n", blockSize, formatBandwidth(bw), formatTime(timer.elapsed_ms() / iterations));

        CHECK_CUDA(cudaFree(d_src));
        CHECK_CUDA(cudaFree(d_dst));
    }

    printf("\nNote: Optimal blockSize balances occupancy, register pressure, and shared memory usage.\n");
}

// =============================================================================
// 3. Memory Clock and Theoretical Bandwidth
// =============================================================================

void runTheoreticalBandwidthAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. Memory Clock and Theoretical Bandwidth Analysis\n");
    printf("================================================================================\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("\n--- GPU Memory Information ---\n\n");
    printf("GPU:                      %s\n", prop.name);
    printf("Compute Capability:        %d.%d\n", prop.major, prop.minor);
    printf("Global Memory:            %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memory Bus Width:          %d bits\n", prop.memoryBusWidth);
    printf("L2 Cache Size:            %d bytes (%.2f MB)\n", prop.l2CacheSize, prop.l2CacheSize / (1024.0 * 1024.0));

    // RTX 5080 Laptop typical memory bandwidth: ~800 GB/s (estimated)
    // Using a typical value since memoryClockRate is deprecated
    double theoreticalBandwidthGB = 800.0;  // Estimated peak bandwidth

    printf("\n--- Theoretical vs Actual Bandwidth ---\n\n");
    printf("Est. Theoretical Bandwidth: %.2f GB/s\n", theoreticalBandwidthGB);

    // Run actual bandwidth test
    const int blockSize = 256;
    const size_t N = 16 * 1024 * 1024;  // 64 MB
    const int iterations = 100;

    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    GPUTimer timer;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        lowRegisterKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double actualBandwidth = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Measured Bandwidth:         %.2f GB/s\n", actualBandwidth);
    printf("Efficiency:                 %.1f%%\n", actualBandwidth / theoreticalBandwidthGB * 100.0);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// 4. PCIe Bandwidth Test
// =============================================================================

void runPCIeBandwidthTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("4. PCIe Bandwidth Test (Host-Device Transfer)\n");
    printf("================================================================================\n");

    const size_t N = 32 * 1024 * 1024;  // 128 MB
    const int iterations = 10;

    float *h_src, *h_dst;
    float *d_src, *d_dst;

    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    for (size_t i = 0; i < N; i++) {
        h_src[i] = (float)i;
    }

    GPUTimer timer;

    // Host to Device
    printf("\n--- Host to Device (H2D) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double h2d_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Pageable H2D:      %s (%.3f ms/transfer)\n", formatBandwidth(h2d_bw), timer.elapsed_ms() / iterations);

    // Pinned memory H2D
    float *h_pinned_src;
    CHECK_CUDA(cudaMallocHost(&h_pinned_src, N * sizeof(float)));
    memcpy(h_pinned_src, h_src, N * sizeof(float));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(d_src, h_pinned_src, N * sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double pinned_h2d_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Pinned H2D:        %s (%.3f ms/transfer)\n", formatBandwidth(pinned_h2d_bw), timer.elapsed_ms() / iterations);

    // Device to Host
    printf("\n--- Device to Host (D2H) ---\n\n");

    CHECK_CUDA(cudaMemset(d_dst, 0, N * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double d2h_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Pageable D2H:      %s (%.3f ms/transfer)\n", formatBandwidth(d2h_bw), timer.elapsed_ms() / iterations);

    float *h_pinned_dst;
    CHECK_CUDA(cudaMallocHost(&h_pinned_dst, N * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(h_pinned_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double pinned_d2h_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Pinned D2H:        %s (%.3f ms/transfer)\n", formatBandwidth(pinned_d2h_bw), timer.elapsed_ms() / iterations);

    // Device to Device (baseline)
    printf("\n--- Device to Device (D2D) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(d_dst, d_src, N * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double d2d_bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("D2D:               %s (%.3f ms/transfer)\n", formatBandwidth(d2d_bw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFreeHost(h_dst));
    CHECK_CUDA(cudaFreeHost(h_pinned_src));
    CHECK_CUDA(cudaFreeHost(h_pinned_dst));

    printf("\nNote: PCIe Gen4 x16 theoretical peak is ~32 GB/s (bidirectional).\n");
}

// =============================================================================
// 5. Bank Conflict Analysis
// =============================================================================

void runBankConflictAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("5. Bank Conflict Analysis (Shared Memory)\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;

    printf("\n--- Shared Memory Access Patterns ---\n\n");

    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128};
    const int numStrides = sizeof(strides) / sizeof(strides[0]);

    printf("%-12s %-15s %-15s\n", "Stride", "Bandwidth", "Relative");
    printf("%-12s %-15s %-15s\n", "------------", "---------------", "---------------");

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    // Baseline: sequential
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    GPUTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        sequentialSharedAccessKernel<float><<<numBlocks, blockSize>>>(d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double baseline = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("%-12s %-15s %-15s\n", "1 (baseline)", formatBandwidth(baseline), "100%");

    for (int s = 1; s < numStrides; s++) {
        int stride = strides[s];
        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedSharedAccessKernel<float><<<numBlocks, blockSize>>>(d_dst, N, stride);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        double relative = bw / baseline * 100.0;
        printf("%-12d %-15s %-12.1f%%\n", stride, formatBandwidth(bw), relative);
    }

    CHECK_CUDA(cudaFree(d_dst));

    printf("\nNote: Bank conflicts reduce effective shared memory bandwidth.\n");
    printf("RTX 5080 has 32 banks, each 4 bytes wide.\n");
}

// =============================================================================
// 6. Branch Divergence Analysis
// =============================================================================

void runBranchDivergenceAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("6. Branch Divergence Impact Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    int *h_pred;
    CHECK_CUDA(cudaMallocHost(&h_pred, 256 * sizeof(int)));
    for (int i = 0; i < 256; i++) {
        h_pred[i] = (i % 2);  // 50% branches taken
    }

    int *d_pred;
    CHECK_CUDA(cudaMalloc(&d_pred, 256 * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_pred, h_pred, 256 * sizeof(int), cudaMemcpyHostToDevice));

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    GPUTimer timer;

    // No divergence (all threads take same branch based on block)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        noDivergenceKernel<float><<<numBlocks, blockSize>>>(d_pred, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double noDivBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    // High divergence (threads within warp take different branches)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        highDivergenceKernel<float><<<numBlocks, blockSize>>>(d_pred, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double highDivBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Branch Divergence Cost ---\n\n");
    printf("No Divergence:     %s (%.3f ms/kernel)\n", formatBandwidth(noDivBw), timer.elapsed_ms() / iterations);
    printf("High Divergence:   %s (%.3f ms/kernel)\n", formatBandwidth(highDivBw), timer.elapsed_ms() / iterations);
    printf("Divergence Cost:   %.1f%% slowdown\n", (noDivBw / highDivBw - 1.0) * 100.0);

    CHECK_CUDA(cudaFree(d_pred));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_pred));

    printf("\nNote: Branch divergence causes warp to execute both paths serially.\n");
}

// =============================================================================
// 7. Atomic Operations Performance
// =============================================================================

void runAtomicOperationsTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("7. Atomic Operations Performance\n");
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
    CHECK_CUDA(cudaMalloc(&d_result, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // atomicAdd (global)
    CHECK_CUDA(cudaMemset(d_result, 0, numBlocks * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicAddKernel<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double atomicAddBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    // Block-level reduction then atomic
    CHECK_CUDA(cudaMemset(d_result, 0, numBlocks * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicAddBlockKernel<<<numBlocks, blockSize>>>(d_src, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double blockAtomicBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Atomic Operation Bandwidth ---\n\n");
    printf("atomicAdd (global):       %s\n", formatBandwidth(atomicAddBw));
    printf("Block reduction + atomic: %s\n", formatBandwidth(blockAtomicBw));
    printf("Speedup:                  %.1fx\n", atomicAddBw / blockAtomicBw);

    // atomicCAS test
    printf("\n--- atomicCAS (Compare-And-Swap) ---\n\n");

    unsigned int *d_atomic_data, *d_atomic_result;
    CHECK_CUDA(cudaMalloc(&d_atomic_data, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_atomic_result, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_atomic_data, 0, sizeof(unsigned int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        atomicCASKernel<<<numBlocks, blockSize>>>(d_atomic_data, d_atomic_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double casBw = N * sizeof(unsigned int) * iterations / (timer.elapsed_ms() * 1e6);
    printf("atomicCAS:                %s\n", formatBandwidth(casBw));

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFree(d_atomic_data));
    CHECK_CUDA(cudaFree(d_atomic_result));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: atomicAdd to same location serializes all atomic operations.\n");
}

// =============================================================================
// 8. Constant Memory Performance
// =============================================================================

void runConstantMemoryTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("8. Constant Memory Bandwidth\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    // Fill constant memory using cudaMemcpyToSymbol
    float h_const_data[4096];
    for (int i = 0; i < 4096; i++) {
        h_const_data[i] = (float)i;
    }
    CHECK_CUDA(cudaMemcpyToSymbol(const_data, h_const_data, sizeof(const_data)));

    float *d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

    GPUTimer timer;

    // Broadcast read (all threads read same value)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        constantMemoryKernel<float><<<numBlocks, blockSize>>>(d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double broadcastBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    // Strided read
    timer.start();
    for (int i = 0; i < iterations; i++) {
        constantMemoryStrideKernel<float><<<numBlocks, blockSize>>>(d_dst, N, 16);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double strideBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Constant Memory Access ---\n\n");
    printf("Broadcast (same location):  %s\n", formatBandwidth(broadcastBw));
    printf("Strided (16 locations):     %s\n", formatBandwidth(strideBw));

    CHECK_CUDA(cudaFree(d_dst));

    printf("\nNote: Constant memory is cached and broadcast to all threads in a warp.\n");
}

// =============================================================================
// 9. Instruction Latency Analysis
// =============================================================================

void runInstructionLatencyAnalysis() {
    printf("\n");
    printf("================================================================================\n");
    printf("9. Instruction Latency Analysis\n");
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
    timer.start();
    for (int i = 0; i < iterations; i++) {
        singleFMAKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double latencyBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    // Independent FMA (tests throughput)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        independentFMAKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double throughputBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- FMA Latency vs Throughput ---\n\n");
    printf("Dependent FMA (latency):    %s\n", formatBandwidth(latencyBw));
    printf("Independent FMA (throughput): %s\n", formatBandwidth(throughputBw));
    printf("Latency/Throughput Ratio:  %.1fx\n", throughputBw / latencyBw);

    CHECK_CUDA(cudaFree(d_data));

    printf("\nNote: Dependent operations are limited by instruction latency (~10 cycles for FMA).\n");
    printf("Independent operations can be overlapped by the GPU's many parallel units.\n");
}

// =============================================================================
// 10. Memory vs Compute Bound Analysis
// =============================================================================

void runMemoryVsComputeBoundTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("10. Memory Bound vs Compute Bound Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, N * sizeof(float)));

    GPUTimer timer;

    // Memory bound kernel
    timer.start();
    for (int i = 0; i < iterations; i++) {
        memoryBoundKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double memoryBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    // Compute bound kernel
    timer.start();
    for (int i = 0; i < iterations; i++) {
        computeBoundKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double computeBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Kernel Bound Analysis ---\n\n");
    printf("Memory Bound Kernel:   %s (%.3f ms/kernel)\n", formatBandwidth(memoryBw), timer.elapsed_ms() / iterations);
    printf("Compute Bound Kernel: %s (%.3f ms/kernel)\n", formatBandwidth(computeBw), timer.elapsed_ms() / iterations);

    double ratio = computeBw / memoryBw;
    if (ratio > 2.0) {
        printf("\nKernel is Memory Bound: compute can hide memory latency.\n");
    } else if (ratio < 0.5) {
        printf("\nKernel is Compute Bound: limited by arithmetic throughput.\n");
    } else {
        printf("\nKernel is Balanced: both memory and compute are limiting.\n");
    }

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// Main Entry
// =============================================================================

void runAdvancedResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) Advanced Research                     #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    // Note: Tensor Core WMMA requires specific hardware configuration
    // Skipping for now - will add in next session
    // runTensorCoreWMMATest();
    runOccupancyAnalysis();
    runTheoreticalBandwidthAnalysis();
    runPCIeBandwidthTest();
    runBankConflictAnalysis();
    runBranchDivergenceAnalysis();
    runAtomicOperationsTest();
    runConstantMemoryTest();
    // runInstructionLatencyAnalysis();
    // runMemoryVsComputeBoundTest();

    printf("\n");
    printf("================================================================================\n");
    printf("Advanced Research Complete!\n");
    printf("================================================================================\n");
}
