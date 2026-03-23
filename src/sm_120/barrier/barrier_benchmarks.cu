#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "barrier_kernels.cu"

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
// C.1 __syncthreads() Overhead Measurement
// =============================================================================

void runSyncOverheadTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("C.1 __syncthreads() Overhead Measurement\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

    GPUTimer timer;

    // No sync (baseline)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        noSyncKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double noSyncTime = timer.elapsed_ms() / iterations;
    double noSyncBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Sync Overhead ---\n\n");
    printf("No Sync (baseline):     %.3f ms, %s\n",
           noSyncTime, formatBandwidth(noSyncBw));

    // Single sync per iteration
    timer.start();
    for (int i = 0; i < iterations; i++) {
        singleSyncKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double singleSyncTime = timer.elapsed_ms() / iterations;
    double singleSyncBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Single __syncthreads():  %.3f ms, %s\n",
           singleSyncTime, formatBandwidth(singleSyncBw));
    printf("Overhead per sync:      %.3f us\n", (singleSyncTime - noSyncTime) * 1000.0);

    // Multiple syncs per iteration
    timer.start();
    for (int i = 0; i < iterations; i++) {
        multiSyncKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double multiSyncTime = timer.elapsed_ms() / iterations;
    double multiSyncBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Multiple __syncthreads(): %.3f ms, %s\n",
           multiSyncTime, formatBandwidth(multiSyncBw));
    printf("2x sync overhead:        %.3f us\n", (multiSyncTime - noSyncTime) * 1000.0 / 2.0);

    CHECK_CUDA(cudaFree(d_data));

    printf("\nNote: __syncthreads() has overhead of ~1-5 microseconds depending on\n");
    printf("block size and GPU architecture. Minimize syncthreads when possible.\n");
}

// =============================================================================
// C.2 Barrier Stall Analysis
// =============================================================================

void runBarrierStallTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("C.2 Barrier Stall Analysis\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

    GPUTimer timer;

    // No stall case
    timer.start();
    for (int i = 0; i < iterations; i++) {
        barrierNoStallKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double noStallTime = timer.elapsed_ms() / iterations;
    double noStallBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Barrier Stall Patterns ---\n\n");
    printf("No Divergence (all reach barrier together): %.3f ms, %s\n",
           noStallTime, formatBandwidth(noStallBw));

    // Divergent case
    int *h_pred;
    CHECK_CUDA(cudaMallocHost(&h_pred, 256 * sizeof(int)));
    for (int i = 0; i < 256; i++) {
        h_pred[i] = (i % 2);  // 50% divergent
    }

    int *d_pred;
    CHECK_CUDA(cudaMalloc(&d_pred, 256 * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_pred, h_pred, 256 * sizeof(int), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        barrierDivergentKernel<<<numBlocks, blockSize>>>(d_data, N, d_pred);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double divergentTime = timer.elapsed_ms() / iterations;
    double divergentBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Divergent Paths (reach barrier at different times): %.3f ms, %s\n",
           divergentTime, formatBandwidth(divergentBw));
    printf("Stall Overhead: %.1f%%\n", (divergentTime / noStallTime - 1.0) * 100.0);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_pred));
    CHECK_CUDA(cudaFreeHost(h_pred));

    printf("\nNote: When threads in a warp take different paths and reach\n");
    printf("__syncthreads() at different times, warp issue stalls occur.\n");
}

// =============================================================================
// C.3 Block Size vs Barrier Efficiency
// =============================================================================

void runBlockSizeBarrierEfficiencyTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("C.3 Block Size vs Barrier Efficiency\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    const int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

    printf("\n--- Block Size vs Sync Efficiency ---\n\n");

    printf("%-12s %-15s %-15s %-15s\n", "BlockSize", "Bandwidth", "Time/kernel", "Relative");
    printf("%-12s %-15s %-15s %-15s\n", "------------", "---------------", "---------------", "---------------");

    double baselineBw = 0;

    for (int b = 0; b < numBlockSizes; b++) {
        int blockSize = blockSizes[b];
        int numBlocks = (N + blockSize - 1) / blockSize;
        if (numBlocks > 65535) numBlocks = 65535;

        float *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

        GPUTimer timer;

        timer.start();
        for (int i = 0; i < iterations; i++) {
            simpleBarrierTest<<<numBlocks, blockSize>>>(d_data, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
        double time = timer.elapsed_ms() / iterations;

        if (b == 0) baselineBw = bw;

        printf("%-12d %-15s %-15.3f ms %-12.1f%%\n",
               blockSize, formatBandwidth(bw), time, bw / baselineBw * 100.0);

        CHECK_CUDA(cudaFree(d_data));
    }

    printf("\nNote: Larger blocks have fewer blocks, reducing synchronization overhead\n");
    printf("but may reduce occupancy. Optimal block size depends on kernel complexity.\n");
}

// =============================================================================
// C.4 Multi-Block Synchronization Patterns
// =============================================================================

void runMultiBlockSyncTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("C.4 Multi-Block Synchronization (Flag-Based Spin-Wait)\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 10;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    int *d_flag;
    CHECK_CUDA(cudaMalloc(&d_flag, sizeof(int)));

    GPUTimer timer;

    // Efficient pattern: No inter-block sync
    CHECK_CUDA(cudaMemset(d_dst, 0, N * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        noGridSyncKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double efficientTime = timer.elapsed_ms() / iterations;
    double efficientBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Multi-Block Sync Patterns ---\n\n");
    printf("No Grid Sync (efficient):     %.3f ms, %s\n",
           efficientTime, formatBandwidth(efficientBw));

    // Inefficient pattern: Spin-wait for other blocks
    CHECK_CUDA(cudaMemset(d_flag, 0, sizeof(int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemset(d_flag, 0, sizeof(int)));
        gridFlagSyncKernel<<<numBlocks, blockSize>>>(d_src, d_dst, d_flag, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double inefficientTime = timer.elapsed_ms() / iterations;
    double inefficientBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Flag-Based Spin-Wait (INEFFICIENT): %.3f ms, %s\n",
           inefficientTime, formatBandwidth(inefficientBw));
    printf("Slowdown from Spin-Wait: %.1fx\n", inefficientTime / efficientTime);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_flag));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nWARNING: Inter-block synchronization using spin-waits is EXTREMELY\n");
    printf("inefficient. Use separate kernel launches or CUDA streams instead.\n");
}

// =============================================================================
// C.5 Warp-Level Synchronization Primitives
// =============================================================================

void runWarpSyncPrimitiveTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("C.5 Warp-Level Synchronization Primitives\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, numBlocks * (blockSize / 32) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // Warp shuffle reduction (no barrier needed)
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpShuffleReductionKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double shuffleTime = timer.elapsed_ms() / iterations;
    double shuffleBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Warp-Level Primitives ---\n\n");
    printf("Warp Shuffle Reduction:  %.3f ms, %s\n",
           shuffleTime, formatBandwidth(shuffleBw));
    printf("(No __syncthreads() needed within warp)\n");

    // Warp vote/ballot test
    int *d_pred, *d_result;
    CHECK_CUDA(cudaMalloc(&d_pred, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_result, (N / 32) * 3 * sizeof(int)));

    for (size_t i = 0; i < N; i++) ((int*)h_src)[i] = (i % 3 == 0);
    CHECK_CUDA(cudaMemcpy(d_pred, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpVoteBallotKernel<<<numBlocks, blockSize>>>(d_pred, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double voteTime = timer.elapsed_ms() / iterations;
    double voteBw = N * sizeof(int) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Warp Vote/Ballot:        %.3f ms, %s\n",
           voteTime, formatBandwidth(voteBw));

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_pred));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: Warp-level primitives (__shfl, __any, __all, __ballot) do not\n");
    printf("require __syncthreads() as all threads in a warp execute synchronously.\n");
}

// =============================================================================
// C.6 CTA Cooperative Synchronization
// =============================================================================

void runCooperativeSyncTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("C.6 CTA Cooperative Synchronization\n");
    printf("================================================================================\n");

    printf("\n--- Cooperative Grid Sync ---\n\n");

    printf("Note: Cooperative grid synchronization requires:\n");
    printf("1. CUDA runtime support (cuda::thread_block, cuda::grid_sync)\n");
    printf("2. Kernel launch with cooperative dimension\n");
    printf("3. Sufficient SM resources\n\n");

    printf("Example launch:\n");
    printf("  cudaLaunchCooperativeKernel(kernel, dim3(blocks), dim3(threads), ..., sharedSize);\n\n");

    printf("This is typically used for:\n");
    printf("- Multi-block reductions\n");
    printf("- Wavefront-style algorithms\n");
    printf("- Algorithms requiring all blocks to complete a phase\n");

    // Note: Full implementation requires cooperative groups and specific launches
    printf("\nSkipping actual test - requires cooperative kernel launch setup.\n");
}

// =============================================================================
// Main Entry
// =============================================================================

void runBarrierBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) Barrier Synchronization Research     #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    runSyncOverheadTest();
    runBarrierStallTest();
    runBlockSizeBarrierEfficiencyTest();
    runMultiBlockSyncTest();
    runWarpSyncPrimitiveTest();
    runCooperativeSyncTest();

    printf("\n");
    printf("================================================================================\n");
    printf("Barrier Synchronization Research Complete!\n");
    printf("================================================================================\n");
}
