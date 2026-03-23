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
// B.7 bar.red Reduction Barrier Tests
// =============================================================================

void runBarRedTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.7 bar.red Reduction Barrier\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    // Prepare predicate array (50% true)
    int *h_pred;
    CHECK_CUDA(cudaMallocHost(&h_pred, N * sizeof(int)));
    for (size_t i = 0; i < N; i++) {
        h_pred[i] = (i % 2);
    }

    int *d_pred, *d_result;
    unsigned int *d_popc_result;

    CHECK_CUDA(cudaMalloc(&d_pred, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_result, numBlocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_popc_result, numBlocks * sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(d_pred, h_pred, N * sizeof(int), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // bar.red.popc test
    printf("\n--- bar.red.popc (Population Count Reduction) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        barRedPopcKernel<<<numBlocks, blockSize>>>(d_pred, d_popc_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double popcTime = timer.elapsed_ms() / iterations;
    printf("bar.red.popc: %.3f ms per kernel\n", popcTime);
    printf("Reduces predicate to thread count in one barrier instruction.\n");

    // bar.red.and test
    printf("\n--- bar.red.and (All-True Reduction) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        barRedAndKernel<<<numBlocks, blockSize>>>(d_pred, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double andTime = timer.elapsed_ms() / iterations;
    printf("bar.red.and: %.3f ms per kernel\n", andTime);
    printf("Returns 1 if ALL threads had predicate=true.\n");

    // bar.red.or test
    printf("\n--- bar.red.or (Any-True Reduction) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        barRedOrKernel<<<numBlocks, blockSize>>>(d_pred, d_result, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double orTime = timer.elapsed_ms() / iterations;
    printf("bar.red.or: %.3f ms per kernel\n", orTime);
    printf("Returns 1 if ANY thread had predicate=true.\n");

    printf("\nPTX Reference:\n");
    printf("  bar.red.popc.u32 Rd, Nt, Pd;  // count true predicates\n");
    printf("  bar.red.and.pred Rd, Nt, Pd;  // AND reduction\n");
    printf("  bar.red.or.pred  Rd, Nt, Pd;  // OR reduction\n");

    CHECK_CUDA(cudaFree(d_pred));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFree(d_popc_result));
    CHECK_CUDA(cudaFreeHost(h_pred));
}

// =============================================================================
// B.8 bar.arrive vs bar.sync vs bar.wait
// =============================================================================

void runArriveSyncWaitTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.8 bar.arrive vs bar.sync vs bar.wait\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

    GPUTimer timer;

    // bar.sync (blocking)
    printf("\n--- bar.sync (Blocking) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        barSyncBlockingKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double syncTime = timer.elapsed_ms() / iterations;
    double syncBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("bar.sync (blocking): %.3f ms, %s\n", syncTime, formatBandwidth(syncBw));
    printf("All threads arrive AND wait at barrier.\n");

    // bar.arrive + bar.wait (non-blocking)
    printf("\n--- bar.arrive + bar.wait (Non-blocking) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        barArriveWaitKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double arriveWaitTime = timer.elapsed_ms() / iterations;
    double arriveWaitBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("bar.arrive + bar.wait: %.3f ms, %s\n", arriveWaitTime, formatBandwidth(arriveWaitBw));
    printf("Threads arrive (decrement counter) without waiting, then wait later.\n");

    // Producer-consumer with two barriers
    printf("\n--- Producer-Consumer (Two Barriers) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        producerConsumerKernel<<<numBlocks, blockSize>>>(d_data, N, 0);
        producerConsumerKernel<<<numBlocks, blockSize>>>(d_data, N, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double pcTime = timer.elapsed_ms() / iterations;
    double pcBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Producer-Consumer: %.3f ms, %s\n", pcTime, formatBandwidth(pcBw));
    printf("Uses separate barriers for produce/consume phases.\n");

    printf("\nPTX Reference:\n");
    printf("  bar.sync  Id, Nt;     // Arrive AND wait (blocking)\n");
    printf("  bar.arrive Id, Nt;    // Arrive only (non-blocking)\n");
    printf("  bar.wait  Id, Nt;     // Wait for arrive count\n");
    printf("\nNote: bar.arrive/wait enables work between arrive and wait,\n");
    printf("      hiding latency vs blocking bar.sync.\n");

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// B.9 Named Barriers (SM90+)
// =============================================================================

void runNamedBarrierTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.9 Named Barriers (SM90+)\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

    GPUTimer timer;

    printf("\n--- Named Barrier (ID 8) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        namedBarrierKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double namedTime = timer.elapsed_ms() / iterations;
    double namedBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("Named Barrier: %.3f ms, %s\n", namedTime, formatBandwidth(namedBw));

    printf("\nNamed Barrier Features:\n");
    printf("- 16 named barriers (ID 0-15) available on SM90+\n");
    printf("- ID 0 reserved for __syncthreads() compatibility\n");
    printf("- IDs 1-7 reserved for system (CUTLASS uses 1-7)\n");
    printf("- User barriers start at ID 8\n");
    printf("- Can be used for inter-CTA synchronization\n");

    printf("\nPTX Reference:\n");
    printf("  bar.sync  Id, Nt;     // Id = 0-15 for named barriers\n");
    printf("  bar.arrive Id, Nt;\n");
    printf("  bar.wait  Id, Nt;\n");

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// B.10 mbarrier Operations
// =============================================================================

void runMbarrierTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.10 mbarrier (Memory Barrier) Operations\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    uint64_t *d_mbarrier;
    float *d_data;

    CHECK_CUDA(cudaMalloc(&d_mbarrier, numBlocks * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

    GPUTimer timer;

    // mbarrier.init + mbarrier.test_wait
    printf("\n--- mbarrier.init + test_wait ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mbarrierInitWaitKernel<<<numBlocks, blockSize>>>(d_mbarrier, d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double initWaitTime = timer.elapsed_ms() / iterations;
    printf("mbarrier.init + test_wait: %.3f ms\n", initWaitTime);

    // mbarrier.arrive (non-blocking)
    printf("\n--- mbarrier.arrive (Non-blocking) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mbarrierArriveKernel<<<numBlocks, blockSize>>>(d_mbarrier, d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double arriveTime = timer.elapsed_ms() / iterations;
    printf("mbarrier.arrive: %.3f ms\n", arriveTime);

    // mbarrier.expect_tx (transaction count)
    printf("\n--- mbarrier.expect_tx (Transaction Counting) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        mbarrierExpectTxKernel<<<numBlocks, blockSize>>>(d_mbarrier, d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double expectTxTime = timer.elapsed_ms() / iterations;
    printf("mbarrier.expect_tx: %.3f ms\n", expectTxTime);

    printf("\nmbarrier Operations:\n");
    printf("- mbarrier.init: Initialize barrier with arrival count\n");
    printf("- mbarrier.arrive: Decrement arrival count (non-blocking)\n");
    printf("- mbarrier.test_wait: Blocking wait for barrier\n");
    printf("- mbarrier.try_wait: Non-blocking query\n");
    printf("- mbarrier.expect_tx: Declare expected transaction bytes\n");
    printf("- mbarrier.complete_tx: Complete transaction bytes\n");

    printf("\nPTX Reference:\n");
    printf("  mbarrier.init.shared::cta.b64 [addr], count;\n");
    printf("  mbarrier.arrive.shared::cta.b64 _, [addr];\n");
    printf("  mbarrier.test_wait.parity.shared::cta.b64 P, [addr], phase;\n");
    printf("  mbarrier.expect_tx.shared::cta.b64 [addr], bytes;\n");
    printf("  mbarrier.complete_tx.shared::cta.b64 [addr], bytes;\n");

    CHECK_CUDA(cudaFree(d_mbarrier));
    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// B.11 cp.async + mbarrier Pipeline
// =============================================================================

void runCpAsyncMbarrierTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.11 cp.async + mbarrier Pipeline\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const size_t block_size = 256 * 4;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_dst;
    uint64_t *d_mbarrier;

    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mbarrier, numBlocks * sizeof(uint64_t)));

    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    printf("\n--- Producer (cp.async + mbarrier.arrive) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        cpAsyncProducerKernel<<<numBlocks, blockSize, 0>>>(d_src, d_dst, d_mbarrier, N, block_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double producerTime = timer.elapsed_ms() / iterations;
    double producerBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("cp.async Producer: %.3f ms, %s\n", producerTime, formatBandwidth(producerBw));

    printf("\nAsync Copy + mbarrier Pattern:\n");
    printf("1. cp.async.ca.shared::cta.b32 [smem], [gm], size;\n");
    printf("2. cp.async.commit_group; // Group async ops\n");
    printf("3. cp.async.mbarrier.arrive [barrier]; // Arrive on mbarrier\n");
    printf("4. cp.async.wait_group 0; // Wait for group\n");
    printf("5. mbarrier.test_wait [barrier]; // Final sync\n");

    printf("\nThis pattern enables:\n");
    printf("- Overlap memory copy with computation\n");
    printf("- Producer-consumer pipelines across blocks\n");
    printf("- Latency hiding for global memory access\n");

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_mbarrier));
    CHECK_CUDA(cudaFreeHost(h_src));
}

// =============================================================================
// B.12 __threadfence vs __syncthreads
// =============================================================================

void runThreadFenceTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("B.12 __threadfence vs __syncthreads\n");
    printf("================================================================================\n");

    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 1, N * sizeof(float)));

    GPUTimer timer;

    // __threadfence (memory fence only)
    printf("\n--- __threadfence (Memory Fence Only) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        threadFenceOnlyKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double fenceTime = timer.elapsed_ms() / iterations;
    printf("__threadfence: %.3f ms (UNSAFE for cross-thread communication)\n", fenceTime);

    // __syncthreads (fence + sync)
    printf("\n--- __syncthreads (Fence + Synchronization) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        syncthreadsKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double syncthreadsTime = timer.elapsed_ms() / iterations;
    double syncthreadsBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("__syncthreads: %.3f ms, %s\n", syncthreadsTime, formatBandwidth(syncthreadsBw));

    // __threadfence_block
    printf("\n--- __threadfence_block (Block Fence Only) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        threadFenceBlockKernel<<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double fenceBlockTime = timer.elapsed_ms() / iterations;
    double fenceBlockBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);
    printf("__threadfence_block: %.3f ms, %s\n", fenceBlockTime, formatBandwidth(fenceBlockBw));

    printf("\nMemory Ordering Comparison:\n");
    printf("+--------------------+------+---------+--------+\n");
    printf("| Primitive          | Fence|  Sync   | Scope  |\n");
    printf("+--------------------+------+---------+--------+\n");
    printf("| __threadfence      | Yes  | No      | Grid   |\n");
    printf("| __threadfence_block| Yes  | No      | Block  |\n");
    printf("| __syncthreads      | Yes  | Yes     | Block  |\n");
    printf("+--------------------+------+---------+--------+\n");

    printf("\nUse cases:\n");
    printf("- __threadfence: Ensure memory ordering without synchronization\n");
    printf("- __threadfence_block: Cheaper fence when only block-scope needed\n");
    printf("- __syncthreads: When both ordering AND synchronization needed\n");

    CHECK_CUDA(cudaFree(d_data));
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

    // Basic Tests
    runSyncOverheadTest();
    runBarrierStallTest();
    runBlockSizeBarrierEfficiencyTest();
    runMultiBlockSyncTest();
    runWarpSyncPrimitiveTest();
    runCooperativeSyncTest();

    // Deep Barrier Research (B.7-B.12)
    runBarRedTest();
    runArriveSyncWaitTest();
    runNamedBarrierTest();
    runMbarrierTest();
    runCpAsyncMbarrierTest();
    runThreadFenceTest();

    printf("\n");
    printf("================================================================================\n");
    printf("Barrier Synchronization Research Complete!\n");
    printf("================================================================================\n");
}
