#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "../common/timer.h"
#include "cooperative_groups_research_kernel.cu"

// =============================================================================
// Cooperative Groups Research Benchmarks
// =============================================================================
//
// Cooperative Groups enable thread cooperation across:
// - Thread Block (same block threads)
// - Grid (all threads in kernel)
// - Multi-GPU (across GPUs)
//
// Key APIs:
// - cooperative_groups::this_thread_block()
// - cooperative_groups::this_grid()
// - cooperative_groups::this_multi_grid()
// =============================================================================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// =============================================================================
// Utility Functions
// =============================================================================

// Check if cooperative groups launch is supported
static bool checkCooperativeGroupsSupport() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Cooperative groups supported: %s\n",
           prop.cooperativeLaunch ? "Yes" : "No");
    printf("Multi-device cooperative groups: %s\n",
           prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");

    return prop.cooperativeLaunch != 0;
}

// =============================================================================
// Thread Block Synchronization Tests
// =============================================================================

static void runThreadBlockSyncTests(size_t N) {
    printf("\n--- Thread Block Synchronization Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 1, bytes));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 1: Thread block sync
    printf("\n[Test 1] Thread Block Sync:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        threadBlockSyncKernel<float><<<numBlocks, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Thread block sync: %.3f ms for %d iterations\n",
           timer.elapsed_ms(), iterations);

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// Grid-Level Synchronization Tests
// =============================================================================

static void runGridSyncTests(size_t N) {
    printf("\n--- Grid-Level Synchronization Tests ---\n");

    if (!checkCooperativeGroupsSupport()) {
        printf("  Cooperative groups not supported, skipping tests\n");
        return;
    }

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));

    // Initialize input
    float* h_input = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    for (size_t i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_input));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;
    size_t shared_bytes = blockSize * sizeof(float);

    GPUTimer timer;

    // Test 2: Grid reduction
    printf("\n[Test 2] Grid Reduction:\n");
    timer.start();
    gridReduceKernel<float><<<numBlocks, blockSize, shared_bytes>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Grid reduce: %.3f ms\n", timer.elapsed_ms());

    // Verify result
    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Result: %.2f (expected: %.2f)\n", result, (float)N);

    // Test 3: Cooperative load
    printf("\n[Test 3] Cooperative Load:\n");
    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_src, 1, bytes));
    CHECK_CUDA(cudaMemset(d_dst, 0, bytes));

    timer.start();
    cooperativeLoadKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Cooperative load: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// Grid Barrier Tests
// =============================================================================

static void runGridBarrierTests(size_t N) {
    printf("\n--- Grid Barrier Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 4: Grid barrier with memset
    printf("\n[Test 4] Grid Barrier with Memset:\n");
    timer.start();
    gridBarrierMemsetKernel<float><<<numBlocks, blockSize>>>(d_data, 1.0f, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Grid barrier memset: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// Multi-Block Reduction Tests
// =============================================================================

static void runMultiBlockReduceTests(size_t N) {
    printf("\n--- Multi-Block Reduction Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    float *d_result = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    // Initialize input with sequential values
    float* h_input = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    for (size_t i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_input));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 5: Multi-block reduction
    printf("\n[Test 5] Multi-Block Reduction:\n");
    timer.start();
    multiBlockReduceKernel<float><<<numBlocks, blockSize>>>(d_input, d_result, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Multi-block reduce: %.3f ms\n", timer.elapsed_ms());

    // Verify result (sum of 1 to N)
    float expected = static_cast<float>(N * (N + 1) / 2);
    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Result: %.2f (expected: %.2f)\n", result, expected);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_result));
}

// =============================================================================
// Two-Phase Cooperative Tests
// =============================================================================

static void runTwoPhaseTests(size_t N) {
    printf("\n--- Two-Phase Cooperative Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 1, bytes));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 6: Two-phase kernel
    printf("\n[Test 6] Two-Phase Kernel:\n");
    timer.start();
    twoPhaseKernel<float><<<numBlocks, blockSize>>>(d_data, N, 0);
    twoPhaseKernel<float><<<numBlocks, blockSize>>>(d_data, N, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Two-phase: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// Broadcast Tests
// =============================================================================

static void runBroadcastTests(size_t N) {
    printf("\n--- Broadcast Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_src = nullptr;
    float *d_dst = nullptr;

    CHECK_CUDA(cudaMalloc(&d_src, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_src, 42, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dst, 0, bytes));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 7: Broadcast from lane 0
    printf("\n[Test 7] Broadcast from Thread 0:\n");
    timer.start();
    broadcastKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, 0, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Broadcast: %.3f ms\n", timer.elapsed_ms());

    // Verify
    float* h_dst = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_dst, bytes));
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    bool all_42 = true;
    for (size_t i = 0; i < N; i++) {
        if (h_dst[i] != 42.0f) {
            all_42 = false;
            break;
        }
    }
    printf("  Verification: %s\n", all_42 ? "PASS" : "FAIL");
    CHECK_CUDA(cudaFreeHost(h_dst));

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// Even-Odd Synchronization Tests
// =============================================================================

static void runEvenOddSyncTests(size_t N) {
    printf("\n--- Even-Odd Synchronization Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int num_phases = 10;

    // Test 8: Even-odd sync pattern
    printf("\n[Test 8] Even-Odd Sync Pattern:\n");
    timer.start();
    evenOddSyncKernel<float><<<numBlocks, blockSize>>>(d_data, N, num_phases);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Even-odd sync (%d phases): %.3f ms\n", num_phases, timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// Barrier Efficiency Tests
// =============================================================================

static void runBarrierEfficiencyTests(size_t N) {
    printf("\n--- Barrier Efficiency Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 1, bytes));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 9: Barrier efficiency
    printf("\n[Test 9] Barrier Efficiency:\n");
    timer.start();
    barrierEfficiencyKernel<float><<<numBlocks, blockSize>>>(d_data, N, iterations);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Barrier efficiency (%d iters): %.3f ms\n", iterations, timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// Vectorized Cooperative Load Tests
// =============================================================================

static void runVectorizedCoopLoadTests(size_t N) {
    printf("\n--- Vectorized Cooperative Load Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_src = nullptr;
    float *d_dst = nullptr;

    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_src, 1, bytes));
    CHECK_CUDA(cudaMemset(d_dst, 0, bytes));

    const int blockSize = 256;
    int numBlocks = (N / 4 + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 10: Vectorized cooperative load
    printf("\n[Test 10] Vectorized Cooperative Load:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorizedCoopLoadKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Vectorized coop load: %.3f ms for %d iterations\n",
           timer.elapsed_ms(), iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

void runCooperativeGroupsBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("Cooperative Groups Research Benchmarks\n");
    printf("========================================\n");
    printf("Concepts: Thread block sync, Grid sync,\n");
    printf("          Cooperative load, Broadcast\n");
    printf("========================================\n");

    // Get device info
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("Cooperative Launch: %s\n", prop.cooperativeLaunch ? "Supported" : "Not Supported");
    printf("Multi-Device Cooperative: %s\n", prop.cooperativeMultiDeviceLaunch ? "Supported" : "Not Supported");
    printf("========================================\n");

    // Run all test categories
    runThreadBlockSyncTests(N);
    runGridSyncTests(N);
    runGridBarrierTests(N);
    runMultiBlockReduceTests(N);
    runTwoPhaseTests(N);
    runBroadcastTests(N);
    runEvenOddSyncTests(N);
    runBarrierEfficiencyTests(N);
    runVectorizedCoopLoadTests(N);

    printf("\n--- Cooperative Groups Research Complete ---\n");
    printf("NCU Profiling Hints:\n");
    printf("  ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe coop\n");
    printf("  ncu --set full --metrics sm__average_active_warps_per_sm ./gpupeek.exe coop\n");
}
