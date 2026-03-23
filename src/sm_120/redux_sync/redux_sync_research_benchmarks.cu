#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "../common/timer.h"
#include "redux_sync_research_kernel.cu"

// =============================================================================
// Redux.sync Research Benchmarks
// =============================================================================
//
// redux.sync performs warp-level reduction operations.
// Supported: ADD, MIN, MAX, AND, OR, XOR
//
// Key advantage over shuffle-based reduction:
// - Single instruction instead of multiple shuffles
// - Hardware-accelerated
// - Lower latency
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
// Basic Redux Operations Tests
// =============================================================================

static void runReduxBasicTests(size_t N) {
    printf("\n--- Redux.sync Basic Operations ---\n");

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, (N / 32) * sizeof(float)));

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

    GPUTimer timer;
    const int iterations = 100;

    // Test 1: Redux ADD
    printf("\n[Test 1] Redux ADD (conceptual):\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        reduxAddKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Redux ADD: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Test 2: Redux MIN
    printf("\n[Test 2] Redux MIN:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        reduxMinKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Redux MIN: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Test 3: Redux MAX
    printf("\n[Test 3] Redux MAX:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        reduxMaxKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Redux MAX: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// Bitwise Redux Tests
// =============================================================================

static void runReduxBitwiseTests(size_t N) {
    printf("\n--- Redux.sync Bitwise Operations ---\n");

    size_t bytes = N * sizeof(unsigned int);
    unsigned int *d_input = nullptr;
    unsigned int *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, (N / 32) * sizeof(unsigned int)));

    // Initialize input
    unsigned int* h_input = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    for (size_t i = 0; i < N; i++) {
        h_input[i] = 0xFFFFFFFF;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_input));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 4: Redux AND
    printf("\n[Test 4] Redux AND:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        reduxAndKernel<unsigned int><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Redux AND: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Test 5: Redux OR
    printf("\n[Test 5] Redux OR:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        reduxOrKernel<unsigned int><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Redux OR: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Test 6: Redux XOR
    printf("\n[Test 6] Redux XOR:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        reduxXorKernel<unsigned int><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Redux XOR: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// Redux Performance Comparison
// =============================================================================

static void runReduxPerfComparison(size_t N) {
    printf("\n--- Redux Performance Comparison ---\n");

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, (N / 32) * sizeof(float)));

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
    const int iterations = 100;

    // Test 7: Shuffle-based reduction (baseline)
    printf("\n[Test 7a] Shuffle Reduction (baseline):\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        shuffleReductionKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Shuffle reduction: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Test 8: Butterfly reduction pattern
    printf("\n[Test 7b] Butterfly Reduction:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        butterflyReductionKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Butterfly reduction: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Test 9: Redux conceptual (simulated)
    printf("\n[Test 7c] Redux Conceptual (simulated):\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        reduxConceptualKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Redux conceptual: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Verify results
    float* h_output = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_output, (N / 32) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, (N / 32) * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Partial sum[0]: %.2f (expected: ~496)\n", h_output[0]);
    CHECK_CUDA(cudaFreeHost(h_output));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// Redux for Atomic Operations
// =============================================================================

static void runReduxAtomicTests(size_t N) {
    printf("\n--- Redux with Atomic Operations ---\n");

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    float *d_global_sum = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_global_sum, sizeof(float)));

    // Initialize input
    float* h_input = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    for (size_t i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_input));

    CHECK_CUDA(cudaMemset(d_global_sum, 0, sizeof(float)));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 10: Redux + Atomic Add
    printf("\n[Test 8] Redux + Atomic Add:\n");
    timer.start();
    reduxAtomicKernel<float><<<numBlocks, blockSize>>>(d_input, d_global_sum, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Redux + Atomic: %.3f ms\n", timer.elapsed_ms());

    // Verify result
    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_global_sum, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Global sum: %.2f (expected: %.2f)\n", result, (float)N);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_global_sum));
}

// =============================================================================
// Vote Operations Tests
// =============================================================================

static void runVoteTests(size_t N) {
    printf("\n--- Warp Vote Operations ---\n");

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    int *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, (N / 32) * sizeof(int)));

    // Initialize input (all non-zero)
    float* h_input = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    for (size_t i = 0; i < N; i++) {
        h_input[i] = (i % 2 == 0) ? 1.0f : 0.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_input));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 11: __any_sync
    printf("\n[Test 9a] Warp Vote ANY:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpVoteAnyKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  __any_sync: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Test 12: __all_sync
    printf("\n[Test 9b] Warp Vote ALL:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpVoteAllKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  __all_sync: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// Match Operations Tests
// =============================================================================

static void runMatchTests(size_t N) {
    printf("\n--- Match Operations ---\n");

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    unsigned int *d_matched = nullptr;
    unsigned int *d_count = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_matched, (N / 32) * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_count, (N / 32) * sizeof(unsigned int)));

    // Initialize input with repeating pattern
    float* h_input = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    for (size_t i = 0; i < N; i++) {
        h_input[i] = static_cast<float>((i % 4) + 1);  // Values 1,2,3,4
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_input));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 13: Match sync (simulated)
    printf("\n[Test 10] Match Sync (simulated):\n");
    timer.start();
    matchSyncKernel<float><<<numBlocks, blockSize>>>(d_input, d_matched, d_count, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Match sync: %.3f ms\n", timer.elapsed_ms());

    // Show some results
    unsigned int* h_count = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_count, (N / 32) * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(h_count, d_count, (N / 32) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("  First warp match counts: %u, %u, %u, %u\n",
           h_count[0], h_count[1], h_count[2], h_count[3]);
    CHECK_CUDA(cudaFreeHost(h_count));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_matched));
    CHECK_CUDA(cudaFree(d_count));
}

// =============================================================================
// Block-Level Reduction with Redux
// =============================================================================

static void runBlockReduceTests(size_t N) {
    printf("\n--- Block-Level Reduction with Redux ---\n");

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, (N / blockSize) * sizeof(float)));

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
    const int iterations = 100;

    // Test 14: Block reduction with redux concept
    printf("\n[Test 11] Block Reduction with Redux:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        blockReduceReduxKernel<float><<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("  Block reduce: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Verify partial results
    float* h_output = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_output, (N / blockSize) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, (N / blockSize) * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Block sum[0]: %.2f (expected: %.2f)\n", h_output[0], (float)(blockSize * (blockSize + 1) / 2));
    CHECK_CUDA(cudaFreeHost(h_output));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

void runReduxSyncBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("Redux.sync Research Benchmarks\n");
    printf("========================================\n");
    printf("Concepts: warp-level reduction, redux.sync\n");
    printf("          ADD/MIN/MAX/AND/OR/XOR\n");
    printf("========================================\n");

    // Get device info
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("========================================\n");

    // Run all test categories
    runReduxBasicTests(N);
    runReduxBitwiseTests(N);
    runReduxPerfComparison(N);
    runReduxAtomicTests(N);
    runVoteTests(N);
    runMatchTests(N);
    runBlockReduceTests(N);

    printf("\n--- Redux.sync Research Complete ---\n");
    printf("NCU Profiling Hints:\n");
    printf("  ncu --set full --metrics sm__inst_executed.redux_sync.sum ./gpupeek.exe redux\n");
    printf("  ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe redux\n");
}
