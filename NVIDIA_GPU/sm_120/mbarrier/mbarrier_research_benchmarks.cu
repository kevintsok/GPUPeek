#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "../../common/timer.h"
#include "mbarrier_research_kernel.cu"

// =============================================================================
// Mbarrier Research Benchmarks
// =============================================================================
//
// Mbarrier provides synchronization for asynchronous memory operations.
// Key for: cp.async, cp.async.bulk, st.async, WGMMA
//
// Topics:
// 1. Basic mbarrier operations
// 2. Async copy synchronization
// 3. Pipeline synchronization
// 4. Producer-consumer patterns
// 5. Transaction counting
// 6. Memory fence comparison
// 7. Grid dependency control
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
// Basic Mbarrier Tests
// =============================================================================

static void runBasicMbarrierTests(size_t N) {
    printf("\n--- Basic Mbarrier Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    unsigned int *d_mbarrier_state = nullptr;
    unsigned int *h_mbarrier_state = nullptr;

    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMalloc(&d_mbarrier_state, sizeof(unsigned int)));
    CHECK_CUDA(cudaMallocHost(&h_mbarrier_state, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));
    CHECK_CUDA(cudaMemset(d_mbarrier_state, 0, sizeof(unsigned int)));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 1: Basic atomic-based synchronization
    printf("\n[Test 1] Basic Atomic-based Sync:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        mbarrier_init_wait_kernel<float><<<numBlocks, blockSize>>>(
            d_data, d_mbarrier_state, N, i % 2);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Atomic sync: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    // Test 2: Async copy concept test
    printf("\n[Test 2] Async Copy Mbarrier Concept:\n");
    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_src, 1, bytes));
    CHECK_CUDA(cudaMemset(d_mbarrier_state, 0, sizeof(unsigned int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        async_copy_mbarrier_kernel<float><<<numBlocks, blockSize>>>(
            d_src, d_dst, d_mbarrier_state, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Async copy: %.3f ms for %d iterations\n", timer.elapsed_ms(), iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    // Test 3: Pipeline synchronization
    printf("\n[Test 3] Pipeline Synchronization:\n");
    CHECK_CUDA(cudaMemset(d_mbarrier_state, 0, sizeof(unsigned int)));

    timer.start();
    mbarrier_pipeline_kernel<float><<<numBlocks, blockSize>>>(
        d_data, d_mbarrier_state, N, 4);
    cudaDeviceSynchronize();
    timer.stop();
    printf("  4-stage pipeline: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_mbarrier_state));
    CHECK_CUDA(cudaFreeHost(h_mbarrier_state));
}

// =============================================================================
// Reduction with Synchronization
// =============================================================================

static void runReduceSyncTests(size_t N) {
    printf("\n--- Reduction with Synchronization ---\n");

    size_t bytes = N * sizeof(float);
    float *d_input = nullptr;
    float *d_output = nullptr;
    unsigned int *d_sync_array = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_sync_array, bytes));

    for (size_t i = 0; i < N; i++) {
        float* temp = nullptr;
        CHECK_CUDA(cudaMallocHost(&temp, bytes));
        for (size_t j = 0; j < N; j++) {
            temp[j] = static_cast<float>(i + j);
        }
        CHECK_CUDA(cudaMemcpy(d_input, temp, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaFreeHost(temp));
        break;
    }

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 4: Synchronized reduction
    printf("\n[Test 4] Mbarrier-based Reduction:\n");
    CHECK_CUDA(cudaMemset(d_sync_array, 0, bytes));

    timer.start();
    mbarrier_reduce_kernel<float><<<numBlocks, blockSize>>>(
        d_input, d_output, d_sync_array, N);
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Reduction: %.3f ms\n", timer.elapsed_ms());

    // Verify result
    float result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Result[0]: %.2f\n", result);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_sync_array));
}

// =============================================================================
// Producer-Consumer Tests
// =============================================================================

static void runProducerConsumerTests(size_t N) {
    printf("\n--- Producer-Consumer Synchronization ---\n");

    size_t bytes = N * sizeof(float);
    size_t buffer_size = 1024;

    float *d_buffer = nullptr;
    unsigned int *d_prod_idx = nullptr;
    unsigned int *d_cons_idx = nullptr;
    unsigned int *d_mbarrier_state = nullptr;

    CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_prod_idx, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_cons_idx, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_mbarrier_state, sizeof(unsigned int)));

    CHECK_CUDA(cudaMemset(d_buffer, 0, buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_prod_idx, 0, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_cons_idx, 0, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_mbarrier_state, 0, sizeof(unsigned int)));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 5: Producer-Consumer
    printf("\n[Test 5] Producer-Consumer with Mbarrier:\n");
    timer.start();
    mbarrier_producer_consumer_kernel<float><<<numBlocks, blockSize>>>(
        d_buffer, d_prod_idx, d_cons_idx, d_mbarrier_state, N, buffer_size);
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Producer-Consumer: %.3f ms\n", timer.elapsed_ms());

    // Check indices
    unsigned int prod_val, cons_val;
    CHECK_CUDA(cudaMemcpy(&prod_val, d_prod_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&cons_val, d_cons_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("  Producer idx: %u, Consumer idx: %u\n", prod_val, cons_val);

    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaFree(d_prod_idx));
    CHECK_CUDA(cudaFree(d_cons_idx));
    CHECK_CUDA(cudaFree(d_mbarrier_state));
}

// =============================================================================
// Transaction Counting Tests
// =============================================================================

static void runTransactionCountTests(size_t N) {
    printf("\n--- Transaction Counting Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    unsigned int *d_tx_count = nullptr;
    unsigned int *d_tx_complete = nullptr;

    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMalloc(&d_tx_count, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_tx_complete, sizeof(unsigned int)));

    CHECK_CUDA(cudaMemset(d_data, 1, bytes));
    CHECK_CUDA(cudaMemset(d_tx_count, 0, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_tx_complete, 0, sizeof(unsigned int)));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 10;

    // Test 6: Transaction counting
    printf("\n[Test 6] Transaction Counting:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        mbarrier_tx_count_kernel<float><<<numBlocks, blockSize>>>(
            d_data, d_tx_count, d_tx_complete, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  %d iterations: %.3f ms\n", iterations, timer.elapsed_ms());

    // Check counts
    unsigned int tx_count, tx_complete;
    CHECK_CUDA(cudaMemcpy(&tx_count, d_tx_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&tx_complete, d_tx_complete, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("  Arrivals: %u, Completions: %u\n", tx_count, tx_complete);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_tx_count));
    CHECK_CUDA(cudaFree(d_tx_complete));
}

// =============================================================================
// Memory Fence Comparison Tests
// =============================================================================

static void runFenceComparisonTests(size_t N) {
    printf("\n--- Memory Fence Comparison ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    int *d_flag = nullptr;

    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMalloc(&d_flag, sizeof(int)));

    CHECK_CUDA(cudaMemset(d_data, 1, bytes));
    CHECK_CUDA(cudaMemset(d_flag, 0, sizeof(int)));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 7: No fence
    printf("\n[Test 7a] No Fence:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        memory_fence_variants_kernel<float><<<numBlocks, blockSize>>>(
            d_data, d_flag, N, 0);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  No fence: %.3f ms\n", timer.elapsed_ms());

    // Test 8: __threadfence_block
    printf("\n[Test 7b] __threadfence_block:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        memory_fence_variants_kernel<float><<<numBlocks, blockSize>>>(
            d_data, d_flag, N, 2);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  __threadfence_block: %.3f ms\n", timer.elapsed_ms());

    // Test 9: __threadfence (GPU-to-GPU)
    printf("\n[Test 7c] __threadfence:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        memory_fence_variants_kernel<float><<<numBlocks, blockSize>>>(
            d_data, d_flag, N, 3);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  __threadfence: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_flag));
}

// =============================================================================
// Grid Dependency Control Tests
// =============================================================================

static void runGridDepControlTests(size_t N) {
    printf("\n--- Grid Dependency Control ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    volatile int *d_ready = nullptr;

    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMalloc(&d_ready, sizeof(int)));

    CHECK_CUDA(cudaMemset(d_data, 1, bytes));
    CHECK_CUDA(cudaMemset((void*)d_ready, 0, sizeof(int)));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 10: Grid dependency control
    printf("\n[Test 8] Grid Dependency Control:\n");
    timer.start();
    grid_dep_control_kernel<float><<<numBlocks, blockSize>>>(d_data, d_ready, N);
    timer.stop();
    printf("  Grid dep (immediate): %.3f ms\n", timer.elapsed_ms());

    // Set ready flag and run again
    CHECK_CUDA(cudaMemset((void*)d_ready, 1, sizeof(int)));
    timer.start();
    grid_dep_control_kernel<float><<<numBlocks, blockSize>>>(d_data, d_ready, N);
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Grid dep (after flag): %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree((void*)d_ready));
}

// =============================================================================
// Cluster Barrier Tests
// =============================================================================

static void runClusterBarrierTests(size_t N) {
    printf("\n--- Cluster Barrier Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;

    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 1, bytes));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // Test 11: Cluster barrier concept
    printf("\n[Test 9] Cluster Barrier (using __syncthreads as reference):\n");
    timer.start();
    cluster_barrier_kernel<float><<<numBlocks, blockSize>>>(d_data, N);
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Cluster barrier: %.3f ms\n", timer.elapsed_ms());
    printf("  Note: __cluster_barrier requires CUDA 12.0+ and sm_90+\n");

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

void runMbarrierResearchBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("Mbarrier Research Benchmarks\n");
    printf("========================================\n");
    printf("Concepts: mbarrier, async copy sync,\n");
    printf("          transaction counting, fences\n");
    printf("========================================\n");

    // Get device info
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("========================================\n");

    // Run all test categories
    runBasicMbarrierTests(N);
    runReduceSyncTests(N);
    runProducerConsumerTests(N);
    runTransactionCountTests(N);
    runFenceComparisonTests(N);
    runGridDepControlTests(N);
    runClusterBarrierTests(N);

    printf("\n--- Mbarrier Research Complete ---\n");
    printf("NCU Profiling Hints:\n");
    printf("  ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe mbarrier\n");
    printf("  ncu --set full --metrics sm__inst_executed.mbarrier.sum ./gpupeek.exe mbarrier\n");
}
