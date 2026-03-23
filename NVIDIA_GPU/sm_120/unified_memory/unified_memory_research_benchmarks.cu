#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "../../common/timer.h"
#include "unified_memory_research_kernel.cu"

// =============================================================================
// Unified Memory Research Benchmarks
// =============================================================================
//
// Unified Memory Concepts:
// - cudaMallocManaged: Single allocation accessible from GPU and CPU
// - cudaMemPrefetchAsync: Explicit data migration to device/host
// - cudaMemAdvise: Hints about data usage patterns
// - Page fault: On-demand migration triggers
// - Access counters: Track which device accessed data
//
// Use cases:
// - Simplified memory management
// - GPU memory expansion beyond device memory
// - Heterogeneous computing
// - Out-of-core processing
// =============================================================================

// Helper to check unified memory pointer attributes
static void checkManagedPointer(const void* ptr) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err == cudaSuccess) {
        printf("  Pointer type: %s\n",
               attrs.type == cudaMemoryTypeManaged ? "Managed" : "Other");
        if (attrs.type == cudaMemoryTypeManaged) {
            printf("  Device: %d, Host: %p\n", attrs.device, attrs.hostPointer);
        }
    }
}

// =============================================================================
// Basic Managed Memory Tests
// =============================================================================

static void runBasicManagedMemoryTests(size_t N) {
    printf("\n--- Basic Managed Memory Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *managed_data = nullptr;

    // Test 1: Basic cudaMallocManaged
    printf("\n[Test 1] cudaMallocManaged:\n");
    cudaError_t err = cudaMallocManaged(&managed_data, bytes);
    if (err != cudaSuccess) {
        printf("  FAILED: cudaMallocManaged error: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("  Allocated %.2f MB managed memory\n", bytes / (1024.0 * 1024.0));

    // Initialize on host
    for (size_t i = 0; i < N; i++) {
        managed_data[i] = static_cast<float>(i);
    }

    // Check pointer attributes
    printf("  Pointer attributes:\n");
    checkManagedPointer(managed_data);

    // Prefetch to GPU
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaMemPrefetchAsync(managed_data, bytes, deviceId, 0);
    cudaDeviceSynchronize();
    printf("  Prefetched to GPU %d\n", deviceId);

    // Run kernel on GPU
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorScaleKernel<float><<<numBlocks, blockSize>>>(managed_data, 2.0f, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  GPU vector scale: %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Prefetch back to host
    cudaMemPrefetchAsync(managed_data, bytes, cudaCpuDeviceId, 0);
    cudaDeviceSynchronize();
    printf("  Prefetched back to CPU\n");

    // Verify data on host
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        sum += managed_data[i];
    }
    printf("  Verification (sum): %.2f (expected: %.2f)\n", sum, N * (N - 1));

    cudaFree(managed_data);
    printf("  Freed managed memory\n");
}

// =============================================================================
// Page Fault Detection Tests
// =============================================================================

static void runPageFaultTests(size_t N) {
    printf("\n--- Page Fault Detection Tests ---\n");

    size_t bytes = N * sizeof(float);

    // Test 2: Touch all pages to trigger page faults
    printf("\n[Test 2] Page Fault Detection (First Touch):\n");
    float *managed_data = nullptr;
    cudaMallocManaged(&managed_data, bytes);

    // Don't initialize - just touch each page
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;

    // First kernel pass - should trigger page faults
    timer.start();
    touchAllPagesKernel<float><<<numBlocks, blockSize>>>(managed_data, N);
    cudaDeviceSynchronize();
    timer.stop();
    printf("  First pass (page fault): %.3f ms\n", timer.elapsed_ms());

    // Second kernel pass - pages already faulted
    timer.start();
    for (int i = 0; i < 10; i++) {
        touchAllPagesKernel<float><<<numBlocks, blockSize>>>(managed_data, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Second pass (cached): %.3f ms\n", timer.elapsed_ms() / 10.0);

    cudaFree(managed_data);
}

// =============================================================================
// Memory Access Pattern Tests
// =============================================================================

static void runAccessPatternTests(size_t N) {
    printf("\n--- Managed Memory Access Pattern Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *managed_data = nullptr;
    cudaMallocManaged(&managed_data, bytes);

    for (size_t i = 0; i < N; i++) {
        managed_data[i] = 1.0f;
    }

    // Prefetch to GPU
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaMemPrefetchAsync(managed_data, bytes, deviceId, 0);
    cudaDeviceSynchronize();

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Sequential access
    printf("\n[Test 3] Sequential Access:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        sequentialAccessKernel<float><<<numBlocks, blockSize>>>(managed_data, N, 1);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Sequential: %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Strided access
    printf("\n[Test 4] Strided Access (stride=64):\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        stridedAccessKernel<float><<<numBlocks, blockSize>>>(managed_data, N, 64, 1);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Strided: %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    cudaFree(managed_data);
}

// =============================================================================
// Prefetch and Advice Tests
// =============================================================================

static void runPrefetchAdviceTests(size_t N) {
    printf("\n--- Prefetch and Advice Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *managed_data = nullptr;
    cudaMallocManaged(&managed_data, bytes);

    // Initialize
    for (size_t i = 0; i < N; i++) {
        managed_data[i] = 1.0f;
    }

    int deviceId;
    cudaGetDevice(&deviceId);

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 5: No prefetch (let system manage)
    printf("\n[Test 5] No Prefetch (System Managed):\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorScaleKernel<float><<<numBlocks, blockSize>>>(managed_data, 2.0f, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  No prefetch: %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Test 6: Explicit GPU prefetch
    printf("\n[Test 6] Explicit GPU Prefetch:\n");
    cudaMemPrefetchAsync(managed_data, bytes, deviceId, 0);
    cudaDeviceSynchronize();
    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorScaleKernel<float><<<numBlocks, blockSize>>>(managed_data, 2.0f, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  With prefetch: %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Test 7: Set read-mostly advice
    printf("\n[Test 7] cudaMemAdvise (Read Mostly):\n");
    cudaMemAdvise(managed_data, bytes, cudaMemAdviseSetReadMostly, deviceId);
    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorScaleKernel<float><<<numBlocks, blockSize>>>(managed_data, 2.0f, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Read mostly: %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    cudaFree(managed_data);
}

// =============================================================================
// Write Combining Tests
// =============================================================================

static void runWriteCombiningTests(size_t N) {
    printf("\n--- Write Combining Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *managed_data = nullptr;
    cudaMallocManaged(&managed_data, bytes);

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    GPUTimer timer;
    const int iterations = 100;

    // Test 8: Sequential writes
    printf("\n[Test 8] Sequential Writes:\n");
    timer.start();
    for (int i = 0; i < iterations; i++) {
        writeCombiningKernel<float><<<numBlocks, blockSize>>>(managed_data, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Sequential write: %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // Scatter writes
    printf("\n[Test 9] Scatter Writes:\n");
    size_t *indices = nullptr;
    cudaMallocManaged(&indices, N * sizeof(size_t));
    for (size_t i = 0; i < N; i++) {
        indices[i] = (i * 17) % N;  // Pseudo-random scatter
    }

    timer.start();
    for (int i = 0; i < iterations; i++) {
        writeScatterKernel<float><<<numBlocks, blockSize>>>(managed_data, N, indices);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Scatter write: %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    cudaFree(indices);
    cudaFree(managed_data);
}

// =============================================================================
// GPU-CPU Synchronization Tests
// =============================================================================

static void runSyncTests(size_t N) {
    printf("\n--- GPU-CPU Synchronization Tests ---\n");

    size_t bytes = N * sizeof(float);
    float *managed_data = nullptr;
    cudaMallocManaged(&managed_data, bytes);

    for (size_t i = 0; i < N; i++) {
        managed_data[i] = 1.0f;
    }

    // Prefetch to GPU
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaMemPrefetchAsync(managed_data, bytes, deviceId, 0);
    cudaDeviceSynchronize();

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    // Test 10: Spin-wait synchronization
    printf("\n[Test 10] GPU Spin-Wait Synchronization:\n");

    volatile int *flag = nullptr;
    cudaMallocManaged(&flag, sizeof(int));
    *flag = 0;

    // Launch kernel that spins waiting for flag
    spinWaitKernel<float><<<numBlocks, blockSize>>>(managed_data, flag, N);

    // CPU work then set flag
    GPUTimer timer;
    timer.start();
    for (int i = 0; i < 1000; i++) {
        // Simulate CPU work
    }
    *flag = 1;
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Spin-wait complete: %.3f ms (CPU work + GPU sync)\n", timer.elapsed_ms());

    cudaFree(flag);
    cudaFree(managed_data);
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

void runUnifiedMemoryResearchBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("Unified Memory Research Benchmarks\n");
    printf("========================================\n");
    printf("Concepts: cudaMallocManaged, cudaMemPrefetchAsync,\n");
    printf("          cudaMemAdvise, Page Faults, Access Counters\n");
    printf("========================================\n");

    // Run all test categories
    runBasicManagedMemoryTests(N);
    runPageFaultTests(N);
    runAccessPatternTests(N);
    runPrefetchAdviceTests(N);
    runWriteCombiningTests(N);
    runSyncTests(N);

    printf("\n--- Unified Memory Research Complete ---\n");
    printf("NCU Profiling Hints:\n");
    printf("  ncu --set full --metrics dram__bytes.sum,uops__issue_active.sum ./gpupeek.exe unified\n");
}
