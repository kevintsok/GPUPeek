#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "../common/timer.h"
#include "multi_stream_research_kernel.cu"

// =============================================================================
// Multi-Stream Research Benchmarks
// =============================================================================
//
// Multi-Stream Concepts:
// - Stream creation with priorities
// - Stream synchronization patterns
// - Concurrent kernel execution
// - Memory transfer + compute overlap
// - Event-based synchronization
// - Stream dependencies
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

static void getStreamPriorityRange(int* lowestPriority, int* highestPriority) {
    int priorityRange = 0;
    cudaDeviceGetStreamPriorityRange(lowestPriority, highestPriority);
    printf("  Stream priority range: [%d, %d] (lower = higher priority)\n",
           *lowestPriority, *highestPriority);
}

// =============================================================================
// Basic Stream Tests
// =============================================================================

static void runBasicStreamTests(size_t N) {
    printf("\n--- Basic Stream Tests ---\n");

    size_t bytes = N * sizeof(float);
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    // Allocate host and device memory
    float *h_src, *d_src, *d_dst;
    CHECK_CUDA(cudaMallocHost(&h_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));

    for (size_t i = 0; i < N; i++) {
        h_src[i] = static_cast<float>(i);
    }
    CHECK_CUDA(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    // Test 1: Single stream baseline
    printf("\n[Test 1] Single Stream Baseline:\n");
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    GPUTimer timer;
    timer.start();
    for (int i = 0; i < 10; i++) {
        streamVectorAddKernel<float><<<numBlocks, blockSize, 0, stream>>>(d_src, d_src, d_dst, N);
    }
    cudaStreamSynchronize(stream);
    timer.stop();
    printf("  10x vector add (single stream): %.3f ms\n", timer.elapsed_ms());

    cudaStreamDestroy(stream);

    // Test 2: Multiple streams (sequential)
    printf("\n[Test 2] Multiple Streams Sequential:\n");
    const int numStreams = 4;
    cudaStream_t streams[numStreams];

    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    timer.start();
    for (int i = 0; i < numStreams; i++) {
        streamVectorAddKernel<float><<<numBlocks, blockSize, 0, streams[i]>>>(d_src, d_src, d_dst, N);
    }
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    timer.stop();
    printf("  4x vector add (sequential streams): %.3f ms\n", timer.elapsed_ms());

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // Test 3: Multiple streams (concurrent - if supported)
    printf("\n[Test 3] Multiple Streams Concurrent:\n");
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    timer.start();
    for (int i = 0; i < numStreams; i++) {
        streamVectorAddKernel<float><<<numBlocks, blockSize, 0, streams[i]>>>(d_src, d_src, d_dst, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  4x vector add (concurrent streams): %.3f ms\n", timer.elapsed_ms());

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));
}

// =============================================================================
// Stream Priority Tests
// =============================================================================

static void runStreamPriorityTests(size_t N) {
    printf("\n--- Stream Priority Tests ---\n");

    int lowestPriority, highestPriority;
    getStreamPriorityRange(&lowestPriority, &highestPriority);

    if (highestPriority == 0 && lowestPriority == 0) {
        printf("  Priority not supported on this device\n");
        return;
    }

    size_t bytes = N * sizeof(float);
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_src, 1, bytes));

    // Test 4: High priority stream
    printf("\n[Test 4] High Priority Stream:\n");
    cudaStream_t highPriorityStream;
    cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, highestPriority);

    GPUTimer timer;
    timer.start();
    for (int i = 0; i < 10; i++) {
        streamVectorScaleKernel<float><<<numBlocks, blockSize, 0, highPriorityStream>>>(d_src, 2.0f, N);
    }
    cudaStreamSynchronize(highPriorityStream);
    timer.stop();
    printf("  High priority (10x scale): %.3f ms\n", timer.elapsed_ms());

    cudaStreamDestroy(highPriorityStream);

    // Test 5: Low priority stream
    printf("\n[Test 5] Low Priority Stream:\n");
    cudaStream_t lowPriorityStream;
    cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamNonBlocking, lowestPriority);

    timer.start();
    for (int i = 0; i < 10; i++) {
        streamVectorScaleKernel<float><<<numBlocks, blockSize, 0, lowPriorityStream>>>(d_src, 2.0f, N);
    }
    cudaStreamSynchronize(lowPriorityStream);
    timer.stop();
    printf("  Low priority (10x scale): %.3f ms\n", timer.elapsed_ms());

    cudaStreamDestroy(lowPriorityStream);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// Stream Dependency Tests (Event-based)
// =============================================================================

static void runEventDependencyTests(size_t N) {
    printf("\n--- Event-based Stream Dependencies ---\n");

    size_t bytes = N * sizeof(float);
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_src, 1, bytes));

    // Test 6: cudaStreamWaitEvent
    printf("\n[Test 6] cudaStreamWaitEvent (Synchronization):\n");
    cudaStream_t stream1, stream2;
    cudaEvent_t event;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEventCreate(&event);

    // Stream 1 does scale
    streamVectorScaleKernel<float><<<numBlocks, blockSize, 0, stream1>>>(d_src, 2.0f, N);
    cudaEventRecord(event, stream1);

    // Stream 2 waits for event then does add
    cudaStreamWaitEvent(stream2, event, 0);
    streamVectorAddKernel<float><<<numBlocks, blockSize, 0, stream2>>>(d_src, d_src, d_dst, N);

    GPUTimer timer;
    timer.start();
    cudaStreamSynchronize(stream2);
    timer.stop();
    printf("  Event sync (add after scale): %.3f ms\n", timer.elapsed_ms());

    cudaEventDestroy(event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Test 7: Event query
    printf("\n[Test 7] cudaEventQuery (Polling):\n");
    cudaEvent_t event2;
    cudaEventCreate(&event2);

    cudaStream_t stream3;
    cudaStreamCreate(&stream3);

    streamVectorAddKernel<float><<<numBlocks, blockSize, 0, stream3>>>(d_src, d_src, d_dst, N);
    cudaEventRecord(event2, stream3);

    int pollCount = 0;
    cudaError_t err;
    do {
        err = cudaEventQuery(event2);
        pollCount++;
    } while (err == cudaErrorNotReady);
    printf("  Event poll count: %d (err=%s)\n", pollCount, cudaGetErrorString(err));

    cudaEventDestroy(event2);
    cudaStreamDestroy(stream3);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
}

// =============================================================================
// Overlap Tests (Memory + Compute)
// =============================================================================

static void runOverlapTests(size_t N) {
    printf("\n--- Memory Transfer + Compute Overlap ---\n");

    size_t bytes = N * sizeof(float);
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *h_src, *h_dst, *d_src, *d_dst;
    CHECK_CUDA(cudaMallocHost(&h_src, bytes));
    CHECK_CUDA(cudaMallocHost(&h_dst, bytes));
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));

    for (size_t i = 0; i < N; i++) {
        h_src[i] = static_cast<float>(i);
    }

    // Test 8: Serial (no overlap)
    printf("\n[Test 8] Serial (No Overlap):\n");
    cudaStream_t serialStream;
    cudaStreamCreate(&serialStream);

    GPUTimer timer;

    timer.start();
    CHECK_CUDA(cudaMemcpyAsync(d_src, h_src, bytes, cudaMemcpyHostToDevice, serialStream));
    streamVectorAddKernel<float><<<numBlocks, blockSize, 0, serialStream>>>(d_src, d_src, d_dst, N);
    CHECK_CUDA(cudaMemcpyAsync(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost, serialStream));
    cudaStreamSynchronize(serialStream);
    timer.stop();
    printf("  Serial H2D-Compute-D2H: %.3f ms\n", timer.elapsed_ms());

    cudaStreamDestroy(serialStream);

    // Test 9: Two streams with event sync (partial overlap)
    printf("\n[Test 9] Two Streams with Event Sync:\n");
    cudaStream_t computeStream, memcpyStream;
    cudaEvent_t computeDone;

    cudaStreamCreate(&computeStream);
    cudaStreamCreate(&memcpyStream);
    cudaEventCreate(&computeDone);

    timer.start();
    // Start memcpy H2D
    CHECK_CUDA(cudaMemcpyAsync(d_src, h_src, bytes, cudaMemcpyHostToDevice, memcpyStream));
    // Wait for memcpy to complete, then compute
    cudaStreamWaitEvent(computeStream, computeDone, 0);
    streamVectorAddKernel<float><<<numBlocks, blockSize, 0, computeStream>>>(d_src, d_src, d_dst, N);
    cudaEventRecord(computeDone, computeStream);
    // Wait for compute, then memcpy D2H
    cudaStreamWaitEvent(memcpyStream, computeDone, 0);
    CHECK_CUDA(cudaMemcpyAsync(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost, memcpyStream));
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Overlapped H2D-Compute-D2H: %.3f ms\n", timer.elapsed_ms());

    cudaEventDestroy(computeDone);
    cudaStreamDestroy(computeStream);
    cudaStreamDestroy(memcpyStream);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFreeHost(h_dst));
}

// =============================================================================
// Concurrent Kernel Execution Tests
// =============================================================================

static void runConcurrentKernelTests(size_t N) {
    printf("\n--- Concurrent Kernel Execution ---\n");

    size_t bytes = N * sizeof(float);
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_data1, *d_data2;
    CHECK_CUDA(cudaMalloc(&d_data1, bytes));
    CHECK_CUDA(cudaMalloc(&d_data2, bytes));
    CHECK_CUDA(cudaMemset(d_data1, 1, bytes));
    CHECK_CUDA(cudaMemset(d_data2, 2, bytes));

    // Test 10: Sequential kernels
    printf("\n[Test 10] Sequential Kernels:\n");
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    GPUTimer timer;
    timer.start();
    streamMemoryIntensiveKernel<float><<<numBlocks, blockSize, 0, stream1>>>(d_data1, N, 10);
    streamComputeIntensiveKernel<float><<<numBlocks, blockSize, 0, stream2>>>(d_data2, N, 10);
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Sequential (memory + compute): %.3f ms\n", timer.elapsed_ms());

    // Test 11: Concurrent kernels
    printf("\n[Test 11] Concurrent Kernels:\n");
    timer.start();
    streamMemoryIntensiveKernel<float><<<numBlocks, blockSize, 0, stream1>>>(d_data1, N, 10);
    streamComputeIntensiveKernel<float><<<numBlocks, blockSize, 0, stream2>>>(d_data2, N, 10);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    timer.stop();
    printf("  Concurrent (memory + compute): %.3f ms\n", timer.elapsed_ms());

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    CHECK_CUDA(cudaFree(d_data1));
    CHECK_CUDA(cudaFree(d_data2));
}

// =============================================================================
// Pipeline Tests
// =============================================================================

static void runPipelineTests(size_t N) {
    printf("\n--- Pipeline Tests (3-Stage) ---\n");

    size_t bytes = N * sizeof(float);
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_input, *d_temp1, *d_temp2, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp1, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp2, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    CHECK_CUDA(cudaMemset(d_input, 1, bytes));

    // Test 12: Serial pipeline
    printf("\n[Test 12] Serial 3-Stage Pipeline:\n");
    cudaStream_t pipelineStream;
    cudaStreamCreate(&pipelineStream);

    GPUTimer timer;
    timer.start();
    pipelineLoadKernel<float><<<numBlocks, blockSize, 0, pipelineStream>>>(d_input, d_temp1, N, 0);
    pipelineProcessKernel<float><<<numBlocks, blockSize, 0, pipelineStream>>>(d_temp1, d_temp2, N, 0);
    pipelineStoreKernel<float><<<numBlocks, blockSize, 0, pipelineStream>>>(d_temp2, d_output, N, 0);
    cudaStreamSynchronize(pipelineStream);
    timer.stop();
    printf("  Serial pipeline: %.3f ms\n", timer.elapsed_ms());

    // Test 13: Overlapped pipeline (chunked)
    printf("\n[Test 13] Overlapped Pipeline (Chunked):\n");
    const size_t chunkSize = N / 4;
    cudaStream_t loadStream, processStream, storeStream;
    cudaEvent_t loadDone, processDone;

    cudaStreamCreate(&loadStream);
    cudaStreamCreate(&processStream);
    cudaStreamCreate(&storeStream);
    cudaEventCreate(&loadDone);
    cudaEventCreate(&processDone);

    timer.start();
    for (size_t offset = 0; offset < N; offset += chunkSize) {
        // Load
        pipelineLoadKernel<float><<<numBlocks, blockSize, 0, loadStream>>>(d_input, d_temp1, N, offset);
        cudaEventRecord(loadDone, loadStream);

        // Wait for load, then process
        cudaStreamWaitEvent(processStream, loadDone, 0);
        pipelineProcessKernel<float><<<numBlocks, blockSize, 0, processStream>>>(d_temp1, d_temp2, N, offset);
        cudaEventRecord(processDone, processStream);

        // Wait for process, then store
        cudaStreamWaitEvent(storeStream, processDone, 0);
        pipelineStoreKernel<float><<<numBlocks, blockSize, 0, storeStream>>>(d_temp2, d_output, N, offset);
    }
    cudaDeviceSynchronize();
    timer.stop();
    printf("  Overlapped pipeline: %.3f ms\n", timer.elapsed_ms());

    cudaEventDestroy(loadDone);
    cudaEventDestroy(processDone);
    cudaStreamDestroy(loadStream);
    cudaStreamDestroy(processStream);
    cudaStreamDestroy(storeStream);
    cudaStreamDestroy(pipelineStream);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_temp1));
    CHECK_CUDA(cudaFree(d_temp2));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// Stream Query and Synchronization Tests
// =============================================================================

static void runStreamSyncTests(size_t N) {
    printf("\n--- Stream Synchronization Tests ---\n");

    size_t bytes = N * sizeof(float);
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 1, bytes));

    // Test 14: cudaStreamQuery (non-blocking)
    printf("\n[Test 14] cudaStreamQuery (Non-blocking):\n");
    cudaStream_t queryStream;
    cudaStreamCreate(&queryStream);

    streamVectorScaleKernel<float><<<numBlocks, blockSize, 0, queryStream>>>(d_data, 2.0f, N);

    int queryCount = 0;
    while (cudaStreamQuery(queryStream) == cudaErrorNotReady) {
        queryCount++;
    }
    printf("  Stream query poll count: %d\n", queryCount);

    cudaStreamDestroy(queryStream);

    // Test 15: cudaStreamSynchronize (blocking)
    printf("\n[Test 15] cudaStreamSynchronize (Blocking):\n");
    cudaStream_t syncStream;
    cudaStreamCreate(&syncStream);

    GPUTimer timer;
    timer.start();
    streamVectorScaleKernel<float><<<numBlocks, blockSize, 0, syncStream>>>(d_data, 2.0f, N);
    cudaStreamSynchronize(syncStream);
    timer.stop();
    printf("  Stream synchronize: %.3f ms\n", timer.elapsed_ms());

    cudaStreamDestroy(syncStream);

    CHECK_CUDA(cudaFree(d_data));
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

void runMultiStreamResearchBenchmarks(size_t N) {
    printf("\n========================================\n");
    printf("Multi-Stream Research Benchmarks\n");
    printf("========================================\n");
    printf("Concepts: Stream priorities, Event sync,\n");
    printf("          Concurrent kernels, Overlap\n");
    printf("========================================\n");

    // Check device support
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("========================================\n");

    runBasicStreamTests(N);
    runStreamPriorityTests(N);
    runEventDependencyTests(N);
    runOverlapTests(N);
    runConcurrentKernelTests(N);
    runPipelineTests(N);
    runStreamSyncTests(N);

    printf("\n--- Multi-Stream Research Complete ---\n");
    printf("NCU Profiling Hints:\n");
    printf("  ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe multi_stream\n");
    printf("  ncu --set full --metrics cuda_stream_kernel_launched.sum ./gpupeek.exe multi_stream\n");
}
