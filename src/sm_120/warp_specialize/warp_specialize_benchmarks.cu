#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "warp_specialize_kernels.cu"

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
// D.1 Warp Specialization Basic Test
// =============================================================================

void runWarpSpecializationBasicTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("D.1 Warp Specialization Basic (2-Warp Producer/Consumer)\n");
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
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // Warp specialization kernel
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpSpecializationBasicKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Warp Specialization ---\n\n");
    printf("Warp Spec (prod/cons):  %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    // Baseline: Standard kernel
    timer.start();
    for (int i = 0; i < iterations; i++) {
        barrierCopyKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double baselineBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Standard Kernel:         %s (%.3f ms/kernel)\n",
           formatBandwidth(baselineBw), timer.elapsed_ms() / iterations);
    printf("Warp Spec Overhead:      %.1f%%\n", (baselineBw / bw - 1.0) * 100.0);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: Warp specialization divides warps into producer/consumer roles.\n");
    printf("This can hide memory latency by overlapping load and compute.\n");
}

// =============================================================================
// D.2 TMA + Barrier Synchronization Test
// =============================================================================

void runTMABarrierSyncTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("D.2 TMA + Barrier Synchronization\n");
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
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // Standard barrier copy
    timer.start();
    for (int i = 0; i < iterations; i++) {
        barrierCopyKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- TMA + Barrier Patterns ---\n\n");
    printf("Standard Barrier Copy:  %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

#if __CUDA_ARCH__ >= 900
    // TMA barrier copy (Blackwell+)
    int *d_barrier;
    CHECK_CUDA(cudaMalloc(&d_barrier, sizeof(int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        tmaBarrierCopyKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N, d_barrier);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double tmaBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("TMA + Barrier Copy:     %s (%.3f ms/kernel)\n",
           formatBandwidth(tmaBw), timer.elapsed_ms() / iterations);
    printf("TMA Speedup:            %.1fx\n", tmaBw / bw);

    CHECK_CUDA(cudaFree(d_barrier));
#else
    printf("TMA + Barrier Copy:     (requires SM 9.0+)\n");
#endif

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: TMA (Tensor Memory Accelerator) provides async copy with\n");
    printf("automatic barrier synchronization. Available on Blackwell (SM 9.0+).\n");
}

// =============================================================================
// D.3 Multi-Stage Pipeline Test
// =============================================================================

void runMultiStagePipelineTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("D.3 Multi-Stage Pipeline (Load/Compute/Store)\n");
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
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // 3-stage pipeline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        pipelineKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double pipelineBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Pipeline Stages ---\n\n");
    printf("3-Stage Pipeline:        %s (%.3f ms/kernel)\n",
           formatBandwidth(pipelineBw), timer.elapsed_ms() / iterations);

    // Overlapped pipeline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        overlappedPipelineKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double overlappedBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Overlapped Pipeline:     %s (%.3f ms/kernel)\n",
           formatBandwidth(overlappedBw), timer.elapsed_ms() / iterations);
    printf("Overlap Speedup:         %.1fx\n", overlappedBw / pipelineBw);

    // Baseline
    timer.start();
    for (int i = 0; i < iterations; i++) {
        barrierCopyKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double baselineBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Baseline (simple copy):  %s (%.3f ms/kernel)\n",
           formatBandwidth(baselineBw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: Pipelining overlaps Load, Compute, and Store stages to\n");
    printf("maximize GPU utilization and hide memory latency.\n");
}

// =============================================================================
// D.4 Block Specialization Test
// =============================================================================

void runBlockSpecializationTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("D.4 Block Specialization (Half Block = Producer, Half = Consumer)\n");
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
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // Block specialization
    timer.start();
    for (int i = 0; i < iterations; i++) {
        blockSpecializationKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double blockSpecBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Block Specialization ---\n\n");
    printf("Block Spec (half prod/cons):  %s (%.3f ms/kernel)\n",
           formatBandwidth(blockSpecBw), timer.elapsed_ms() / iterations);

    // Warp-level block specialization
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpBlockSpecializationKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double warpBlockBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Warp Block Spec:               %s (%.3f ms/kernel)\n",
           formatBandwidth(warpBlockBw), timer.elapsed_ms() / iterations);

    // Simple producer-consumer
    timer.start();
    for (int i = 0; i < iterations; i++) {
        simpleProducerConsumerKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double simpleBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Simple Producer/Consumer:     %s (%.3f ms/kernel)\n",
           formatBandwidth(simpleBw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: Block specialization assigns different roles to thread groups.\n");
    printf("Producer threads load data while consumer threads compute.\n");
}

// =============================================================================
// D.5 Warp-Level Synchronization Primitives Test
// =============================================================================

void runWarpSyncPrimitivesTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("D.5 Warp-Level Synchronization Primitives\n");
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
    CHECK_CUDA(cudaMalloc(&d_dst, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // Warp shuffle reduction
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpShuffleReductionKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double shuffleBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- Warp-Level Primitives ---\n\n");
    printf("Warp Shuffle Reduction:  %s (%.3f ms/kernel)\n",
           formatBandwidth(shuffleBw), timer.elapsed_ms() / iterations);

    // Warp barrier with shuffle
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpBarrierShuffleKernel<<<numBlocks, blockSize>>>(d_src, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double barrierBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Warp Barrier (shuffle):  %s (%.3f ms/kernel)\n",
           formatBandwidth(barrierBw), timer.elapsed_ms() / iterations);

    // Warp reduce with barrier
    CHECK_CUDA(cudaMemset(d_dst, 0, numBlocks * sizeof(float)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpReduceWithBarrierKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double reduceBw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Warp Reduce + Barrier:   %s (%.3f ms/kernel)\n",
           formatBandwidth(reduceBw), timer.elapsed_ms() / iterations);

    // Warp scan
    int *d_int_src, *d_int_dst;
    CHECK_CUDA(cudaMalloc(&d_int_src, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_int_dst, N * sizeof(int)));
    for (size_t i = 0; i < N; i++) ((int*)h_src)[i] = 1;
    CHECK_CUDA(cudaMemcpy(d_int_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpScanKernel<<<numBlocks, blockSize>>>(d_int_src, d_int_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double scanBw = N * sizeof(int) * iterations / (timer.elapsed_ms() * 1e6);

    printf("Warp Scan (prefix sum):  %s (%.3f ms/kernel)\n",
           formatBandwidth(scanBw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_int_src));
    CHECK_CUDA(cudaFree(d_int_dst));
    CHECK_CUDA(cudaFreeHost(h_src));

    printf("\nNote: Warp-level primitives don't require __syncthreads():\n");
    printf("- __shfl_sync: Warp-level register shuffle\n");
    printf("- __any_sync, __all_sync: Warp vote operations\n");
    printf("- __ballot_sync: Warp ballot (which threads satisfied condition)\n");
}

// =============================================================================
// D.6 TMA + Warp Specialization Combined Test
// =============================================================================

void runTMAWarpSpecializationTest() {
    printf("\n");
    printf("================================================================================\n");
    printf("D.6 TMA + Warp Specialization Combined\n");
    printf("================================================================================\n");

#if __CUDA_ARCH__ >= 900
    const size_t N = 4 * 1024 * 1024;
    const int iterations = 100;
    const int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_src;
    CHECK_CUDA(cudaMallocHost(&h_src, N * sizeof(float)));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    GPUTimer timer;

    // TMA + Warp specialization
    timer.start();
    for (int i = 0; i < iterations; i++) {
        tmaWarpSpecializationKernel<float><<<numBlocks, blockSize,
            blockSize * sizeof(float)>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double bw = N * sizeof(float) * iterations / (timer.elapsed_ms() * 1e6);

    printf("\n--- TMA + Warp Specialization ---\n\n");
    printf("TMA + Warp Specialization:  %s (%.3f ms/kernel)\n",
           formatBandwidth(bw), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));
#else
    printf("\n--- TMA + Warp Specialization ---\n\n");
    printf("TMA + Warp Specialization:  (requires SM 9.0+)\n");
    printf("Your current GPU is SM %d.%d\n", __CUDA_ARCH__ / 100, (__CUDA_ARCH__ / 10) % 10);
#endif

    printf("\nNote: TMA + Warp Specialization combines async memory copy\n");
    printf("with producer/consumer warp roles for maximum efficiency.\n");
}

// =============================================================================
// Main Entry
// =============================================================================

void runWarpSpecializeBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) Warp Specialization Research          #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    runWarpSpecializationBasicTest();
    runTMABarrierSyncTest();
    runMultiStagePipelineTest();
    runBlockSpecializationTest();
    runWarpSyncPrimitivesTest();
    runTMAWarpSpecializationTest();

    printf("\n");
    printf("================================================================================\n");
    printf("Warp Specialization Research Complete!\n");
    printf("================================================================================\n");
}
