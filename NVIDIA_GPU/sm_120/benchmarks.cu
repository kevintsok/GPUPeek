#include <cuda_runtime.h>
#include <stdio.h>
#include "arch_kernels.cu"
#include "arch.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

void runSM120Benchmarks(size_t N) {
    printf("\n=== SM 12.0 (Blackwell) Specific Benchmarks ===\n");

    sm_120::printArchSpecificInfo();

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    size_t bytes = N * sizeof(float);
    size_t int_bytes = N * sizeof(int);

    // Allocate memory
    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_src, 1, bytes));

    // For kernels that need int input
    int *d_int_src, *d_int_dst;
    CHECK_CUDA(cudaMalloc(&d_int_src, int_bytes));
    CHECK_CUDA(cudaMalloc(&d_int_dst, int_bytes));
    CHECK_CUDA(cudaMemset(d_int_src, 1, int_bytes));

    GPUTimer timer;

    // Enhanced Warp Shuffle (uses int*)
    printf("\n--- Enhanced Warp Operations ---\n");
    timer.start();
    for (int i = 0; i < 100; i++) {
        sm_120::enhancedWarpShuffleKernel<<<numBlocks, blockSize>>>(d_int_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Enhanced Shuffle:    %.2f GB/s (%.3f ms)\n",
           bytes * 100 / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / 100);

    // Async Copy
    printf("\n--- Memory Operations ---\n");
    timer.start();
    for (int i = 0; i < 100; i++) {
        sm_120::asyncCopyKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Async Copy:          %.2f GB/s (%.3f ms)\n",
           bytes * 100 / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / 100);

    // L2 Streaming
    size_t stride = 16;
    timer.start();
    for (int i = 0; i < 100; i++) {
        sm_120::l2StreamingKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N, stride);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("L2 Streaming:        %.2f GB/s (%.3f ms)\n",
           bytes * 100 / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / 100);

    // Register Bandwidth
    timer.start();
    for (int i = 0; i < 100; i++) {
        sm_120::registerBandwidthKernel<<<numBlocks, blockSize>>>(d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Register Bandwidth:  %.2f GB/s (%.3f ms)\n",
           bytes * 100 / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / 100);

    // Prefetch pattern
    timer.start();
    for (int i = 0; i < 100; i++) {
        sm_120::prefetchKernel<<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Software Prefetch:   %.2f GB/s (%.3f ms)\n",
           bytes * 100 / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / 100);

    // Reduced Precision
    __half *d_half_dst;
    CHECK_CUDA(cudaMalloc(&d_half_dst, N * sizeof(__half)));
    timer.start();
    for (int i = 0; i < 100; i++) {
        sm_120::reducedPrecisionKernel<<<numBlocks, blockSize>>>(d_src, d_half_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Reduced Precision:   %.2f GB/s (%.3f ms)\n",
           bytes * 100 / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / 100);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_int_src));
    CHECK_CUDA(cudaFree(d_int_dst));
    CHECK_CUDA(cudaFree(d_half_dst));
}
