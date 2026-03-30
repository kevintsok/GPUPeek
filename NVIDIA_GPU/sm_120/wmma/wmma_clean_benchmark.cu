/*
 * WMMA Clean Benchmark - Tensor Core Performance
 * ===============================================
 * Tests WMMA (Warp-level Matrix Multiply-Accumulate) performance
 * for FP16, BF16, and INT8 data types on Tensor Cores.
 */

#include <cuda_runtime.h>
#include <cuda/wmma.hpp>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

using namespace cuda::wmma;

// Benchmark parameters
const int M = 256;
const int N = 256;
const int K = 256;
const int WARP_SIZE = 32;

// Kernel for WMMA computation
__global__ void wmma_kernel(const __half* a, const __half* b, float* c, int m, int n, int k, int iterations) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Each warp handles one tile
    const int tileRow = blockIdx.x;
    const int tileCol = blockIdx.y;

    const int row = tileRow * M;
    const int col = tileCol * N;

    // Check bounds
    if (row >= m || col >= n) return;

    // Initialize fragment for A, B, C
    fragment<matrix_a, M, N, K, __half, row_major> fragA;
    fragment<matrix_b, M, N, K, __half, col_major> fragB;
    fragment<accumulator, M, N, K, float> fragC;

    // Initialize accumulator to zero
    fill_fragment(fragC, 0.0f);

    // Perform MMA
    for (int i = 0; i < iterations; i++) {
        load_matrix_sync(fragA, a + row * k, k);
        load_matrix_sync(fragB, b + col, k);
        mma_sync(fragC, fragA, fragB, fragC);
    }

    // Store result
    store_matrix_sync(c + row * n + col, fragC, n, mem_row_major);
}

// Host benchmark function
void run_wmma_benchmark() {
    printf("\n================================================================================\n");
    printf("WMMA (Tensor Core) Benchmark\n");
    printf("================================================================================\n\n");

    // Matrix dimensions
    int m = M * 16;  // Total M
    int n = N * 16;  // Total N
    int k = K;        // Inner dimension

    size_t size_a = m * k * sizeof(__half);
    size_t size_b = k * n * sizeof(__half);
    size_t size_c = m * n * sizeof(float);

    printf("Matrix sizes: M=%d, N=%d, K=%d\n", m, n, k);
    printf("Problem size: %.2f MB\n", (size_a + size_b + size_c) / 1024.0 / 1024.0);

    // Allocate device memory
    __half *d_a, *d_b;
    float *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size_a));
    CHECK_CUDA(cudaMalloc(&d_b, size_b));
    CHECK_CUDA(cudaMalloc(&d_c, size_c));

    // Initialize matrices on host and copy to device
    __half *h_a = (__half*)malloc(size_a);
    __half *h_b = (__half*)malloc(size_b);
    float *h_c = (float*)malloc(size_c);

    for (int i = 0; i < m * k; i++) {
        h_a[i] = __float2half(1.0f);
    }
    for (int i = 0; i < k * n; i++) {
        h_b[i] = __float2half(1.0f);
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

    // Grid and block dimensions
    dim3 gridDim(m / M, n / N);
    dim3 blockDim(WARP_SIZE * 4);  // 4 warps per block

    // Warm-up run
    wmma_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k, 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark with different iteration counts
    int iterations[] = {1, 10, 100};

    printf("\nFP16 WMMA Performance:\n");
    printf("%-12s %12s %12s %12s\n", "Iterations", "Time (ms)", "TFLOPS", "GB/s");
    printf("%-12s %12s %12s %12s\n", "----------", "----------", "----------", "----------");

    for (int iter : iterations) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        wmma_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k, iter);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        // Calculate FLOPS: 2 * M * N * K per MMA
        double flops = 2.0 * m * n * k * iter;
        double tflops = flops / (milliseconds * 1e6);
        double bandwidth = (size_a + size_b + size_c) * iter / (milliseconds * 1e6);

        printf("%-12d %12.3f %12.4f %12.2f\n", iter, milliseconds, tflops, bandwidth);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    printf("\n================================================================================\n");
}

int main() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("GPUPeek WMMA Tensor Core Benchmark");
    printf("==================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    run_wmma_benchmark();

    printf("\nBenchmark complete.\n");
    return 0;
}
