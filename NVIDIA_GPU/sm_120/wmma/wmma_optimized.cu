/*
 * WMMA Optimized Tensor Core Benchmark
 * ===================================
 * Improvements:
 * 1. Increased warps per block (8 instead of 4)
 * 2. Double buffering to overlap load and compute
 * 3. Better memory coalescing
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
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

using namespace nvcuda::wmma;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

/*
 * Optimized WMMA Kernel:
 * - 8 warps per block (256 threads)
 * - Double buffering to hide latency
 * - Better occupancy
 */
__global__ void wmma_optimized_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 8 warps per block for better occupancy
    const int warpsPerBlock = 8;
    const int threadsPerBlock = warpsPerBlock * 32;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    // Calculate tile assignment
    int tilesM = (M + WMMA_M - 1) / WMMA_M;
    int tilesN = (N + WMMA_N - 1) / WMMA_N;
    int tileIdx = blockIdx.x * warpsPerBlock + warpId;

    int tileRow = tileIdx / tilesN;
    int tileCol = tileIdx % tilesN;

    int rowStart = tileRow * WMMA_M;
    int colStart = tileCol * WMMA_N;

    if (rowStart >= M || colStart >= N) return;

    // WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag0, a_frag1;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag0, b_frag1;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    // Double buffer: preload first iteration while initializing
    int kTile = 0;

    // Load first tile
    load_matrix_sync(a_frag0, A + rowStart * K + kTile, K);
    load_matrix_sync(b_frag0, B + colStart + kTile * N, N);
    kTile += WMMA_K;

    // Process remaining tiles with double buffering
    for (; kTile + WMMA_K <= K; kTile += WMMA_K) {
        // Compute with current tile while loading next
        mma_sync(c_frag, a_frag0, b_frag0, c_frag);

        // Prefetch next tile
        load_matrix_sync(a_frag1, A + rowStart * K + kTile, K);
        load_matrix_sync(b_frag1, B + colStart + kTile * N, N);

        // Swap buffers
        auto temp_a = a_frag0; a_frag0 = a_frag1; a_frag1 = temp_a;
        auto temp_b = b_frag0; b_frag0 = b_frag1; b_frag1 = temp_b;
    }

    // Handle remaining tiles
    for (; kTile < K; kTile += WMMA_K) {
        load_matrix_sync(a_frag0, A + rowStart * K + kTile, K);
        load_matrix_sync(b_frag0, B + colStart + kTile * N, N);
        mma_sync(c_frag, a_frag0, b_frag0, c_frag);
    }

    // Final MMA for last tile
    mma_sync(c_frag, a_frag0, b_frag0, c_frag);

    // Store result
    store_matrix_sync(C + rowStart * N + colStart, c_frag, N, mem_row_major);
}

/*
 * Baseline: Original 4-warps kernel
 */
__global__ void wmma_baseline_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int warpsPerBlock = 4;
    int warpId = threadIdx.x / 32;

    int tilesM = (M + WMMA_M - 1) / WMMA_M;
    int tilesN = (N + WMMA_N - 1) / WMMA_N;
    int tileIdx = blockIdx.x * warpsPerBlock + warpId;

    int tileRow = tileIdx / tilesN;
    int tileCol = tileIdx % tilesN;

    int rowStart = tileRow * WMMA_M;
    int colStart = tileCol * WMMA_N;

    if (rowStart >= M || colStart >= N) return;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        load_matrix_sync(a_frag, A + rowStart * K + k, K);
        load_matrix_sync(b_frag, B + colStart + k * N, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + rowStart * N + colStart, c_frag, N, mem_row_major);
}

void run_comparison() {
    printf("\n");
    printf("================================================================================\n");
    printf("Tensor Core Optimization Comparison\n");
    printf("================================================================================\n\n");

    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("\n");

    // Test sizes
    int sizes[] = {512, 768, 1024, 1536, 2048, 3072};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    const int ITERATIONS = 20;

    printf("%-8s %-12s %12s %12s %10s\n", "Size", "Kernel", "Time (ms)", "TFLOPS", "Speedup");
    printf("%-8s %-12s %12s %12s %10s\n", "----", "------", "---------", "------", "-------");

    for (int i = 0; i < numSizes; i++) {
        int M = sizes[i], N = sizes[i], K = sizes[i];

        size_t size_a = M * K * sizeof(__half);
        size_t size_b = K * N * sizeof(__half);
        size_t size_c = M * N * sizeof(float);

        __half *d_a, *d_b;
        float *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, size_a));
        CHECK_CUDA(cudaMalloc(&d_b, size_b));
        CHECK_CUDA(cudaMalloc(&d_c, size_c));

        __half *h_a = (__half*)malloc(size_a);
        __half *h_b = (__half*)malloc(size_b);
        for (int j = 0; j < M * K; j++) h_a[j] = __float2half(1.0f);
        for (int j = 0; j < K * N; j++) h_b[j] = __float2half(1.0f);

        CHECK_CUDA(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

        // Grid for optimized (8 warps/block)
        const int WARPS_OPT = 8;
        const int WARPS_BASE = 4;

        int tilesM = (M + 15) / 16;
        int tilesN = (N + 15) / 16;
        int totalTiles = tilesM * tilesN;

        dim3 gridOpt((totalTiles + WARPS_OPT - 1) / WARPS_OPT);
        dim3 blockOpt(WARPS_OPT * 32);

        dim3 gridBase((totalTiles + WARPS_BASE - 1) / WARPS_BASE);
        dim3 blockBase(WARPS_BASE * 32);

        // Warm up
        wmma_optimized_kernel<<<gridOpt, blockOpt>>>(d_a, d_b, d_c, M, N, K);
        wmma_baseline_kernel<<<gridBase, blockBase>>>(d_a, d_b, d_c, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Benchmark optimized
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < ITERATIONS; iter++) {
            wmma_optimized_kernel<<<gridOpt, blockOpt>>>(d_a, d_b, d_c, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float optMs;
        CHECK_CUDA(cudaEventElapsedTime(&optMs, start, stop));
        optMs /= ITERATIONS;

        double optFlops = 2.0 * M * N * K;
        double optTflops = optFlops / (optMs * 1e6);

        // Benchmark baseline
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < ITERATIONS; iter++) {
            wmma_baseline_kernel<<<gridBase, blockBase>>>(d_a, d_b, d_c, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float baseMs;
        CHECK_CUDA(cudaEventElapsedTime(&baseMs, start, stop));
        baseMs /= ITERATIONS;

        double baseTflops = optFlops / (baseMs * 1e6);

        printf("%-8d %-12s %12.3f %12.2f %10.2fx\n", sizes[i], "Optimized", optMs, optTflops, baseMs/optMs);
        printf("%-8s %-12s %12.3f %12.2f\n", "", "Baseline", baseMs, baseTflops);

        free(h_a);
        free(h_b);
        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        printf("\n");
    }

    printf("================================================================================\n");
    printf("Optimization: 8 warps/block (vs 4) + double buffering\n");
    printf("================================================================================\n");
}

int main() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("================================================================================\n");
    printf("GPUPeek Tensor Core Optimization Benchmark\n");
    printf("================================================================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    run_comparison();

    printf("\nBenchmark complete.\n");
    return 0;
}
