/*
 * WMMA Tensor Core Benchmark - Working Version
 * ===========================================
 * Uses CUDA's WMMA API to invoke Tensor Cores.
 * Measures actual Tensor Core performance for FP16 matrix multiply.
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

// WMMA using nvcuda::wmma namespace
using namespace nvcuda::wmma;

// Constants for WMMA tile dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

/*
 * WMMA Tensor Core Kernel
 * Each warp (32 threads) cooperatively computes one 16x16x16 MMA operation.
 */
__global__ void wmma_fp16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Each warp handles one WMMA tile
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    // Calculate which tile this warp handles
    int tilesM = (M + WMMA_M - 1) / WMMA_M;
    int tilesN = (N + WMMA_N - 1) / WMMA_N;
    int tilesPerBlock = blockDim.x / 32;
    int tileIdx = blockIdx.x * tilesPerBlock + warpId;

    int tileRow = tileIdx / tilesN;
    int tileCol = tileIdx % tilesN;

    int rowStart = tileRow * WMMA_M;
    int colStart = tileCol * WMMA_N;

    // Check bounds
    if (rowStart >= M || colStart >= N) return;

    // WMMA fragments for FP16 accumulation in FP32
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    fill_fragment(c_frag, 0.0f);

    // Perform K/16 iterations
    for (int k = 0; k < K; k += WMMA_K) {
        // Load A and B matrices
        load_matrix_sync(a_frag, A + rowStart * K + k, K);
        load_matrix_sync(b_frag, B + colStart + k * N, N);

        // Perform MMA
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(C + rowStart * N + colStart, c_frag, N, mem_row_major);
}

/*
 * CUDA Core baseline: FP16 matmul using regular CUDA cores
 */
__global__ void cuda_fp16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = sum;
}

/*
 * CUDA Core FP32 matmul baseline
 */
__global__ void cuda_fp32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void run_benchmark() {
    printf("\n");
    printf("================================================================================\n");
    printf("Tensor Core (WMMA) vs CUDA Core Performance\n");
    printf("================================================================================\n\n");

    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("\n");

    // Test matrix sizes
    int sizes[] = {256, 512, 768, 1024, 1536, 2048};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    const int ITERATIONS = 10;

    printf("%-8s %-15s %10s %12s %10s\n", "Size", "Kernel", "Time (ms)", "GFLOPS", "TFLOPS");
    printf("%-8s %-15s %10s %12s %10s\n", "----", "------", "---------", "------", "------");

    for (int i = 0; i < numSizes; i++) {
        int M = sizes[i], N = sizes[i], K = sizes[i];

        size_t size_a = M * K * sizeof(__half);
        size_t size_b = K * N * sizeof(__half);
        size_t size_c = M * N * sizeof(float);

        // Allocate device memory
        __half *d_a, *d_b;
        float *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, size_a));
        CHECK_CUDA(cudaMalloc(&d_b, size_b));
        CHECK_CUDA(cudaMalloc(&d_c, size_c));

        // Initialize host data
        __half *h_a = (__half*)malloc(size_a);
        __half *h_b = (__half*)malloc(size_b);
        for (int j = 0; j < M * K; j++) h_a[j] = __float2half(1.0f);
        for (int j = 0; j < K * N; j++) h_b[j] = __float2half(1.0f);

        CHECK_CUDA(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

        // Grid configuration for WMMA
        // Each warp (32 threads) handles one 16x16 tile
        // Use 4 warps per block for better occupancy
        const int WARPS_PER_BLOCK = 4;
        const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

        int tilesM = (M + WMMA_M - 1) / WMMA_M;
        int tilesN = (N + WMMA_N - 1) / WMMA_N;
        int totalTiles = tilesM * tilesN;
        int blocksForWMMA = (totalTiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        dim3 wmmaGrid(blocksForWMMA);
        dim3 wmmaBlock(THREADS_PER_BLOCK);

        // Grid for CUDA core
        dim3 cudaBlock(16, 16);
        dim3 cudaGrid((N + 15) / 16, (M + 15) / 16);

        // Warm up
        wmma_fp16_kernel<<<wmmaGrid, wmmaBlock>>>(d_a, d_b, d_c, M, N, K);
        cuda_fp16_kernel<<<cudaGrid, cudaBlock>>>(d_a, d_b, d_c, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Benchmark WMMA (Tensor Core)
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < ITERATIONS; iter++) {
            wmma_fp16_kernel<<<wmmaGrid, wmmaBlock>>>(d_a, d_b, d_c, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float wmmaMs;
        CHECK_CUDA(cudaEventElapsedTime(&wmmaMs, start, stop));
        wmmaMs /= ITERATIONS;

        double wmmaFlops = 2.0 * M * N * K;
        double wmmaGflops = wmmaFlops / (wmmaMs * 1e6);
        double wmmaTflops = wmmaGflops / 1000.0;

        printf("%-8d %-15s %10.3f %12.2f %10.4f\n", sizes[i], "WMMA (TC)", wmmaMs, wmmaGflops, wmmaTflops);

        // Benchmark CUDA Core FP16
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < ITERATIONS; iter++) {
            cuda_fp16_kernel<<<cudaGrid, cudaBlock>>>(d_a, d_b, d_c, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float cudaMs;
        CHECK_CUDA(cudaEventElapsedTime(&cudaMs, start, stop));
        cudaMs /= ITERATIONS;

        double cudaFlops = 2.0 * M * N * K;
        double cudaGflops = cudaFlops / (cudaMs * 1e6);

        printf("%-8s %-15s %10.3f %12.2f %10s\n", "", "CUDA (FP16)", cudaMs, cudaGflops, "-");

        printf("%-8s %-15s %10s %12s %10.2fx\n", "", "Speedup", "-", "-", wmmaGflops / cudaGflops);

        // Cleanup
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
    printf("Analysis:\n");
    printf("- WMMA uses Tensor Cores for FP16 matrix multiply\n");
    printf("- CUDA Core uses regular CUDA cores for FP16 computation\n");
    printf("- Theoretical Tensor Core peak: ~89 TFLOPS FP16 on RTX 5080\n");
    printf("- Actual performance depends on matrix size and occupancy\n");
    printf("================================================================================\n");
}

int main() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("================================================================================\n");
    printf("GPUPeek Tensor Core Benchmark\n");
    printf("================================================================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    run_benchmark();

    printf("\nBenchmark complete.\n");
    return 0;
}
