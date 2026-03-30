/*
 * WMMA Tensor Core Benchmark - Using Real MMA.sync Instructions
 * =============================================================
 * Tests actual Tensor Core performance using mma.sync PTX instruction.
 * For Blackwell (sm_90+), uses HMMA.164 (FP16) instruction format.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// WMMA tile dimensions for Blackwell
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel using WMMA API for FP16 Tensor Core computation
__global__ void wmma_tensor_core_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int iterations
) {
    // Each warp processes one WMMA tile (16x16x16)
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Calculate tile position
    int tileRow = blockIdx.x;
    int tileCol = blockIdx.y;

    int row = tileRow * WMMA_M;
    int col = tileCol * WMMA_N;

    if (row >= M || col >= N) return;

    // WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);

    // Perform MMA
    for (int iter = 0; iter < iterations; iter++) {
        load_matrix_sync(a_frag, a + row * K, K);
        load_matrix_sync(b_frag, b + col, K);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(c + row * N + col, c_frag, N, mem_row_major);
}

// Kernel for baseline CUDA core matmul
__global__ void cuda_core_matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int iterations
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
    }
    c[row * N + col] = sum;
}

// Size sweep benchmark
void run_size_sweep() {
    printf("\n");
    printf("================================================================================\n");
    printf("Tensor Core Size Sweep Benchmark\n");
    printf("================================================================================\n\n");

    // Matrix sizes to test
    int sizes[] = {256, 512, 768, 1024, 1536, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int iterations = 10;

    printf("%-8s %-10s %12s %12s %10s\n", "Size", "Status", "Time (ms)", "TFLOPS", "Efficiency");
    printf("%-8s %-10s %12s %12s %10s\n", "----", "------", "---------", "------", "----------");

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i];
        int N = sizes[i];
        int K = sizes[i];

        size_t size_a = M * K * sizeof(__half);
        size_t size_b = K * N * sizeof(__half);
        size_t size_c = M * N * sizeof(float);

        // Allocate device memory
        __half *d_a, *d_b;
        float *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, size_a));
        CHECK_CUDA(cudaMalloc(&d_b, size_b));
        CHECK_CUDA(cudaMalloc(&d_c, size_c));

        // Initialize matrices
        __half *h_a = (__half*)malloc(size_a);
        __half *h_b = (__half*)malloc(size_b);
        for (int j = 0; j < M * K; j++) h_a[j] = __float2half(1.0f);
        for (int j = 0; j < K * N; j++) h_b[j] = __float2half(1.0f);

        CHECK_CUDA(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

        // Grid configuration for WMMA
        dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
        dim3 blockDim(256);  // 8 warps per block

        // Warm up
        wmma_tensor_core_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K, 1);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        wmma_tensor_core_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K, iterations);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        // Calculate TFLOPS: 2*M*N*K FLOPS per iteration
        double flops = 2.0 * M * N * K * iterations;
        double tflops = flops / (ms * 1e6);

        // Theoretical peak: 89 TFLOPS for FP16 on RTX 5080
        double efficiency = tflops / 89.0 * 100.0;

        printf("%-8d %-10s %12.3f %12.4f %10.1f%%\n", M, "OK", ms, tflops, efficiency);

        // Cleanup
        free(h_a);
        free(h_b);
        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    printf("\n");
}

// FP16 vs FP32 comparison
void run_precision_comparison() {
    printf("\n");
    printf("================================================================================\n");
    printf("Precision Comparison: FP16 Tensor Core vs FP32 CUDA Core\n");
    printf("================================================================================\n\n");

    int M = 1024, N = 1024, K = 1024;
    int iterations = 10;

    printf("Matrix Size: %dx%d (%s)\n\n", M, N, "FP16 Tensor Core vs FP32 CUDA Core");

    // Allocate memory for FP16
    size_t size_a = M * K * sizeof(__half);
    size_t size_b = K * N * sizeof(__half);
    size_t size_c = M * N * sizeof(float);

    __half *d_a_fp16, *d_b_fp16;
    float *d_c_fp16;
    CHECK_CUDA(cudaMalloc(&d_a_fp16, size_a));
    CHECK_CUDA(cudaMalloc(&d_b_fp16, size_b));
    CHECK_CUDA(cudaMalloc(&d_c_fp16, size_c));

    // Initialize FP16 matrices
    __half *h_a = (__half*)malloc(size_a);
    __half *h_b = (__half*)malloc(size_b);
    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);

    CHECK_CUDA(cudaMemcpy(d_a_fp16, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_fp16, h_b, size_b, cudaMemcpyHostToDevice));

    // Allocate memory for FP32
    float *d_a_fp32, *d_b_fp32, *d_c_fp32;
    CHECK_CUDA(cudaMalloc(&d_a_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_fp32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c_fp32, M * N * sizeof(float)));

    float *h_a_f = (float*)malloc(M * K * sizeof(float));
    float *h_b_f = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) h_a_f[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_b_f[i] = 1.0f;

    CHECK_CUDA(cudaMemcpy(d_a_fp32, h_a_f, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_fp32, h_b_f, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Grid for WMMA
    dim3 gridDim_wmma((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 blockDim_wmma(256);

    // Grid for CUDA core
    dim3 blockDim_cuda(16, 16);
    dim3 gridDim_cuda((N + 15) / 16, (M + 15) / 16);

    // Benchmark FP16 Tensor Core
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    wmma_tensor_core_kernel<<<gridDim_wmma, blockDim_wmma>>>(d_a_fp16, d_b_fp16, d_c_fp16, M, N, K, 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    wmma_tensor_core_kernel<<<gridDim_wmma, blockDim_wmma>>>(d_a_fp16, d_b_fp16, d_c_fp16, M, N, K, iterations);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fp16;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp16, start, stop));

    double flops_fp16 = 2.0 * M * N * K * iterations;
    double tflops_fp16 = flops_fp16 / (ms_fp16 * 1e6);

    // Benchmark FP32 CUDA Core
    cuda_core_matmul_kernel<<<gridDim_cuda, blockDim_cuda>>>(d_a_fp32, d_b_fp32, d_c_fp32, M, N, K, 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    cuda_core_matmul_kernel<<<gridDim_cuda, blockDim_cuda>>>(d_a_fp32, d_b_fp32, d_c_fp32, M, N, K, iterations);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fp32;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp32, start, stop));

    double flops_fp32 = 2.0 * M * N * K * iterations;
    double gflops_fp32 = flops_fp32 / (ms_fp32 * 1e6);

    printf("%-20s %12s %12s %10s\n", "Precision", "Time (ms)", "Performance", "Speedup");
    printf("%-20s %12s %12s %10s\n", "---------", "---------", "-----------", "-------");
    printf("%-20s %12.3f %12.2f GFLOPS %10.1fx\n", "FP32 (CUDA Core)", ms_fp32, gflops_fp32, 1.0);
    printf("%-20s %12.3f %12.2f TFLOPS %10.1fx\n", "FP16 (Tensor Core)", ms_fp16, tflops_fp16, tflops_fp16 / gflops_fp32);

    printf("\nNote: Tensor Core theoretical peak = 89 TFLOPS FP16 on RTX 5080\n");
    printf("      Current measurement shows CUDA core performance (WMMA API overhead)\n");

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_a_f);
    free(h_b_f);
    CHECK_CUDA(cudaFree(d_a_fp16));
    CHECK_CUDA(cudaFree(d_b_fp16));
    CHECK_CUDA(cudaFree(d_c_fp16));
    CHECK_CUDA(cudaFree(d_a_fp32));
    CHECK_CUDA(cudaFree(d_b_fp32));
    CHECK_CUDA(cudaFree(d_c_fp32));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
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
    printf("Tensor Cores: %s\n", prop.major >= 7 ? "Available" : "Not Available");
    printf("Warp Size: %d\n", prop.warpSize);

    run_size_sweep();
    run_precision_comparison();

    printf("\n================================================================================\n");
    printf("Benchmark complete.\n");
    printf("================================================================================\n");

    return 0;
}
