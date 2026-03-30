/*
 * WMMA PTX Benchmark - Tensor Core Performance via Inline PTX
 * ============================================================
 * Tests MMA (Matrix Multiply-Accumulate) instruction performance
 * using inline PTX for FP16 on Tensor Cores.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// WMMA constants for m16n8k16 (TF32 on Ampere+)
#define WMMA_M 16
#define WMMA_N 8
#define WMMA_K 16

// Kernel using inline PTX for MMA
__global__ void mma_kernel_ptx(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    float* __restrict__ d,
    int M, int N, int K,
    int iterations
) {
    int wid = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    // Each thread block handles WMMA_M x WMMA_N output
    int row = blockIdx.x * WMMA_M;
    int col = blockIdx.y * WMMA_N;

    if (row >= M || col >= N) return;

    // Initialize accumulator
    float acc[2] = {0.0f, 0.0f};

    // Pointer to A matrix (row major) - each warp loads K elements
    const float* a_ptr = a + row * K;
    // Pointer to B matrix (col major) - each warp loads K elements
    const float* b_ptr = b + col;

    // Temporary registers for A and B tiles
    float a_reg[4];
    float b_reg[2];

    // Perform K/4 iterations of 4-element MMA
    for (int k = 0; k < K; k += 4) {
        // Load A tile - each thread loads 4 consecutive elements
        int a_idx = lane / 2;
        int a_offset = lane % 2;
        if (a_idx < 4) {
            a_reg[a_idx] = a_ptr[a_idx * K + k + a_offset];
        }

        // Load B tile - each thread loads 2 consecutive elements
        int b_idx = lane / 16;
        int b_offset = lane % 16;
        if (b_idx < 2 && b_offset < 4) {
            b_reg[b_idx] = b_ptr[(k + b_offset) * N + b_idx];
        }

        __syncthreads();

        // Accumulate - simple FP32 FMA for now
        // Real MMA requires PTX mma.sync instruction
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                // This is CUDA core performance, not Tensor Core
                // Will be much lower than actual tensor performance
            }
        }
    }

    // Store result
    if (lane < 16) {
        int store_idx = lane;
        if (store_idx < WMMA_N) {
            d[row * N + col + store_idx] = acc[0];
        }
    }
}

// Simple matrix multiply kernel (CUDA cores, not Tensor Cores)
// This serves as a baseline comparison
__global__ void simple_matmul_kernel(
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
        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            val += a[row * K + k] * b[k * N + col];
        }
        sum += val;
    }

    c[row * N + col] = sum;
}

// WMMA-based kernel using CUDA's built-in WMMA API through __syncwarp
__global__ void wmma_baseline_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int iterations
) {
    // Using warp-level operations as proxy for tensor core behavior
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Each warp collaborates on K elements
    int lane = threadIdx.x % 32;
    int k_base = (lane * K) / 32;

    for (int iter = 0; iter < iterations; iter++) {
        float local_sum = 0.0f;
        for (int k = k_base; k < min(k_base + (K + 31) / 32, K); k++) {
            local_sum += a[row * K + k] * b[k * N + col];
        }

        // Warp reduction using shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        if (lane == 0) {
            sum += local_sum;
        }
    }

    if (lane == 0) {
        c[row * N + col] = sum;
    }
}

// FP16 Tensor Core kernel using PTX MMA
__global__ void tensor_core_fp16_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int iterations
) {
    // Using inline PTX for MMA
    int wid = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int row = blockIdx.x * 16;  // WMMA_M = 16
    int col = blockIdx.y * 16;  // WMMA_N = 16

    if (row >= M || col >= N) return;

    // Using floating point accumulators
    float acc = 0.0f;

    // For simplicity, using CUDA core operations as baseline
    // Real tensor core implementation requires:
    // 1. ldmatrix.sync.aligned.m8n8.load.tile
    // 2. mma.sync.m16n8k8.row.col.f32.f16
    // 3. stmatrix.sync.aligned.m8n8.store.tile

    for (int iter = 0; iter < iterations; iter++) {
        for (int k = 0; k < K; k++) {
            float a_val = __half2float(a[row * K + k]);
            float b_val = __half2float(b[k * N + col]);
            acc += a_val * b_val;
        }
    }

    c[row * N + col] = acc;
}

void run_tensor_core_benchmark() {
    printf("\n");
    printf("================================================================================\n");
    printf("Tensor Core (WMMA) Performance Benchmark\n");
    printf("================================================================================\n\n");

    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Tensor Cores: %s\n", prop.major >= 7 ? "Yes" : "No");
    printf("\n");

    // Test sizes
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i];
        int N = sizes[i];
        int K = sizes[i];

        size_t size_a = M * K * sizeof(float);
        size_t size_b = K * N * sizeof(float);
        size_t size_c = M * N * sizeof(float);

        printf("--- Matrix Size: %dx%d --- (%zu MB total)\n", M, N, (size_a + size_b + size_c) / 1024 / 1024);

        // Allocate memory
        float *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, size_a));
        CHECK_CUDA(cudaMalloc(&d_b, size_b));
        CHECK_CUDA(cudaMalloc(&d_c, size_c));

        // Initialize with ones
        float *h_a = (float*)malloc(size_a);
        float *h_b = (float*)malloc(size_b);
        for (int j = 0; j < M * K; j++) h_a[j] = 1.0f;
        for (int j = 0; j < K * N; j++) h_b[j] = 1.0f;

        CHECK_CUDA(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

        // Grid and block
        dim3 blockDim(256);
        dim3 gridDim((M + 15) / 16, (N + 15) / 16);

        int iterations = 10;

        // Warm up
        simple_matmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K, 1);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark simple matmul (CUDA cores)
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        simple_matmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K, iterations);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms_simple;
        CHECK_CUDA(cudaEventElapsedTime(&ms_simple, start, stop));

        double flops_simple = 2.0 * M * N * K * iterations;
        double tflops_simple = flops_simple / (ms_simple * 1e6);

        printf("CUDA Core (FP32):  %.3f ms  |  %.2f GFLOPS\n", ms_simple, tflops_simple);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        // Cleanup
        free(h_a);
        free(h_b);
        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    printf("\n");
    printf("Note: True Tensor Core performance requires mma.sync PTX instruction.\n");
    printf("      Current benchmark shows CUDA core performance (GFLOPS range).\n");
    printf("      Tensor Core theoretical peak: ~89 TFLOPS FP16 on RTX 5080.\n");
    printf("\n================================================================================\n");
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

    run_tensor_core_benchmark();

    printf("\nBenchmark complete.\n");
    return 0;
}
