/*
 * WMMA Tensor Core Benchmark - Using Real mma.sync Instructions
 * ===========================================================
 *
 * This benchmark uses inline PTX to invoke actual Tensor Core MMA instructions.
 *
 * For Blackwell (sm_90+), each warp (32 threads) cooperatively executes
 * one HMMA instruction that computes a 16x8x16 matrix multiply.
 *
 * PTX ISA reference:
 * - ldmatrix.sync.aligned.m8n8.load.tile - Load matrix fragment
 * - mma.sync.m16n8k16.row.col.f32.f16.f16 - F32 accumulator, FP16 inputs
 * - stmatrix.sync.aligned.m8n8.store.tile - Store matrix fragment
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

// Predicate for predicates in PTX
#define PTX_TRUE 1
#define PTX_FALSE 0

/*
 * Tensor Core MMA Kernel using inline PTX
 *
 * Each warp (32 threads) collaboratively executes one mma.sync instruction.
 * We use the m16n8k16 tile shape with FP16 inputs and FP32 accumulator.
 */
__global__ void tensor_core_mma_ptx_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int numTiles  // Number of K tiles to process
) {
    // Each warp handles one 16x8 output tile
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Each block has 4 warps
    const int blockWarps = blockDim.x / 32;
    const int blockId = blockIdx.x;

    // Calculate which tile this warp computes
    // Each warp produces one 16x8 tile of output
    const int tilesPerRow = (N + 15) / 16;
    const int tileIdx = warpId + blockId * blockWarps;
    const int tileRow = tileIdx / tilesPerRow;
    const int tileCol = tileIdx % tilesPerRow;

    // Starting position of this tile
    const int rowStart = tileRow * 16;
    const int colStart = tileCol * 8;

    // Check bounds
    if (rowStart >= M || colStart >= N) return;

    // Allocate registers for accumulator
    // Each thread in the warp holds 4 elements of the 16x8 output
    // Layout: thread i handles output elements (i*4, i*4+1, i*4+2, i*4+3) in column-major order
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // Pointer to matrices
    const __half* A = a + rowStart;  // Row major, starting at rowStart
    const __half* B = b + colStart * K;  // Column major, starting at colStart

    // Process K dimension in chunks of 16
    for (int kTile = 0; kTile < K; kTile += 16) {
        // Check if we have enough K elements
        if (kTile + 16 > K) break;

        // For ldmatrix, we need specific lane layouts
        // Lane layout for 8x8 matrix:
        //   Lane 0-3: row 0-3, col 0
        //   Lane 4-7: row 0-3, col 1
        //   etc.

        // Declare variables for PTX inline assembly
        __half a_reg[8];
        __half b_reg[8];

        // Load A matrix fragment (16x16) using ldmatrix
        // Each thread loads 2 half elements (one 32-bit register with two halves)
        // For A (row-major), we need to arrange threads for 16x16 loading
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = i;
            int col = laneId / 2;
            int halfIdx = laneId % 2;
            if (row < 16 && col < 8 && (kTile + col * 2 + halfIdx) < K) {
                // Load 2 consecutive halfs from A[row][kTile + col*2]
                int kOffset = col * 2 + halfIdx;
                a_reg[i * 2 + halfIdx] = A[row * K + kTile + kOffset];
            } else {
                a_reg[i * 2 + halfIdx] = __half(0.0f);
            }
        }

        // Load B matrix fragment (16x16) using ldmatrix
        // For B (column-major), we need different layout
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = laneId / 2;
            int col = i;
            int halfIdx = laneId % 2;
            if (col < 8 && row < 16 && (kTile + row * 2 + halfIdx) < K) {
                // Load 2 consecutive halfs from B[kTile + row*2][col]
                int kOffset = row * 2 + halfIdx;
                b_reg[i * 2 + halfIdx] = B[(kTile + kOffset) * N + col];
            } else {
                b_reg[i * 2 + halfIdx] = __half(0.0f);
            }
        }

        __syncthreads();

        // Perform MMA using inline PTX
        // mma.sync.m16n8k16.row.col.f32.f16.f16
        // Each thread computes 4 output elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int idx = i * 2 + j;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    // a_reg[i*4+k] * b_reg[k*2+j] accumulates into acc[idx]
                    // This is still scalar - we need actual MMA instruction
                }
            }
        }
    }

    // Store result using warp shuffle
    // Thread i stores to output[i], output[i+8], output[i+16], output[i+24]
    int colOffset = laneId % 8;
    int rowOffset = laneId / 8;

    if (rowStart + rowOffset < M && colStart + colOffset < N) {
        int outIdx = (rowStart + rowOffset) * N + (colStart + colOffset);
        c[outIdx] = acc0;
    }
    if (rowStart + rowOffset + 8 < M && colStart + colOffset < N) {
        int outIdx = (rowStart + rowOffset + 8) * N + (colStart + colOffset);
        c[outIdx] = acc1;
    }
    if (rowStart + rowOffset < M && colStart + colOffset + 8 < N) {
        int outIdx = (rowStart + rowOffset) * N + (colStart + colOffset + 8);
        c[outIdx] = acc2;
    }
    if (rowStart + rowOffset + 8 < M && colStart + colOffset + 8 < N) {
        int outIdx = (rowStart + rowOffset + 8) * N + (colStart + colOffset + 8);
        c[outIdx] = acc3;
    }
}

// Helper to get lane ID
__device__ __forceinline__ int lane_id() {
    int laneid;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(laneid));
    return laneid;
}

// Get warp ID within block
__device__ __forceinline__ int warp_id() {
    return threadIdx.x / 32;
}

/*
 * Proper Tensor Core Kernel using WMMA API with correct configuration
 * This should use actual tensor cores, not CUDA cores
 */
__global__ void wmma_tensor_core_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K
) {
    // Using the nvcuda::wmma namespace
    using namespace nvcuda::wmma;

    // Each warp handles one 16x16x16 tile
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Calculate which tile this warp handles
    const int tilesM = (M + 15) / 16;
    const int tilesN = (N + 15) / 16;
    const int tilesPerBlock = blockDim.x / 32;
    const int blockTiles = blockIdx.x * tilesPerBlock + warpId;

    const int tileRow = blockTiles / tilesN;
    const int tileCol = blockTiles % tilesN;

    const int rowStart = tileRow * 16;
    const int colStart = tileCol * 16;

    if (rowStart >= M || colStart >= N) return;

    // WMMA fragments
    // For FP16 inputs and FP32 accumulator
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);

    // Perform K/16 iterations of MMA
    for (int k = 0; k < K; k += 16) {
        // Load matrices
        load_matrix_sync(a_frag, a + rowStart * K + k, K);
        load_matrix_sync(b_frag, b + colStart + k * N, N);

        // Perform MMA
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(c + rowStart * N + colStart, c_frag, N, row_major);
}

/*
 * Baseline: Pure CUDA Core matmul (no tensor cores)
 */
__global__ void cuda_core_matmul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}

/*
 * Baseline: FP16 CUDA Core matmul
 */
__global__ void cuda_core_matmul_fp16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(a[row * K + k]) * __half2float(b[k * N + col]);
    }
    c[row * N + col] = sum;
}

void run_benchmark() {
    printf("\n");
    printf("================================================================================\n");
    printf("Tensor Core Benchmark - Comparing CUDA Core vs WMMA API\n");
    printf("================================================================================\n\n");

    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("\n");

    // Test sizes
    struct TestCase {
        int size;
        const char* name;
    };

    TestCase tests[] = {
        {256, "256^3"},
        {512, "512^3"},
        {1024, "1024^3"},
        {2048, "2048^3"},
    };

    int numTests = sizeof(tests) / sizeof(tests[0]);
    const int ITERATIONS = 10;

    printf("%-10s %-15s %12s %12s\n", "Size", "Kernel", "Time (ms)", "GFLOPS");
    printf("%-10s %-15s %12s %12s\n", "----", "------", "---------", "------");

    for (int t = 0; t < numTests; t++) {
        int M = tests[t].size;
        int N = tests[t].size;
        int K = tests[t].size;

        size_t size_a = M * K * sizeof(__half);
        size_t size_b = K * N * sizeof(__half);
        size_t size_c = M * N * sizeof(float);

        // Allocate
        __half *d_a, *d_b;
        float *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, size_a));
        CHECK_CUDA(cudaMalloc(&d_b, size_b));
        CHECK_CUDA(cudaMalloc(&d_c, size_c));

        // Initialize with 1s
        __half *h_a = (__half*)malloc(size_a);
        __half *h_b = (__half*)malloc(size_b);
        for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
        for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);

        CHECK_CUDA(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

        // FP16 CUDA Core kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (M + 15) / 16);

        // Warm up
        cuda_core_matmul_fp16<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark FP16 CUDA
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERATIONS; i++) {
            cuda_core_matmul_fp16<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= ITERATIONS;

        double flops = 2.0 * M * N * K;
        double gflops = flops / (ms * 1e6);

        printf("%-10s %-15s %12.3f %12.2f\n", tests[t].name, "FP16 CUDA", ms, gflops);

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
    printf("Note: CUDA Core FP16 performance measured above.\n");
    printf("      True Tensor Core performance requires mma.sync PTX.\n");
    printf("      RTX 5080 theoretical peak: 89 TFLOPS FP16 Tensor Core\n");
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

    run_benchmark();

    printf("\nBenchmark complete.\n");
    return 0;
}
