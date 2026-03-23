#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/timer.h"
#include "wgmma_research_kernel.cu"

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
// WGMMA Research Benchmarks
// =============================================================================
//
// PTX ISA Section 9.7.15 - WGMMA (Warpgroup Matrix Multiply Async)
//
// WGMMA vs WMMA/MMA:
// - WMMA: Warp-level, synchronous, m16n16k16 shape
// - MMA: Warp-level, synchronous, multiple shapes
// - WGMMA: Warpgroup-level, ASYNCHRONOUS, larger shapes
//
// Shapes:
// - m64nNk16 (N = K/16, K=16)
// - m64nNk8 (K=8)
// - m64nNk32 (K=32)
// - m64nNk256 (K=256)
//
// Data Types:
// - .f16, .bf16, .tf32, .f64
// - .s8, .u8 (integer)
//
// Async Operations:
// - wgmma.fence - ordering
// - wgmma.commit_group - commit
// - wgmma.wait_group n - wait
//
// NCU Metrics:
// - sm__pipe_tensor_cycles_active.pct - Tensor utilization
// - sm__inst_executed.wgmma.sum - WGMMA instruction count
// =============================================================================

// =============================================================================
// Section 1: Basic WGMMA Tests
// =============================================================================

void runWGMMAFP16Tests() {
    printf("\n");
    printf("================================================================================\n");
    printf("1. WGMMA FP16 Tests (m64nNk16)\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const size_t bytes_a = M * K * sizeof(__half);
    const size_t bytes_b = K * N * sizeof(__half);
    const size_t bytes_c = M * N * sizeof(float);
    const int iterations = 100;

    __half *d_a, *d_b;
    float *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMalloc(&d_c, bytes_c));

    CHECK_CUDA(cudaMemset(d_a, 1, bytes_a));
    CHECK_CUDA(cudaMemset(d_b, 2, bytes_b));
    CHECK_CUDA(cudaMemset(d_c, 0, bytes_c));

    dim3 gridDim(M / 64, N / 64);
    dim3 blockDim(128);
    size_t shared_size = (64 * 16 + 64 * 16) * sizeof(__half);

    GPUTimer timer;

    // WGMMA FP16
    printf("\n--- WGMMA FP16 (m64nNk16) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wgmma_fp16_kernel<__half><<<gridDim, blockDim, shared_size>>>(
            d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;
    printf("WGMMA FP16:           %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// =============================================================================
// Section 2: WGMMA Data Types
// =============================================================================

void runWGMMADataTypeTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. WGMMA Data Type Tests\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const int iterations = 100;

    // BF16
    {
        __half *d_a, *d_b;
        float *d_c;

        CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemset(d_a, 1, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_b, 2, K * N * sizeof(__half)));

        dim3 gridDim(M / 64, N / 64);
        dim3 blockDim(128);
        size_t shared_size = (64 * 16 + 64 * 16) * sizeof(__half);

        GPUTimer timer;

        timer.start();
        for (int i = 0; i < iterations; i++) {
            wgmma_bf16_kernel<__half><<<gridDim, blockDim, shared_size>>>(
                d_a, d_b, d_c, M, N, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double flops = 2.0 * M * N * K * iterations;
        printf("WGMMA BF16:           %.2f GFLOPS (%.3f ms)\n",
               flops / (timer.elapsed_ms() * 1e6),
               timer.elapsed_ms() / iterations);

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // FP64
    {
        __half *d_a, *d_b;
        double *d_c;

        CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(double)));

        CHECK_CUDA(cudaMemset(d_a, 1, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_b, 2, K * N * sizeof(__half)));

        dim3 gridDim(M / 64, N / 32);
        dim3 blockDim(128);
        size_t shared_size = (64 * 8 + 32 * 8) * sizeof(__half);

        GPUTimer timer;

        timer.start();
        for (int i = 0; i < iterations; i++) {
            wgmma_fp64_kernel<__half><<<gridDim, blockDim, shared_size>>>(
                d_a, d_b, d_c, M, N, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double flops = 2.0 * M * N * K * iterations;
        printf("WGMMA FP64:           %.2f GFLOPS (%.3f ms)\n",
               flops / (timer.elapsed_ms() * 1e6),
               timer.elapsed_ms() / iterations);

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // INT8
    {
        __half *d_a, *d_b;
        int32_t *d_c;

        CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(int32_t)));

        CHECK_CUDA(cudaMemset(d_a, 1, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_b, 2, K * N * sizeof(__half)));

        dim3 gridDim(M / 64, N / 64);
        dim3 blockDim(128);
        size_t shared_size = (64 * 32 + 64 * 32) * sizeof(__half);

        GPUTimer timer;

        timer.start();
        for (int i = 0; i < iterations; i++) {
            wgmma_int8_kernel<__half><<<gridDim, blockDim, shared_size>>>(
                d_a, d_b, d_c, M, N, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();

        double ops = 2.0 * M * N * K * iterations;
        printf("WGMMA INT8:           %.2f GOPS (%.3f ms)\n",
               ops / (timer.elapsed_ms() * 1e6),
               timer.elapsed_ms() / iterations);

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }
}

// =============================================================================
// Section 3: WGMMA Sparse
// =============================================================================

void runWGMMA_SparseTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. WGMMA Sparse Tests (2:4 Structured Sparsity)\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const size_t bytes_a = M * K * sizeof(__half);
    const size_t bytes_b = K * N * sizeof(__half);
    const size_t bytes_meta = (M / 4) * (K / 4) * sizeof(__half);
    const int iterations = 100;

    __half *d_a, *d_b, *d_meta;
    float *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMalloc(&d_meta, bytes_meta));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_a, 1, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMemset(d_c, 0, M * N * sizeof(float)));

    dim3 gridDim(M / 64, N / 64);
    dim3 blockDim(128);
    size_t shared_size = (64 * 32 + 64 * 32 + 16 * 8) * sizeof(__half);

    GPUTimer timer;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wgmma_sparse_fp16_kernel<__half><<<gridDim, blockDim, shared_size>>>(
            d_a, d_b, d_meta, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;
    printf("WGMMA Sparse FP16:    %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);
    printf("  (With 2:4 structured sparsity, ~2x speedup vs dense)\n");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_meta));
    CHECK_CUDA(cudaFree(d_c));
}

// =============================================================================
// Section 4: WGMMA Pipeline
// =============================================================================

void runWGMMA_PipelineTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("4. WGMMA Pipeline Tests (Hide Memory Latency)\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const size_t bytes_a = M * K * sizeof(__half);
    const size_t bytes_b = K * N * sizeof(__half);
    const int iterations = 100;

    __half *d_a, *d_b;
    float *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_a, 1, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMemset(d_c, 0, M * N * sizeof(float)));

    dim3 gridDim(M / 64, N / 64);
    dim3 blockDim(128);
    size_t shared_size = 2 * (64 * 16 + 64 * 16) * sizeof(__half);

    GPUTimer timer;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wgmma_pipeline_kernel<__half><<<gridDim, blockDim, shared_size>>>(
            d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;
    printf("WGMMA Pipeline FP16:  %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// =============================================================================
// Section 5: Comparison with WMMA
// =============================================================================

void runWMMAComparisonTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("5. WGMMA vs WMMA Comparison\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const size_t bytes_a = M * K * sizeof(__half);
    const size_t bytes_b = K * N * sizeof(__half);
    const int iterations = 100;

    __half *d_a, *d_b;
    float *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_a, 1, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMemset(d_c, 0, M * N * sizeof(float)));

    GPUTimer timer;

    // WGMMA
    dim3 gridDim_wgmma(M / 64, N / 64);
    dim3 blockDim_wgmma(128);
    size_t shared_size_wgmma = (64 * 16 + 64 * 16) * sizeof(__half);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wgmma_fp16_kernel<__half><<<gridDim_wgmma, blockDim_wgmma, shared_size_wgmma>>>(
            d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;
    printf("WGMMA FP16 (m64):     %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // WMMA baseline
    dim3 gridDim_wmma(M / 16, N / 16);
    dim3 blockDim_wmma(128);
    size_t shared_size_wmma = (16 * 16 + 16 * 16) * sizeof(__half);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        wmma_baseline_fp16_kernel<__half><<<gridDim_wmma, blockDim_wmma, shared_size_wmma>>>(
            d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("WMMA FP16 (m16):      %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    printf("\nNote: WGMMA operates on 4x larger tiles (64 vs 16) with async support\n");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// =============================================================================
// Section 6: NCU Profiling Reference
// =============================================================================

void runNCUProfilingReference() {
    printf("\n");
    printf("================================================================================\n");
    printf("6. NCU Profiling Reference - WGMMA\n");
    printf("================================================================================\n");

    printf("\n--- Key NCU Metrics for WGMMA ---\n\n");

    printf("WGMMA Instruction Count:\n");
    printf("  ncu --metrics sm__inst_executed.wgmma.sum ./gpupeek wgmma\n\n");

    printf("Tensor Core Utilization:\n");
    printf("  ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek wgmma\n\n");

    printf("Async Efficiency:\n");
    printf("  ncu --metrics sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek wgmma\n\n");

    printf("Memory Analysis:\n");
    printf("  ncu --metrics dram__bytes.sum,lts__tcs_hit_rate.pct ./gpupeek wgmma\n\n");

    printf("--- SASS Instruction Reference ---\n\n");

    printf("| SASS | Description | PTX |\n");
    printf("|------|------------|-----|\n");
    printf("| WGMMA | Warpgroup MMA async | wgmma.mma_async |\n");
    printf("| WGMMA.sp | Warpgroup MMA sparse | wgmma.mma_async.sp |\n");
    printf("| WGMMAF | WGMMA fence | wgmma.fence |\n");
    printf("| WGMMAWG | WGMMA wait group | wgmma.wait_group |\n");
    printf("| WGMMAAG | WGMMA commit group | wgmma.commit_group |\n\n");

    printf("--- PTX Instruction Reference ---\n\n");

    printf("WGMMA (Section 9.7.15):\n\n");

    printf("Basic WGMMA:\n");
    printf("  wgmma.mma_async.sync.aligned.m64nNk16.f16  Rd, Ra, Rb, Rc;\n");
    printf("  wgmma.mma_async.sync.aligned.m64nNk8.f64   Rd, Ra, Rb, Rc;\n");
    printf("  wgmma.mma_async.sync.aligned.m64nNk16.bf16 Rd, Ra, Rb, Rc;\n");
    printf("  wgmma.mma_async.sync.aligned.m64nNk16.tf32 Rd, Ra, Rb, Rc;\n\n");

    printf("Integer WGMMA:\n");
    printf("  wgmma.mma_async.sync.aligned.m64nNk16.s8  Rd, Ra, Rb, Rc;\n");
    printf("  wgmma.mma_async.sync.aligned.m64nNk16.u8  Rd, Ra, Rb, Rc;\n\n");

    printf("Sparse WGMMA:\n");
    printf("  wgmma.mma_async.sp.m64nNk32  Rd, Ra, Rb, Rc, Rm;\n");
    printf("  wgmma.mma_async.sp.m64nNk16  Rd, Ra, Rb, Rc, Rm;\n\n");

    printf("Async Operations:\n");
    printf("  wgmma.fence;\n");
    printf("  wgmma.commit_group;\n");
    printf("  wgmma.wait_group 0;\n\n");

    printf("--- WGMMA vs WMMA vs MMA ---\n\n");

    printf("| Feature | WMMA | MMA | WGMMA |\n");
    printf("|---------|------|-----|-------|\n");
    printf("| Level | Warp | Warp | Warpgroup |\n");
    printf("| Shape | m16n16k16 | m16n8k8, etc | m64nNk16 |\n");
    printf("| Sync | Sync | Sync | ASYNC |\n");
    printf("| Setup | Medium | Medium | Low |\n");
    printf("| Throughput | High | High | Highest |\n\n");

    printf("--- Key Findings Guide ---\n\n");

    printf("1. WGMMA Advantages:\n");
    printf("   - 4x larger tiles (64 vs 16 rows)\n");
    printf("   - Asynchronous execution hides memory latency\n");
    printf("   - Warpgroup-level parallelism (3 warps)\n");
    printf("   - Better utilization of tensor cores\n\n");

    printf("2. Async Pattern Benefits:\n");
    printf("   - Load next tile while computing current tile\n");
    printf("   - Double buffering pipeline\n");
    printf("   - Reduced idle time\n\n");

    printf("3. Use Cases:\n");
    printf("   - Large matrix operations\n");
    printf("   - Deep learning training/inference\n");
    printf("   - Scientific computing\n\n");

    printf("4. Sparse WGMMA:\n");
    printf("   - 2:4 structured sparsity (50%% zeros)\n");
    printf("   - ~2x speedup vs dense\n");
    printf("   - wgmma.mma_async.sp variant\n");
}

// =============================================================================
// Main Entry Point
// =============================================================================

void runWGMMA_ResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) WGMMA Research                     #\n");
    printf("#           Warpgroup Matrix Multiply Async                                   #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    printf("\n");
    printf("================================================================================\n");
    printf("PTX ISA WGMMA (Section 9.7.15) - Warpgroup MMA Async\n");
    printf("================================================================================\n\n");

    printf("Key Difference from WMMA/MMA:\n");
    printf("  - ASYNCHRONOUS execution\n");
    printf("  - Warpgroup level (3 warps = 96 threads)\n");
    printf("  - Larger tiles (m64 vs m16)\n\n");

    printf("Shapes:\n");
    printf("  - m64nNk16 (K=16)\n");
    printf("  - m64nNk8  (K=8)\n");
    printf("  - m64nNk32 (K=32)\n");
    printf("  - m64nNk256 (K=256)\n\n");

    printf("Data Types:\n");
    printf("  - FP16, BF16, TF32, FP64\n");
    printf("  - INT8, UINT8\n\n");

    printf("Async Operations:\n");
    printf("  - wgmma.fence\n");
    printf("  - wgmma.commit_group\n");
    printf("  - wgmma.wait_group\n\n");

    runWGMMAFP16Tests();
    runWGMMADataTypeTests();
    runWGMMA_SparseTests();
    runWGMMA_PipelineTests();
    runWMMAComparisonTests();
    runNCUProfilingReference();

    printf("\n");
    printf("================================================================================\n");
    printf("WGMMA Research Complete!\n");
    printf("================================================================================\n");
    printf("\n");
    printf("For NCU profiling, run:\n");
    printf("  ncu --set full --metrics sm__inst_executed.wgmma.sum ./gpupeek.exe wgmma\n");
    printf("  ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe wgmma\n");
    printf("\n");
}
