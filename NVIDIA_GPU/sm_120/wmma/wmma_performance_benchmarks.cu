#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "../../common/timer.h"
#include "wmma_performance_kernel.cu"

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

// =============================================================================
// WMMA Performance Benchmarks - Proper Tensor Core Utilization
// =============================================================================
//
// Key insight: The original benchmark shows ~257 GFLOPS because:
// 1. Small matrices (256x256) don't saturate tensor cores
// 2. Only 1 warp per block = low occupancy
// 3. Grid of 256 warps vs 960 max concurrent on RTX 5080
//
// This benchmark uses:
// - Large matrices (2048x2048, 4096x4096) to saturate tensor cores
// - Multiple warps per block for better occupancy
// - Proper grid dimensions
// =============================================================================

double compute_gflops(size_t M, size_t N, size_t K, double time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

const char* format_gflops(double gflops) {
    static char buf[64];
    if (gflops >= 1000) {
        snprintf(buf, sizeof(buf), "%.2f TFLOPS", gflops / 1000);
    } else {
        snprintf(buf, sizeof(buf), "%.2f GFLOPS", gflops);
    }
    return buf;
}

// =============================================================================
// Benchmark: Simple WMMA (1 warp per block)
// =============================================================================

static void run_simple_wmma_benchmark(int M, int N, int K, int iterations) {
    printf("\n--- Simple WMMA (1 warp/block) ---\n");
    printf("Matrix: M=%d, N=%d, K=%d\n", M, N, K);

    __half *h_a, *h_b, *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(__half)));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; i++) h_d[i] = __float2half(0.0f);

    __half *d_a, *d_b, *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Grid: M/16 rows, N/16 cols
    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);

    int total_warps = gridDim.x * gridDim.y;
    printf("Grid: %dx%d = %d blocks (%d warps)\n", gridDim.x, gridDim.y, total_warps, total_warps);

    // Warmup
    wmma_fp16_simple_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    CPUTimer timer;
    timer.start();

    for (int i = 0; i < iterations; i++) {
        wmma_fp16_simple_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    timer.stop();
    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = compute_gflops(M, N, K, time_ms);

    printf("Time: %.3f ms, Performance: %s\n", time_ms, format_gflops(gflops));

    // Verify result
    CHECK_CUDA(cudaMemcpy(h_d, d_d, M * N * sizeof(__half), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) sum += __half2float(h_d[i]);
    printf("Result sum: %.2f (expected: %.2f)\n", sum, (float)M * N * K);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Benchmark: Perf WMMA (4 warps per block)
// =============================================================================

static void run_perf_wmma_benchmark(int M, int N, int K, int iterations) {
    printf("\n--- Perf WMMA (4 warps/block) ---\n");
    printf("Matrix: M=%d, N=%d, K=%d\n", M, N, K);

    __half *h_a, *h_b, *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(__half)));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; i++) h_d[i] = __float2half(0.0f);

    __half *d_a, *d_b, *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Grid dimensions for 4 warps per block (2x2 tiles)
    dim3 gridDim = get_wmma_grid_perf(M, N, 4);
    dim3 blockDim(128);  // 4 warps

    int total_warps = gridDim.x * gridDim.y * 4;
    printf("Grid: %dx%d, Block: %d warps\n", gridDim.x, gridDim.y, 4);
    printf("Total warps: %d\n", total_warps);

    // Warmup
    wmma_fp16_perf_kernel<4><<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    CPUTimer timer;
    timer.start();

    for (int i = 0; i < iterations; i++) {
        wmma_fp16_perf_kernel<4><<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    timer.stop();
    double time_ms = timer.elapsed_ms() / iterations;
    double gflops = compute_gflops(M, N, K, time_ms);

    printf("Time: %.3f ms, Performance: %s\n", time_ms, format_gflops(gflops));

    // Theoretical peak
    printf("Theoretical FP16 peak: ~89 TFLOPS\n");
    printf("Efficiency: %.1f%%\n", gflops / 89000.0 * 100);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_d));
}

// =============================================================================
// Size Sweep Benchmark
// =============================================================================

static void run_size_sweep() {
    printf("\n");
    printf("================================================================================\n");
    printf("WMMA Size Sweep Benchmark\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Matrix sizes: 256, 512, 1024, 2048, 4096\n");
    printf("Iterations per size: 10\n");
    printf("\n");

    // Matrix sizes to test
    int sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int iterations = 10;

    printf("%-8s %-12s %-12s %-12s %-12s\n", "Size", "Simple", "Perf-4W", "Speedup", "Efficiency");
    printf("%-8s %-12s %-12s %-12s %-12s\n", "", "(GFLOPS)", "(GFLOPS)", "(x)", "(%)");

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i];
        int N = sizes[i];
        int K = sizes[i];

        // Simple WMMA
        __half *h_a, *h_b, *h_d;
        CHECK_CUDA(cudaMallocHost(&h_a, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMallocHost(&h_b, K * N * sizeof(__half)));
        CHECK_CUDA(cudaMallocHost(&h_d, M * N * sizeof(__half)));

        for (int j = 0; j < M * K; j++) h_a[j] = __float2half(1.0f);
        for (int j = 0; j < K * N; j++) h_b[j] = __float2half(1.0f);

        __half *d_a, *d_b, *d_d;
        CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(__half)));

        CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));

        // Simple kernel
        dim3 gridDim(M / 16, N / 16);
        dim3 blockDim(32);

        // Warmup
        wmma_fp16_simple_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CPUTimer timer;
        timer.start();
        for (int j = 0; j < iterations; j++) {
            wmma_fp16_simple_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_d, M, N, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double simple_time = timer.elapsed_ms() / iterations;
        double simple_gflops = compute_gflops(M, N, K, simple_time);

        // Perf kernel (4 warps)
        dim3 perfGridDim = get_wmma_grid_perf(M, N, 4);
        dim3 perfBlockDim(128);

        // Warmup
        wmma_fp16_perf_kernel<4><<<perfGridDim, perfBlockDim>>>(d_a, d_b, d_d, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        timer.start();
        for (int j = 0; j < iterations; j++) {
            wmma_fp16_perf_kernel<4><<<perfGridDim, perfBlockDim>>>(d_a, d_b, d_d, M, N, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.stop();
        double perf_time = timer.elapsed_ms() / iterations;
        double perf_gflops = compute_gflops(M, N, K, perf_time);

        double speedup = simple_gflops / perf_gflops;
        double efficiency = perf_gflops / 89000.0 * 100;  // vs 89 TFLOPS peak

        printf("%-8d %-12.0f %-12.0f %-12.2fx %-12.1f%%\n",
               M, simple_gflops, perf_gflops, speedup, efficiency);

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_d));
        CHECK_CUDA(cudaFreeHost(h_a));
        CHECK_CUDA(cudaFreeHost(h_b));
        CHECK_CUDA(cudaFreeHost(h_d));
    }

    printf("\nNote: Theoretical FP16 tensor peak on RTX 5080: ~89 TFLOPS\n");
}

// =============================================================================
// Main Runner
// =============================================================================

void runWMMA_performance_benchmarks() {
    printf("\n");
    printf("================================================================================\n");
    printf("WMMA Performance Benchmarks - Proper Tensor Core Utilization\n");
    printf("================================================================================\n");
    printf("\n");
    printf("GPU: RTX 5080 Laptop (Blackwell SM 12.0)\n");
    printf("Tensor Peak: ~89 TFLOPS FP16\n");
    printf("\n");

    // Run size sweep
    run_size_sweep();

    printf("\n");
    printf("================================================================================\n");
    printf("Analysis\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Key findings:\n");
    printf("1. Small matrices (256x256) show ~257 GFLOPS because tensor cores are barely used\n");
    printf("2. With proper large matrices and multi-warp blocks, tensor cores should saturate\n");
    printf("3. Expected performance with proper GEMM: 50-89 TFLOPS\n");
    printf("\n");
    printf("Why 257 GFLOPS instead of ~89 TFLOPS?\n");
    printf("- 256 warps x 512 FLOPS/cycle x 1.9 GHz = ~247 GFLOPS\n");
    printf("- This is essentially CUDA core performance, NOT tensor core\n");
    printf("- Tensor cores require:\n");
    printf("  a) Large matrices (2048+) for full utilization\n");
    printf("  b) Multiple warps per block for occupancy\n");
    printf("  c) Proper memory coalescing and prefetching\n");
    printf("\n");
}

void runWMMA_size_sweep_benchmark() {
    printf("\n");
    printf("================================================================================\n");
    printf("WMMA Size Sweep - Tensor Core Saturation Analysis\n");
    printf("================================================================================\n");

    run_size_sweep();

    printf("\n================================================================================\n");
    printf("Interpretation Guide\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Efficiency levels:\n");
    printf("  < 10%%:   Tensor cores barely utilized (current 257 GFLOPS = 0.3%%)\n");
    printf("  10-30%%:  Partial tensor utilization, memory bound\n");
    printf("  30-60%%:  Good tensor utilization\n");
    printf("  60-90%%:  Excellent tensor utilization\n");
    printf("  > 90%%:   Near-peak tensor performance (requires cuBLAS-level optimization)\n");
    printf("\n");
}
