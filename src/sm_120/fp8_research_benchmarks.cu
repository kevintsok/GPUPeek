#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/timer.h"
#include "fp8_research_kernel.cu"

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
// FP8 / TCGen05 Block Scaling Research Benchmarks
// =============================================================================
//
// PTX ISA Section 9.7.16 - TCGen05 (TensorCore 5th Generation)
//
// FP8 Formats:
// - E4M3 (.e4m3): 4-bit exponent, 3-bit mantissa (range: 0-240)
// - E5M2 (.e5m2): 5-bit exponent, 2-bit mantissa (range: 0-57344)
//
// TCGen05 Features:
// - Block scaling for weight-only quantization
// - FP8 support
// - Scaled MMA operations
//
// Variants:
// - tcgen05.mma - basic MMA
// - tcgen05.mma.sp - sparse MMA
// - tcgen05.mma.ws - weight-only scaling
// - tcgen05.mma.ws.sp - weight-only + sparse
//
// Block Scaling:
// - W8A16: 8-bit weights, 16-bit activations
// - W8A8: 8-bit weights, 8-bit activations
// =============================================================================

// =============================================================================
// Section 1: FP8 Format Conversion Tests
// =============================================================================

void runFP8ConversionTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("1. FP8 Format Conversion Tests\n");
    printf("================================================================================\n");

    const size_t N = 1 << 20;
    const size_t bytes = N * sizeof(float);
    const int iterations = 100;

    float *h_src, *d_src;
    unsigned char *d_e4m3, *d_e5m2;

    CHECK_CUDA(cudaMallocHost(&h_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_e4m3, N));
    CHECK_CUDA(cudaMalloc(&d_e5m2, N));

    // Initialize with random data
    for (size_t i = 0; i < N; i++) {
        h_src[i] = (rand() % 100) / 10.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    // FP32 to E4M3
    timer.start();
    for (int i = 0; i < iterations; i++) {
        convert_to_fp8_e4m3<float><<<gridDim, blockDim>>>(d_src, d_e4m3, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("FP32 -> E4M3:        %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // FP32 to E5M2
    timer.start();
    for (int i = 0; i < iterations; i++) {
        convert_to_fp8_e5m2<float><<<gridDim, blockDim>>>(d_src, d_e5m2, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("FP32 -> E5M2:        %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_e4m3));
    CHECK_CUDA(cudaFree(d_e5m2));
}

// =============================================================================
// Section 2: Block Scaling Quantization Tests
// =============================================================================

void runBlockScalingTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("2. Block Scaling Quantization Tests\n");
    printf("================================================================================\n");

    const size_t N = 1 << 18;
    const size_t bytes = N * sizeof(float);
    const size_t block_dim = 32;
    const size_t num_blocks = (N + block_dim - 1) / block_dim;
    const int iterations = 100;

    float *d_src, *h_src;
    int8_t *d_quant;
    float *d_scales;

    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMallocHost(&h_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_quant, N));
    CHECK_CUDA(cudaMalloc(&d_scales, num_blocks * sizeof(float)));

    for (size_t i = 0; i < N; i++) {
        h_src[i] = (rand() % 100 - 50) / 10.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);

    GPUTimer timer;

    // W8A16 quantization
    printf("\n--- W8A16 Block Scaling (32 elements/block) ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        block_scale_quantize_kernel<float><<<gridDim, blockDim>>>(
            d_src, d_quant, d_scales, N, block_dim);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("W8A16 Quantize:      %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // W8A8 quantization
    timer.start();
    for (int i = 0; i < iterations; i++) {
        block_scale_quantize_w8a8_kernel<float><<<gridDim, blockDim>>>(
            d_src, d_quant, d_scales, N, block_dim);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("W8A8 Quantize:      %.2f GB/s (%.3f ms)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_quant));
    CHECK_CUDA(cudaFree(d_scales));
}

// =============================================================================
// Section 3: FP8 GEMM Tests
// =============================================================================

void runFP8GEMMTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("3. FP8 GEMM Tests\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const size_t bytes_a = M * K;
    const size_t bytes_b = K * N;
    const size_t bytes_c = M * N * sizeof(float);
    const int iterations = 10;

    unsigned char *d_a, *d_b;
    float *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, bytes_a));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
    CHECK_CUDA(cudaMalloc(&d_c, bytes_c));

    // Initialize with random FP8-like data
    unsigned char* h_a = (unsigned char*)malloc(bytes_a);
    unsigned char* h_b = (unsigned char*)malloc(bytes_b);
    for (size_t i = 0; i < bytes_a; i++) h_a[i] = rand() % 200;
    for (size_t i = 0; i < bytes_b; i++) h_b[i] = rand() % 200;
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(256);

    GPUTimer timer;

    // FP8 E4M3 GEMM
    printf("\n--- FP8 E4M3 GEMM ---\n\n");

    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp8_gemm_e4m3_kernel<float><<<gridDim, blockDim>>>(
            d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;
    printf("FP8 E4M3 GEMM:       %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // FP8 E5M2 GEMM
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp8_gemm_e5m2_kernel<float><<<gridDim, blockDim>>>(
            d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("FP8 E5M2 GEMM:       %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// =============================================================================
// Section 4: Weight-Only Quantization Inference
// =============================================================================

void runWeightOnlyQuantTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("4. Weight-Only Quantization Inference\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const size_t block_size = 32;
    const size_t num_weight_blocks = (M * K + block_size - 1) / block_size;
    const int iterations = 10;

    float *d_weights, *h_weights;
    int8_t *d_quant_w;
    float *d_scales;
    __half *d_activations, *d_output;

    CHECK_CUDA(cudaMalloc(&d_weights, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_weights, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_quant_w, M * K));
    CHECK_CUDA(cudaMalloc(&d_scales, num_weight_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_activations, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_output, M * N * sizeof(__half)));

    // Initialize
    for (size_t i = 0; i < M * K; i++) {
        h_weights[i] = (rand() % 100 - 50) / 10.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_activations, 1, K * N * sizeof(__half)));

    dim3 gridDimQuant((M * K + 255) / 256);
    dim3 blockDimQuant(256);
    dim3 gridDimMMA(M / 16, N / 16);
    dim3 blockDimMMA(256);

    GPUTimer timer;

    // Quantize weights
    printf("\n--- W8A16 Inference Pipeline ---\n\n");

    weight_only_quant_kernel<float><<<gridDimQuant, blockDimQuant>>>(
        d_weights, d_quant_w, d_scales, M * K, block_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // W8A16 MMA
    timer.start();
    for (int i = 0; i < iterations; i++) {
        w8a16_mma_kernel<float><<<gridDimMMA, blockDimMMA>>>(
            d_quant_w, d_activations, d_output, d_scales, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;
    printf("W8A16 MMA:           %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFreeHost(h_weights));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_quant_w));
    CHECK_CUDA(cudaFree(d_scales));
    CHECK_CUDA(cudaFree(d_activations));
    CHECK_CUDA(cudaFree(d_output));
}

// =============================================================================
// Section 5: Baseline Comparisons
// =============================================================================

void runBaselineComparisonTests() {
    printf("\n");
    printf("================================================================================\n");
    printf("5. Baseline Comparisons\n");
    printf("================================================================================\n");

    const size_t M = 256, N = 256, K = 256;
    const int iterations = 10;

    float *d_a, *d_b, *d_c;
    __half *d_a_fp16, *d_b_fp16, *d_c_fp16;

    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a_fp16, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b_fp16, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_c_fp16, M * N * sizeof(__half)));

    CHECK_CUDA(cudaMemset(d_a, 1, M * K * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b, 1, K * N * sizeof(float)));

    dim3 gridDim(N / 16, M / 16);
    dim3 blockDim(16, 16);

    GPUTimer timer;

    printf("\n--- FP32 vs FP16 vs FP8 vs W8A16 ---\n\n");

    // FP32 GEMM
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp32_baseline_gemm_kernel<float><<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    double flops = 2.0 * M * N * K * iterations;
    printf("FP32 GEMM:           %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    // FP16 GEMM
    timer.start();
    for (int i = 0; i < iterations; i++) {
        fp16_baseline_gemm_kernel<__half><<<gridDim, blockDim>>>(d_a_fp16, d_b_fp16, d_c_fp16, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("FP16 GEMM:           %.2f GFLOPS (%.3f ms)\n",
           flops / (timer.elapsed_ms() * 1e6),
           timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_a_fp16));
    CHECK_CUDA(cudaFree(d_b_fp16));
    CHECK_CUDA(cudaFree(d_c_fp16));
}

// =============================================================================
// Section 6: NCU Profiling Reference
// =============================================================================

void runNCUProfilingReference() {
    printf("\n");
    printf("================================================================================\n");
    printf("6. NCU Profiling Reference - FP8 / TCGen05 Block Scaling\n");
    printf("================================================================================\n");

    printf("\n--- Key NCU Metrics ---\n\n");

    printf("Tensor Core Utilization:\n");
    printf("  ncu --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek fp8\n\n");

    printf("FP8 Instructions:\n");
    printf("  ncu --metrics sm__inst_executed.fp8.sum ./gpupeek fp8\n\n");

    printf("Memory Bandwidth:\n");
    printf("  ncu --metrics dram__bytes.sum ./gpupeek fp8\n\n");

    printf("--- SASS Instruction Reference ---\n\n");

    printf("| SASS | Description | PTX |\n");
    printf("|------|------------|-----|\n");
    printf("| HMMA | Half MMA | wmma.mma.f16 |\n");
    printf("| IMMA | INT MMA | wmma.mma.s32 |\n");
    printf("| BMMA | BF16 MMA | wmma.mma.bf16 |\n");
    printf("| WGMMA | Warpgroup MMA | wgmma.mma_async |\n");
    printf("| TCGEN05 | 5th Gen Tensor | tcgen05.mma |\n\n");

    printf("--- PTX Instruction Reference ---\n\n");

    printf("TCGen05 MMA (Section 9.7.16.10):\n");
    printf("  tcgen05.mma.sync.aligned.kind.f16       Rd, Ra, Rb, Rc;\n");
    printf("  tcgen05.mma.sync.aligned.kind.f8f6f4   Rd, Ra, Rb, Rc;\n");
    printf("  tcgen05.mma.sync.aligned.kind.mxf8f6f4 Rd, Ra, Rb, Rc, Rs;\n\n");

    printf("Block Scaling (Section 9.7.16.10.7):\n");
    printf("  tcgen05.mma.ws  - Weight-only scaling\n");
    printf("  tcgen05.mma.ws.sp - Weight-only + sparse\n\n");

    printf("FP8 Formats:\n");
    printf("  .e4m3 - 4-bit exponent, 3-bit mantissa (range 0-240)\n");
    printf("  .e5m2 - 5-bit exponent, 2-bit mantissa (range 0-57344)\n\n");

    printf("--- FP8 vs Other Formats ---\n\n");

    printf("| Format | Bits | Range | Use Case |\n");
    printf("|--------|------|-------|----------|\n");
    printf("| FP32 | 32 | 1e-38 to 1e38 | Training |\n");
    printf("| FP16 | 16 | 1e-5 to 1e4 | Inference |\n");
    printf("| BF16 | 16 | 1e-38 to 1e38 | ML |\n");
    printf("| TF32 | 19 | 1e-38 to 1e4 | ML |\n");
    printf("| FP8 E4M3 | 8 | 0-240 | Inference |\n");
    printf("| FP8 E5M2 | 8 | 0-57344 | Inference |\n\n");

    printf("--- Key Findings ---\n\n");

    printf("1. FP8 Benefits:\n");
    printf("   - 2x memory bandwidth vs FP16\n");
    printf("   - 2x throughput vs FP16\n");
    printf("   - Blackwell-native support\n\n");

    printf("2. Block Scaling (W8A16):\n");
    printf("   - 8-bit quantized weights\n");
    printf("   - Per-block scales\n");
    printf("   - FP16 activations\n");
    printf("   - ~4x memory reduction vs FP32\n\n");

    printf("3. TCGen05 Features:\n");
    printf("   - Built-in block scaling\n");
    printf("   - FP8 support (.e4m3, .e5m2)\n");
    printf("   - Sparse matrix support\n");
    printf("   - Weight-only quantization\n\n");

    printf("4. Recommended Use Cases:\n");
    printf("   - LLM inference (W8A16)\n");
    printf("   - Vision transformers (FP8 E4M3)\n");
    printf("   - Recommendation systems (FP8 E5M2)\n");
}

// =============================================================================
// Main Entry Point
// =============================================================================

void runFP8ResearchBenchmarks(size_t N) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                                                                              #\n");
    printf("#           RTX 5080 (Blackwell SM 12.0) FP8 / TCGen05 Research             #\n");
    printf("#           FP8 Formats & Block Scaling for Inference                         #\n");
    printf("#                                                                              #\n");
    printf("################################################################################\n");

    printf("\n");
    printf("================================================================================\n");
    printf("PTX ISA TCGen05 / FP8 (Section 9.7.16)\n");
    printf("================================================================================\n\n");

    printf("FP8 Formats:\n");
    printf("  - E4M3: 4-bit exponent, 3-bit mantissa (0-240)\n");
    printf("  - E5M2: 5-bit exponent, 2-bit mantissa (0-57344)\n\n");

    printf("TCGen05 Variants:\n");
    printf("  - tcgen05.mma - basic MMA\n");
    printf("  - tcgen05.mma.sp - sparse MMA\n");
    printf("  - tcgen05.mma.ws - weight-only scaling\n");
    printf("  - tcgen05.mma.ws.sp - weight-only + sparse\n\n");

    printf("Block Scaling:\n");
    printf("  - W8A16: 8-bit weights, 16-bit activations\n");
    printf("  - W8A8: 8-bit weights, 8-bit activations\n\n");

    runFP8ConversionTests();
    runBlockScalingTests();
    runFP8GEMMTests();
    runWeightOnlyQuantTests();
    runBaselineComparisonTests();
    runNCUProfilingReference();

    printf("\n");
    printf("================================================================================\n");
    printf("FP8 / TCGen05 Research Complete!\n");
    printf("================================================================================\n");
    printf("\n");
    printf("For NCU profiling, run:\n");
    printf("  ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe fp8\n");
    printf("  ncu --metrics dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustainedTesla ./gpupeek.exe fp8\n");
    printf("\n");
}
