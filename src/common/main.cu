#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_info.h"
#include "timer.h"
#include "../generic/bandwidth_kernel.cu"
#include "../generic/compute_kernel.cu"
#include "../generic/warp_kernel.cu"

// Architecture-specific includes (conditionally included based on detected architecture)
#include "../sm_120/arch_kernels.cu"
#include "../sm_120/benchmarks.cu"
#include "../sm_120/memory_research_benchmarks.cu"
#include "../sm_120/deep_research_benchmarks.cu"
#include "../sm_120/advanced_research_benchmarks.cu"
// Research modules with compilation issues - DISABLED until fixed
// #include "../sm_120/ncu_profiling_benchmarks.cu"
// #include "../sm_120/cuda_core_benchmarks.cu"
// #include "../sm_120/atomic_benchmarks.cu"
// #include "../sm_120/barrier_benchmarks.cu"
// #include "../sm_120/warp_specialize_benchmarks.cu"
// #include "../sm_120/mma_research_benchmarks.cu"  // DISABLED - WMMA kernel needs redesign
// #include "../sm_120/tensor_mem_research_benchmarks.cu"
// #include "../sm_120/wgmma_research_benchmarks.cu"
// #include "../sm_120/dp4a_research_benchmarks.cu"
// #include "../sm_120/fp8_research_benchmarks.cu"
// #include "../sm_120/cuda_graph_research_benchmarks.cu"
// #include "../sm_120/unified_memory_research_benchmarks.cu"
#include "../sm_120/multi_stream_research_benchmarks.cu"
// #include "../sm_120/mbarrier_research_benchmarks.cu"
// #include "../sm_120/cooperative_groups_research_benchmarks.cu"
// #include "../sm_120/redux_sync_research_benchmarks.cu"
#include "../sm_120/fp4_fp6_research_benchmarks.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void printUsage(const char* program) {
    printf("Usage: %s [benchmark] [size]\n", program);
    printf("\nBenchmarks:\n");
    printf("  generic             - Generic GPU benchmarks (all architectures)\n");
    printf("  arch                - Architecture-specific benchmarks (auto-detected)\n");
    printf("  memory              - Memory research (size vs BW, L1/L2, TMA, patterns)\n");
    printf("  deep                - Deep research (L2 cache, Tensor Core, Warp, instructions)\n");
    printf("  advanced            - Advanced research (Occupancy, PCIe, atomics, etc.)\n");
    printf("  ncu                 - NCU profiling kernels (for Nsight Compute analysis)\n");
    printf("  cuda                - CUDA Core arithmetic research (FP64/32/16, INT, vectors)\n");
    printf("  atomic              - Atomic operations deep research\n");
    printf("  barrier             - Barrier synchronization research\n");
    printf("  warp                - Warp specialization and producer/consumer patterns\n");
    printf("  mma                 - MMA (Tensor Core) research (WMMA/MMA/WGMMA/TCGen05)\n");
    printf("  tensor_mem          - Tensor memory operations (LDMATRIX/STMATRIX/cp.async)\n");
    printf("  dp4a                - DP4A (INT8 dot product of 4 bytes)\n");
    printf("  wgmma               - WGMMA (Warpgroup MMA Async)\n");
    printf("  fp8                 - FP8 / TCGen05 Block Scaling (E4M3/E5M2)\n");
    printf("  graph               - CUDA Graph (kernel launch optimization)\n");
    printf("  unified             - Unified Memory (managed memory, prefetch, page faults)\n");
    printf("  multi_stream        - Multi-Stream concurrency (priorities, events, overlap)\n");
    printf("  mbarrier            - Mbarrier operations (async sync, transactions, fences)\n");
    printf("  coop                - Cooperative Groups (grid sync, broadcast, reduce)\n");
    printf("  redux               - Redux.sync warp-level reduction (ADD/MIN/MAX/AND/OR/XOR)\n");
    printf("  fp4                 - FP4/FP6 low-precision MMA (Blackwell new formats)\n");
    printf("  all                 - Run all benchmarks (default)\n");
    printf("\nSize: Number of elements (default: 1M)\n");
    printf("\nSupported Architectures:\n");
    printf("  SM 12.0 (Blackwell)\n");
}

void runBandwidthBenchmarks(size_t N) {
    printf("\n=== Memory Bandwidth Benchmarks (Generic) ===\n");

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;

    size_t bytes = N * sizeof(float);
    printf("Config: %d blocks x %d threads = %zu threads, %.2f MB\n",
           numBlocks, blockSize, (size_t)numBlocks * blockSize, (double)(bytes / (1024.0*1024.0)));

    float *d_src, *h_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMallocHost(&h_src, bytes));

    for (size_t i = 0; i < N; i++) {
        h_src[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    GPUTimer timer;
    const int iterations = 100;

    // Sequential Read
    timer.start();
    for (int i = 0; i < iterations; i++) {
        sequentialReadKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Sequential Read:    %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    // Sequential Write
    timer.start();
    for (int i = 0; i < iterations; i++) {
        sequentialWriteKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Sequential Write:   %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    // Read-Modify-Write
    timer.start();
    for (int i = 0; i < iterations; i++) {
        readModifyWriteKernel<float><<<numBlocks, blockSize>>>(d_src, d_dst, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Read-Modify-Write:  %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));
}

void runComputeBenchmarks(size_t N) {
    printf("\n=== Compute Throughput Benchmarks (Generic) ===\n");

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;
    size_t bytes = N * sizeof(float);

    float *d_a, *d_b, *d_c, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    CHECK_CUDA(cudaMemset(d_a, 1, bytes));
    CHECK_CUDA(cudaMemset(d_b, 2, bytes));
    CHECK_CUDA(cudaMemset(d_c, 3, bytes));

    GPUTimer timer;
    const int iterations = 100;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        fmaKernel<float><<<numBlocks, blockSize>>>(d_a, d_b, d_c, d_out, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("FP32 FMA:           %.2f GFLOPS (%.3f ms per kernel)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    int *d_ia, *d_ib, *d_iout;
    CHECK_CUDA(cudaMalloc(&d_ia, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ib, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_iout, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_ia, 1, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_ib, 2, N * sizeof(int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        intArithmeticKernel<<<numBlocks, blockSize>>>(d_ia, d_ib, d_iout, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("INT32 Arithmetic:   %.2f GIOPS (%.3f ms per kernel)\n",
           N * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_ia));
    CHECK_CUDA(cudaFree(d_ib));
    CHECK_CUDA(cudaFree(d_iout));
}

void runWarpBenchmarks(size_t N) {
    printf("\n=== Warp-Level Benchmarks (Generic) ===\n");

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535;
    size_t bytes = N * sizeof(float);

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemset(d_input, 1, bytes));

    GPUTimer timer;
    const int iterations = 100;

    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpShuffleKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Warp Shuffle:       %.2f GB/s (%.3f ms per kernel)\n",
           bytes * iterations / (timer.elapsed_ms() * 1e6), timer.elapsed_ms() / iterations);

    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpReductionKernel<<<numBlocks, blockSize>>>(d_input, d_output, N / blockSize);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Warp Reduction:     %.3f ms per kernel\n", timer.elapsed_ms() / iterations);

    int *d_int_input, *d_int_output;
    CHECK_CUDA(cudaMalloc(&d_int_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_int_output, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_int_input, 1, N * sizeof(int)));

    timer.start();
    for (int i = 0; i < iterations; i++) {
        warpVoteKernel<<<numBlocks, blockSize>>>(d_int_input, d_int_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    printf("Warp Vote:          %.3f ms per kernel\n", timer.elapsed_ms() / iterations);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_int_input));
    CHECK_CUDA(cudaFree(d_int_output));
}

int main(int argc, char** argv) {
    const char* benchmark = "all";
    size_t N = 1 << 20;  // 1M elements

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        benchmark = argv[1];
    }
    if (argc > 2) {
        N = atoll(argv[2]);
    }

    // Print GPU info
    GPUInfo info = getGPUInfo(0);
    printGPUInfo(info);

    int sm = info.computeCapabilityMajor * 10 + info.computeCapabilityMinor;
    printf("\nDetected Compute Capability: SM %d.%d (0x%02X)\n",
           info.computeCapabilityMajor, info.computeCapabilityMinor, sm);

    printf("\nRunning benchmark: %s with %zu elements (%.2f MB)\n",
           benchmark, N, N * sizeof(float) / (1024.0 * 1024.0));

    // Run generic benchmarks (work on all architectures)
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "generic") == 0) {
        runBandwidthBenchmarks(N);
        runComputeBenchmarks(N);
        runWarpBenchmarks(N);
    }

    // Run architecture-specific benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "arch") == 0) {
        switch (sm) {
            case 120:
                printf("\n[Using SM 12.0 (Blackwell) specific benchmarks]\n");
                runSM120Benchmarks(N);
                break;
            default:
                printf("\n[No architecture-specific benchmarks for SM %d.%d]\n",
                       info.computeCapabilityMajor, info.computeCapabilityMinor);
                printf("Consider adding support for this architecture.\n");
                break;
        }
    }

    // Run memory research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "memory") == 0) {
        switch (sm) {
            case 120:
                printf("\n[Running SM 12.0 Memory Research Benchmarks]\n");
                runMemoryResearchBenchmarks(N);
                break;
            default:
                printf("\n[Memory research not available for SM %d.%d]\n",
                       info.computeCapabilityMajor, info.computeCapabilityMinor);
                break;
        }
    }

    // Run deep research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "deep") == 0) {
        switch (sm) {
            case 120:
                printf("\n[Running SM 12.0 Deep Research Benchmarks]\n");
                runDeepResearchBenchmarks(N);
                break;
            default:
                printf("\n[Deep research not available for SM %d.%d]\n",
                       info.computeCapabilityMajor, info.computeCapabilityMinor);
                break;
        }
    }

    // Run advanced research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "advanced") == 0) {
        switch (sm) {
            case 120:
                printf("\n[Running SM 12.0 Advanced Research Benchmarks]\n");
                runAdvancedResearchBenchmarks(N);
                break;
            default:
                printf("\n[Advanced research not available for SM %d.%d]\n",
                       info.computeCapabilityMajor, info.computeCapabilityMinor);
                break;
        }
    }

    // Run NCU profiling benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "ncu") == 0) {
        printf("\n[NCU profiling disabled - requires compilation fix]\n");
    }

    // Run CUDA Core research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "cuda") == 0) {
        printf("\n[CUDA Core research disabled - requires compilation fix]\n");
    }

    // Run Atomic research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "atomic") == 0) {
        printf("\n[Atomic research disabled - requires compilation fix]\n");
    }

    // Run Barrier research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "barrier") == 0) {
        printf("\n[Barrier research disabled - requires compilation fix]\n");
    }

    // Run Warp Specialization research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "warp") == 0) {
        printf("\n[Warp Specialization research disabled - requires compilation fix]\n");
    }

    // Run MMA (Tensor Core) research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "mma") == 0) {
        printf("\n[MMA research disabled - kernel requires redesign]\n");
    }

    // Run Tensor Memory research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "tensor_mem") == 0) {
        printf("\n[Tensor memory research disabled - requires compilation fix]\n");
    }

    // Run DP4A research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "dp4a") == 0) {
        printf("\n[DP4A research disabled - requires compilation fix]\n");
    }

    // Run WGMMA research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "wgmma") == 0) {
        printf("\n[WGMMA research disabled - requires compilation fix]\n");
    }

    // Run FP8 research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "fp8") == 0) {
        printf("\n[FP8 research disabled - requires compilation fix]\n");
    }

    // Run CUDA Graph research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "graph") == 0) {
        printf("\n[CUDA Graph research disabled - requires compilation fix]\n");
    }

    // Run Unified Memory research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "unified") == 0) {
        printf("\n[Unified Memory research disabled - requires compilation fix]\n");
    }

    // Run Multi-Stream research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "multi_stream") == 0) {
        switch (sm) {
            case 120:
                printf("\n[Running SM 12.0 Multi-Stream Research]\n");
                runMultiStreamResearchBenchmarks(N);
                break;
            default:
                printf("\n[Multi-Stream research not available for SM %d.%d]\n",
                       info.computeCapabilityMajor, info.computeCapabilityMinor);
                break;
        }
    }

    // Run Mbarrier research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "mbarrier") == 0) {
        printf("\n[Mbarrier research disabled - requires compilation fix]\n");
    }

    // Run Cooperative Groups research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "coop") == 0) {
        printf("\n[Cooperative Groups research disabled - requires compilation fix]\n");
    }

    // Run Redux.sync research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "redux") == 0) {
        printf("\n[Redux.sync research disabled - requires compilation fix]\n");
    }

    // Run FP4/FP6 research benchmarks
    if (strcmp(benchmark, "all") == 0 || strcmp(benchmark, "fp4") == 0) {
        switch (sm) {
            case 120:
                printf("\n[Running SM 12.0 FP4/FP6 Research]\n");
                printf("For NCU analysis: ncu --set full --metrics sm__pipe_tensor_cycles_active.pct ./gpupeek.exe fp4\n");
                runFP4FP6ResearchBenchmarks(N);
                break;
            default:
                printf("\n[FP4/FP6 research not available for SM %d.%d]\n",
                       info.computeCapabilityMajor, info.computeCapabilityMinor);
                break;
        }
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
