#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#include "tensor_mem_research_benchmarks.cu"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Size sweep benchmark - runs tests at different sizes and outputs CSV
void runSizeSweepBenchmark() {
    printf("\n");
    printf("================================================================================\n");
    printf("Size Sweep Benchmark - Memory Copy Bandwidth\n");
    printf("================================================================================\n\n");

    // Size sweep: 1KB to 256MB
    size_t sizes[] = {
        1 << 10,       // 1KB
        1 << 12,       // 4KB
        1 << 14,       // 16KB
        1 << 16,       // 64KB
        1 << 18,       // 256KB
        1 << 20,       // 1MB
        1 << 22,       // 4MB
        1 << 24,       // 16MB
        1 << 26,       // 64MB
        1 << 28,       // 256MB
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    const int iterations = 100;

    // CSV header
    printf("size_bytes,naive_gb/s,shared_gb/s,regular_gb/s,cp_async_gb/s\n");

    for (int i = 0; i < num_sizes; i++) {
        size_t N = sizes[i];
        size_t bytes = N * sizeof(float);

        // Allocate memory
        float *d_src, *d_dst;
        CHECK_CUDA(cudaMalloc(&d_src, bytes));
        CHECK_CUDA(cudaMalloc(&d_dst, bytes));
        CHECK_CUDA(cudaMemset(d_src, 1, bytes));

        float naive_bw = 0, shared_bw = 0, regular_bw = 0, cp_async_bw = 0;

        // Naive global load
        dim3 gridDim(256);
        dim3 blockDim(256);

        // Use CPU timer for reliable measurement
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < iterations; j++) {
            naive_load_kernel<float><<<gridDim, blockDim>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        naive_bw = bytes * iterations / (elapsed * 1e6);

        // Shared memory load
        size_t shared_size = bytes;
        start_time = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < iterations; j++) {
            shared_load_kernel<float><<<gridDim, blockDim, shared_size>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        shared_bw = bytes * iterations / (elapsed * 1e6);

        // Regular copy kernel
        start_time = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < iterations; j++) {
            regular_copy_kernel<<<gridDim, blockDim, shared_size>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        regular_bw = bytes * iterations / (elapsed * 1e6);

        // cp.async 16B kernel
        start_time = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < iterations; j++) {
            cp_async_true_kernel<<<gridDim, blockDim, shared_size>>>(d_src, d_dst, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        cp_async_bw = bytes * iterations / (elapsed * 1e6);

        CHECK_CUDA(cudaFree(d_src));
        CHECK_CUDA(cudaFree(d_dst));

        // Print size label
        const char* size_label;
        if (N <= 1024) size_label = "1KB";
        else if (N <= 4*1024) size_label = "4KB";
        else if (N <= 16*1024) size_label = "16KB";
        else if (N <= 64*1024) size_label = "64KB";
        else if (N <= 256*1024) size_label = "256KB";
        else if (N <= 1024*1024) size_label = "1MB";
        else if (N <= 4*1024*1024) size_label = "4MB";
        else if (N <= 16*1024*1024) size_label = "16MB";
        else if (N <= 64*1024*1024) size_label = "64MB";
        else size_label = "256MB";

        printf("%-12s %-12.2f %-12.2f %-12.2f %-12.2f\n",
               size_label, naive_bw, shared_bw, regular_bw, cp_async_bw);
        printf("%zu,%.2f,%.2f,%.2f,%.2f\n",
               bytes, naive_bw, shared_bw, regular_bw, cp_async_bw);
    }
}

int main(int argc, char** argv) {
    size_t N = 1 << 20;  // 1M elements default

    // Check for size sweep mode
    if (argc > 1 && strcmp(argv[1], "--size-sweep") == 0) {
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

        printf("GPUPeek Memory Bandwidth Size Sweep Benchmark\n");
        printf("=========================================\n");
        printf("Device: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("=========================================\n\n");

        runSizeSweepBenchmark();

        printf("\nBenchmark complete.\n");
        return 0;
    }

    if (argc > 1) {
        N = atoll(argv[1]);
    }

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("GPUPeek Tensor Mem Research Benchmark\n");
    printf("=====================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Elements: %zu (%.2f MB)\n", N, N * sizeof(float) / (1024.0 * 1024.0));
    printf("=====================================\n\n");

    runTensorMemResearchBenchmarks(N);

    printf("\nBenchmark complete.\n");
    return 0;
}
