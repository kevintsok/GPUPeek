#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Forward declarations
void runWMMAResearchBenchmarks(size_t N);
void runWMMA_performance_benchmarks();
void runWMMA_size_sweep_benchmark();

int main(int argc, char** argv) {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("GPUPeek WMMA Research Benchmark\n");
    printf("================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("================================\n\n");

    // Parse command line arguments
    if (argc > 1) {
        if (strcmp(argv[1], "--perf") == 0) {
            // Performance benchmark - proper tensor core utilization
            runWMMA_performance_benchmarks();
        } else if (strcmp(argv[1], "--size-sweep") == 0) {
            // Size sweep benchmark
            runWMMA_size_sweep_benchmark();
        } else {
            // Legacy: treat as element count
            size_t N = atoll(argv[1]);
            runWMMAResearchBenchmarks(N);
        }
    } else {
        // Default: run research benchmarks
        size_t N = 1 << 20;
        runWMMAResearchBenchmarks(N);
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
