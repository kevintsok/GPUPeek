#include <cuda_runtime.h>
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

int main(int argc, char** argv) {
    size_t N = 1 << 20;  // 1M elements default

    if (argc > 1) {
        N = atoll(argv[1]);
    }

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("GPUPeek FP8 Research Benchmark\n");
    printf("===============================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Elements: %zu (%.2f MB)\n", N, N * sizeof(float) / (1024.0 * 1024.0));
    printf("===============================\n\n");

    runFP8ResearchBenchmarks(N);

    printf("\nBenchmark complete.\n");
    return 0;
}
