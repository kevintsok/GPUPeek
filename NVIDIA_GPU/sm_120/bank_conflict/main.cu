#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declaration
void runBankConflictResearchBenchmarks();

int main(int argc, char** argv) {
    printf("GPUPeek Bank Conflict Research Module\n");
    printf("======================================\n\n");

    // Get GPU info
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared Memory: %zu bytes\n", prop.sharedMemPerBlock);
    printf("\n");

    // Run all benchmarks
    runBankConflictResearchBenchmarks();

    printf("\n");
    printf("Bank Conflict Research Complete!\n");

    return 0;
}
