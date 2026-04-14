#include "gpu_info.h"
#include <cuda_runtime.h>
#include <stdio.h>

GPUInfo getGPUInfo(int deviceId) {
    GPUInfo info = {};
    info.deviceId = deviceId;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    info.name = prop.name;
    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;
    info.numSMs = prop.multiProcessorCount;
    info.globalMemTotal = prop.totalGlobalMem;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.regsPerBlock = prop.regsPerBlock;
    info.warpSize = prop.warpSize;

    // Blackwell has 128 cores per SM
    info.numCoresPerSM = 128;

    // Max threads per SM varies by architecture
    info.maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

    return info;
}

void printGPUInfo(const GPUInfo& info) {
    printf("=== GPU Info (Device %d) ===\n", info.deviceId);
    printf("  Name:                  %s\n", info.name.c_str());
    printf("  Compute Capability:    %d.%d\n",
           info.computeCapabilityMajor, info.computeCapabilityMinor);
    printf("  Number of SMs:         %d\n", info.numSMs);
    printf("  Cores per SM:          %d\n", info.numCoresPerSM);
    printf("  Total Cores:           %d\n", info.numSMs * info.numCoresPerSM);
    printf("  Global Memory:         %.2f GB\n",
           info.globalMemTotal / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Mem per Block:   %zu KB\n",
           info.sharedMemPerBlock / 1024);
    printf("  Max Threads per Block: %d\n", info.maxThreadsPerBlock);
    printf("  Max Threads per SM:    %d\n", info.maxThreadsPerSM);
    printf("  Registers per Block:   %d\n", info.regsPerBlock);
    printf("  Warp Size:             %d\n", info.warpSize);
    printf("===========================\n");
}
