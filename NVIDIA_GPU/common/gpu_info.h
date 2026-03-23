#pragma once
#include <cuda_runtime.h>
#include <string>

struct GPUInfo {
    int deviceId;
    std::string name;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int numSMs;
    int numCoresPerSM;
    size_t globalMemTotal;
    size_t sharedMemPerBlock;
    int maxThreadsPerBlock;
    int maxThreadsPerSM;
    int regsPerBlock;
    int warpSize;
};

GPUInfo getGPUInfo(int deviceId = 0);
void printGPUInfo(const GPUInfo& info);
