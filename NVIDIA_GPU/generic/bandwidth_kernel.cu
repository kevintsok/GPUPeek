#pragma once
#include <cuda_runtime.h>

// Memory bandwidth benchmark kernels

// Simple sequential read - measures memory read bandwidth
template <typename T>
__global__ void sequentialReadKernel(T* __restrict__ data, T* output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    T acc = 0;
    for (size_t i = idx; i < N; i += stride) {
        acc += data[i];
    }
    output[idx] = acc;
}

// Simple sequential write - measures memory write bandwidth
template <typename T>
__global__ void sequentialWriteKernel(T* __restrict__ data, T* output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        output[i] = data[i];
    }
}

// Read-modify-write (add)
template <typename T>
__global__ void readModifyWriteKernel(T* __restrict__ src, T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i] + src[i] * 2;
    }
}

// Streaming store benchmark - measures write combine bandwidth
template <typename T>
__global__ void streamingStoreKernel(const T* __restrict__ src, T* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

// Random read pattern (simulated with thread-local offset)
template <typename T>
__global__ void randomReadKernel(T* __restrict__ data, T* output, size_t N, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int local_seed = seed ^ (idx * 0x9e3779b9);
    size_t offset = local_seed % N;

    output[idx] = data[offset];
}

