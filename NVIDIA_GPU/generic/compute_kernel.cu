#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Compute throughput benchmark kernels

// FMA (fused multiply-add) - measures floating point throughput
template <typename T>
__global__ void fmaKernel(const T* __restrict__ a, const T* __restrict__ b,
                          const T* __restrict__ c, T* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        output[i] = a[i] * b[i] + c[i];
    }
}

// Integer arithmetic throughput
__global__ void intArithmeticKernel(const int* __restrict__ a,
                                   const int* __restrict__ b,
                                   int* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        int t1 = a[i] * b[i];
        int t2 = a[i] + b[i];
        int t3 = a[i] - b[i];
        output[i] = (t1 + t2) ^ t3;
    }
}

// Half-precision FMA (FP16)
__global__ void fp16FmaKernel(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               const __half* __restrict__ c,
                               __half* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        output[i] = __hfma(a[i], b[i], c[i]);
    }
}

// Branch divergence test
__global__ void branchDivergenceKernel(const int* __restrict__ cond,
                                       int* __restrict__ output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        if (cond[i] > 0) {
            output[i] = cond[i] * 2;
        } else if (cond[i] < 0) {
            output[i] = cond[i] / 2;
        } else {
            output[i] = 0;
        }
    }
}

// Memory coalesced access test
template <typename T>
__global__ void coalescedAccessKernel(const T* __restrict__ input,
                                      T* __restrict__ output, size_t N) {
    // Each thread reads consecutive elements
    size_t tid = threadIdx.x;
    size_t blockStart = blockIdx.x * blockDim.x * 4;  // 4 elements per thread

    T sum = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        size_t idx = blockStart + tid + j * blockDim.x;
        if (idx < N) {
            sum += input[idx];
        }
    }
    output[blockIdx.x * blockDim.x + tid] = sum;
}

// Shared memory bank conflict test
template <int BLOCK_SIZE>
__global__ void sharedMemoryKernel(const float* __restrict__ input,
                                   float* __restrict__ output, size_t N) {
    __shared__ float shared[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Load to shared memory
    if (idx < N) {
        shared[tid] = input[idx];
    }
    __syncthreads();

    // Process with shared memory access
    if (tid > 0 && tid < BLOCK_SIZE - 1) {
        output[idx] = shared[tid - 1] + shared[tid] + shared[tid + 1];
    } else if (idx < N) {
        output[idx] = shared[tid];
    }
}

